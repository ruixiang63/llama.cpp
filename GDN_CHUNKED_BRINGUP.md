# Chunked GDN verify — bring-up recipe (math validated, ggml-op decomposition)

The chunked math is bitwise-correct (gdn_chunked_oracle.py vs sequential, err ~1e-13). It needs NO
hand-written CUDA kernel: ggml has cumsum + tri + solve_tri + mul_mat on BOTH CPU and CUDA. Build the
verify GDN (n_seqs==1, N=draft_max+1 ≤ ~16, single chunk) as a ggml subgraph; validate on the CPU
backend locally (build/bin/libggml*.dylib already built — write a standalone test linking it), then
it runs on CUDA for free.

## Inputs (from ggml_gated_delta_net): q,k,v [S_v,H,N,1]; g [S_v,H,N,1] (kda); beta [1,H,N,1];
## state S0 [S_v,S_v,H,1]. Output [S_v*H, N + S_v] (cols 0..N-1 = attn, cols N..N+S_v-1 = new state).

## Op recipe (per the VALIDATED oracle; A_r[i]=prod_{s<=r} a_s, a=exp(g)):
1. Permute q,k,v,g to per-head token-matrices Xp [S_v, N, H] (ne0=dim i, ne1=token r, ne2=head).
2. A: put tokens on ne0 -> g2 [N, S_v, H]; L = ggml_cumsum(g2) over ne0 (tokens); A = exp(L); 
   permute A back to [S_v, N, H]. (cumsum is ne0-only, hence the shuffle.)
3. Kbar = A⊙kp ; Qbar = A⊙qp ; Ktil = kp ⊙ exp(-L_perm)   (= kp/A, but form via exp(-L) to stay fp32-safe).
4. U_carry = mul_mat(S0[i,j,H], Kbar[i,r,H]) -> [j, r, H]  (contracts i). O_carry = scale·mul_mat(S0, Qbar).
5. KK = mul_mat(Ktil[i,s,H], Kbar[i,r,H]) -> [s, r, H]; want T[r,s]=beta_r·KK[s,r]. 
   Tfull = beta(broadcast over s) ⊙ transpose(KK to [r,s,H]); T = ggml_tri(Tfull, LOWER strict).
6. rhs = beta ⊙ (vp_as[r,j] - U_carry[j,r]^T)  -> shape [j? r?]; keep as [N, S_v, H] (r on ne0) for solve.
   Dmat = ggml_solve_tri(A=I+T [N,N,H] unit-lower, B=rhs, left=true, lower=true, uni=true) -> [N, S_v, H].
7. QK = ggml_tri(transpose(mul_mat(Ktil,Qbar)) to [r,s,H], LOWER_DIAG incl diagonal);
   O = O_carry + scale·mul_mat(QK[s,r,H]?, Dmat[s,j,H]) -> [r,j,H]  (contracts s). 
8. S_out[i,j,H] = (A_end[i] ⊙ S0[i,j]) + mul_mat(Kw[i? ], Dmat) ; Kw_r = (A_end/A_r)⊙k_r.
9. Reassemble into the [S_v*H, N+S_v] output layout (attn cols = O reshaped, state cols = S_out).

## Validation (do BEFORE wiring into the model):
- Standalone CPU test: random q,k,v,g,beta,S0; run ggml_gated_delta_net (sequential, reference) and
  build_gated_delta_net_chunked; ggml_backend_cpu; compare max|diff| < 1e-4. Iterate layout bugs here
  (mul_mat is a^T b contracting ne0; transposes/permutes are where bugs hide).
- Then bitwise-vs-sequential for the TRACE rows too (DFlash rewind needs per-token state; the chunked
  path gives only the final S_out + per-token O, NOT per-token state -> for the rewind we still need
  per-token states. EITHER also emit per-token states (S after each token = O_carry-style partial), OR
  keep trace on the sequential path. RESOLVE THIS before shipping: the rewind/promote depends on the
  per-token trace; chunked must reproduce it or the verify can't use trace+promote.)

## Then: wire as the verify fast path (n_seqs==1, N small) behind a flag; bench draft-max 8/12/16 on
## Blackwell/H100; expect verify cost ~flat in N -> larger blocks affordable -> accept_len ~6 -> ~2.5-3x.

## OPEN RISK (important): the DFlash rewind (trace+promote) needs per-TOKEN states. The chunked form
## naturally yields only the chunk-final state. Per-token states within the chunk can be recovered
## (S_t = decay(S0,t) + intra-chunk updates up to t) but that's extra work; OR run chunked for speed
## and the sequential trace only when a partial-accept rewind is actually needed. Must be designed.

---
## STATUS: chunked GDN ggml-graph VALIDATED (commit 9c1f082b8). Integration plan below.

tests/test-gdn-chunked.cpp: chunked-vs-sequential ALL PASS (N=1..16, S_v=64/128, fp32). The graph is
portable (cumsum/tri/solve_tri/mul_mat on CPU+CUDA; the path that also extends Metal/Vulkan/WebGPU).

### Wiring (next):
1. Lift build_chunked() from the test into a reusable builder, e.g. build_gated_delta_net_chunked()
   in src/models/ (or a ggml helper), returning the SAME [S_v*H, N+S_v] output as ggml_gated_delta_net.
2. In src/models/qwen35.cpp build_layer_attn_linear: when (n_seqs==1 && n_seq_tokens>1 && verify),
   call the chunked builder instead of ggml_gated_delta_net. Keep the fused sequential op for prefill
   and single-token decode (chunked wins only for a multi-token block).
3. Gate behind a flag (e.g. cparams.gdn_chunked or env) so it's opt-in until GPU-validated.

### TRACE / rewind (the one real design decision):
DFlash rewind (trace+promote) needs the per-TOKEN state S_t for the accepted position. The chunked
path yields only S_out (after all N). Resolution: keep the chunked path for the fast verify FORWARD
(attn + S_out); on a PARTIAL accept (acc<N, ~38% of rounds), recompute S_acc by running the chunked
builder on just tokens [0..acc] (acc<=N, cheap) and promote that. Full-accept rounds (the majority)
never touch the trace. This removes the per-token trace buffer + its O(N) cpy nodes entirely.
ALT: emit per-token states from chunked via a masked cumulative outer-product (S_t = A_t*(S0 + prefix
of Ktil_s (x) Dmat_s)) - exact but O(N*D*D), negating part of the win. Prefer recompute-on-partial.

### Validation on GPU (after wiring):
- bitwise: identical accepted tokens vs the sequential-GDN build (LLAMA_SPEC_TRACE path), greedy.
- speed: draft-max 8/12/16 on Blackwell/H100 - verify cost should go ~flat in N -> accept_len ~6 ->
  toward 2.5-3x (the SGLang regime). This is the payoff.

### NUMERICAL: fp32 err grows mildly with N,D (1e-5 at N16/D128). On GPU keep fp32 accumulation in the
matmuls/solve. If a stronger-decay model underflows k/A, switch to the log-space ratio form (DESIGN.md).

---
## REASSEMBLY into the [S_v*H, N+S_v] combined output (the drop-in detail, derived + checked)
ggml_gated_delta_net's data = [attn region (S_v*H*N) | state region (S_v*S_v*H)], flat:
  attn[h,t,j]  at (t*H + h)*S_v + j        -> column-major [S_v*H, N], row=h*S_v+j, col=t
  state[h,i,j] at attn_elems + h*S_v*S_v + j*S_v + i  -> EXACTLY a [S_v(i),S_v(j),H] contiguous tensor
So the chunked builder's cs ([i,j,H], already contiguous) IS the state region as-is. For attn:
  O is [t, j, H] -> permute(2,0,1,3) -> [j, H, t] -> cont -> reshape [S_v*H, N].
Combined = reshape( ggml_concat( reshape(O_attn,1D[S_v*H*N]), reshape(cs,1D[S_v*S_v*H]), dim0 ),
                    [S_v*H, N+S_v] ). Drop-in for ggml_gated_delta_net's result.

## GQA: q/k have num_k_heads (ssm_n_group), v/g/beta/state have num_v_heads (ssm_dt_rank). In the
## builder: ggml_repeat q/k from Hk to H (interleaved h%Hk) BEFORE the per-head ops. VALIDATED.

## DONE this session: chunked GDN ggml builder VALIDATED on CPU bitwise vs sequential, incl GQA
## (tests/test-gdn-chunked.cpp, ALL PASS N=1..16, S_v=64/128, H_v=4, H_k=1/2). It is portable
## (CUDA/Metal/Vulkan/WebGPU via the op kernels). REMAINING: (1) lift build_chunked into
## delta-net-base.cpp + reassembly above, gate by env, fall back to sequential when gdn_trace!=null;
## (2) GPU build + bitwise vs sequential + draft-max 8/12/16 speed (verify ~flat in N -> ~3x);
## (3) trace/rewind: compute S_acc on partial-accept from kept A/Ktil/Dmat (no per-token buffer).

---
## GPU BRING-UP RESULTS (eva01 V100, Qwen3.5-4B-Q8_0, 24 GDN + 8 attn) — DECISIVE, lever #3 verdict

Ran the wired chunked path on CUDA. Three findings settle whether chunked GDN is a speedup lever:

1. **Qwen3.5 GDN uses a SCALAR gate, not the vector (KDA) gate.** Gate diagnostic at the dispatch
   site: `g->ne[0]=1, S_v=128`. The builder + tests/test-gdn-chunked.cpp were written for the
   per-channel VECTOR gate (`g->ne[0]==S_v`), so the original `g->ne[0]==S_v` trigger NEVER fired on
   Qwen3.5 — every "chunked vs seq" comparison before this was a silent no-op (identical numbers).
   Fix: generalized the builder to accept the scalar gate (A=[1,N,H] broadcasts across S_v; AendB
   width from A->ne[0]; full-size tensor first in every A-multiply so [1,N,H] repeats into [S_v,N,H]).
   Now fires: `[GDN-CHUNKED] active: N=16 S_v=128 H=32 Hk=16 gate=scalar`, assert-free on CUDA.

2. **Chunked is SLOWER than the fused CUDA kernel** (llama-bench pp, t/s, r=8):
   | N (pp) |  fused (off) | chunked (on) |
   |--------|-------------:|-------------:|
   | 8      | 514          | 413          |
   | 16     | 830          | 672          |
   | 32     | 1373         | 1129         |
   | 64     | 1507         | 1343         |
   CUDA already ships a native `GGML_OP_GATED_DELTA_NET` kernel (ggml-cuda/gated_delta_net.cu) that
   the default multi-token path uses; the decomposed ggml graph (cumsum/exp/diag/tri/solve_tri +
   several mul_mat) cannot beat it. **Worse, GDN is not the verify bottleneck at all**: baseline pp
   throughput is identical whether the GDN op is the fused kernel or not — the 4B transformer's
   attention+MLP matmuls dominate at N<=64, the GDN scan over <=16 tokens is negligible.

3. **Single-chunk is numerically valid only for SMALL N.** The builder treats the whole block as ONE
   chunk, so `Ainv = exp(-cumsum(g))` overflows for long sequences. A chat-template-inflated prefill
   (~30-40 tok) already produces `?????` garbage under LLAMA_GDN_CHUNKED=1. It is correct only for a
   true small verify block (<=16, as the CPU test covers); using it for prefill needs real multi-chunk
   tiling (chunk into 16-64 blocks) which is NOT implemented.

### VERDICT (lever #3): chunked GDN is a PORTABILITY artifact, NOT a speed lever.
- On CUDA it loses to the fused kernel and GDN isn't the bottleneck -> zero speedup toward SGLang's ~3x.
- Its only real use is backends WITHOUT a fused GDN kernel (WebGPU/Metal/Vulkan) AND only for small
  blocks. For the speedup goal, the real lever is #2 (CUDA graphs for cheap large-block verify),
  not GDN chunking.
- Code kept opt-in behind LLAMA_GDN_CHUNKED (default OFF, gdn_trace==null only) so normal serving is
  untouched. Scalar-gate model-path correctness at small N is UNVERIFIED (llama-cli is confounded by
  template prefill length + single-chunk overflow); do not claim it correct without a unit harness.

---
## PORTABLE + CORRECT (multi-chunk tiling) — what it took to make chunked GDN actually work on CUDA

Goal: a pure-ggml chunked GDN that is CORRECT on every backend (verify path for fused-kernel-less
backends: WebGPU/Metal/Vulkan). Two deliverables: a unit test proving correctness, and multi-chunk
tiling for sequences longer than one block. Both done; the road there found three real issues.

1. **Scalar gate proven correct (CPU + CUDA).** tests/test-gdn-chunked.cpp now covers the scalar
   (Gated DeltaNet, g->ne[0]==1) gate as well as the vector (KDA) gate, GQA, and runs on a SELECTABLE
   backend (GDN_BACKEND=CUDA). Match vs ggml_gated_delta_net is ~1e-8 (fp32) on both CPU and CUDA.

2. **Multi-chunk tiling.** build_delta_net_chunked tiles the N tokens into blocks of C, threads the
   recurrent state forward, concats the per-block attn. For a verify block (N<=C) the loop runs once
   = the original single-chunk path. Capped at N<=128 in the dispatch gate (LLAMA_GDN_CHUNKED_MAXN):
   the loop unrolls ceil(N/C) subgraphs per layer, so a long prefill (the 512-token ubatch reserve)
   would explode the static graph (GGML_ASSERT(obj_new) — ctx pool exhausted) -> fall through to the
   fused op there. On CUDA the fused kernel is faster for long prefill anyway (see verdict above).

3. **The test's random inputs hid TWO bugs that only the real model exposed:**
   - **Unnormalized keys diverge.** Random k with ||k||^2 ~ S_v*sc^2 >> 2 violates the delta-rule
     stability bound beta*||k||^2 < 2, so the TRUE recurrence diverges and ref vs chunked blow up in
     the unstable directions for long N (looked like a tiling bug: error grew ~30%/token). Fix:
     l2-normalize q,k in the test (delta-net normalizes them) -> stable, machine-eps match.
   - **Deflation precision sets a small max chunk size.** A=exp(+/-cumsum(g)) has wide dynamic range;
     Ktil=k*exp(-cumsum), Kbar=k*exp(+cumsum) individually span many orders of magnitude even though
     their product is bounded, so the KK matmul loses fp32 precision when a chunk is too long. The
     test used mild gates (g~-0.2) and passed at C=16/32; the REAL model has strong-decay heads and
     **garbles at C>=16, is clean at C<=8** (verified: greedy output identical to the fused baseline
     at C=4 and C=8; "??????" at C=16/24). **Deployed default C=8.** The test now uses C=8 to match.

### Net: chunked GDN is CORRECT on CUDA (greedy output bit-identical to the fused kernel across
short/medium/long prompts) and portable. Confirmed NOT a speedup on CUDA (the fused kernel is faster
and GDN isn't the bottleneck); its role is the verify path on backends without a fused GDN kernel.
Validation harness: tests/test-gdn-chunked.cpp (GDN_BACKEND=CPU|CUDA), all PASS on both.
