# Chunk-parallel Gated-DeltaNet (GDN) for DFlash speculative VERIFY

Status: DESIGN + reviewable CUDA skeleton (no GPU available; not yet compiled/validated).
Scope: the **verify** path only — single sequence (`n_seqs == 1`), a block of `N` tokens
(N = draft-max + 1, up to ~16, design supports up to 32). Prefill and single-token decode keep
using the existing sequential kernel.

Files:
- existing sequential kernel: `ggml/src/ggml-cuda/gated_delta_net.cu`
- CPU reference (exact math): `ggml/src/ggml-cpu/ops.cpp`
  (`ggml_compute_forward_gated_delta_net_one_chunk`)
- new skeleton: `ggml/src/ggml-cuda/gated_delta_net_chunked.cu`
- dispatch hook: `ggml/src/ggml-cuda/ggml-cuda.cu` (`GGML_OP_GATED_DELTA_NET`, ~L2934)

---

## 1. The exact recurrence (from the CPU reference)

Per head, per sequence. Let `S_k = S_v = D` (the existing kernel assumes square state; head dim
`D in {16,32,64,128}`). The recurrent state is a `D x D` matrix `S` with `S[i][j]`, where `i` indexes
the **key** dimension and `j` indexes the **value** dimension.

> Storage detail: both the CPU ref and the CUDA kernel store `S` **transposed** as
> `M[j][i] = S[i][j]` so that "row j of M" (contiguous) is "column j of S". The math below is in the
> mathematical `S[i][j]` convention; the kernel maps it to the transposed layout.

Inputs at token `t` (all f32): `q_t, k_t in R^D` (key dim), `v_t in R^D` (value dim),
`beta_t in R` (scalar), gate `g_t`:
- **scalar gate** (non-KDA): `g_t in R`, decay `a_t = exp(g_t)` applied to the whole state.
- **KDA / vector gate**: `g_t in R^D` indexed by the **key** dim `i`, decay `a_t[i] = exp(g_t[i])`
  applied per key-row.

The update (matching `ops.cpp` lines 10522-10552 exactly):

```
1. decay:   S[i][j] <- a_t[i] * S[i][j]                 (a_t[i] = exp(g_t[i]); scalar: a_t[i]=exp(g_t) for all i)
2. kv:      u_t[j]  =  sum_i S[i][j] * k_t[i]   =  (S^T k_t)[j]
3. delta:   d_t[j]  =  (v_t[j] - u_t[j]) * beta_t
4. update:  S[i][j] <- S[i][j] + k_t[i] * d_t[j]        (rank-1: S += k_t d_t^T)
5. output:  o_t[j]  =  scale * sum_i S[i][j] * q_t[i] = scale * (S^T q_t)[j]      (scale = 1/sqrt(D))
```

Substituting (3) into (4), with `S_{t-1}` the post-(prev-token) state and `S_t` the post-update
state, this is the **gated delta rule**:

```
S_t = diag(a_t) S_{t-1}  +  k_t ( beta_t (v_t - (diag(a_t) S_{t-1})^T k_t) )^T
o_t = scale * S_t^T q_t
```

Define the **effective value** (a.k.a. "new value" / pseudo-value in DeltaNet) so the update becomes
a plain (non-recursive-in-S) rank-1 add:

```
w_t   = beta_t * k_t                                   (D, key dim)         <- write key
u_t   = (diag(a_t) S_{t-1})^T k_t                      (D, value dim)       <- what's already stored
d_t   = beta_t * v_t  -  beta_t * u_t                  (D, value dim)       <- delta value
S_t   = diag(a_t) S_{t-1}  +  k_t d_t^T
```

`o_t` reads `S_t` (post-update, includes the current token — note step 5 runs *after* step 4).

This is the per-token recurrence the existing CUDA kernel runs `N` times sequentially in the verify
block. Cost of verify ~ `N` sequential steps, each `O(D^2)` work. We want to cut the **sequential
depth** from `N` to `N/C`.

---

## 2. Chunked derivation (chunked delta-rule / chunked linear attention)

Reference: FLA `chunk_delta_rule` / `chunk_gated_delta_rule`; Yang et al. "Parallelizing Linear
Transformers with the Delta Rule over Sequence Length" (DeltaNet) and "Gated DeltaNet".

Split the `N` tokens into chunks of size `C` (e.g. `C = 16`, so a 16-token verify block is **one
chunk**; a 32-token block is two). Index tokens within a chunk by `r = 0..C-1` (global token
`t = chunk_base + r`). Let `S_in` be the state entering the chunk (the carry).

### 2.1 Cumulative gate products inside the chunk

For the **KDA / vector gate**, the per-key-dim decay is multiplicative, so define inclusive cumulative
products along the chunk (per key dim `i`):

```
A_r[i] = prod_{s=0..r} a_s[i]          (inclusive, decay applied up to and including token r)
```

with `A_{-1}[i] = 1`. The decay from "just after token s applied" to "the chunk boundary after token
C-1" is `A_{C-1}[i] / A_s[i]`. For the **scalar gate**, `a_s` is a scalar and `A_r` collapses to a
scalar per token — same formulas, broadcast over `i`.

To keep the rank-1 writes commutable, **pre-scale** each token's write key into a common reference
frame (the chunk start). Define:

```
k~_r[i] = k_r[i] / A_r[i]              (deflated write key — "undo" the decay it will accumulate)
q~_r[i] = q_r[i] * A_r[i]             (inflated query  — apply decay the carry would have gotten)
```

Intuition: a rank-1 contribution `k_s d_s^T` written at token `s` gets multiplicatively decayed by
`A_{r}[i]/A_s[i]` (key dim) by the time we read at token `r >= s`. Folding `1/A_s` into the key and
`A_r` into the query realizes that decay through a single elementwise scale per token, so the
intra-chunk interactions become plain matmuls. (This is exactly the FLA "secondary chunking" trick;
do the cumprod in log space — see section 5.)

### 2.2 Intra-chunk parallel form

Stack the chunk into matrices (rows = tokens within the chunk):
`K, Q, V in R^{C x D}` (rows `k_r, q_r, v_r`), `K~, Q~` the deflated/inflated versions, `beta in R^C`.

**(a) Carry read (contribution of `S_in` to every token's `u` and `o`):**

The "already stored" value seen by token `r` from the *incoming* state is
`(diag(A_r) S_in)^T k_r = S_in^T (A_r (.) k_r)`. So with `Kbar_r = A_r (.) k_r` (the **inflated** read
key) stacked into `Kbar in R^{C x D}`:

```
U_carry = Kbar @ S_in            in R^{C x D}    (each row = u_r^carry, value dim)
```

The carry contribution to the output uses the *post-update* state, but since `S_in` is constant
within the chunk its output contribution is `O_carry = scale * (Qbar @ S_in)` with
`Qbar_r = A_r (.) q_r`.

**(b) Intra-chunk token-token interactions (the delta-rule coupling):**

Within the chunk, token `r`'s delta `d_r` depends on the writes of all earlier tokens `s < r` (and on
the carry). Build the **strictly-lower-triangular** decayed attention matrix between deflated keys:

```
T[r][s] = beta_r * ( k~_r . k~_s )      for s < r,   else 0          in R^{C x C}
```

The delta-rule "un-mixing" is the classic `(I + tril(T,-1))^{-1}` solve (forward substitution over the
chunk, `C` sequential micro-steps but only on a `C x C` system, cheap and in shared memory). Let

```
W = (I + strict_tril(T))^{-1}                                  in R^{C x C}
Dmat = W @ ( beta (.) (V - U_carry) )                          in R^{C x D}   (rows = d_r, the resolved deltas)
```

(`beta (.) V` is row-scaling `V` by `beta_r`; `U_carry` from (a).) `Dmat` rows are exactly the
per-token delta values `d_r` consistent with the sequential recurrence — now computed by two matmuls +
one small triangular solve instead of `C` rank-1 steps.

**(c) Per-token output:**

```
O = O_carry  +  scale * tril( Q~ @ K~^T ) @ Dmat              in R^{C x D}
```
The `tril(Q~ K~^T)` term sums the intra-chunk writes that token `r` should see. The output reads the
**post-update** state, so the current token's own write (`s == r`) must be included — use the
lower-triangle **including** the diagonal for this output term, while the `T` solve in (b) stays
**strictly** lower. Mapping the exact diagonal handling is the one subtlety to nail against the
reference (see section 6 validation).

### 2.3 Inter-chunk state carry (the only sequential part)

After the chunk, the new boundary state:

```
S_out = diag(A_{C-1}) S_in  +  Kw^T @ Dmat
      = diag(A_{C-1}) S_in  +  sum_r ((A_{C-1}/A_r) (.) k_r) d_r^T
```
with `Kw_r = (A_{C-1}/A_r) (.) k_r` stacked into `Kw in R^{C x D}` (each write key carried forward to
the chunk end). This is one `D x D` update per chunk.

**Sequential depth = number of chunks = ceil(N/C).** With `N <= C` (verify block <= 16 and `C = 16`)
the whole verify is **a single chunk**: zero inter-chunk recurrence, everything is matmuls + one
`C x C` triangular solve. That is the win.

---

## 3. CUDA kernel structure (`gated_delta_net_chunked_cuda`)

One CUDA **block per (head, sequence)** — for verify `sequence` is fixed (n_seqs==1), so grid is
`(H, 1, 1)`. Each block owns the chunk's `C x D` tiles and the `D x D` carry state in shared memory.

Tiling (for the verify regime: `C <= 32`, `D in {16,32,64,128}`):
- Shared mem holds: `S` (`D x D` f32), `K,Q,V,K~,Q~,Kbar,Qbar` chunk tiles (`C x D` each), `T`/`W`
  (`C x C`), `Dmat` (`C x D`), `A` cumprods (`C x D` for KDA, `C` for scalar). For `D=128, C=16` that
  is `128*128*4 = 64KB` for `S` alone — at the edge of the 48-96KB smem budget, so for `D=128` either
  keep `S` in registers (sharded across the warp like the existing kernel) or cap `C` smaller / use
  the host-decomposition fallback (3.2). For `D <= 64` everything fits comfortably.
- Threads: a 2D thread block, `D` lanes x `num_warps` (mirror the existing
  `block_dims(min(warp,D), num_warps)`). Matmuls are done cooperatively; the `C x C` triangular solve
  is done by a single warp (C <= 32 fits one warp) via forward substitution.

Phases inside the kernel (single chunk; the multi-chunk loop wraps phases 2-6):
1. **Load + gate cumprod.** Load `g`, compute `a_r = exp(g_r)`, inclusive cumprod `A_r` along the
   chunk **in f32 / log space** (Hillis-Steele scan across `C`). Build `k~,q~,Kbar,Qbar,Kw`.
2. **U_carry = Kbar @ S_in**, **O_carry = scale . (Qbar @ S_in)** — two `C x D . D x D` matmuls.
3. **T = strict_tril(beta (.) (K~ K~^T))** — a `C x D . D x C` matmul, mask to strict lower.
4. **Solve `W (I+T)`:** forward-substitution to get `Dmat = (I+T)^{-1} (beta (.) (V - U_carry))`
   (C sequential micro-steps on the small `C x C` system, one warp).
5. **Output:** `O = O_carry + scale . tril(Q~ K~^T) @ Dmat`; write `O` rows to `attn_data` (same
   `[S_v.H]`-strided layout as the sequential kernel) and, if `trace != nullptr`, materialize the
   per-token state trace (see 3.3).
6. **Carry:** `S_out = diag(A_{C-1}) S_in + Kw^T @ Dmat`; write back transposed `M[j][i]`.

### 3.1 Dispatch hook

In `ggml/src/ggml-cuda/ggml-cuda.cu`, `GGML_OP_GATED_DELTA_NET` (~L2934) currently calls
`ggml_cuda_op_gated_delta_net`. Add inside that op (in `gated_delta_net.cu`'s
`ggml_cuda_op_gated_delta_net`) a guarded fast path:

```
if (n_seqs == 1 && n_tokens >= GDN_CHUNK_MIN && n_tokens <= GDN_CHUNK_MAX && S_v <= GDN_CHUNK_DMAX)
    launch_gated_delta_net_chunked<KDA>(...);     // new path (verify block)
else
    launch_gated_delta_net<KDA>(...);             // existing sequential path (prefill / single decode)
```

`GDN_CHUNK_MIN` ~ 2 (no point for a single token), `GDN_CHUNK_MAX` ~ 32, `GDN_CHUNK_DMAX` initially
64 (raise to 128 once the smem/register strategy for `D=128` is validated). The trace output and the
final-state writeback use the **same** dst layout (`[attn_scores | new_states]`) and the same
transposed state convention, so nothing downstream changes.

### 3.2 Host-decomposition fallback

If a single monolithic kernel is too much for a first cut, the same math maps onto existing ggml CUDA
ops as a host-side graph (per head, single chunk): `ggml_mul_mat` for `Kbar@S`, `Q~K~^T`, `Kw^T@Dmat`;
elementwise muls for gating; a tiny custom kernel only for the `C x C` triangular solve. Slower than
the fused kernel (extra global-memory round trips) but a correctness oracle and a quick path to a
working verify. The skeleton notes this decomposition.

### 3.3 Trace compatibility (DFlash rewind)

DFlash needs the **per-token** state `S_t` for partial-acceptance rewind (`src[6]` trace). The chunked
kernel does not naturally produce per-token `S_t` (it jumps chunk->chunk). Two options:
- **(preferred)** After computing `Dmat`, materialize `S_r = diag(A_r) S_in + Kw(->r)^T @ Dmat[0..r]`
  for each `r` via a small cumulative pass (the prefix of the chunk update) and write the trace rows.
  Costs `C` light steps but they're independent across `r` (can be a parallel segmented scan).
- **(fallback)** When `trace != nullptr`, route to the sequential kernel (it already writes the trace
  near-free). Verify still benefits whenever the harness doesn't request a trace; partial-accept paths
  pay the sequential cost. Start here, then implement the prefix-trace.

---

## 4. Expected speedup

- Sequential kernel verify cost ~ `N` sequential GDN steps (latency-bound: each step's rank-1 update
  + two reductions depends on the previous). For the 24 GDN layers this is the dominant verify cost
  and is why single-stream speedup caps at ~1.5-1.7x.
- Chunked: sequential **depth** drops to `ceil(N/C)`. For `N <= 16, C = 16` -> **depth 1**. The
  remaining work is matmuls (`C x D x D`) + one `C x C` solve, which are throughput-bound and overlap
  well; on a modern GPU the `C x D x D` matmuls for `C,D <= 128` are far below peak and hide behind
  issue latency.
- Net: verify GDN latency goes from `O(N)` serial to `O(N/C)` serial + parallel intra-chunk. This is
  the same structural change that lets SGLang reach ~3x — we expect the single-stream cap to move from
  ~1.5-1.7x toward the ~2.5-3x regime, gated by how much of end-to-end time is GDN verify vs the
  full-attention layers and sampling.
- The arithmetic *work* slightly increases (the `(I+T)^{-1}` solve + extra matmuls), but it converts
  serial-dependent work into parallel work, which is the right trade for a latency-bound verify.

---

## 5. Numerical-stability concerns

- **fp32 accumulation everywhere** — mirrors the top-level CLAUDE.md lesson (fp16 mean-pool overflowed
  to +/-inf on attention-sink channels and poisoned every downstream lstsq). All cumprods, matmul
  accumulators, the `C x C` solve, and the state must accumulate in **f32** (the sequential kernel is
  already all-f32; keep parity). Never down-cast the state or the gate products to fp16.
- **Cumulative gate product underflow/overflow.** `A_r[i] = prod a_s[i]` with `a_s = exp(g_s)`. Over a
  chunk of 16 tokens with strongly negative `g` (heavy decay), `A_{C-1}` can underflow and the
  deflated key `k~_r = k_r / A_r` can blow up — the classic instability the FLA chunked kernels guard.
  Mitigations: (a) keep `C` modest (16) so the product spans few tokens; (b) work in **log space** for
  the cumulative gate (`L_r[i] = sum_{s<=r} g_s[i]`, then `A_r = exp(L_r)`, and form ratios as
  `exp(L_r - L_s)` rather than dividing two exponentials) — this is the numerically safe way to get
  `A_r/A_s` and `A_{C-1}/A_r` without ever materializing a tiny denominator; (c) f32 throughout.
- **Triangular solve conditioning.** `(I + strict_tril(T))` is unit-lower-triangular, so always
  invertible and forward-substitution is stable; just accumulate in f32.
- **Diagonal/self-token bookkeeping** is the main *correctness* (not stability) risk — the output reads
  the **post-update** state, so the current token's own write must be included. Validate against the
  reference (section 6) rather than reasoning it through once.

---

## 6. Step-by-step plan to production-correct + validation

1. **CPU oracle first.** Implement the chunked math as a second CPU function next to
   `ggml_compute_forward_gated_delta_net_one_chunk` (or a standalone test harness) and assert it
   reproduces the sequential CPU reference (f32, `|delta| < 1e-4` per element) on random
   `q,k,v,g,beta,S_in` for both scalar and KDA gates, for `D in {16,32,64,128}` and
   `C in {1,2,4,8,16}`. This nails the diagonal/self-token and the gate-ratio direction *before* any
   CUDA.
2. **Single-chunk CUDA kernel** for `n_seqs==1`, `N==C`, `D <= 64`. Compare its `attn` output and
   `S_out` against the sequential CUDA kernel on the same inputs (host-side max-abs diff `< 1e-3` f32).
3. **Multi-chunk loop** (`N` = a few chunks); re-check the inter-chunk carry matches sequential.
4. **Trace path** (3.3 preferred): verify the per-token trace rows equal the sequential kernel's trace
   element-for-element (this is what DFlash rewind reads — must match exactly).
5. **D=128 strategy**: pick register-sharded `S` (like the existing kernel) or smem; re-validate.
6. **Wire dispatch** behind the `n_seqs==1 && N in [MIN,MAX] && D <= DMAX` guard; keep the sequential
   path as the default so prefill/decode are untouched. Add an env/define kill-switch
   (`GGML_CUDA_GDN_CHUNKED=0`) to fall back at runtime during bring-up.
7. **End-to-end**: run the Qwen3.5-4B DFlash verify on a real prompt, confirm accepted-token sequences
   are identical to the sequential-verify build (greedy + fixed seed), then measure tok/s. Validation =
   *identical accepted tokens* + improved verify latency.

Validation harness lives alongside the existing GDN tests (search `test-backend-ops` /
`gated_delta_net` test cases); add a chunked-vs-sequential equivalence case there.

---
## CORRECTION (validated by gdn_chunked_oracle.py, bitwise vs sequential, max err ~1e-13)

The pairwise inter-token decay (s -> r) is **A_r / A_s**, NOT 1/(A_r·A_s). So the deflation must be
ASYMMETRIC: the LATER token r carries A_r (Kbar/Qbar = A⊙k, A⊙q), the EARLIER token s carries 1/A_s
(Ktil = k/A). The dot Kbar_r·Ktil_s = sum_i k_r k_s · A_r/A_s (bounded ≤1 for s<r). Corrected forms:

  Kbar = A⊙k ,  Qbar = A⊙q ,  Ktil = k/A          (A_r[i] = prod_{s<=r} a_s[i], inclusive)
  U_carry = Kbar @ S0 ,  O_carry = scale·(Qbar @ S0)
  T   = strict_tril( beta ⊙ (Kbar @ Ktil^T) )       # NOT Ktil@Ktil  (that blows up ~1e9 at N=16)
  Dmat = (I+T)^-1 @ ( beta ⊙ (V - U_carry) )         # unit-lower-tri solve / fwd-subst
  O   = O_carry + scale · tril(Qbar @ Ktil^T, incl-diag) @ Dmat    # NOT Qtil@Ktil
  S_out = diag(A_end)·S0 + Kw^T @ Dmat ,  Kw = (A_end/A)⊙k

NUMERICAL NOTE: Ktil = k/A can overflow fp32 when A is tiny (strong decay × long block). The kernel
should form the bounded ratio A_r/A_s directly in log space (exp(L_r - L_s), L = cumsum(g)) rather
than materialize k/A. The oracle uses explicit k/A and stays exact for mild gates (a~0.9, N<=16);
validate the log-space path against it before shipping. Bring-up order: (1) this oracle is the gold
reference; (2) port to a CUDA single-chunk kernel (N<=C, one block/head); (3) bitwise vs sequential.
