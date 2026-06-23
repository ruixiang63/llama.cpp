#include "gated_delta_net.cuh"

// NOTE: SKELETON ONLY - not yet compilable (pseudo-helpers/TODOs). Guarded out of the
// build so the branch compiles; see DESIGN.md. Remove this #if 0 once the kernel is
// brought up per the validation plan.
#if 0

// =============================================================================
// Chunk-parallel Gated-DeltaNet forward for the DFlash speculative VERIFY path.
//
// SCOPE: single sequence (n_seqs == 1), a block of N tokens (N = draft_max + 1,
//        up to ~32). Prefill and single-token decode keep using the existing
//        SEQUENTIAL kernel in gated_delta_net.cu. This file is the CHUNK-PARALLEL
//        variant that cuts the verify's sequential depth from N to ceil(N/C).
//
// STATUS: reviewable SKELETON. The structure, tiling, and math->thread mapping
//         are concrete; the cooperative matmul/scan/solve bodies are sketched
//         with pseudo-helpers (cg_* / smem tiles) and are NOT yet a compilable,
//         numerically-verified kernel. See DESIGN.md (repo root), sections 2-6,
//         for the derivation and the bring-up/validation plan.
//
// MATH (mathematical S[i][j] convention; i = key dim, j = value dim, D = head dim):
//   per token t, with decay a_t[i] = exp(g_t[i]) (KDA) or exp(g_t) (scalar):
//     decay:  S[i][j] <- a_t[i] * S[i][j]
//     kv:     u_t[j]   = sum_i S[i][j] * k_t[i]
//     delta:  d_t[j]   = (v_t[j] - u_t[j]) * beta_t
//     update: S[i][j] += k_t[i] * d_t[j]
//     out:    o_t[j]   = scale * sum_i S[i][j] * q_t[i]          scale = 1/sqrt(D)
//
//   chunked (chunk size C, S_in = state entering the chunk; see DESIGN.md 2.2):
//     A_r[i]   = prod_{s<=r} a_s[i]                  (inclusive cumprod, log space)
//     Kbar_r   = A_r (.) k_r        Qbar_r = A_r (.) q_r     (carry-read keys/queries)
//     k~_r     = k_r / A_r          q~_r   = q_r * A_r        (deflated/inflated)
//     Kw_r     = (A_{C-1}/A_r) (.) k_r                        (carry-forward write key)
//     U_carry  = Kbar @ S_in                                  [C x D]
//     T        = strict_tril( beta (.) (K~ K~^T) )            [C x C]
//     Dmat     = (I + T)^{-1} @ ( beta (.) (V - U_carry) )    [C x D]   (fwd-subst solve)
//     O        = scale*(Qbar @ S_in) + scale*tril(Q~ K~^T) @ Dmat   [C x D]
//     S_out    = diag(A_{C-1}) S_in + Kw^T @ Dmat             [D x D]
//
// STATE STORAGE: same transposed layout as the sequential kernel and CPU ref:
//   M[j*D + i] = S[i][j]  (row j of M is column j of S, contiguous).
//
// IMPORTANT (see top-level CLAUDE.md fp16 pooling lesson): ALL accumulation here
// is f32. Gate cumprods are done in LOG space (sums of g) and ratios formed as
// exp(L_r - L_s) so we never divide by a tiny exp() denominator.
// =============================================================================

// ---- tunables (mirror DESIGN.md 3.1) ---------------------------------------
#define GDN_CHUNK_C        16   // chunk size; a <=16-token verify block is ONE chunk
#define GDN_CHUNK_MIN       2   // below this, sequential decode wins
#define GDN_CHUNK_MAX      32   // largest verify block we accept on this path
#define GDN_CHUNK_DMAX     64   // start with D<=64; D=128 needs the register-shard variant

// Pseudo-helpers used in the sketch (to be replaced with real cooperative impls):
//   smem_matmul_AB(out, A, B, M, K, N) : out[MxN] = A[MxK] @ B[KxN], f32 accum, block-cooperative
//   smem_matmul_ABt(out, A, B, M, K, N): out[MxN] = A[MxK] @ B[NxK]^T
//   tri_mask_strict(M, C)              : zero the upper triangle incl. diagonal
//   tri_mask_incl(M, C)                : zero the strict upper triangle (keep diagonal)
//   warp_fwd_subst(W_or_inplace, T, C) : solve (I + strict_tril(T)) X = RHS by forward substitution

// -----------------------------------------------------------------------------
// One CUDA block per (head, sequence). For verify n_seqs==1 -> grid (H,1,1).
// Template on S_v (=D) and KDA exactly like the sequential kernel so the same
// dispatch switch can pick the instantiation.
// -----------------------------------------------------------------------------
template <int D, bool KDA>
__global__ void gated_delta_net_chunked_cuda(
        const float * q,          // [D, H, T]  (key dim, head, token)   strides sq*
        const float * k,          // [D, H, T]
        const float * v,          // [D, H, T]  (value dim, head, token)
        const float * g,          // [1|D, H, T]   gate (scalar or KDA vector over key dim)
        const float * beta,       // [1,  H, T]
        const float * curr_state, // [D, D, H]     incoming state S_in (transposed M[j][i])
        float *       dst,        // [attn_scores | new_states] (same layout as sequential kernel)
        float *       trace,      // optional per-token state trace (n_seqs==1), may be nullptr
        int64_t       H,
        int64_t       n_tokens,   // N (the verify block length)
        int64_t       sq1, int64_t sq2,   // q/k strides (floats): sq1 over head, sq2 over token
        int64_t       sv1, int64_t sv2,   // v strides
        int64_t       sb1, int64_t sb2,   // beta/g base strides
        float         scale) {

    const int h_idx = blockIdx.x;     // head this block owns
    const int lane  = threadIdx.x;    // 0..D-1  (value/key column)
    const int warp  = threadIdx.y;    // 0..num_warps-1

    // ---- shared-memory tiles (DESIGN.md 3, "Tiling") ------------------------
    // For D<=64, C=16 these fit in <=48KB. For D=128 use the register-shard
    // variant for S (like the sequential kernel) instead of smem S.
    __shared__ float s_S    [D][D];           // incoming/outgoing state (M[j][i] = S[i][j])
    __shared__ float s_K    [GDN_CHUNK_C][D]; // raw chunk tiles
    __shared__ float s_Q    [GDN_CHUNK_C][D];
    __shared__ float s_V    [GDN_CHUNK_C][D];
    __shared__ float s_L    [GDN_CHUNK_C][D]; // cumulative LOG gate L_r[i] = sum_{s<=r} g_s[i] (KDA)
                                              // (scalar gate: column 0 used, broadcast over i)
    __shared__ float s_beta [GDN_CHUNK_C];
    __shared__ float s_Ucar [GDN_CHUNK_C][D]; // U_carry
    __shared__ float s_T    [GDN_CHUNK_C][GDN_CHUNK_C]; // intra-chunk coupling / solve workspace
    __shared__ float s_Dmat [GDN_CHUNK_C][D]; // resolved per-token deltas d_r
    __shared__ float s_O    [GDN_CHUNK_C][D]; // outputs

    const int C = (int) (n_tokens < GDN_CHUNK_C ? n_tokens : GDN_CHUNK_C);

    // base pointers for this (head) — n_seqs==1 so sequence offset is 0
    const float * q_h = q + h_idx * sq1;
    const float * k_h = k + h_idx * sq1;
    const float * v_h = v + h_idx * sv1;
    const float * gb_base = (const float *) nullptr; // gate/beta offset computed per token below
    const int64_t gb_h = h_idx * sb1;

    float * attn_data = dst + h_idx * D;                  // [.. + token*D*H], value rows
    const int64_t attn_score_elems = (int64_t) D * H * n_tokens; // n_seqs==1
    float * state_out = dst + attn_score_elems + (int64_t) h_idx * D * D;

    // =========================================================================
    // Outer loop over chunks. Sequential DEPTH = ceil(N/C). For N<=C this runs once.
    // S_in for chunk 0 is curr_state; for later chunks it's the previous S_out.
    // =========================================================================
    // load S_in (transposed) into s_S
    for (int j = warp; j < D; j += blockDim.y) {
        s_S[j][lane] = curr_state[(int64_t) (h_idx * D + j) * D + lane];
    }
    __syncthreads();

    for (int chunk_base = 0; chunk_base < n_tokens; chunk_base += GDN_CHUNK_C) {
        const int cc = (int) min((int64_t) GDN_CHUNK_C, n_tokens - chunk_base);

        // -- Phase 1: load chunk tiles + cumulative LOG gate (Hillis-Steele scan) --
        // Load k_r, q_r, v_r, beta_r, g_r for r=0..cc-1 into smem. Then prefix-sum
        // g over r (log space) -> s_L[r][i] = sum_{s<=r} g_s[i].  scalar gate:
        // s_L[r][0] = sum_{s<=r} g_s, broadcast at use sites.
        for (int r = warp; r < cc; r += blockDim.y) {
            const int t = chunk_base + r;
            s_K[r][lane] = k_h[t * sq2 + lane];
            s_Q[r][lane] = q_h[t * sq2 + lane];
            s_V[r][lane] = v_h[t * sv2 + lane];
            const int64_t gb = gb_h + (int64_t) t * sb2;
            if (lane == 0) s_beta[r] = beta[gb];
            // KDA gate is a length-D vector over key dim; scalar gate is length 1
            s_L[r][lane] = KDA ? g[gb * D + lane] : (lane == 0 ? g[gb] : 0.0f);
        }
        __syncthreads();
        // inclusive prefix sum of s_L along r (one warp marches r; cheap, C<=32)
        // gdn_prefix_sum_logspace(s_L, cc, D, KDA);   // <- TODO real scan
        __syncthreads();

        // Convenience: A_r[i]      = exp(s_L[r][i])
        //              A_last[i]   = exp(s_L[cc-1][i])
        //              ratio(r,i)  = exp(s_L[cc-1][i] - s_L[r][i])  (= A_last/A_r, carry-forward)
        // Build the derived keys/queries on the fly inside the matmuls below to
        // avoid extra smem; shown here named for clarity:
        //   Kbar_r[i] = exp(s_L[r][i]) * s_K[r][i]
        //   Qbar_r[i] = exp(s_L[r][i]) * s_Q[r][i]
        //   k~_r[i]   = exp(-s_L[r][i]) * s_K[r][i]
        //   q~_r[i]   = exp(+s_L[r][i]) * s_Q[r][i]   (== Qbar; same for query)
        //   Kw_r[i]   = ratio(r,i)      * s_K[r][i]

        // -- Phase 2: U_carry = Kbar @ S_in ;  O_carry = scale*(Qbar @ S_in) --
        // s_Ucar[r][j] = sum_i Kbar_r[i] * S[i][j]
        //             = sum_i (exp(L[r][i]) * s_K[r][i]) * s_S[j][i]   (M transposed!)
        for (int r = warp; r < cc; r += blockDim.y) {
            float acc = 0.0f;
            for (int i = 0; i < D; ++i) {
                const float Kbar = expf(s_L[r][i]) * s_K[r][i];
                acc += Kbar * s_S[lane][i];   // s_S[j][i], here j=lane (value dim column)
            }
            s_Ucar[r][lane] = acc;
            // O_carry accumulates into s_O below (Qbar @ S_in == same with q)
            float oacc = 0.0f;
            for (int i = 0; i < D; ++i) {
                const float Qbar = expf(s_L[r][i]) * s_Q[r][i];
                oacc += Qbar * s_S[lane][i];
            }
            s_O[r][lane] = scale * oacc;      // O_carry; intra-chunk term added in Phase 5
        }
        __syncthreads();

        // -- Phase 3: T[r][s] = beta_r * (k~_r . k~_s), strict lower triangle --
        // k~_r[i] = exp(-L[r][i]) * s_K[r][i].  Build [C x C], mask s>=r to 0.
        for (int r = warp; r < cc; r += blockDim.y) {
            // each lane handles a subset of columns s
            for (int s = lane; s < cc; s += blockDim.x) {
                if (s < r) {
                    float dot = 0.0f;
                    for (int i = 0; i < D; ++i) {
                        const float kr = expf(-s_L[r][i]) * s_K[r][i];
                        const float ks = expf(-s_L[s][i]) * s_K[s][i];
                        dot += kr * ks;
                    }
                    s_T[r][s] = s_beta[r] * dot;
                } else {
                    s_T[r][s] = 0.0f;   // strict lower only
                }
            }
        }
        __syncthreads();

        // -- Phase 4: solve Dmat = (I + strict_tril(T))^{-1} @ RHS,  RHS = beta(.)(V - U_carry) --
        // Unit-lower-triangular -> forward substitution, C sequential micro-steps
        // on the C x C system (one warp). RHS lives in s_Dmat initially.
        for (int r = warp; r < cc; r += blockDim.y) {
            s_Dmat[r][lane] = s_beta[r] * (s_V[r][lane] - s_Ucar[r][lane]); // RHS row r
        }
        __syncthreads();
        // forward substitution: for r = 0..cc-1:  Dmat[r] -= sum_{s<r} T[r][s] * Dmat[s]
        // (diagonal is 1, so no division). Done by a single warp over the value cols.
        if (warp == 0) {
            for (int r = 0; r < cc; ++r) {
                float acc = s_Dmat[r][lane];
                for (int s = 0; s < r; ++s) {
                    acc -= s_T[r][s] * s_Dmat[s][lane];
                }
                s_Dmat[r][lane] = acc;
            }
        }
        __syncthreads();

        // -- Phase 5: O += scale * tril(Q~ K~^T) @ Dmat  (diagonal INCLUDED) --
        // P[r][s] = q~_r . k~_s  for s<=r  (self term s==r included: output reads
        // post-update state). Then O[r][j] += scale * sum_{s<=r} P[r][s] * Dmat[s][j].
        for (int r = warp; r < cc; r += blockDim.y) {
            float oacc = 0.0f;
            for (int s = 0; s <= r; ++s) {
                // P[r][s] = sum_i (exp(L[r][i]) q_r[i]) * (exp(-L[s][i]) k_s[i])
                float p = 0.0f;
                for (int i = 0; i < D; ++i) {
                    const float qrt = expf(s_L[r][i]) * s_Q[r][i];
                    const float kst = expf(-s_L[s][i]) * s_K[s][i];
                    p += qrt * kst;
                }
                oacc += p * s_Dmat[s][lane];
            }
            s_O[r][lane] += scale * oacc;
        }
        __syncthreads();

        // write outputs for this chunk: attn_data[(chunk_base+r) layout] = s_O[r]
        for (int r = warp; r < cc; r += blockDim.y) {
            const int t = chunk_base + r;
            attn_data[(int64_t) t * D * H + lane] = s_O[r][lane];
        }

        // -- Phase 6: carry  S_out = diag(A_last) S_in + Kw^T @ Dmat --
        // S[i][j] <- A_last[i]*S[i][j] + sum_r Kw_r[i] * Dmat[r][j]
        // transposed: s_S[j][i] <- A_last[i]*s_S[j][i] + sum_r Kw_r[i]*Dmat[r][j]
        for (int j = warp; j < D; j += blockDim.y) {
            const int i = lane;
            const float Alast = expf(s_L[cc - 1][i]);
            float acc = Alast * s_S[j][i];
            for (int r = 0; r < cc; ++r) {
                const float Kw = expf(s_L[cc - 1][i] - s_L[r][i]) * s_K[r][i]; // ratio*k
                acc += Kw * s_Dmat[r][j];
            }
            s_S[j][i] = acc;
        }
        __syncthreads();

        // -- (optional) per-token trace for DFlash rewind (DESIGN.md 3.3) -------
        // Preferred: materialize S_r = diag(A_r) S_in + Kw(->r)^T @ Dmat[0..r] per r
        // and write each into trace[(t*H + h)*D*D ...]. Sketched as a prefix pass;
        // omitted in this skeleton (start by routing trace!=nullptr to sequential).
        (void) trace;
    }

    // -- final state writeback (transposed layout, same as sequential kernel) --
    for (int j = warp; j < D; j += blockDim.y) {
        state_out[(int64_t) j * D + lane] = s_S[j][lane];
    }
}

// -----------------------------------------------------------------------------
// Host launcher. Mirrors launch_gated_delta_net<KDA> in gated_delta_net.cu but
// uses a (H,1,1) grid with one block per head. Called from the guarded fast path
// in ggml_cuda_op_gated_delta_net (see DESIGN.md 3.1):
//
//   if (n_seqs == 1 && n_tokens >= GDN_CHUNK_MIN && n_tokens <= GDN_CHUNK_MAX
//                   && S_v <= GDN_CHUNK_DMAX && trace_d == nullptr)
//       launch_gated_delta_net_chunked<KDA>(...);
//   else
//       launch_gated_delta_net<KDA>(...);   // existing sequential kernel
// -----------------------------------------------------------------------------
template <bool KDA>
void launch_gated_delta_net_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * trace_d,
        int64_t S_v, int64_t H, int64_t n_tokens,
        int64_t sq1, int64_t sq2,
        int64_t sv1, int64_t sv2,
        int64_t sb1, int64_t sb2,
        float scale, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3 grid_dims((unsigned) H, 1, 1);
    dim3 block_dims((unsigned) (warp_size <= S_v ? warp_size : S_v), num_warps, 1);

    switch (S_v) {
        case 16:
            gated_delta_net_chunked_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, trace_d, H, n_tokens,
                sq1, sq2, sv1, sv2, sb1, sb2, scale);
            break;
        case 32:
            gated_delta_net_chunked_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, trace_d, H, n_tokens,
                sq1, sq2, sv1, sv2, sb1, sb2, scale);
            break;
        case 64:
            gated_delta_net_chunked_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, trace_d, H, n_tokens,
                sq1, sq2, sv1, sv2, sb1, sb2, scale);
            break;
        // case 128: needs the register-shard S variant (smem S is 64KB); see DESIGN.md 3.
        default:
            GGML_ABORT("gated_delta_net_chunked: unsupported S_v (use sequential path)");
            break;
    }
}

// explicit instantiations so the dispatcher in gated_delta_net.cu can link them
template void launch_gated_delta_net_chunked<true >(const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float,cudaStream_t);
template void launch_gated_delta_net_chunked<false>(const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,float,cudaStream_t);

#endif // skeleton guard
