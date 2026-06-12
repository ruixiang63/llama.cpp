#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Chunk-parallel GDN forward for the DFlash speculative VERIFY path (single sequence).
// Cuts the verify's sequential depth from N to ceil(N/C). See DESIGN.md (repo root)
// and gated_delta_net_chunked.cu. SKELETON — not yet wired into the dispatcher.
template <bool KDA>
void launch_gated_delta_net_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * trace_d,
        int64_t S_v, int64_t H, int64_t n_tokens,
        int64_t sq1, int64_t sq2,
        int64_t sv1, int64_t sv2,
        int64_t sb1, int64_t sb2,
        float scale, cudaStream_t stream);
