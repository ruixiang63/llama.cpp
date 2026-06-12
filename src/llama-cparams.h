#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool eagle3_extract_enabled;  // enable layer extraction for EAGLE3 speculative decoding
    bool dflash_extract_enabled;  // enable layer extraction for DFlash speculative decoding
    bool out_argmax;              // emit on-device argmax of the output logits (greedy verify path)
    bool out_spec_sample;         // emit on-device temp-softmax prob of each draft token (sampling verify)
    float spec_temp;              // temperature baked into the in-graph softmax for out_spec_sample
    int32_t spec_topk;            // >0: emit top-K candidate logits per row (top-k/top-p verify) instead of pdraft
    bool pipeline_parallel;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
