#include "models.h"

// graph-reuse guard for the device-cache decoder path: the feat/mask/position tensor shapes are
// baked from (cross->n_enc, dflash->feat_bucket) at build time; force a rebuild when either moves
class llm_graph_input_dflash_dev : public llm_graph_input_i {
public:
    llm_graph_input_dflash_dev(const llama_cross * cross, const llama_dflash * df,
                               int64_t n_enc_built, int32_t bucket_built)
        : cross(cross), df(df), n_enc_built(n_enc_built), bucket_built(bucket_built) {}

    void set_input(const llama_ubatch * ubatch) override { GGML_UNUSED(ubatch); }

    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        return cross->n_enc == n_enc_built && df->feat_bucket == bucket_built;
    }

    const llama_cross  * cross;
    const llama_dflash * df;
    int64_t n_enc_built;
    int32_t bucket_built;
};

ggml_tensor * llm_build_dflash_encode::build_inp_embd() const {
    const int64_t n_target_layer_ids = (int64_t) hparams.dflash_target_layer_ids.size();
    const int64_t n_embd_target_features = n_target_layer_ids * n_embd;

    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_target_features);
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_target_features, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

llm_build_dflash_encode::llm_build_dflash_encode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd();

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "hidden_norm_out", -1);

    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

llm_build_dflash_decode::llm_build_dflash_decode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Noise tokens [MASK]
    GGML_ASSERT(model.target_tok_embd != nullptr && "DFlash decoder requires target model's tok_embd");
    ggml_tensor * noise_embd = build_inp_embd(model.target_tok_embd);
    cb(noise_embd, "inp_noise_embd", -1);

    ggml_tensor * target_ctx = nullptr;
    int64_t       n_ctx      = 0;

    if (dflash != nullptr && dflash->cross_dev != nullptr) {
        // encoder folded into the decoder graph: the NEW tokens' raw target features arrive as an
        // input, get fc+norm'ed here and appended into the persistent device cache (cross_dev) via
        // set_rows; the attention context is a view of that cache. this removes the separate
        // encoder llama_encode + the host round trip of the encoded features per draft round.
        const int64_t n_feat    = (int64_t) hparams.dflash_target_layer_ids.size() * n_embd;
        // padded feature rows; extra rows land in the scratch row via the index input. bucketed
        // (8 for normal rounds, 256 for the prompt round) so the fc GEMM does not waste flops on
        // padding - the graph rebuilds once when the bucket changes
        const int64_t n_new_max = dflash->feat_bucket;

        ggml_tensor * feat = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_feat, n_new_max);
        ggml_set_input(feat);
        cb(feat, "dflash_feat_new", -1);

        ggml_tensor * fidx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_new_max);
        ggml_set_input(fidx);
        cb(fidx, "dflash_feat_idx", -1);

        ggml_tensor * enc = build_lora_mm(model.fc, feat);
        enc = build_norm(enc, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
        cb(enc, "dflash_enc_new", -1);

        ggml_tensor * cache = ggml_set_rows(ctx0, dflash->cross_dev, enc, fidx);
        cb(cache, "dflash_cross_cache", -1);

        n_ctx      = cross->n_enc; // bucketed valid+padding rows (padding masked out)
        target_ctx = ggml_view_2d(ctx0, cache, n_embd, n_ctx, cache->nb[1], 0);

        res->add_input(std::make_unique<llm_graph_input_dflash_dev>(cross, dflash, n_ctx, (int32_t) n_new_max));
    } else {
        // legacy host-mediated path: accumulated context uploaded via llama_cross
        target_ctx = build_inp_cross_embd();
        n_ctx      = target_ctx->ne[1];
    }

    ggml_tensor * inpL = noise_embd;

    const int64_t n_tokens_kv = n_ctx + n_tokens;

    // Position tensor covering target_ctx + noise
    ggml_tensor * inp_pos_full = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens_kv);
    ggml_set_input(inp_pos_full);
    cb(inp_pos_full, "inp_pos_full", -1);

    // Q positions: last n_tokens entries (noise only)
    ggml_tensor * inp_pos_q = ggml_view_1d(ctx0, inp_pos_full, n_tokens,
            n_ctx * ggml_element_size(inp_pos_full));

    // Additive attention mask over [target_ctx (n_ctx) ++ noise (n_tokens)] for the noise queries.
    // target_ctx is a fixed-capacity buffer so the graph shape stays constant across speculative
    // rounds (enables graph reuse); the padding rows are masked out. Values are filled per round in
    // llama_context::process_ubatch (named "dflash_kq_mask"). Requires the eager soft_max path,
    // which is why flash attention is disabled for the DFlash decoder context.
    ggml_tensor * dflash_kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens_kv, n_tokens);
    ggml_set_input(dflash_kq_mask);
    cb(dflash_kq_mask, "dflash_kq_mask", -1);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        // Q from noise only
        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        if (layer.wq_b) { Qcur = ggml_add(ctx0, Qcur, layer.wq_b); }
        cb(Qcur, "Qcur", il);

        // K = concat(k_proj(target_ctx), k_proj(noise))
        ggml_tensor * K_tgt   = build_lora_mm(layer.wk, target_ctx);
        ggml_tensor * K_noise = build_lora_mm(layer.wk, noise_norm);
        if (layer.wk_b) {
            K_tgt   = ggml_add(ctx0, K_tgt,   layer.wk_b);
            K_noise = ggml_add(ctx0, K_noise, layer.wk_b);
        }
        ggml_tensor * Kcur = ggml_concat(ctx0, K_tgt, K_noise, 1);
        cb(Kcur, "Kcur", il);

        // V = concat(v_proj(target_ctx), v_proj(noise))
        ggml_tensor * V_tgt   = build_lora_mm(layer.wv, target_ctx);
        ggml_tensor * V_noise = build_lora_mm(layer.wv, noise_norm);
        if (layer.wv_b) {
            V_tgt   = ggml_add(ctx0, V_tgt,   layer.wv_b);
            V_noise = ggml_add(ctx0, V_noise, layer.wv_b);
        }
        ggml_tensor * Vcur = ggml_concat(ctx0, V_tgt, V_noise, 1);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens_kv);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens_kv);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        cb(Kcur, "Kcur_normed", il);

        // RoPE: K uses full positions [0..n_ctx+n_tokens-1], Q uses last n_tokens
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos_full, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        cb(Kcur, "Kcur_rope", il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos_q, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        cb(Qcur, "Qcur_rope", il);

        // Full attention (no causal mask)
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);

        ggml_tensor * cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, dflash_kq_mask, nullptr, nullptr, kq_scale, il);
        cb(cur, "kqv_out", il);

        cur = build_lora_mm(layer.wo, cur);
        if (layer.wo_b) { cur = ggml_add(ctx0, cur, layer.wo_b); }
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "attn_res", il);

        ggml_tensor * ffn_inp = cur;
        cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    if (model.target_output) {
        cur = build_lora_mm(model.target_output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        // GPU argmax over the block logits: the DFlash draft is greedy top-1, so downloading
        // block_size ints instead of n_vocab x block_size floats (~5 MB/round at vocab 248k)
        // removes the per-round logits host copy + CPU scan entirely
        ggml_tensor * am = ggml_argmax(ctx0, cur);
        cb(am, "result_argmax", -1);
        res->t_argmax = am;
        ggml_build_forward_expand(gf, am);
    }

    ggml_build_forward_expand(gf, cur);
}