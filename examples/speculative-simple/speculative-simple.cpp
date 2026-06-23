#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "chat.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>
#include <random>
#include <cmath>
#include <algorithm>

// Build the filtered candidate distribution for one verify row from its top-K (token, logit)
// pairs: apply temperature, softmax, top-k and top-p, renormalize. Fills `cand` with the kept
// (token, prob) pairs sorted by descending prob. This is the exact target sampler restricted to
// the top-K candidates - the top-p nucleus is a subset of the top-K for any realistic params.
static void spec_build_candidates(
        const int32_t * ids, const float * logits, int32_t k, int32_t n_vocab,
        float temp, int32_t top_k, float top_p,
        std::vector<std::pair<llama_token, float>> & cand) {
    // store RAW (un-temped) logits; the sampler order matches llama.cpp's "top_k;top_p;temp" chain:
    // top_k and top_p operate on the pre-temperature logits/probs, temperature is applied last.
    cand.clear();
    cand.reserve(k);
    for (int32_t j = 0; j < k; ++j) {
        // skip padding slots: the logits vocab can be padded beyond the real vocabulary
        if (ids[j] < 0 || ids[j] >= n_vocab) { continue; }
        cand.emplace_back((llama_token) ids[j], logits[j]); // raw logit
    }
    std::sort(cand.begin(), cand.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });
    // 1) top-k cut (logit order, temperature-invariant)
    if (top_k > 0 && (int32_t) cand.size() > top_k) {
        cand.resize(top_k);
    }
    // 2) top-p cut on the temperature-1 softmax (the nucleus is defined pre-temperature)
    if (top_p < 1.0f && !cand.empty()) {
        const float maxl = cand[0].second;
        double sum = 0.0;
        std::vector<double> p(cand.size());
        for (size_t j = 0; j < cand.size(); ++j) { p[j] = std::exp(cand[j].second - maxl); sum += p[j]; }
        double cum = 0.0;
        size_t keep = cand.size();
        for (size_t j = 0; j < cand.size(); ++j) {
            cum += p[j] / sum;
            if (cum >= top_p) { keep = j + 1; break; }
        }
        cand.resize(keep);
    }
    // 3) temperature, then final softmax over the kept candidates
    const float inv_t = 1.0f / (temp > 0.0f ? temp : 1.0f);
    const float maxl = cand.empty() ? 0.0f : cand[0].second * inv_t;
    double z = 0.0;
    for (auto & c : cand) { c.second = (float) std::exp(c.second * inv_t - maxl); z += c.second; }
    if (z > 0.0) { for (auto & c : cand) { c.second = (float) (c.second / z); } }
}

// Sample one token from a single device-resident verify logits row, applying temperature and,
// for the residual case, excluding the rejected draft token. Returns the sampled token id.
// This is the host side of the sampling speculative verify: only ONE logits row is fetched per
// block (on the first rejection, or for the bonus when everything is accepted) instead of the
// whole n_vocab x block matrix. The temperature distribution is reproduced exactly, so the
// output is lossless to the target's sampling distribution.
static llama_token spec_sample_row(
        llama_context * ctx, int32_t row, llama_token exclude, float temp, int32_t n_vocab,
        std::vector<float> & buf, std::mt19937 & rng) {
    buf.resize(n_vocab);
    if (!llama_dflash_fetch_logits_row(ctx, row, buf.data(), n_vocab)) {
        return 0;
    }
    const float inv_t = 1.0f / (temp > 0.0f ? temp : 1.0f);
    float maxl = -INFINITY;
    for (int32_t v = 0; v < n_vocab; ++v) {
        if (v == exclude) { continue; }
        buf[v] *= inv_t;
        if (buf[v] > maxl) { maxl = buf[v]; }
    }
    double sum = 0.0;
    for (int32_t v = 0; v < n_vocab; ++v) {
        if (v == exclude) { buf[v] = 0.0f; continue; }
        buf[v] = std::exp(buf[v] - maxl);
        sum += buf[v];
    }
    // inverse-CDF categorical sample
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    double r = u01(rng) * sum;
    for (int32_t v = 0; v < n_vocab; ++v) {
        r -= buf[v];
        if (r <= 0.0) { return (llama_token) v; }
    }
    return (llama_token) (n_vocab - 1);
}

struct spec_checkpoint {
    int64_t n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }

    bool empty() const {
        return data.empty();
    }
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    if (params.speculative.mparams_dft.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // DFlash/EAGLE3 on a hybrid/recurrent target can't partial-seq-rm, so speculative decoding
    // checkpoints the target state on every step. Reserve a 2nd sequence slot so that checkpoint
    // can live ON-DEVICE (seq_cp) instead of a ~50 MiB GPU<->host round-trip per step.
    if (params.speculative.dflash || params.speculative.eagle3) {
        params.n_parallel = std::max(params.n_parallel, 2);
    }

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    // check if the context supports partial sequence removal
    const auto ctx_seq_rm = common_context_can_seq_rm(ctx_tgt);
    const bool use_ckpt = (ctx_seq_rm == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    // when the context has a spare sequence slot, keep the speculative checkpoint on-device by
    // copying the active sequence into a scratch sequence (seq_cp) instead of serializing ~50 MiB
    // to host and back every step. Profiling showed the host round-trip was ~22% of decode time.
    const llama_seq_id SEQ_CKPT = 1;
    // LLAMA_SPEC_NO_SEQCP forces the host-checkpoint path (for validating the on-device path).
    const bool use_seq_cp = use_ckpt && llama_n_seq_max(ctx_tgt) > SEQ_CKPT
        && !(getenv("LLAMA_SPEC_NO_SEQCP") && std::string(getenv("LLAMA_SPEC_NO_SEQCP")) != "0");

    if (use_ckpt) {
        LOG_INF("speculative decoding will use checkpoints (%s)\n",
                use_seq_cp ? "on-device seq_cp" : "host state round-trip");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    llama_model_ptr model_dft;

    // TODO: simplify this logic
    {
        const auto & params_spec = params.speculative;

        auto params_dft = params;

        params_dft.n_parallel   = 1;
        params_dft.n_ctx        = params_spec.n_ctx;
        params_dft.n_batch      = llama_n_ctx_seq(ctx_tgt);
        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams_dft;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        params.speculative.model_tgt = model_tgt;
        params.speculative.model_dft = model_dft.get();
        params.speculative.cparams_dft = common_context_params_to_llama(params_dft);

        if (params.speculative.eagle3) {
            llama_set_eagle3(ctx_tgt, model_dft.get());
        }
        if (params.speculative.dflash) {
            llama_set_dflash(ctx_tgt, model_dft.get());
        }
    }

    // DFlash recurrent rewind (LLAMA_SPEC_TRACE=1): trace per-token recurrent states during the
    // verify decode so a partial acceptance promotes the state at the accepted position instead of
    // checkpoint-restore + re-decode of the accepted tokens (the "commit-forward").
    const bool use_state_trace = params.speculative.dflash && use_ckpt &&
        getenv("LLAMA_SPEC_TRACE") && std::string(getenv("LLAMA_SPEC_TRACE")) != "0";
    if (use_state_trace) {
        llama_set_dflash_state_trace(ctx_tgt, params.speculative.n_max + 1);
        LOG_INF("DFlash recurrent state trace enabled (promote instead of re-decode)\n");
    }

    // GPU greedy verify (LLAMA_SPEC_GPU_VERIFY=1, GREEDY SAMPLING ONLY): the target emits an
    // on-device argmax of the verify-block logits and the host logits copy (n_vocab x block
    // floats per round) is skipped; acceptance compares block_size ints.
    const bool use_gpu_verify = params.speculative.dflash &&
        getenv("LLAMA_SPEC_GPU_VERIFY") && std::string(getenv("LLAMA_SPEC_GPU_VERIFY")) != "0";
    if (use_gpu_verify) {
        LOG_INF("GPU greedy verify enabled (target logits stay on-device)\n");
    }

    // async draft feed (LLAMA_SPEC_ASYNC=1, requires GPU verify): draft tokens go to the verify
    // batch device-to-device; one host synchronization per round instead of two
    const bool use_async_feed = use_gpu_verify &&
        getenv("LLAMA_SPEC_ASYNC") && std::string(getenv("LLAMA_SPEC_ASYNC")) != "0";
    if (use_async_feed) {
        LOG_INF("async draft feed enabled (single sync per round)\n");
    }

    // sampling speculative verify (LLAMA_SPEC_GPU_SAMPLE=1, temperature > 0): the target emits the
    // temp-softmax probability of each draft token on-device; the host does rejection sampling on
    // those probs and fetches a single logits row for the residual/bonus sample. Lossless to the
    // target's temperature distribution. SGLang's tree_speculative_sampling_target_only, ported.
    const float spec_temp = params.sampling.temp;
    const int32_t spec_top_k = params.sampling.top_k;
    const float   spec_top_p = params.sampling.top_p;
    // top-k/top-p would emit top-K candidates (cap 256), but the on-device top-K path does not yet
    // match the host sampler's acceptance closely enough to beat it - keep it experimental behind
    // LLAMA_SPEC_GPU_SAMPLE_TOPK. By default GPU sampling verify is temperature-only (where it is a
    // clear win); any top-k/top-p config falls back to the (correct, faster) host sampler path.
    const bool spec_filtered = (spec_top_k > 0) || (spec_top_p < 1.0f);
    const bool allow_topk = getenv("LLAMA_SPEC_GPU_SAMPLE_TOPK") &&
        std::string(getenv("LLAMA_SPEC_GPU_SAMPLE_TOPK")) != "0";
    const int32_t spec_topk_cap = (spec_filtered && allow_topk) ? std::max(256, spec_top_k) : 0;
    const bool use_gpu_sample = params.speculative.dflash && spec_temp > 0.0f && !use_async_feed &&
        (!spec_filtered || allow_topk) &&
        getenv("LLAMA_SPEC_GPU_SAMPLE") && std::string(getenv("LLAMA_SPEC_GPU_SAMPLE")) != "0";
    std::mt19937 spec_rng((uint32_t) (params.sampling.seed == LLAMA_DEFAULT_SEED ? 0xC0FFEE : params.sampling.seed));
    std::vector<float> spec_logits_buf;
    const int32_t spec_n_vocab = llama_vocab_n_tokens(vocab);
    if (use_gpu_sample) {
        LOG_INF("GPU sampling verify enabled (temp=%.2f, residual on-device prob + 1-row fetch)\n", spec_temp);
    }

    // Apply chat template for EAGLE3 / DFlash if available which can increase the acceptance rate
    std::string prompt = params.prompt;
    if (params.speculative.eagle3 || params.speculative.dflash) {
        auto chat_templates = common_chat_templates_init(model_tgt, params.chat_template);
        if (common_chat_templates_was_explicit(chat_templates.get())) {
            std::vector<common_chat_msg> chat_msgs;
            common_chat_msg user_msg;
            user_msg.role = "user";
            user_msg.content = params.prompt;
            chat_msgs.push_back(user_msg);

            common_chat_templates_inputs inputs;
            inputs.messages = chat_msgs;
            inputs.add_generation_prompt = true;
            // Disable thinking mode can improve accept rate
            if (const char * nt = std::getenv("LLAMA_SPEC_NO_THINK"); nt && std::string(nt) != "0") {
                // Qwen3 / 3.5
                inputs.enable_thinking = false;
                // gpt-oss
                inputs.chat_template_kwargs["reasoning_effort"] = "\"low\"";
            }
            prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            LOG_INF("%s: %s chat template applied\n", __func__, params.speculative.eagle3 ? "EAGLE3" : "DFlash");
        }
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }


    // eval the prompt
    llama_token id_last;
    llama_tokens prompt_tgt;
    int n_past;

    // TODO: simplify
    if (params.speculative.eagle3 || params.speculative.dflash) {
        // Target model decodes full prompt and sample first token and intermediate features are extracted
        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size()));

        id_last = common_sampler_sample(smpl.get(), ctx_tgt, -1);
        common_sampler_accept(smpl.get(), id_last, true);

        // from now on the verify loop only needs the on-device argmax (the initial sample above
        // still consumed host logits, so the flag is enabled only after it)
        if (use_gpu_verify) {
            llama_set_out_argmax(ctx_tgt, true);
        }
        if (use_gpu_sample) {
            llama_set_out_spec_sample(ctx_tgt, true, spec_temp, spec_topk_cap);
        }
        LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
        n_predict++;

        // all tokens currently in the target context
        prompt_tgt.assign(inp.begin(), inp.end());
        prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

        n_past = inp.size();
    } else {
        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

        // note: keep the last token separate!
        id_last = inp.back();

        // all tokens currently in the target context
        prompt_tgt.assign(inp.begin(), inp.end() - 1);
        prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

        n_past = inp.size() - 1;
    }

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    common_speculative_begin(spec, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    size_t n_draft = 0;

    llama_tokens draft;
    spec_checkpoint spec_ckpt;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            // generate a new draft
            draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);

            if ((int) draft.size() > params_spec.n_max) {
                LOG_WRN("draft size %zu exceeds max %d, truncating\n", draft.size(), params_spec.n_max);
                draft.resize(params_spec.n_max);
            }

            if ((int) draft.size() < params_spec.n_min) {
                LOG_DBG("ignoring small draft: %zu < %d\n", draft.size(), params_spec.n_min);
                draft.clear();
            }

            // save the original draft size
            n_draft = draft.size();

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty() && use_state_trace) {
                // recurrent rewind: no checkpoint needed - the verify decode traces per-token
                // states and a partial acceptance promotes the right one (see below)
                spec_ckpt.n_tokens = (int64_t) prompt_tgt.size();
            } else if (!draft.empty() && use_seq_cp) {
                // on-device checkpoint: copy the active sequence (0) into the scratch sequence.
                // The subsequent draft decode on seq 0 advances its state; seq SEQ_CKPT keeps the
                // pre-draft state (recurrent state is copy-on-write), so we can restore from it.
                auto * mem = llama_get_memory(ctx_tgt);
                llama_memory_seq_rm(mem, SEQ_CKPT, -1, -1);
                llama_memory_seq_cp(mem, 0, SEQ_CKPT, -1, -1);
                spec_ckpt.n_tokens = (int64_t) prompt_tgt.size();
            } else if (!draft.empty() && use_ckpt) {
                const size_t ckpt_size = llama_state_seq_get_size_ext(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                spec_ckpt.data.resize(ckpt_size);

                const size_t n = llama_state_seq_get_data_ext(ctx_tgt, spec_ckpt.data.data(), ckpt_size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                GGML_ASSERT(n == ckpt_size);

                spec_ckpt.n_tokens = (int64_t) prompt_tgt.size();
                LOG_DBG("created speculative checkpoint (n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        spec_ckpt.n_tokens, (float) spec_ckpt.data.size() / 1024 / 1024);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            // (for the on-device path the checkpoint lives in seq SEQ_CKPT, not in spec_ckpt.data)
            if (use_ckpt && !use_seq_cp) {
                GGML_ASSERT(!spec_ckpt.empty());
            }
        }

        GGML_ASSERT(n_draft > 0);

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            // do not waste time on small drafts
            if (draft.size() < (size_t) params_spec.n_min) {
                draft.clear();
            }

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt);

            // debug: validate the state-trace mechanics (trace[last] must equal the live cell)
            if (use_state_trace && getenv("LLAMA_DFLASH_DEBUG")) {
                llama_dflash_trace_check(ctx_tgt, batch_tgt.n_tokens);
            }
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        llama_tokens ids;
        if (use_gpu_sample) {
            // sampling speculative verify (lossless to the target temperature distribution):
            // the DFlash drafter proposes greedily (q = delta), so accept draft d_i with prob
            // p_i(d_i) (the target temp-softmax prob, computed on-device); on the first rejection
            // sample the replacement from the residual p_k with d_k removed; if all accepted, the
            // bonus is sampled from the last position's full target distribution.
            std::uniform_real_distribution<float> u01(0.0f, 1.0f);

            // sample a token from a filtered candidate distribution, optionally excluding one token
            auto sample_cand = [&](const std::vector<std::pair<llama_token, float>> & cand,
                                   llama_token exclude) -> llama_token {
                double z = 0.0;
                for (const auto & c : cand) { if (c.first != exclude) { z += c.second; } }
                if (z <= 0.0) { return cand.empty() ? 0 : cand[0].first; }
                double r = u01(spec_rng) * z;
                for (const auto & c : cand) {
                    if (c.first == exclude) { continue; }
                    r -= c.second;
                    if (r <= 0.0) { return c.first; }
                }
                return cand.back().first;
            };

            size_t i = 0;
            bool rejected = false;

            if (spec_topk_cap > 0) {
                // top-k/top-p verify: per row, rebuild the filtered candidate distribution on the
                // host from the on-device top-K, then run the standard speculative rejection test.
                int32_t n_rows = 0, kk = 0;
                const float * tvals = nullptr;
                const int32_t * tidx = llama_get_dflash_topk(ctx_tgt, &n_rows, &kk, &tvals);
                GGML_ASSERT(tidx != nullptr && n_rows >= (int32_t) draft.size() + 1 && "spec topk missing");
                std::vector<std::pair<llama_token, float>> cand;
                for (; i < draft.size(); ++i) {
                    spec_build_candidates(tidx + (size_t) i * kk, tvals + (size_t) i * kk, kk, spec_n_vocab,
                                          spec_temp, spec_top_k, spec_top_p, cand);
                    float p = 0.0f;
                    for (const auto & c : cand) { if (c.first == draft[i]) { p = c.second; break; } }
                    if (u01(spec_rng) < p) {
                        ids.push_back(draft[i]); // accept
                    } else {
                        ids.push_back(sample_cand(cand, draft[i])); // residual (d_i removed)
                        rejected = true;
                        break;
                    }
                }
                if (!rejected) {
                    spec_build_candidates(tidx + draft.size() * kk, tvals + draft.size() * kk, kk, spec_n_vocab,
                                          spec_temp, spec_top_k, spec_top_p, cand);
                    ids.push_back(sample_cand(cand, -1)); // bonus
                }
            } else {
                // temperature-only verify: accept d_i with prob p_i(d_i) from the on-device gather
                int32_t n_pd = 0;
                const float * pd = llama_get_dflash_pdraft(ctx_tgt, &n_pd);
                GGML_ASSERT(pd != nullptr && n_pd >= (int32_t) draft.size() + 1 && "spec pdraft missing");
                for (; i < draft.size(); ++i) {
                    if (u01(spec_rng) < pd[i]) {
                        ids.push_back(draft[i]);
                    } else {
                        ids.push_back(spec_sample_row(ctx_tgt, (int32_t) i, draft[i], spec_temp,
                                                      spec_n_vocab, spec_logits_buf, spec_rng));
                        rejected = true;
                        break;
                    }
                }
                if (!rejected) {
                    ids.push_back(spec_sample_row(ctx_tgt, (int32_t) draft.size(), -1, spec_temp,
                                                  spec_n_vocab, spec_logits_buf, spec_rng));
                }
            }
        } else if (use_gpu_verify) {
            // greedy accept from the on-device argmax: identical semantics to
            // common_sampler_sample_and_accept_n with a greedy sampler (token at each position up
            // to and including the first mismatch; bonus token if everything matched)
            int32_t n_am = 0;
            const int32_t * am = llama_get_dflash_argmax(ctx_tgt, &n_am);
            GGML_ASSERT(am != nullptr && n_am >= (int32_t) draft.size() + 1 && "target argmax missing");

            // async feed mode: the draft vector holds placeholders (the real tokens never touched
            // the host before the verify); refill it now from the drafter's extracted argmax.
            // by this point the target sync above has fenced all earlier drafter work too.
            if (use_async_feed && !draft.empty()) {
                int32_t n_dam = 0;
                const int32_t * dam = llama_get_dflash_argmax(common_speculative_get_dflash_decoder(spec), &n_dam);
                GGML_ASSERT(dam != nullptr && n_dam >= (int32_t) draft.size() + 1 && "drafter argmax missing");
                for (size_t i = 0; i < draft.size(); ++i) {
                    draft[i] = (llama_token) dam[i + 1];
                }
            }
            size_t i = 0;
            for (; i < draft.size(); ++i) {
                ids.push_back((llama_token) am[i]);
                if (draft[i] != (llama_token) am[i]) {
                    break;
                }
            }
            if (i == draft.size()) {
                ids.push_back((llama_token) am[i]);
            }
        } else {
            ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        }

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_state_trace && ids.size() - 1 < draft.size()) {
            // recurrent rewind: promote the traced state at the accepted position. the verify batch
            // was [id_last @ P, draft0 @ P+1, ...] with P == prompt_tgt.size(); accepting `acc`
            // drafts means the state after batch token `acc` is the correct one (trace slot `acc`),
            // ending at position P + acc. then fall through to the normal commit path - the
            // loop-tail llama_memory_seq_rm(0, n_past, -1) truncates the attention KV of the
            // rejected tail and now succeeds because the recurrent cell pos was rewound.
            const int32_t   acc      = (int32_t) ids.size() - 1;
            const llama_pos pos_last = (llama_pos) prompt_tgt.size() + acc;

            if (!llama_dflash_promote_state(ctx_tgt, acc, pos_last, 0)) {
                LOG_ERR("%s: DFlash state promote failed (idx=%d)\n", __func__, acc);
                return 1;
            }
            // fall through to the commit path below
        } else if (use_ckpt && ids.size() - 1 < draft.size()) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, draft.size());

            draft = std::move(ids);

            if (use_seq_cp) {
                auto * mem = llama_get_memory(ctx_tgt);
                llama_memory_seq_rm(mem, 0, -1, -1);           // drop the speculative advance on seq 0
                llama_memory_seq_cp(mem, SEQ_CKPT, 0, -1, -1); // restore the pre-draft state from scratch seq
            } else {
                const size_t n = llama_state_seq_set_data_ext(ctx_tgt, spec_ckpt.data.data(), spec_ckpt.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                GGML_ASSERT(n == spec_ckpt.size());

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, spec_ckpt.n_tokens, -1);
            }

            prompt_tgt.resize(spec_ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            const bool rm_ok = llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
            if (!rm_ok && use_state_trace) {
                // in trace mode this MUST succeed (the recurrent cell pos was rewound by promote);
                // a failure means the rejected verify KV is still in the attention cache -> corruption
                LOG_ERR("%s: post-accept seq_rm(0, %d, -1) FAILED in trace mode\n", __func__, n_past);
                return 1;
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
