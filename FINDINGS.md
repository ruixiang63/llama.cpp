# CUDA graphs for the DFlash verify decode — investigation + hardening

Status: code analysis only (no GPU here). CPU-only build of the touched file passes.
Scope: the per-round **verify** decode on the TARGET context (hybrid Qwen3.5-4B: 24 Gated-DeltaNet
recurrent layers + 8 full-attention), driven by `examples/speculative-simple` with `--spec dflash`.

## TL;DR

The DFlash verify graph is **structurally eligible** for CUDA-graph capture+replay on Ampere+ — every
candidate blocker from the brief checks out as *non*-blocking (see below). The real per-round CPU
friction is the scheduler **uid churn**: `ggml_backend_sched_split_graph` mints a fresh `uid`
(`ggml_graph_next_uid()`) for each split on every call, even when the split is byte-identical to last
round, which defeats the CUDA backend's `uid` fast-path (`ggml-cuda.cu: ggml_cuda_graph_update_required`
~L3141) and forces the full O(n_nodes) node-property walk + warmup churn — the mechanism behind the
prior "-6% / warmup keeps resetting" whole-model result.

**A stable-uid fix for this already exists in the tree** (committed at HEAD `5395cb8c8`): the split
struct carries `prev_uid`/`prev_sig`, and the uid loop reuses the previous uid when a per-slot topology
signature matches. This investigation (a) confirms that mechanism is the correct lever and that nothing
else in the DFlash verify graph blocks capture, and (b) **hardens the signature** against collisions.

The remaining first-order cap on single-stream speedup (~1.5-1.7x) is the **sequential O(N) Gated-
DeltaNet verify kernel** (see `DESIGN.md`), not graph-launch overhead. The uid fix is the second-order
lever; the GDN chunk-parallel kernel is the first-order one. Both are needed to reach SGLang's ~3x.

## Candidate blockers from the brief — all verified NON-blocking

1. **`ggml_cuda_graph_check_compability` (ggml-cuda.cu ~L3089).** Disables graphs only on split buffers
   and `MUL_MAT_ID` (non-quantized / large-batch). The verify graph (`src/models/qwen35.cpp` GDN +
   `src/models/dflash.cpp` attention) is **dense** — `build_lora_mm` / `build_ffn(SILU, PAR)`, no MoE,
   no `MUL_MAT_ID`, no split buffers. => compatible.

2. **DFlash-specific nodes are capturable.** `ggml_set_rows(cross_dev,...)` (`dflash.cpp` L90), the
   per-token conv/state trace `ggml_cpy` nodes (`qwen35.cpp` L301-317, L380-396), `ggml_argmax`
   (`dflash.cpp` L227), and the top-k `argsort` verify path all map to normal CUDA ops with **no host
   stream sync** (checked `set-rows.cu` / `argmax.cu` / `argsort.cu` — only `cudaMemcpyAsync` D2D, which
   is capturable). => none disables capture.

3. **Stable destinations / offsets.** `cross_dev`, `trace_s[il]`, `trace_r[il]` are persistent tensors
   (allocated once in `dflash_cross_ctx` / `dflash_trace_buf`), so the trace/set_rows dst ptrs are
   constant. The recurrent state write offset `kv_head * n_embd_s` (`qwen35.cpp` L395) is constant for a
   single sequence (`get_head()` fixed for seq 0). => node props stable round-to-round.

4. **The graph key is stable.** The CUDA graph is keyed by `cgraph->nodes[0]`. The verify ubatch is a
   **constant** `block_size` tokens (the drafter always emits `block_size-1` drafts:
   `speculative.cpp result.assign(block_size-1,0)`; `speculative-simple.cpp` L457-469). So
   `llm_graph_result::can_reuse` holds (constant `n_tokens`/`n_outputs`/`cross`/samplers; recurrent
   `head`/`rs_z` constant), and `llm_graph_result::reset()` reuses `buf_compute_meta` in place (same
   `.data()` => tensors placement-allocated at the same offsets). => `nodes[0]` is the same pointer
   across rounds, even when a rebuild happens.

5. **Double-buffering is a non-issue here.** `cur_copy` only flips in `ggml_backend_sched_alloc_graph`
   (skipped on the reuse path), and a single-GPU DFlash target runs `pipeline_parallel=false =>
   n_copies=1`, so input-copy pointers don't alternate.

Conclusion: on Ampere+ the verify graph already captures and stays warm (the `cuda_graph` object keyed
by the stable `nodes[0]` keeps `node_props` across rounds; eviction is 10 s, rounds are ms apart). The
warmup does NOT reset for the verify on an identical graph.

## Root cause of the residual per-round CPU cost (and the whole-model -6%)

`ggml_graph_view` zeroes the uid; the tail of `ggml_backend_sched_split_graph` then assigns a fresh
monotonic uid per split every call. The CUDA backend's fast-path
(`if (cgraph->uid != 0 && cgraph->uid == graph->uid) return false;`) can therefore only skip the
property walk when the *higher-level* graph reuse keeps `split_graph` from running at all. Any reuse
miss re-runs `split_graph`, bumps the uid, and forces the full walk. On the ~1800-node whole-model graph
that is the measured ~-6%; on the hundreds-of-nodes verify graph it is smaller but non-zero.

## The existing fix, and what this change adds

Existing at HEAD (`ggml/src/ggml-backend.cpp`):
- `struct ggml_backend_sched_split` carries `prev_uid` / `prev_sig`.
- The uid loop computes a per-slot topology signature; if it matches the previous round's, it reuses
  `prev_uid` instead of minting a fresh one. `GGML_SCHED_STABLE_UID=0` opts out (on by default).
- Grown `splits` slots are zeroed after `realloc` so `prev_uid`/`prev_sig` start clean.

This change (hardening only):
- The signature was `backend_id + n_nodes + nodes[0] + nodes[n-1]` (endpoints only). A "same count +
  same endpoints but different middle" collision would let the backend reuse a **stale captured graph**
  (a silent correctness bug). Strengthened it to also fold in a **strided sample of up to ~16 interior
  node pointers**, making such a collision effectively impossible while staying O(1)-ish per split.
- Updated the in-code comment to match.

Why safe: the uid is a pure optimization hint. A matching uid only skips a walk that would have found no
change anyway (signature matched on stable placement-allocated pointers); any mismatch falls back to the
full walk + recapture. The fast-path's `node_props.size() == n_nodes` assert holds because `n_nodes` is
in the signature.

## Files changed by this investigation

- `ggml/src/ggml-backend.cpp` — `ggml_backend_sched_split_graph()`: strengthen the per-split topology
  signature (strided interior-node sampling); comment fix. No struct/ABI change beyond what HEAD already
  had; no CUDA file touched.

## What to validate on GPU (remote Ampere+ box; V100 needs `GGML_CUDA_GRAPHS_VOLTA`)

1. Build with CUDA. On V100 also pass `GGML_CUDA_GRAPHS_VOLTA=<n>` (n >= verify node count, or 1).
2. Run `speculative-simple --spec dflash` for draft-max in {8, 12, 16}, comparing tokens/sec with vs
   without the uid stabilization (`GGML_SCHED_STABLE_UID=0` disables). Expect the fix to remove the
   per-round split walk -> higher t/s, and to make the larger draft blocks (12/16) viable toward the
   SGLang accept_len ~6.6 regime (combined with the DESIGN.md chunked GDN kernel).
3. Debug build of `ggml-cuda.cu` (`-DCMAKE_BUILD_TYPE=Debug`): confirm
   `GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", ...)` fires every steady-state verify round.
4. Correctness: greedy verify output must be **token-identical** with and without
   `GGML_SCHED_STABLE_UID` (a divergence would mean a signature collision — not expected after the
   hardening).
5. Cross-check whole-model decode (no spec): the same stabilization should turn the prior ~-6% CUDA-graph
   regression neutral/positive.

GPU validation command (example):

```
GGML_SCHED_STABLE_UID=0 ./build/bin/llama-speculative-simple \
    -m target.gguf -md draft.gguf --spec dflash --draft-max 16 --draft-min 1 -n 256 -p "<prompt>"
GGML_SCHED_STABLE_UID=1 ./build/bin/llama-speculative-simple \
    -m target.gguf -md draft.gguf --spec dflash --draft-max 16 --draft-min 1 -n 256 -p "<prompt>"
```
