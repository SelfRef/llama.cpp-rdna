# Changelog

Branch `rdna` has no releases: it is upstream master plus the fork's carried
work, moved forward in maintenance rounds. Changes are grouped by date, newest
first. The measurements behind each decision are in the README section of the
same round.

## 2026-10-07

- **Renamed** from llama.cpp-rdna3 to llama.cpp-rdna, default branch `rdna3` →
  `rdna`. The scope is unchanged: RDNA3 and RDNA3.5 (gfx1100, gfx1101, gfx1151).
- Base moved to upstream master `36a73916` (41 commits), mainly for
  EmbeddingGemma 2 (#30054).
- Upstream #30049 (AMD iGPU reads of uncached host-visible memory): the fork's
  own UMA readback guard is kept instead. It covers every UMA device and still
  reads ≤ 64 KiB directly.
- #27332's new head (32-token density gate on NVIDIA) not re-merged; the fork's
  `device->uma` narrowing is unchanged.
- Speed flat and greedy output byte-identical on the three reference models;
  the −1.75 % 32k prefill on gfx1101 noted on 2026-10-05 is gone.

## 2026-10-06

- **jina-embeddings-v5-omni** (small and nano): text, image and audio
  embeddings at torch parity (cos ≥ 0.998 image, ≥ 0.999 audio). Ported from
  jina-ai's `feat-v5-omni` branch where upstream does not already have it:
  - Qwen2.5-Omni chunked audio encoder (`n_window` attention), gated on
    `clip.audio.n_window`; variable-length input; audio markers taken from the
    vocab;
  - image min/max pixels from the mmproj without padding after the resize,
    gated on `clip.vision.image_{min,max}_pixels`;
  - Qwen wrapper tokens missing from a text vocab are dropped, not BPE-split.
- New `scripts/merge-mmproj.py`: combines a vision and an audio mmproj into
  one mixed-modality file that upstream already loads (instead of jina's
  multi-`--mmproj`).
- Send audio at 16 kHz; video goes through upstream's ffmpeg path (cos 0.94).

## 2026-10-05

- Base moved to upstream master `e117148a` (259 commits).
- **qwen4exp is upstream's again, byte for byte.** Dropped as superseded:
  #28243 (MTP head; upstream #29761), #28213 (gather-based QSA decode), #28699
  (pooled-key cache), the on-disk n-gram/PLE reader and MTP-draft tensor
  borrowing. The `--ngram-on-disk` / `--model-ple` flags and
  `gguf_extract_ple.py` / `gguf_split_ple_heads.py` are gone.
- Rejection sampling for temperature > 0 drafts replaced by upstream's
  (#27694); adaptive draft depth (`--spec-draft-adaptive`,
  `draft-mtp-adaptive`) and #28333's MTP carrier zeroing re-ported onto it.
- **jina-reranker-v3 / v3.5** on `/v1/rerank`: RANK pooling through the
  projector and the cosine of the document and query marker states;
  `convert_hf_to_gguf.py` converts `JinaForRanking` checkpoints. Merged #26286
  (Qwen3 sliding-window attention pattern), which the model needs.
- #29019 (batch order for layer inputs) arrives with master and no longer
  costs MoE decode.

## 2026-09-24

- Base moved to upstream master `8212c780`.
- **#27952 (int8 coopmat1 MMQ for RDNA3) is upstream now.** The fork keeps
  only what upstream lacks: the A-side `end_k` prefetch clamp (a k-step past a
  row end could read a NaN scale), the per-type cm1 gate and the
  `GGML_VK_NO_CM1_MMQ` opt-out.
- New heads merged for #28956, #28927 and #28243.
- #28943 (HIP masked-KV-tile skip) reverted: closed upstream, and HIP code
  this Vulkan tree never builds.
- Rejected after measurement: #29182 (MoE-aware `mul_mat_id` tiles, −12 % MoE
  prefill on a discrete card) and #29280 (descriptor-set reuse, neutral).

## 2026-09-20

- Documentation links follow the llama-swap-rdna repository rename.

## 2026-09-19

- Merged after measurement on a 7900 XTX:
  - #27183: a lost Vulkan device fails the request instead of aborting the
    process;
  - #28873: `PARTIAL_ONLY` honoured in KV-cache state write/read, 2.1 GB less
    resident with 64 context checkpoints on a 27B model.
- #27332 (density gate for `MUL_MAT_VEC_ID` without coopmat2) narrowed to
  integrated GPUs: as written it cost 47 % of MoE decode on a discrete card.
- #29019 merged and reverted the same day: 57 % of MoE decode on a discrete
  card (returned in its merged form on 2026-10-05).
- Also merged: #28243 (Qwen3.8-Flash-Next MTP draft head), #28699 (QSA pooled-key
  cache), #27210 (`draft-mtp-adaptive`), unslothai#137 (batched readahead for
  lazily read gather tables), unslothai#95 (penalties indexed by token id).
- qwen4exp fixes: per-head PLE layout and the on-disk n-gram reader restored;
  an MTP draft whose hc head uses the trunk tensor names loads; the graph is
  expanded after the recurrent layer, not before it.
- coopmat1 int8 MMQ shader: the A prefetch is clamped to `end_k`.
- `GGML_VK_NO_CM1_MMQ` guidance corrected: the cm1 MMQ path is a prefill win on
  models without an MTP draft and a decode loss on models with one.

## 2026-09-18

- **Fork created** on LaurentZuijdwijk/llama.cpp `11bfe8a6` (upstream
  2026-08-30): the ROCmFPx weight formats (`Q4_0_ROCMFP4*`,
  `Q2/Q3/Q6/Q8_0_ROCMFPX`), the batch-3..8 mat-vec path,
  `--spec-draft-adaptive` and the RADV coopmat LDS pad gate.
- Carried patches, each kept only after a benchmark:
  - three coopmat1 flash-attention patches by Nathan Wilson (P-fragment load
    hoisted out of the hsv_tile loop, query-major Psh store, pinned 32-wide
    subgroup);
  - `server: keep speculative checkpoints on device` (Gaetan Puleo);
  - opt-in KV-cache row padding, `LLAMA_KV_ROW_PAD`, off by default.
- Tried and parked: FA MMQ fp32 narrowing (neutral). Rejected: full checkpoints
  for MTP rollback (−23 % prose decode and a changed greedy output).
- **Re-ported onto upstream master in four measured steps:** to 2026-09-09; through
  #25773 (spec-constant matmul, with the ROCmFPx types re-expressed in
  upstream's per-type registration); to 2026-09-17; and through the Vulkan
  source split (#28732) to `44be98f0`. Result against the base: prefill +6 % at
  32k, decode within −2 % (the cost of upstream #28068, the gated-delta-net L2
  norm fix; perplexity unchanged).
- The carried swiglu-fusion patch dropped, superseded by upstream #27220. The fork's
  `GGML_VK_DENSE_WAVE32` and `GGML_VK_MMID_WG256/WAVE32` knobs went with
  the re-port.
- Restored `ggml_vk_wait_for_fence`, lost in the step-4 conflict resolution:
  `libggml-vulkan.so` failed to `dlopen` and ggml silently fell back to the CPU.
  The README now asks for an `ldd -r` check and a `--list-devices` check in any
  build pipeline.
- Phase 2, upstream PRs on the fork: #27952 (int8 coopmat1 MMQ, with a
  per-type gate and the `GGML_VK_NO_CM1_MMQ` opt-out), #25666 (no MMVQ on
  speculative-decode steps), #28213, #28333, #25592 (checkpoints for
  hybrid/recurrent models), #28265 (Qwen3.5 delta-net output projection),
  #28876 (RANK pooling batch splitting), #28901, #28927, #28943, #28956.
