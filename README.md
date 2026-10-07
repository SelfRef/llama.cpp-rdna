# llama.cpp-rdna3

**A llama.cpp fork for AMD RDNA3 / RDNA3.5 on Vulkan — Radeon RX 7900 XTX (gfx1100),
RX 7800 XT (gfx1101) and Strix Halo / Ryzen AI Max+ 395 (gfx1151). Nothing else.**

It exists because the two things this hardware needs have never been in one tree:

1. **The ROCmFPx weight formats** — `Q4_0_ROCMFP4*`, `Q2/Q3/Q6/Q8_0_ROCMFPX` (ggml type ids
   100-107), which stock llama.cpp cannot even *load*. They come from
   [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) (originally charlie12345/ROCmFPX) via
   [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), which is the base
   of this branch. On a 7900 XTX, Qwen3.8-27B at ROCmFP4-FAST decodes 33-38 % faster than the same
   model as a Q4_K_M on stock llama.cpp, in 2.4 GiB less VRAM, for +4.3 % perplexity.
2. **RDNA3 Vulkan work that was never upstreamed** — a body of measured patches that exists only as
   patch files passed between community forks, with no pull request behind any of them. Carrying
   them means carrying them ourselves.

Upstream is where this should all live, and some of it is on its way there
([#28898](https://github.com/ggml-org/llama.cpp/pull/28898) would put FP8/NVFP4 quant scales in ggml
with a Vulkan implementation; [#27952](https://github.com/ggml-org/llama.cpp/pull/27952), int8
coopmat1 MMQ for RDNA3, merged on 2026-09-24). When it does, this fork should shrink, not grow.

## What branch `rdna3` carries

Base: `LaurentZuijdwijk/llama.cpp` @ `11bfe8a6` (upstream `0190529e`, 2026-08-30) — the ROCmFPx
formats, the batch-3..8 mat-vec path, `--spec-draft-adaptive`, and the RADV ≥ 25.3 coopmat LDS pad
gate — and, since 2026-09-18, **upstream master itself** (last merged 2026-10-07, `36a73916`): the re-port is complete, the fork is no longer behind.
On top of upstream there are five carried patches (the swiglu fusion was dropped once upstream's #27220
superseded it) plus the fork's own ROCmFPx type plumbing, its delta-net concat-transpose kernel, and the
UMA readback guard:

| # | Patch | Author | Why it is here |
|---|---|---|---|
| 1 | `vulkan: hoist the coopmat1 FA P-fragment load out of the hsv_tile loop` | Nathan Wilson | coopmat1 flash-attention is the path RDNA3 actually takes |
| 2 | `vulkan: store coopmat1 FA Psh query-major so the GEMM2 A load vectorizes` | Nathan Wilson | same |
| 3 | `vulkan: pin a 32-wide subgroup for coopmat1 FA where narrowing is free` | Nathan Wilson | same |
| 5 | `server: keep speculative checkpoints on device` | Gaetan Puleo | MTP speculative decoding is on for every model we run |
| 6 | `llama: opt-in KV cache row padding to defeat power-of-2 channel aliasing` | Nathan Wilson | **opt-in, off by default** (`LLAMA_KV_ROW_PAD`); measured null on gfx1100, kept as a knob for gfx1151 |

Every one of them is in the measured set below. **Nothing goes on this branch on inspection alone**
— see "Tried and parked" for what happened the one time it did.

Provenance: 1-3 and 5 are cherry-picked from
[voidsurfer/llama.cpp-nudge](https://github.com/voidsurfer/llama.cpp-nudge) with original authorship
intact; 4 and 6 come via [guevae2/paoai-strix-engine](https://github.com/guevae2/paoai-strix-engine),
which had already rebased them onto this exact base. Every commit carries a `cherry picked from`
line. All sources are MIT, as is this fork.

## Tried and parked

Two further patches by the same authors looked right on inspection — one numerical, one a
correctness fix — and were briefly carried on that basis. They were then measured, and bisected:

| | prose | json | refactor | prefill @32k | output |
|---|---|---|---|---|---|
| `rdna3` (patches 1-6) | 76.2 | 107.7 | 130.6 | 843.6 | matches base |
| `carry/vulkan-fa-mmq-fp32` (+ FA MMQ fp32 narrowing) | 76.0 | 107.7 | 130.4 | 841.4 | matches base |
| + `carry/mtp-full-checkpoints` as well | **58.4** | **80.5** | 120.2 | 806.4 | **differs** |

- **`vulkan: scale the FA MMQ dot product in fp32 before narrowing` — neutral, parked.** It costs
  nothing and changes nothing on an FP4 model, which is expected: MMQ is the integer-dot path the
  K-quants take, not this one. It is worth re-testing on a K-quant, and it is a reasonable candidate
  once the base moves up far enough that one binary serves every model.
- **`Reapply "common: use full checkpoints for MTP rollback"` — rejected.** On its own it accounts
  for the whole **-23 % prose / -24 % json** and it **changes the greedy output**, on a path that is
  supposed to be lossless. Draft acceptance rises (53 % vs 44 %) while throughput falls, i.e. the
  rollback is paying on every accepted token. Do not merge without understanding the output change.

The lesson is on the branch now: a patch gets carried when a benchmark says so, not when the commit
message is persuasive.

## A failure mode every canary missed

The first step-4 build passed the compiler, the ROCmFPx-type and adaptive-drafting canaries and a
smoke request (it answered "pong"), and then benchmarked at **2.7 t/s** — Qwen3.8-27B on sixteen CPU
threads. `--list-devices` printed `(none)`. Cause: the hunk cleanup had cut the only definition of
`ggml_vk_wait_for_fence` (a definition count that mistook the header declaration for one), a shared
library links fine with an undefined symbol, `dlopen` of `libggml-vulkan.so` then fails at runtime,
and ggml registers the CPU backend alone — correct output, silently 25× slower.

Two guards now sit in the build pipeline: `ldd -r` on every `libggml-*.so` must report no undefined
symbol (`ldd` without `-r` does not resolve symbols and sees nothing), and every hop's smoke step
requires `--list-devices` to show `Vulkan0` before any benchmark runs. If you build this tree
yourself, run both.

**Deliberately not carried:** everything DeepSeek-V4-specific (not run here), the DFlash2 draft-cache
patches (DFlash2 loses to the baked MTP head on these cards, and it cannot be combined with it), the
`mul_mat_id` IQ-type pipelines (second half of a two-commit series whose first half does not apply,
and no IQ quant runs here), and anything already in the base — notably the RADV coopmat pad-2 driver
gate, which is present and is worth ~11 % prefill on its own.

## Measured

7900 XTX (gfx1100), RADV / Mesa 26.2.2, Qwen3.8-27B ROCmFP4-FAST, 131k ctx, f16 K / q4_0 V, MTP n4,
greedy, median of 2 — patches 1-6 against the same base without them:

| | prose | json | refactor | prefill @32k |
|---|---|---|---|---|
| ROCmFPx base (`11bfe8a6`, upstream 08-30) | 75.6 | 106.5 | 129.3 | 818.6 t/s |
| the six patches, at that base | 76.2 | 107.7 | 130.6 | **843.6 t/s** |
| base moved to upstream 09-09 (step 1) | 72.1 | 103.9 | 129.8 | 841.6 t/s |
| through #25773 (step 2) | 71.9 | 104.3 | 130.0 | 859.6 t/s |
| base moved to upstream 09-17 (step 3) | 74.4 | 106.1 | 130.4 | 864.3 t/s |
| **this branch** (upstream master 09-18, step 4) | 74.1 | 105.7 | 129.9 | 868.1 t/s |

The 5 % prose / 4 % json decode this branch gives up against the row above is **entirely** upstream
[#28068](https://github.com/ggml-org/llama.cpp/pull/28068), isolated by building with it reverted
(75.6 / 108.3, i.e. reference speed restored). It recomposes the gated-delta-net L2 norm as
`rms_norm(eps/n) · 1/√n` so epsilon is applied the way the reference implementation does, across
`q_conv` and `k_conv` in all 48 GDN layers — two ops where there was one. Perplexity is unmoved
(6.9218 vs 6.9211, ~1 % of one standard error), so the cost buys fidelity that wikitext cannot see.
It stays: reverting would mean carrying a divergence from upstream on model correctness, forever,
at every future hop.

Step 3 (upstream to 09-17) won back part of the #28068 cost: +3.0 % prose / +1.8 % json over step 2 in one
session, with upstream's #28457 small-M kernels (the MTP verify shape) as the visible cause — the prose hash
changes, json's does not. Step 4 (current master, through the Vulkan source split): decode and prefill within noise of step 3 (−0.4…−0.8 %), output byte-identical on all four
presets. The re-port is complete; the fork is at upstream master.

Step 2 (#25773, upstream's spec-constant matmul rewrite) then cost nothing: decode within noise of
step 1, prefill +1.9 %, output byte-identical on all four presets. The ROCmFPx types now register
through upstream's own per-type `X(TYPE, tstr)` idiom (`FOR_EACH_ROCMFPX_TYPE`) instead of the
fork's macro-per-type scheme, and the fork's f16-B routing — which was **on by default**, not an
unused knob — is replaced by upstream doing the same f32→f16 B conversion unconditionally on
coopmat1. Two fork tuning knobs went with it (`GGML_VK_DENSE_WAVE32`, off and measured −5.8 %;
`GGML_VK_MMID_WG256/WAVE32`, off and never measured). **A fused eps-aware `l2_norm` in ggml would recover most of the 5 % — that is
the first thing this fork should try to upstream.**

Decode is a tie; prefill is **+2.8 % at 32k** and +1.3-3.3 % on short prompts, with identical VRAM
and **byte-identical greedy output**. `LLAMA_KV_ROW_PAD=256` measured null on this card and cost
0.6 GiB, so leave it unset unless you are on gfx1151 and have measured otherwise.

## Branches

| Branch | What it is |
|---|---|
| `master` | untouched mirror of `ggml-org/llama.cpp`. Never commit here — it is what keeps "Sync fork" and every cross-fork compare working. |
| `rdna3` | **default**, and what gets built. The base above plus the eight commits. |
| `carry/*` | one branch per carried patch set, so a bad upstream rebase blows up in one place instead of all eight. |

## Roadmap: moving the base

The base is 330 commits behind upstream, and catching up is a **re-port, not a rebase**: since the
fork point upstream rewrote matmul pipeline creation into one spec-constant shader per quant family
([#25773](https://github.com/ggml-org/llama.cpp/pull/25773)) and split the Vulkan sources into
separate files ([#28732](https://github.com/ggml-org/llama.cpp/pull/28732)).

Measured merge cost from this branch, 2026-09-18:

| merge target | conflicting files | conflict hunks | status |
|---|---|---|---|
| upstream just before #25773 (09-09) | 10 | 26 | **done** |
| **at #25773** (spec-constant matmul) | 2 | 19 | **done** — registration work, not shader work |
| just before #28732 (09-17) | 1 | 1 | **done** — `src/CMakeLists.txt` only |
| current master 44be98f0 (09-18), through the source split | 3 | 7 | **merged** — **done** — benchmarked, see "Measured" |

(Costs re-measured from each adopted tip; the original single-jump estimate was 13 files / 47 hunks.)

So roughly **half the work is the single #25773 step**, and it is exactly where the ROCmFPx types
have to be re-expressed in the new `create_mm_pipelines` / spec-constant scheme rather than merged.
`ggml-vulkan.cpp` carries 28 of the 47 hunks at the far end; the rest are `qwen4exp.cpp` (4),
`llama-memory-hybrid-idx.*` (5) and single hunks in converters, CMake and tests.

Do it in those four steps, building and benchmarking each one against the table in "Measured" —
a step that costs decode is a step to stop and understand, not to push through. The prize at the
end is that this tree can take upstream PRs *and* the FP4 types at once, which is what the separate
patched `llama-server` in [llama-swap-rdna](https://github.com/SelfRef/llama-swap-rdna)
exists to work around today. Once it can, that binary goes back to being stock upstream.

## Phase 2: one binary

With the base at master, the tree can take upstream PRs — the thing the separate patched `llama-server`
in [llama-swap-rdna](https://github.com/SelfRef/llama-swap-rdna) exists to work around.
Measured against master on 2026-09-18, of that image's 13 `LLAMA_PATCHES` **eleven merge cleanly**;
two conflict in one hunk each and need a rebase onto this tree:

| PR | what it is | state |
|---|---|---|
| #27952 | int8 coopmat1 MMQ for RDNA3 — a prefill win on entries without an MTP draft, a decode loss on entries with one (see phase 3) | **merged upstream 2026-09-24**; the fork keeps only its per-type gate, `GGML_VK_NO_CM1_MMQ` and the A-side `end_k` clamp (see the 09-24 round) |
| #25666 | no MMVQ on speculative-decode steps — `qwen38-bart`'s draft acceptance | 1 hunk vs master, a device-tuning constant block |
| #28243 | Qwen3.8-Flash-Next MTP head | **gone 2026-10-05**: closed upstream, superseded by #29761 (Qwen4Exp MTP, merged); the fork now runs upstream's qwen4exp as is |

Once those land here, `llama-server` in the image goes back to byte-for-byte upstream and this fork is
the one binary for every entry. Not before: that order was chosen so no production entry ever sees a
regression window.

## Phase 3: four more upstream PRs, and what the cm1 knob actually keys on

Branch `carry/phase3`, 2026-09-19, all four measured on a 7900 XTX (gfx1100) before being kept:

| PR | what it is | verdict |
|---|---|---|
| #27183 | return `GGML_STATUS_FAILED` on Vulkan device loss instead of aborting | **kept.** Hand-resolved against the source split: the PR still carries pre-split copies of the device structs, so only its four real hunks were taken and `device_lost` went into `ggml-vulkan-types.h`. A lost device now fails the request instead of killing the server. |
| #28873 | honour `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` in `llama_kv_cache::state_write`/`state_read_sinfo` | **kept.** Clean merge. On Qwen3.8-27B with `--ctx-checkpoints 64` it takes **2.1 GB off the resident footprint** with a byte-identical output hash — a full-attention cache no longer serialises itself into every checkpoint. |
| #29019 | preserve batch order for layer inputs and unmasked NextN embeddings | **merged, then reverted.** It merges clean and the ordering bug it fixes is real, but on Qwen3.6-35B-A3B it costs **57 % of decode** (146.2 -> 62.3 t/s prose, 181.9 -> 79.6 json) with the same output hash *and* the same draft acceptance, so it buys nothing measurable here. Qwen3.8-27B — dense, same MTP, same `--parallel 2` — is untouched, so the cost scales with graph size. Can come back in a cheaper shape. |
| #27332 | density gate for `MUL_MAT_VEC_ID` on non-coopmat2 devices | **narrowed to `device->uma`** (`c401947ca`). As written it costs **47 % of decode** on Qwen3.6-35B-A3B here (121.6 -> 64.7 t/s prose, same output hash): the widened vector range catches the MTP draft-verify batch, which a discrete card runs faster on the tiled path. Upstream measured +36 % on gfx1151, so the gate stays for integrated parts. |

The same session also corrected what `GGML_VK_NO_CM1_MMQ` is for. The 09-18 rule was "ROCmFPx
entries only"; both arms on one binary say the discriminator is **the MTP draft**, not the weight
format:

| entry | MTP | quant | cm1 |
|---|---|---|---|
| gemma4 | no | UD-Q4_K_XL | **on** — +9 % prose / +29 % json prefill, decode flat |
| ling3 | no | UD-Q4_K_XL | **on** — +19 % / +22 % prefill, decode flat |
| qwen36 | yes | UD-Q4_K_M | **off** — cm1 costs 17-21 % decode |
| qwen38-fast | yes | ROCmFP4-FAST | **off** — cm1 costs 30 % prefill, 37 % decode (09-18) |

This also supersedes the "+18.5 % MoE" figure in the phase 2 table: qwen36 decode was flat under cm1
on 09-18 and is -17 % now. The only cm1-touching commit in between is `aa5e52e44` (clamp the A
prefetch to `end_k`), which added a per-prefetch select inside the unrolled load. That is an untested
hypothesis and it matters, because the clamp is a correctness fix the Strix Halo peer needs.

## Maintenance round 2026-09-24

Base moved to upstream master `8212c780` (2026-09-24). Merge cost: 3 files, 15 hunks, almost all of
them #27952 arriving in its final upstream form on top of the older head this branch carried.
After the merge the coopmat1 MMQ shaders are upstream's byte for byte except one hunk. Upstream's
prefetch clamps only the **B** side to `end_k`, so the fork keeps its **A**-side clamp: a k-step past
the end of a row would otherwise read the next row, or the bytes after the tensor, whose f16
scale can be NaN (NaN × a zeroed B is still NaN). The per-type cm1 gate (`ggml_vk_type_has_cm1_mmq`,
13 types, which matches the generator exactly) and `GGML_VK_NO_CM1_MMQ` are still needed, because
upstream still selects cm1 warptiles per device rather than per type.

| change | what | verdict |
|---|---|---|
| #28956, #28927, #28243 | carried PRs whose authors pushed since the last merge (exact A/B descriptor ranges in `mul_mm`; a clean-up; review fixes + rebase) | new heads merged |
| #28943 | HIP masked-KV-tile skip | **reverted**: closed upstream without merge, and HIP code this Vulkan-only tree never builds |
| #29019 | batch-order fix (reverted in phase 3) | stays out: the author's update only strips unrelated files, nothing addresses the measured MoE cost |
| #29182 | MoE-aware `mul_mat_id` tile selection | **rejected, −12 % prefill**: Qwen3.6-35B-A3B at 32k, 1911 → 1679 t/s, same output hash. At `-ub 256` with top-8 of 256 experts the per-expert row count is exactly 8, so every expert matmul drops from the aligned large tile to the unaligned small one. Upstream measured +10 % on gfx1151; the discrete card pays for it. |
| #29280 | reuse Vulkan descriptor sets when bindings are unchanged | **not carried, neutral**: decode within ±1 % on both a dense 27B and the 35B MoE under MTP. RADV's descriptor updates are evidently cheap enough that skipping them buys nothing measurable. |
| Nathan Wilson's gfx11 coopmat GEMM rewrite (via paoai-strix-engine) | f32 accumulator, register prefetch of the next K tile, 8-wide q6_K/q3_K/q8_0/q5_0 loaders; +3..21 % prefill on gfx1151 with a 27B FP4 | **deferred, needs a re-port**: written against the pre-#25773 per-type loaders and the f16-B/wave32 knobs this branch dropped, so it does not cherry-pick (5 files, 12 hunks). Three independent parts, only measured as a bundle on gfx1151. |

What the base move itself did, on a 7900 XTX (greedy, same session as the reference):

| | prose | json | refactor | prefill @32k | output |
|---|---|---|---|---|---|
| Qwen3.8-27B ROCmFP4-FAST, MTP n4, before | 75.9 | 107.6 | 131.2 | 855.3 (mean of 3 runs) | — |
| same, after | 77.0 | 108.1 | 131.9 | 846.4 (mean of 2) | byte-identical |
| Qwen3.6-35B-A3B UD-Q4_K_M, MTP, before | 166.7 | 190.4 | — | 1910.8 | — |
| same, after | 166.7 | **203.8** | — | 1913.7 | byte-identical |

Decode is a tie or better (+7 % json on the MoE); the dense model's 32k prefill is −1.0 %, which is
on the noise line. **Output changes on every file that carries IQ4_XS tensors, and only on those**
(a Qwen3.8-27B UD-Q4_K_XL with 61 of them, Qwen3.5-4B UD-Q4_K_XL with 10 on the 7800 XT). The cause is
upstream #28415, now in master, which gives IQ4_XS its own q8_1 integer-dot MMQ/MMV path. An earlier
revision of that PR produced subtly broken text on RDNA3 (clauses dropped mid-sentence, draft
acceptance *rising*), so it was checked by reading the output. The current form reads coherent, with
free-form draft acceptance unchanged (57 → 58 %). The 4B's speed is flat. On the 27B the greedy
texts differ, so per-preset decode is not comparable run to run: prose +8 %, json −12 % (its draft
acceptance fell 92 → 66 % on a different answer), refactor +11 %, identical output on refactor.
Check the text again on any IQ4_XS-heavy quant after the next move.

## Maintenance round 2026-10-05

Base moved to upstream master `e117148a` (2026-10-05), 259 commits. Merge cost: 20 files, 68 hunks —
two thirds of them qwen4exp, where upstream had meanwhile landed its own version of nearly everything
the fork carried for that model.

| change | what | verdict |
|---|---|---|
| qwen4exp (#28243 MTP head, #28213 gather-based QSA decode, #28699 pooled-key cache, the on-disk n-gram/PLE reader, MTP-draft tensor borrowing) | all superseded upstream: #29761 (Qwen4Exp MTP), CUDA sparse flash attention and GLM5-Next's shared k-pool cache (the #28213 author re-measured master as equal or faster at every depth), `llama_prefetch_rows` (#29599) for lazily read gather tables | **dropped**: qwen4exp is upstream's, byte for byte. The `--ngram-on-disk`/`--model-ple` flags and `gguf_extract_ple.py`/`gguf_split_ple_heads.py` are gone with it |
| rejection sampling for temperature > 0 drafts (from the DFlash2 base) | upstream merged its own (#27694, `spec_draft_q`) | **replaced by upstream's**. The adaptive draft depth (`--spec-draft-adaptive`, `draft-mtp-adaptive`) and #28333's carrier zeroing are re-ported onto it. A replayed draft after a checkpoint restore is accepted without re-verification at any temperature, as before |
| #28876, #28956, #28751 (supersedes #28927), #29019 | merged upstream | arrive with master. **#29019 no longer costs anything**: the 57 % MoE decode loss it caused in phase 3 is gone in its merged form (table below) |
| #29182 | MoE-aware `mul_mat_id` tile selection, rejected here on 09-24 for −12 % MoE prefill | merged and then **reverted upstream** (#29936) |
| #26286 | Qwen3 sliding-window attention pattern | **merged** — jina-reranker-v3.5 needs it (below) |
| jina-reranker-v3 / v3.5 | the model cannot be served by `/v1/rerank` anywhere else | **added**, see the next section |
| FA verify packing (guevae2/paoai-strix-engine `d8d0b9b74`): 2-8 token GQA batches in one flash-attention call per KV head | −17 % verify time at 32k on gfx1151 | **not carried, noise here**: on Qwen3.8-27B, 32k-depth decode +3.2 % in one session and −4.8 % in the next; on Qwen3.6-35B-A3B json −2.6 %, depth +3.6 %. Branch `test/fa-verify-pack-2026-10-05` |

What the base move did, greedy, same session as the reference:

| | prose | json | refactor | prefill @32k | output |
|---|---|---|---|---|---|
| Qwen3.8-27B ROCmFP4-FAST, MTP n4, 7900 XTX, before | 76.6 | 108.4 | 132.2 | 849.2 | — |
| same, after | 77.1 | 108.8 | 132.5 | 852.7 | byte-identical |
| Qwen3.6-35B-A3B UD-Q4_K_M, MTP, 7900 XTX, before | 167.9 | 201.1 | — | 1904.8 | — |
| same, after | 168.2 | 200.8 | — | 1919.4 | byte-identical |
| Qwen3.5-4B UD-Q4_K_XL, 7800 XT, before | 106.0 | 105.3 | — | 2115.2 | — |
| same, after | 106.0 | 106.2 | — | **2078.3** | byte-identical |

Decode is a tie everywhere. The one loss is **−1.75 % prefill at 32k on the 7800 XT** (gfx1101,
Qwen3.5-4B, reproduced A/B/A), with identical output; neither model on the 7900 XTX shows it. No commit
in the merged range is an obvious cause — upstream's gated-delta-net retune (#29476) leaves the
RADV subgroup-64 configuration unchanged — so it is accepted and noted here, to recheck next round.
The 4B's first greedy reply after a fresh load differs from later ones on both binaries; compare
warm against warm.

## Maintenance round 2026-10-07

Base moved to upstream master `36a73916` (2026-10-07), 41 commits, mainly to pick up EmbeddingGemma 2
(#30054). Two conflicts:

| change | what | verdict |
|---|---|---|
| #30049 | AMD iGPU: route reads of uncached host-visible memory through the device copy | **fork's UMA readback guard kept**: it already does this for every UMA device and still reads ≤ 64 KiB directly to skip the fence round trip; upstream's version is the same rule for AMD without that exception |
| #27332 | new head: the density gate stops at 32 tokens on NVIDIA | **not re-merged**: NVIDIA only, AMD keeps 64; the fork's `device->uma` narrowing is unchanged |
| #29998 / #29822 (MUL_MAT_ID rows when expert ids repeat), #29882 (subgroup `rms_norm`), #29877 (packed f16 FMA without coopmat), #29274 (NVIDIA `rm_id`) | open upstream | **not carried**: still in review, without AMD numbers, or no change on RDNA3 |

Greedy, 7900 XTX unless noted. The reference is the previous build `5fb3ec3`: the 2026-10-05
numbers for what was not re-run, and a run in this session for the output hashes:

| | prose | json | refactor | prefill @32k | output |
|---|---|---|---|---|---|
| Qwen3.8-27B ROCmFP4-FAST, MTP n4, before | 77.4 | 108.7 | 132.2 | 849.2 | — |
| same, after | 76.9 | 108.9 | 132.9 | 856.2 | byte-identical |
| Qwen3.6-35B-A3B UD-Q4_K_M, MTP, before | 167.0 | 203.4 | — | 1904.8 | — |
| same, after (A/B/A, repeat 3) | 166.6 / 167.9 | 202.8 / 205.0 | — | 1917.7 | byte-identical |
| Qwen3.5-4B UD-Q4_K_XL, 7800 XT, before | 105.5 | 105.2 | — | 2115.2 | — |
| same, after | 106.1 | 105.8 | — | 2127.5 | byte-identical |

Flat everywhere. The −1.75 % 32k prefill on the 7800 XT that was noted last round is gone (2127.5).

## jina-reranker-v3 / v3.5

[jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) is a Qwen3-0.6B with
sliding-window layers and a "last but not late" scoring head: the hidden state at `<|embed_token|>`
after a document and at `<|rerank_token|>` after the query go through one two-layer projector
(Linear, ReLU, Linear) and are compared by cosine. Jina's own route is `llama-embedding` plus a Python
scorer; here it is an ordinary `/v1/rerank` model.

```bash
python convert_hf_to_gguf.py jina-reranker-v3.5/ --outtype bf16      # JinaForRanking
llama-quantize jina-reranker-v3.5-BF16.gguf jina-reranker-v3.5-Q8_0.gguf Q8_0
llama-server -m jina-reranker-v3.5-Q8_0.gguf --reranking -ngl all -c 16384 -np 4 -b 4096 -ub 4096
```

- The converter writes the projector as `cls`/`cls.output`, the document marker id as
  `qwen3.rerank.doc_token_id`, and a rerank template. Jina's published GGUFs have neither the projector
  nor the key, so they still need Jina's scorer; convert from the safetensors.
- The model is trained listwise, every document in one prompt. `/v1/rerank` scores one document per
  sequence, so the template is the one-document case of Jina's prompt, cut after the query marker —
  the model is causal, so nothing after that token can change either hidden state.
- A pair must be decoded in one ubatch (the document marker is read from the same ubatch as the
  last token). For such a model the context forces a unified KV cache and `n_batch = n_ubatch`, and the
  server refuses to split the prompt: size `-ub` for the longest pair you send.
- The score is a cosine in [−1, 1], not a probability.

Checked against Jina's PyTorch reference (`AutoModel`, fp32, one document per call): BF16 GGUF on a
7800 XT within 0.0011 on every pair of an EN/PL/technical sanity set, same order everywhere.
Retrieval quality, nDCG@10 re-ranking the top 25 of an EmbeddingGemma stage 1 (MTEB NFCorpus and
SciFact, English and their Polish translations; Q8_0, 7800 XT, documents cut to 2500 characters):

| reranker | NFCorpus-PL | SciFact-PL | NFCorpus | SciFact | mean |
|---|---:|---:|---:|---:|---:|
| none (stage 1 only) | 0.3008 | 0.6882 | 0.3889 | **0.7911** | 0.5422 |
| Jina Reranker v2 base multilingual (278M) | 0.3149 | 0.7226 | 0.3839 | 0.7765 | 0.5495 |
| Qwen3-Reranker-0.6B | 0.3139 | 0.7188 | 0.3995 | 0.7743 | 0.5516 |
| **jina-reranker-v3.5**, one document per sequence | **0.3201** | **0.7249** | **0.4023** | 0.7897 | **0.5592** |

Best of the three rerankers on every task even scored pointwise. It is also the slowest: its prompt
carries Jina's system prompt and the query twice, about 1.35x Qwen3-Reranker's wall time for the same
31k pairs.

## jina-embeddings-v5-omni

[jina-embeddings-v5-omni](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval)
(small: Qwen3-0.6B text, 1024 dims; nano: EuroBERT-210M text, 768 dims) embeds text, images and audio
into one space. Jina ships GGUFs for it, but its own image and audio path needs their
[`feat-v5-omni`](https://github.com/jina-ai/llama.cpp/tree/feat-v5-omni) branch, 21 commits on a
May 2026 master. Ported here against the 2026-10-05 base; most of that branch is upstream by now:

| jina-ai commit(s) | what | here |
|---|---|---|
| `8127970` encoder combined decode | text + media in one batch for a model without a KV cache (nano) | **upstream** — #29969's mixed batch does it, with the 1-D positions nano needs |
| `4a252ec`, `0b9cf28`, `ba0d398` | qwen3vl pos_embed align-corners, Pillow-accurate bicubic resize | **upstream** |
| `37d0f04`, `67ceac2` | image min/max pixels from the mmproj, no padding after the aspect-preserving resize | **ported, gated**: only an mmproj that carries `clip.vision.image_{min,max}_pixels` gets either, so the stock Qwen-VL mmprojs keep their preprocessing; `--image-min/max-tokens` still win |
| `fa8376b` (part) | Qwen wrapper tokens missing from the text vocab | **ported**: wrappers that are not single special tokens are dropped instead of BPE-split (nano) |
| `25b6c8b`, `ff59ead`, `1b91bac` | Qwen2.5-Omni audio: `n_window` chunked attention, per-chunk conv, per-chunk positions, variable length, `<\|audio_start\|>` markers | **rewritten, gated on `clip.audio.n_window`**: the encoder gets only the real mel frames (as torch's `feature_attention_mask` does), so the shorter last chunk needs no extra masks and the token count follows from the input length. Variable length is the default — Jina's model card has since moved to it too. Markers come from the vocab: `<\|audio_bos\|>` if it is a special token, else `<\|audio_start\|>`. Qwen2-Audio and upstream's Qwen2.5-Omni mmprojs (no `n_window`) are untouched |
| `1cfc842` multi-mmproj | `--mmproj` twice to load vision and audio | **not ported**: `scripts/merge-mmproj.py` writes Jina's two mmprojs into one mixed-modality file (`clip.{vision,audio}.projector_type`), which upstream already loads |
| `e192d1b`, `f23e08e` | `videopair_data` request field, video pixel limits | **not ported**: video goes through upstream's ffmpeg path (cos 0.94, see below) |
| `02c8c1e`, `5df11fb`, converter halves | converting the checkpoints yourself | **not ported**: Jina's published GGUFs already carry the metadata |

```bash
python scripts/merge-mmproj.py omni-small-retrieval-vision-mmproj-F16.gguf \
    omni-small-retrieval-audio-mmproj-F16.gguf omni-small-retrieval-omni-mmproj-F16.gguf
llama-server -m jina-embeddings-v5-omni-small-retrieval-Q8_0.gguf \
    --mmproj omni-small-retrieval-omni-mmproj-F16.gguf --embedding --pooling last -ub 4096
```

POST `/embeddings` with `"prompt_string": "Query: ..."` / `"Document: ..."` for text, the server's
`media_marker` (from `/props`) plus `multimodal_data` for an image, and
`"<|im_start|>user\n" + marker + "<|im_end|>\n"` for audio (the model card's chat-template path).

Checked against Jina's PyTorch reference (`AutoModel.embed`, fp32, the model card's raw path), F16
text GGUF, one process with the merged mmproj, 7800 XT:

| cos vs torch | text (3 probes, EN/PL) | image (1600x1598 photo) | audio (7.2 s speech, 181 tokens) |
|---|---:|---:|---:|
| small | ≥ 0.99993 | 0.99790 | 0.99958 |
| nano | ≥ 0.99996 | 0.99829 | 0.99976 |
| small, Q8_0 text | ≥ 0.99978 | 0.99769 | 0.99925 |
| nano, Q8_0 text | ≥ 0.99979 | 0.99812 | 0.99966 |

Jina reports 0.9989-0.9998 for its own branch. **Send audio at 16 kHz**: the same clip at 44.1 kHz,
resampled by llama.cpp's miniaudio instead of librosa, gave 0.986 (small) and 0.37 (nano). Video
(4 frames, 2 fps) through upstream's ffmpeg path embeds at 0.938 vs torch with the `<|vision_start|>`
wrapper: upstream samples 4 fps, keeps the image pixel limits and writes `[0m0.00s]` timestamps
where Qwen3-VL's processor writes `<0.2 seconds>` per frame pair (1521 tokens vs 618).

Unchanged elsewhere: Qwen3.5-4B (qwen3vl mmproj) image embeddings are byte-identical to the
previous build on a square and a 700x333 image. Retrieval quality of the text towers alone (same
harness and tasks as the reranker table, stage 1 only, `Query: `/`Document: ` prompts): small
0.3228 / 0.7059 / 0.3971 / 0.7642, mean 0.5475; text-nano 0.3144 / 0.6681 / 0.3873 / 0.7600,
mean 0.5324; EmbeddingGemma-300M with its prompts 0.5423.

## Build

Vulkan only — the ROCmFPx types ship no HIP kernels.

```bash
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON \
      -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Two invariants worth asserting in any build pipeline — if either fails, the tree has silently lost
the reason it exists and is about to ship as a plain llama.cpp:

```bash
llama-quantize --help | grep -q Q4_0_ROCMFP4_FAST     # the ROCmFPx types
llama-server   --help | grep -q -- --spec-draft-adaptive
```

Both binaries print help to **stdout and exit 1**, so capture before grepping under `pipefail`.

For anything not specific to this fork — general build options, the server's flags, the model zoo —
follow [upstream's documentation](https://github.com/ggml-org/llama.cpp): this tree only adds the
RDNA3 pieces described above and is otherwise current master.

In [SelfRef/llama-swap-rdna](https://github.com/SelfRef/llama-swap-rdna) this tree is the
`llama-rdna3` stage and installs as `llama-server-rdna3`, `llama-cli-rdna3`, `llama-bench-rdna3`,
`llama-quantize-rdna3`, `llama-perplexity-rdna3` (build args `WITH_RDNA3`, `RDNA3_REPO`,
`RDNA3_BRANCH`, `RDNA3_COMMIT` — pin the commit).
