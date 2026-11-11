# Prior work: what in this repo existed before (searched 2026-11-12)

None of the methods in this repo is new. A literature and code search on 2026-11-12, made after
the work was done, found each idea published or shipped elsewhere first, in three cases on the
same model and the same GPUs. This page records what was found, so the other write-ups can be
read as independent measurements and not as first results.

The search was a web search of papers, llama.cpp pull requests and forks. The entries below
are as their pages read on 2026-11-12. Nothing from them was run or compared here.

## Summary

| Work in this repo | Earlier work | Status of ours |
|---|---|---|
| RDNA4 attention patch for small batches at depth (`docs/depth.md` sections 6, 9, 10; F11, F13, F15) | [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts), [issue #45](https://github.com/stew675/llama-cpp-rdna-boosts/issues/45), opened 2026-09-25 and shipped the next day | Same diagnosis and remedy, about six weeks later |
| RDNA4 MMVQ parameters for 2-8 columns (`docs/upstream_mmvq.md`, F8) | Same fork, [issue #56](https://github.com/stew675/llama-cpp-rdna-boosts/issues/56), shipped 2026-09-29 | Same change, about four weeks later |
| Vocabulary shortlist for the MTP drafter (`docs/shortlist.md`, F16) | [FR-Spec](https://arxiv.org/abs/2502.14856) and [VocabTrim](https://arxiv.org/abs/2506.22694) (2025); llama.cpp [PR #31031](https://github.com/ggml-org/llama.cpp/pull/31031) (opened 2026-10-23) and [PR #29143](https://github.com/ggml-org/llama.cpp/pull/29143) (opened 2026-09-19); the fork's `--spec-draft-mtp-vocab` | Known method; a second implementation and measurement |
| Lenient draft acceptance (`docs/lenient.md`, F14) | Typical acceptance in Medusa; a logit-ratio rule in [MARS](https://arxiv.org/pdf/2601.15498.pdf); [a 2026 theory of acceptance rules](https://huggingface.co/papers/2606.30265) | Known family of rules; numbers for one model |
| Dense-to-MoE conversion without training (F3) | [CMoE](https://arxiv.org/abs/2502.04416), [ExpertWeaver](https://arxiv.org/html/2602.15521v1), [MLPMoE](https://www.arxiv.org/pdf/2511.21089), and MoEfication before them | Known idea; the oracle-router bound against the quantization bar is measured here |
| Activation sparsity (TEAL-style) and its loss under speculative decoding (F5) | [TEAL](https://github.com/FasterDecoding/TEAL) lists speculative decoding as untested; [ReLU Strikes Back](https://arxiv.org/abs/2310.04564) treats sparsity aggregated over the verified tokens | No measurement found of training-free masks under an MTP verify batch; F5's 89-100% union is the result here with the least precedent |

## The RDNA patch fork

[stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) is a public
set of RDNA-specific llama.cpp patches. It was not known to this project until the search.

- **Attention at depth.** Issue #45 reports that for GQA 6 the tile kernel uses `ncols2 = 2`,
  so each K/V element is fetched and dequantized three times, and routes batches of up to 8
  tokens to the matrix-core kernel with `ncols2 = 8`. That is the finding and the fix of
  `docs/depth.md` section 10. Its reported numbers: one layer at 102,400 rows, q8_0, 3 tokens,
  1921 -> 893 us; decode at about 110K 28.2 -> 36.0 tok/s. The fork later extended it to f16 and
  added a q8_0 fetch-ahead variant.
- **MMVQ for verify batches.** Issue #56 proposes more rows per block for 2-8 columns on RDNA4,
  4 rows for most types, chosen per weight type from a sweep. Reported: MTP draft 3 +11.5%.
  `docs/upstream_mmvq.md` makes the same change (4 rows, 2 warps) for all types at once.
- **Beyond this repo.** The fork also carries adaptive MTP draft depth, n-gram drafting combined
  with MTP, DFlash work and a draft-vocabulary option, none of which was built here.

The numbers above come from other machines and builds and cannot be set against this repo's
(44.8 tok/s at 120K with MTP and q8_0 KV on one R9700). A like-for-like run of the fork on this
machine has not been done; it is the next step in `docs/roadmap.md`.

## The drafter shortlist

- FR-Spec (Zhao et al., 2025) restricts the draft to a frequency-ranked subset of the vocabulary
  and keeps the full vocabulary for verification, reporting a 75% cut in LM-head computation and
  1.12x over EAGLE-2. VocabTrim (Goel et al., 2025) builds the subset from a calibration run.
- llama.cpp PR #31031 adds `--spec-draft-vocab <file>`: a reduced head built from a token list,
  with unlisted ids at -inf. Its author reports +6.5% on coding for a 27B with a 48K list and
  losses on multilingual text. PR #29143 does the same through a `d2t` tensor in the model file.
  Both were open on 2026-11-12.
- What `docs/shortlist.md` adds is the step-time split on RDNA4 (a drafted token is the read of
  the MTP block plus the output layer, 3.13 ms), the effect on draft length and the confidence
  cutoff, and the numbers at 120K.

## What is still this repo's own

- The measured negative result: a perfect router over FFN blocks loses more than 3-bit
  quantization on Qwen3.5-9B (F3), blocks destroy the per-neuron advantage (F5), and an MTP
  verify batch needs 89-100% of the FFN (F5).
- The quantization and speculative-decoding bar those are held against (F4, F6-F9).
- Independent confirmation, on another build and machine, of the fork's two RDNA4 fixes and of
  the shortlist papers.

## Sources

- stew675/llama-cpp-rdna-boosts: <https://github.com/stew675/llama-cpp-rdna-boosts>, issues
  [#45](https://github.com/stew675/llama-cpp-rdna-boosts/issues/45) and
  [#56](https://github.com/stew675/llama-cpp-rdna-boosts/issues/56)
- FR-Spec: <https://arxiv.org/abs/2502.14856>
- VocabTrim: <https://arxiv.org/abs/2506.22694>
- llama.cpp PR #31031: <https://github.com/ggml-org/llama.cpp/pull/31031>
- llama.cpp PR #29143: <https://github.com/ggml-org/llama.cpp/pull/29143>
- llama.cpp issue #27796 (quantized KV slower than f16 on RDNA4): <https://github.com/ggml-org/llama.cpp/issues/27796>
- CMoE: <https://arxiv.org/abs/2502.04416>
- ExpertWeaver: <https://arxiv.org/html/2602.15521v1>
- MLPMoE: <https://www.arxiv.org/pdf/2511.21089>
- TEAL: <https://github.com/FasterDecoding/TEAL>
- ReLU Strikes Back: <https://arxiv.org/abs/2310.04564>
- MARS: <https://arxiv.org/pdf/2601.15498.pdf>
- A theory of acceptance in speculative decoding: <https://huggingface.co/papers/2606.30265>
- Online Speculative Decoding: <https://arxiv.org/html/2310.07177v1>
- OmniDraft: <https://arxiv.org/abs/2507.02659>
