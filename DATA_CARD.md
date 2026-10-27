# Calix Telco-Bench data card — telecommunications and network-infrastructure evaluation data

## Status

**Release status: PUBLIC_RELEASE (v0.1.0)**

Calix has approved public redistribution and evaluation of the data files
included in this repository. The approval covers both `datasets/heldout/` and
`datasets/train/`, including `response` and `reference_answer` fields and the
source-location metadata that relates records between those directories.
Approval to redistribute is distinct from completion of the provenance,
privacy, and research-quality work listed below. This public release is not a
certification that every record is accurate or suitable for every
benchmarking claim.

The benchmark data in this release is distributed under AGPL-3.0-only, as
described in the repository's [LICENSE](LICENSE), except for any individual
material accompanied by separate terms.

## Intended use

Evaluate answer generation on Calix Telco-Bench's route-labelled telecommunications and
network-infrastructure prompts. The 2,850-record evaluation set includes
`response` reference answers for 1,140 records across `l2_switching`,
`multicast`, `ospf`, `pon_optical`, `qos`, and `wifi_gateway`; the other routes
contain prompts and route metadata without reference answers. The runner uses
the prompt field for generation and does not use `response`. The release has no
executable oracles. Reference answers and rubric-based or human evaluation do
not, by themselves, support automatic accuracy claims.

## Distribution

Calix's public redistribution approval covers the data files in this
repository, including the source files under `datasets/train/`, the evaluation
files under `datasets/heldout/`, and their answer and source-location fields.
The Python package is runner-only and does not distribute this dataset.

## Release-quality and provenance checklist

Public redistribution is approved for the repository data scope described
above. The checklist below tracks provenance and research-quality evidence;
unchecked items are not represented as completed by the redistribution
approval. Results and claims should respect the known limitations below.

- [x] Public redistribution approval for all data files included in this release
- [ ] Per-source license, provenance, and version record
- [ ] Generation, filtering, and human-review process
- [ ] Split method and duplicate/near-duplicate audit
- [ ] PII, secrets, customer-data, and internal-hostname scan
- [ ] SME review and route-label agreement study
- [ ] Cross-domain routing policy
- [ ] Reference answers and/or executable checks for a labelled subset
- [ ] Versioned archive and DOI plan

## Known limitations

**v1 route-label caveat:** `linux_tool_calling` and `linux_tool_agentic` are
kept as separate route labels for compatibility, but they are not an
independently constructed pair of held-out tasks. A future dataset revision
will replace the agentic split; do not treat these labels as independent
evidence.

Route labels define canonical evaluation scope, not exclusive real-world
incident ownership. The data is imbalanced by route and does not claim complete
networking coverage. LLM-judge scores need versioned prompts, recorded model
identity, and human-agreement evidence.

`sample_id` originates in individual source sets and is not globally unique.
The stable public record identity is the pair `(teacher_routing_key, sample_id)`.
