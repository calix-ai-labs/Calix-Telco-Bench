# TelcoBench data card — held-out telecommunications and network-infrastructure prompts

## Status

**Release status: PUBLIC_PREVIEW (v0.1.0-preview)**

This is a preview release approved for public redistribution and evaluation.
The dataset is production-ready for benchmarking, though additional provenance
documentation and quality enhancements are tracked below for future releases.

## Intended use

Evaluate answer generation on TelcoBench's route-labelled telecommunications and
network-infrastructure questions. This data supports reproducible generation and
rubric-based or human evaluation. It does not support automatic accuracy claims
because it has prompts and route metadata, not reference answers or executable
oracles.

## Distribution

Approved TelcoBench benchmark assets are distributed with the canonical GitHub
repository or release archive. The Python package is runner-only and does not
distribute this dataset.

## Release-quality and provenance checklist

Public redistribution of the evaluation dataset is approved. The remaining
items document work needed for stronger provenance, quality, and research
claims; they do not supersede that approval.

- [x] Public redistribution approval for the evaluation dataset
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
