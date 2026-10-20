# TelcoBench evaluation framework: core design and logic

## Purpose

TelcoBench evaluates how a language model answers held-out telecommunications
and network-infrastructure questions. It is deliberately an **evidence-first
generation and rubric-evaluation harness**, rather than an automatic
correctness test.

The released benchmark supplies question text and canonical route labels, but
not reference answers or executable oracles. Consequently, its primary result
is a reproducible, LLM-judged quality score (0–25), plus generation coverage.
It must not be reported as ground-truth accuracy, pass rate, or proof that a
model is operationally safe.

## Architecture at a glance

```text
Versioned config + local environment
               |
               v
Frozen manifest --validate--> selected (route, sample_id) records
               |                         |
               |                         v
               |                 generation protocol
               |                         |
               v                         v
      model / optional LoRA ------> generations.jsonl
                                         |
                              exact coverage, no errors
                                         |
                                         v
                      route-specific rubric + judge protocol
                                         |
                                         v
                judgments.jsonl + raw judge evidence + failures
                                         |
                                         v
                     strict aggregation and comparison report
```

There are four separable layers:

1. **Contract layer** validates the dataset and establishes stable record
   identities.
2. **Generation layer** runs one frozen prompting/decoding protocol against a
   local Hugging Face model (optionally with LoRA) or an OpenAI-compatible
   endpoint.
3. **Judging layer** scores only preserved successful generations using a
   versioned protocol and a route-specific rubric.
4. **Control/reporting layer** manages workers, evidence-gates stages,
   schedules available GPUs, and produces comparable summaries only when their
   provenance is compatible.

## 1. Benchmark contract and identity

`datasets/MANIFEST.json` is the benchmark’s root of trust. Before generation,
the framework validates each route file against its SHA-256 digest, expected
record count, required fields, route label, and uniqueness.

Each record requires:

| Field | Role |
|---|---|
| `prompt` | The question given to the evaluated model. |
| `sample_id` | Source-local sample identifier. |
| `teacher_routing_key` | Canonical route label, which must match the manifest route. |

The durable identity is **`(route, sample_id)`**, rendered internally as
`route:sample_id`; `sample_id` alone is not globally unique. This identity
links input records, generation rows, and judgment rows, and prevents a score
from being accidentally attributed to a similarly named record in another
route.

## 2. Frozen evaluation plan

A YAML configuration specifies the reproducible plan:

- manifest and generation-prompt protocol;
- output root, token cap, and model-loading policy;
- evaluated targets (base model, student adapter, or route specialist);
- artifact digests for local model weights and adapters; and
- judge protocol and route-rubric configuration.

Environment variables are expanded when loading the configuration. Unresolved
variables fail the run, which keeps paths, endpoints, and secrets out of the
published config while making the effective plan explicit at runtime.

For managed local runs, model and adapter files are SHA-256 verified before a
worker starts. A local digest cache reuses a verification only if device,
inode, size, and nanosecond modification time still match. Remote
OpenAI-compatible targets instead persist their endpoint and model identifier
in run metadata.

## 3. Generation logic

### Common protocol

The framework first validates the manifest, loads the versioned prompt
protocol, selects requested routes, and computes the expected record set. It
then writes `run.json` and `generation-progress.json` before loading a model,
so a monitor can distinguish model initialization from a missing worker.

Every completed attempt is appended immediately to `generations.jsonl` as one
of:

- `status: "ok"`: route, sample ID, original prompt, answer, token counts, and
  prompt hash are retained.
- `status: "error"`: route, sample ID, original prompt, exception type, and
  message are retained.

Failures are evidence, not discarded rows. At the end, `coverage.json`
records expected records, total rows, unique identities, missing records,
successful rows, and errors.

Resume is fail-closed: the existing run metadata must match the manifest hash,
selected routes, prompt-protocol hash, model, adapter, and decoding settings.
Only identities already marked `ok` or `error` are skipped.

### Local Hugging Face / PEFT backend

The MOPD-compatible local path uses CUDA and BF16, loads the tokenizer with
`fix_mistral_regex=True`, and renders messages through the model’s native
`apply_chat_template` with a system message, user prompt, and generation
prompt. For greedy protocols it strips sampling-only options before calling
`generate`.

When an adapter is configured, it is loaded under the explicit `student` name,
selected, and enabled. The run refuses to proceed if PEFT exposes no matching
LoRA layers or reports disabled layers. This prevents a supposed student result
from silently being a base-model result.

### OpenAI-compatible backend

The remote path verifies that the configured model appears in the provider’s
model catalog, then sends non-streaming chat-completions requests with the
same system/user structure. Calls use bounded retry with a five-minute delay.
An optional, explicitly recorded fallback can retry empty length-truncated
responses with provider-specific chat-template arguments. Each response stores
finish reason and token usage when supplied by the provider.

## 4. Evidence gate between generation and judging

Judging is normally permitted only after generation has:

- no missing manifest records;
- no generation-error rows; and
- unique generated identities equal to expected identities.

`--allow-incomplete` exists for controlled diagnostic use, but it weakens that
gate and should not be used for a complete comparison. The pipeline controller
enforces the strict form: a target whose generation ends with missing records,
errors, or inconsistent identity coverage fails rather than entering final
judging.

## 5. LLM-as-judge logic

The judge consumes frozen `generations.jsonl`, never regenerates model answers.
It filters to successful rows, loads judge provider/model identity from the
untracked environment, and writes a `judge-run*.json` plan containing hashes
of the generation evidence and judge protocol.

Each pending item is given:

- the versioned global judge protocol;
- its route label;
- a rendered route-specific rubric; and
- the original question and frozen model answer.

The rubric scores five independent integer dimensions from 0 to 5:

| Dimension | Intended interpretation |
|---|---|
| `factual` | Technical correctness in the route’s domain. |
| `completeness` | Coverage of material aspects of the question. |
| `terminology` | Correct standards, command, and domain vocabulary. |
| `hallucinations` | Absence of technically false or fabricated claims. |
| `usefulness` | Practical value to a network engineer. |

The judge must return a JSON array aligned to the input batch. The runner
requires exactly one object per item and an integer in `[0, 5]` for every
dimension; malformed responses, wrong batch lengths, and unsupported providers
are rejected. Successful parsed vectors are appended to `judgments.jsonl` with
their `total` (0–25). Raw provider responses go to `judge-raw.jsonl`; exhausted
batches go to `judge-failures.jsonl` and remain unscored, allowing a later
resume.

Route rubrics are mandatory for every pending route. Missing or empty route
context stops judging before any provider request. The current built-in judge
transport is Anthropic-compatible and fixes temperature at 0.

## 6. Aggregation and comparison

Aggregation strictly joins judgments to successful generations by stable
identity. It rejects an unknown or duplicate judgment, invalid dimension, or a
stored total that disagrees with the dimension sum.

It reports:

- `generated_ok`, `scored`, and `unscored_generated`;
- a **micro mean**: mean total across all scored examples; and
- a **macro route mean**: unweighted mean of per-route means.

The report builder refuses incomplete judgment coverage. For the standard
comparison it also requires both `cpt_base` and `mopd_student_step_2000`, equal
route coverage for those two targets, and exactly identical judge provenance.
It then emits micro, macro, and per-route `MOPD − CPT` deltas. Route-specialist
targets are combined only when their route sets do not overlap.

## 7. Coordinator and lifecycle behavior

`start`, `status`, and `stop` manage only workers launched by this package.
They store a PID, Linux start-time ticks, full command, log path, and start time
in a control file. Status checks both the PID start time and command line to
avoid treating a reused PID as an evaluator; stop signals the worker’s process
group.

`start-all` runs the dynamic coordinator for local GPU targets:

```text
queued target
  -> first truly idle GPU (<=1024 MiB, <=10% utilization)
  -> generation worker
  -> each finished route becomes eligible for judging
  -> one concurrent route judge (gateway protection)
  -> final resume judge writes aggregate target summary
  -> target complete
```

GPU assignments in configuration are fallback metadata rather than a hard
reservation. The coordinator uses `nvidia-smi`, assigns at most one active
evaluator to a GPU, and overlaps network-bound route judging with continued
GPU generation. It persists state atomically in `pipeline-state.json`. A
failure in any target stage makes the pipeline fail instead of emitting an
aggregate comparison that could mix incomplete evidence with complete results.

Remote-only configurations bypass GPU admission and launch their independent
workers directly. The TUI is read-only: it derives progress from persisted
files and does not alter worker state.

## 8. Evidence layout

Each target output directory is a self-contained audit trail:

| Artifact | Contents |
|---|---|
| `run.json` | Generation protocol, hashes, model/adapter identity, runtime, decoding, and prompt hashes. |
| `generation-progress.json` | Live total and per-route progress. |
| `generations.jsonl` | Append-only generation outcomes. |
| `coverage.json` | Final coverage and error counts. |
| `judge-run*.json` | Judge plan, progress, selected routes, and completion state. |
| `judge-raw.jsonl` | Raw judge responses for successful batches. |
| `judge-failures.jsonl` | Batches that exhausted retry attempts. |
| `judgments.jsonl` | Parsed rubric vectors. |
| `score-summary*.json` | Strict aggregated score and judge provenance. |

The final output root additionally contains `comparison-report.json` and a
compact Markdown comparison report after all required targets are fully judged.

## 9. Interpreting results responsibly

Use results to compare answer quality under a named prompt protocol, model or
adapter identity, judge model, and benchmark manifest. Always report those
identities, the neutral versus route-conditioned condition, coverage, errors,
and the distinction between micro and macro scores.

Do not infer operational correctness from a high score. The data card calls out
route imbalance, incomplete provenance/SME validation work, and the fact that
two Linux route labels are not independent held-out task constructions. For
strong research or product claims, audit a blinded stratified sample with
networking SMEs and report judge-human agreement by route and dimension.

## Key implementation modules

| Module | Responsibility |
|---|---|
| `contracts.py`, `dataset.py` | Manifest validation, SHA-256 utilities, identity, record iteration. |
| `plan.py`, `artifacts.py` | Config/environment resolution and local artifact verification. |
| `generation.py` | Local HF/PEFT and OpenAI-compatible generation plus coverage. |
| `rubrics.py`, `judging.py` | Route rubric rendering, judge calls, parsing, retries, and evidence. |
| `scoring.py`, `reporting.py` | Strict aggregation and provenance-compatible comparison. |
| `control.py`, `pipeline.py`, `tui.py` | Managed lifecycle, dynamic scheduling, and read-only monitoring. |
| `cli.py` | Command surface binding the layers together. |

