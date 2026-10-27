# Calix Telco-Bench

**Calix Telco-Bench — A benchmark for evaluating language models on telecommunications
and network-infrastructure tasks.**

Calix Telco-Bench is a reproducible benchmark harness for telecommunications and
network-infrastructure language models. It is a portable extraction of the
MOPD (Multi-expert On-Policy Distillation) evaluation contract:
it loads a frozen base model and optional LoRA adapter, generates one answer per
route-labelled prompt, preserves raw evidence, and aggregates validated rubric
judgments. It intentionally contains no model weights, credentials, internal
service endpoints, or training code.

## Scope and limits

The initial public release (v0.1.0) contains 2,850 evaluation prompts across 14
routes: BGP, OSPF, IS-IS, MPLS VPN, subscriber management, QoS, IP routing,
multicast, Linux diagnostics, networking fundamentals, L2 switching, PON/optical,
and WiFi gateways.

The data includes prompts and canonical route labels. A subset of 1,140 records
across `l2_switching`, `multicast`, `ospf`, `pon_optical`, `qos`, and
`wifi_gateway` also includes a `response` reference answer. The runner uses the
prompt field for generation; it does not use those reference answers. The
release has no executable oracles, and rubric-based scores are not accuracy,
pass rate, or proof of operational correctness. See [DATA_CARD.md](DATA_CARD.md)
for the data contents and release limitations.

## Metric limitations

Calix Telco-Bench currently measures complete generation coverage and rubric-based
quality scores. Although some records include `response` references, the
evaluator does not compare generations against them, and the release has no
executable oracles. Its scores are not ground-truth accuracy, pass-rate, or
proof of operational correctness. Report the judge model, prompt version, and
evaluation configuration alongside any results.

## License and Attribution

Calix Telco-Bench is developed and maintained by **Calix, Inc.** The software
and benchmark data distributed in this repository are licensed under the GNU
Affero General Public License v3.0 only (AGPL-3.0-only), subject to any
separate terms stated with individual third-party materials.

Copyright © 2026 Calix, Inc.

The AGPL-3.0-only terms apply to the software. Benchmark data is also
distributed under AGPL-3.0-only; Calix has approved public redistribution of
the data files included in this release. See [DATA_CARD.md](DATA_CARD.md) for
the approved scope and remaining provenance and quality work.

See the [LICENSE](https://github.com/calix-ai-labs/Calix-Telco-Bench/blob/main/LICENSE) file for the complete license terms.

Public disclosures of benchmark results should follow the [Calix Telco-Bench Citation and Reporting Policy](CITATION_AND_REPORTING_POLICY.md).

### Attribution

Calix Telco-Bench was developed by **Calix, Inc.** as part of its telecommunications AI research and evaluation initiatives.
Redistributions and modifications must preserve applicable copyright and license notices in accordance with AGPL-3.0.

## Distribution

The canonical Calix Telco-Bench distribution is a GitHub repository or release archive
containing the runner code and all benchmark assets approved for public
redistribution. Clone that release to run the benchmark. The `calix-telco-bench` PyPI
package, if published, provides the runner only and does not include benchmark
data, prompts, or evaluation configurations.

Calix has approved public redistribution of the data files included in this
repository, including the source data under `datasets/train` and the reference
answers present in some evaluation records. The [data card](DATA_CARD.md)
records the scope of that approval and distinguishes it from provenance,
privacy, and research-quality evidence that remains incomplete.

## Installation

Validation and aggregation require only PyYAML. Generation additionally needs
CUDA-compatible PyTorch, Transformers, and PEFT.

```bash
git clone https://github.com/calix-ai-labs/Calix-Telco-Bench.git
cd Calix-Telco-Bench
./setup.sh
```

`setup.sh` installs `uv` into the invoking user's Python environment only when
needed, creates `.venv`, installs generation/TUI/judge/test extras, validates the
manifest, and creates an untracked `.env` from `.env.example`. It does not
download model weights or create credentials. It uses CPython 3.11 by default,
matching the MOPD judge runtime; set `CALIX_TELCO_BENCH_PYTHON` only to a
supported CPython 3.10–3.12 interpreter when necessary.

### Manual installation of optional dependencies

If you need to install dependencies separately or the automated setup fails:

```bash
# Activate virtual environment
source .venv/bin/activate

# For model generation (required for evaluation)
pip install -e ".[huggingface]"

# For terminal UI monitoring
pip install -e ".[tui]"

# For LLM-as-judge scoring
pip install -e ".[judge]"

# For development and testing
pip install -e ".[dev]"

# Install all optional dependencies at once
pip install -e ".[huggingface,tui,judge,dev]"
```

**Note**: The `huggingface` extras include PyTorch (2.5GB+) and may take several
minutes to download and install depending on your network connection.

The validator checks file SHA-256 values, record counts, required fields, route
labels, and `(route, sample_id)` uniqueness. Source-local `sample_id` values
are not globally unique by themselves.

Public configurations are stored in [`config/`](config/). Keep provider
credentials, private endpoints, and environment-specific model paths only in
the ignored `config/local/` directory or in environment variables.

For an OpenAI-compatible provider, begin with
[`config/example-openai-compatible.yaml`](config/example-openai-compatible.yaml).
It contains no credential, endpoint, model ID, or local path; copy it to
`config/local/` and supply your own values through environment variables.

## Command usage

Run all commands from the cloned repository using its package-local virtual
environment:

```bash
cd ~/Calix-Telco-Bench
```

| Goal | Command |
|---|---|
| Verify dataset integrity | `.venv/bin/calix-telco-bench validate datasets/MANIFEST.json` |
| List routes and record counts | `.venv/bin/calix-telco-bench list-routes datasets/MANIFEST.json` |
| Verify a configured model/adapter target without loading it | `.venv/bin/calix-telco-bench run-config --config config/default.yaml --target cpt_base --dry-run` |
| Start one managed generation target | `.venv/bin/calix-telco-bench start --config config/default.yaml --target cpt_base` |
| Start the full managed generation → validation → judging pipeline | `.venv/bin/calix-telco-bench start-all --config config/default.yaml` |
| Inspect one target worker | `.venv/bin/calix-telco-bench status --config config/default.yaml --target cpt_base` |
| Inspect every configured worker | `.venv/bin/calix-telco-bench status-all --config config/default.yaml` |
| Stop one managed generation target | `.venv/bin/calix-telco-bench stop --config config/default.yaml --target cpt_base` |
| Stop the end-to-end coordinator, active generation, and judging | `.venv/bin/calix-telco-bench stop-all --config config/default.yaml` |
| Resume an interrupted generation | `.venv/bin/calix-telco-bench start --config config/default.yaml --target cpt_base --resume` |
| Monitor every configured target | `.venv/bin/calix-telco-bench tui --config config/default.yaml --refresh-seconds 2` |
| Validate the full judge plan without API calls | `.venv/bin/calix-telco-bench judge-all --config config/default.yaml --dry-run` |
| Judge one completed target | `.venv/bin/calix-telco-bench judge --config config/default.yaml --target cpt_base` |
| Judge every completed target | `.venv/bin/calix-telco-bench judge-all --config config/default.yaml` |
| Start parallel all-target judging in the background | `.venv/bin/calix-telco-bench judge-all-start --config config/default.yaml --resume --parallelism 4` |
| Inspect background judging | `.venv/bin/calix-telco-bench judge-all-status --config config/default.yaml` |
| Stop background judging | `.venv/bin/calix-telco-bench judge-all-stop --config config/default.yaml` |
| Resume interrupted judging | `.venv/bin/calix-telco-bench judge-all --config config/default.yaml --resume` |
| Build the final comparison reports | `.venv/bin/calix-telco-bench report --config config/default.yaml` |

### Release-grade command sequence

```bash
# Confirm installation, inputs, and local Release_1.0 paths.
.venv/bin/calix-telco-bench validate datasets/MANIFEST.json
.venv/bin/calix-telco-bench run-config --config config/default.yaml --target cpt_base --dry-run

# Start the end-to-end pipeline. It dispatches one model per genuinely free
# GPU, validates each target, and begins its judging immediately on completion.
.venv/bin/calix-telco-bench start-all --config config/default.yaml

# In another terminal, monitor generation and later live judge progress.
.venv/bin/calix-telco-bench tui --config config/default.yaml --refresh-seconds 2

# When the monitor reports pipeline COMPLETE, produce the comparative report.
.venv/bin/calix-telco-bench report --config config/default.yaml
```

Use `--help` after any command for its full options. The TUI is read-only;
`Ctrl+C` exits only the monitor, never a managed generation worker.

The TUI shows the end-to-end coordinator, the separate judge coordinator,
generation coverage, live judge coverage/mean, and active workers. Each target
row separately reports generation and judging state, route, rate, ETA, score,
and failed batches. The pipeline never starts the judge unless every target has
exact frozen-manifest coverage and zero generation errors.

GPU placement is dynamic: the coordinator polls NVML through `nvidia-smi` and
uses only GPUs below its idle memory/utilization thresholds. A target records
its actual assigned CUDA device in its generation evidence; target entries in
`config/default.yaml` is fallback metadata, not a reservation. Judging is network-bound
and starts per target as soon as that target's generation coverage is valid;
it does not wait for unrelated generation targets.

`start-all` is a fresh-run command: unless `--resume` is supplied, it clears
only the configured target evidence directories and stale pipeline/judge state
before starting. This prevents prior partial generations or judgments from
appearing as current progress in the TUI. It does not remove unrelated
root-level validation evidence such as `adapter-effect-check.json`.

## Calix Telco-Bench generation (MOPD-compatible)

The generator follows the MOPD evaluator’s essential behavior:

- CUDA-only BF16 model loading;
- `fix_mistral_regex=True` tokenizer repair;
- either a neutral deployment-like prompt shared by every route or a separately
  labelled route-conditioned diagnostic prompt;
- native `tokenizer.apply_chat_template(..., add_generation_prompt=True)`;
- greedy decoding (`do_sample=false`, temperature 0, `top_p=1`); and
- optional PEFT/LoRA loading for a MOPD student or specialist.

```bash
calix-telco-bench run \
  --manifest datasets/MANIFEST.json \
  --prompts prompts/generation-neutral-v1.yaml \
  --model /path/to/cpt-base \
  --model-revision <immutable-model-revision-or-sha256> \
  --adapter /path/to/mopd-student-adapter \
  --adapter-revision <adapter-sha256> \
  --device cuda:0 \
  --local-files-only \
  --output outputs/mopd-student-v1
```

Omit `--adapter` for the frozen base. Add `--limit-per-route 2` for a smoke
run; `--max-new-tokens N` overrides the prompt-protocol cap. Use
`--trust-remote-code` only after reviewing the model repository. Resume an
interrupted run only with the same model and protocol:

```bash
calix-telco-bench run ... --output outputs/mopd-student-v1 --resume
```

Completed and failed examples are both retained. Never remove failures before
reporting coverage.

`prompts/generation-neutral-v1.yaml` is the primary deployment-like condition:
every record receives exactly the same system prompt and no route label. The
older `prompts/generation-v1.yaml` is a route-conditioned diagnostic. It names
the evaluated expertise (for example, BGP) and must never be presented as a
deployment result. If reported, run both conditions for every comparison model.

## Calix Telco-Bench configuration and terminal monitor

`config/default.yaml` is a versioned execution contract for an approved local artifact
bundle. It pins the expected CPT/base and LoRA SHA-256 identities, neutral
protocol, target devices, and teacher-to-route ownership. Set
`CALIX_TELCO_BENCH_RELEASE_ROOT` in the untracked `.env` to the local bundle;
the committed config contains no user- or organization-specific absolute path.
It writes evidence below the bundle's restricted evaluation tier; that does not
authorize public release of those results.

Validate a target without loading a model:

```bash
.venv/bin/calix-telco-bench run-config --config config/default.yaml --target cpt_base --dry-run
```

Start one managed target in the background. The controller writes a PID record
and `worker.log` in that target's evidence directory; it never uses an MOPD
virtual environment or system service manager:

```bash
.venv/bin/calix-telco-bench start --config config/default.yaml --target mopd_student_step_2000
```

View all persisted progress, worker state, and GPU telemetry in the TUI:

```bash
.venv/bin/calix-telco-bench tui --config config/default.yaml --refresh-seconds 2
```

The TUI is read-only and derives generation state from `run.json` and
`generations.jsonl`. Start/stop is intentionally explicit in another terminal:

```bash
.venv/bin/calix-telco-bench status --config config/default.yaml --target mopd_student_step_2000
.venv/bin/calix-telco-bench stop --config config/default.yaml --target mopd_student_step_2000
```

`stop` sends `SIGTERM` only to the process group recorded by a prior `start`.
To resume an interrupted target with the identical frozen contract, use
`start ... --resume`. For an isolated smoke run, retain `run-config` with
`--limit-per-route 1 --output-suffix smoke-1`.

### Managed evaluation controls

Start the primary CPT-base versus final-MOPD-student comparison:

```bash
cd ~/Calix-Telco-Bench

.venv/bin/calix-telco-bench start --config config/default.yaml --target cpt_base
.venv/bin/calix-telco-bench start --config config/default.yaml --target mopd_student_step_2000
```

See live worker state, persisted-evidence coverage, and GPU telemetry:

```bash
.venv/bin/calix-telco-bench tui --config config/default.yaml --refresh-seconds 2
```

Check or safely stop either managed worker:

```bash
.venv/bin/calix-telco-bench status --config config/default.yaml --target cpt_base
.venv/bin/calix-telco-bench stop --config config/default.yaml --target cpt_base

.venv/bin/calix-telco-bench stop --config config/default.yaml --target mopd_student_step_2000
```

Each `start` writes a PID record and `worker.log` beside that target's
restricted evidence. `stop` sends `SIGTERM` only to a process group previously
started by this framework.

Judge identity is local runtime configuration, not a committed benchmark
setting. Copy `.env.example` to `.env`, set `CALIX_TELCO_BENCH_JUDGE_PROVIDER`,
`CALIX_TELCO_BENCH_JUDGE_MODEL`, and, where available,
`CALIX_TELCO_BENCH_JUDGE_MODEL_REVISION`. The `score` command reads those values
through the package's `.env` utility; CLI flags override them for an explicit
one-off run. Credentials such as `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY`
remain environment-only and are never written to the score report.

For the public Anthropic API, set `ANTHROPIC_API_KEY` in `.env` and leave
`ANTHROPIC_BASE_URL` unset. For a custom Anthropic-compatible endpoint, set both
`ANTHROPIC_BASE_URL` (e.g., `https://api.yourcompany.com/`) and
`ANTHROPIC_AUTH_TOKEN` in `.env`.

### LLM-as-judge stage

The judge is a distinct, resumable stage and never runs while GPU generation is
still mutable. It reads completed `generations.jsonl`, calls the provider named
in `.env`, writes raw provider responses, validated `judgments.jsonl`, failures,
and `score-summary.json` beside the target evidence. It defaults to five answers
per request, temperature zero, and three retries per batch.

The currently supported provider is `anthropic`. An existing checkout that was
created before this feature can install the optional client with:

```bash
uv sync --extra huggingface --extra tui --extra judge --extra dev
```

Before a costly full run, diagnose the local endpoint configuration without a
provider request, then test serial and bounded-parallel gateway behavior. The
script redacts credentials and uses the identical Anthropic/httpx2 transport as
the evaluator:

```bash
.venv/bin/python scripts/diagnose_judge_connection.py
.venv/bin/python scripts/diagnose_judge_connection.py --requests 3 --parallelism 1
.venv/bin/python scripts/diagnose_judge_connection.py --requests 4 --parallelism 4
```

Use the highest probe concurrency with zero failures as the `judge-all`
`--parallelism` value. Do not launch the full run when the serial probe fails.

Before treating a CPT-versus-student result as evidence, verify that the
configured LoRA changes next-token logits on the common backbone. This
read-only one-prompt check is separate from quality evaluation and writes a
small auditable result when `--output` is supplied:

```bash
.venv/bin/python scripts/verify_adapter_effect.py \
  --config config/default.yaml \
  --device cuda:2 \
  --output /data/shared/Release_1.0/restricted_reproduction/evaluations/neutral-v1/adapter-effect-check.json
```

First validate the complete frozen target without making a provider request:

```bash
.venv/bin/calix-telco-bench judge --config config/default.yaml --target cpt_base --dry-run
```

Then run it after the target reaches clean 100% generation coverage:

```bash
.venv/bin/calix-telco-bench judge --config config/default.yaml --target cpt_base
```

Judge the complete CPT, MOPD-student, and six routed-teacher targets under the
same judge identity and protocol with one command. It performs an all-target
coverage preflight before it submits any provider request:

```bash
.venv/bin/calix-telco-bench judge-all --config config/default.yaml --dry-run
.venv/bin/calix-telco-bench judge-all-start --config config/default.yaml --resume --parallelism 4
```

Useful judge options:

| Option | Meaning |
|---|---|
| `--dry-run` | Validate files, coverage, `.env` judge identity, and the exact judge plan; do not call the provider. |
| `--resume` | Continue from existing valid `judgments.jsonl`; previously scored samples are never submitted again. |
| `--batch-size N` | Answers per provider request; default `5`. Lower this only to work around provider output limits. |
| `--retries N` | Retries per failed/malformed batch; default `3`. |
| `--parallelism N` | With `judge-all`, independent target judges allowed concurrently; default `4`. |
| `--max-items N` | Judge at most `N` currently unjudged answers; useful only for a controlled smoke check. Follow it with `--resume` for the full result. |
| `--targets a,b` | With `judge-all`, restrict work to a named target subset. This is not a paper-comparison result. |
| `--allow-incomplete` | Permit judging a still-growing/failed generation set. Avoid for the final result. |

### Per-route rubric

The judge scores every frozen answer with the same five dimensions, but the
*definition* of each dimension is route-specific. The generic part — the five
5/3/1/0 scale anchors, the completeness and usefulness guidance, and the
guardrail that the `hallucinations` dimension only penalizes claims that are
factually false — lives in code as a shared template. What varies by route is the
domain scope: which facts count as correct and which terms count as correct
terminology. The config `prompts/rubric_config.yaml` stores only that scope; the
judge renders it into each item's block at judging time.

Each route scope is written as the full *range* of correct behavior and
terminology for the route, not a sample list and not a single reference document —
no single standard completely defines a protocol, so entries name the scope (for
example, "all BGP facts and behavior across all variants") and let the judge judge
any correct answer within it. The routes here are exactly the 14 in `MANIFEST.json`
(`bgp`, `ospf`, `isis`, `mpls_vpn`, `bng_subscriber_mgmt`, `qos`, `ip_routing`,
`multicast`, `network_fundamentals`, `l2_switching`, `pon_optical`, `wifi_gateway`,
`linux_tool_calling`, `linux_tool_agentic`), with `default_route` as a fallback.

Judging is refused, before any provider request, when a route with pending
generation has no rubric scope defined. `--dry-run` and the normal judge pipeline
enforce this, so no route is ever silently scored without an anchored rubric.
Add or extend a route by editing one short scope entry in `prompts/rubric_config.yaml`
— no code change is required.

The normal release-grade sequence is:

```bash
# 1. No external request: every target must be complete, error-free, and resolvable.
.venv/bin/calix-telco-bench judge-all --config config/default.yaml --dry-run

# 2. Submit all frozen outputs under the same judge contract.
.venv/bin/calix-telco-bench judge-all --config config/default.yaml

# 3. If interrupted, continue without resubmitting valid judgments.
.venv/bin/calix-telco-bench judge-all-start --config config/default.yaml --resume --parallelism 4

# 4. Produce the comparative, paper-ready aggregate.
.venv/bin/calix-telco-bench report --config config/default.yaml
```

After every target is fully judged, create the provenance-checked CPT versus
MOPD versus routed-teacher aggregate report:

```bash
.venv/bin/calix-telco-bench report --config config/default.yaml
```

Use `--resume` after an interrupted judge run. The TUI displays judged/valid
count and the provisional mean score as new validated judgment records arrive.
While a non-dry judge run has unscored valid answers, the target state changes
from `COMPLETE` to `JUDGING`. Run the judge in one terminal and monitor it in
another:

```bash
.venv/bin/calix-telco-bench judge-all-start --config config/default.yaml --resume --parallelism 4
.venv/bin/calix-telco-bench tui --config config/default.yaml --refresh-seconds 2
```

Do not present a provisional live score as the final result; retain the final
`score-summary.json` after judging is complete.

## Outputs and comparison controls

| File | Contents |
|---|---|
| `run.json` | Exact prompt text and SHA-256 per route, manifest/protocol hashes, tokenizer chat-template text and hash, decoding, model/adapter revisions, device, and runtime metadata. |
| `generations.jsonl` | One answer or captured error per example, with token counts and the exact system-prompt hash for successes. |
| `coverage.json` | Expected, generated, missing, successful, and failed counts. |
| `judge-run.json` | Provider/model identity, protocol and generation hashes, deterministic judge parameters, and coverage at judge start; never credentials. |
| `judgments.jsonl` | Validated per-example rubric vectors and totals. |
| `judge-raw.jsonl` | Raw provider response for audit/reparse, paired with the submitted route/sample IDs. Restrict access according to provider/data policy. |
| `judge-failures.jsonl` | Failed or malformed batches after bounded retries. |
| `score-summary.json` | Per-target micro/macro rubric summary and remaining unscored count. |
| `comparison-report.json` / `.md` | CPT, MOPD student, and non-overlapping routed-teacher aggregate comparison; emitted only after compatible complete judge evidence exists. |

Two systems are comparable only when benchmark-manifest and prompt-protocol
hashes match and rendering, decoding, token budget, model/adapter identity, and
judge contract are held fixed. Supply `--model-revision` and
`--adapter-revision` with immutable revisions or SHA-256 values; these values
are recorded rather than inferred from a mutable path.

## Rubric and scoring

The rubric has five integer dimensions in `[0,5]`: `factual` (technical
correctness), `completeness` (sufficient coverage), `terminology` (networking
and command precision), `hallucinations` (absence of technically false or
fabricated claims), and `usefulness` (safe operational value). `total` is their
sum, from 0–25. For diagnosis-only questions, penalize state-changing commands
unless remediation was explicitly requested.

Create external, versioned judge or blinded-human JSONL records like:

```json
{"route":"ospf","sample_id":"OSPF-EVAL-0001","factual":5,"completeness":4,"terminology":5,"hallucinations":5,"usefulness":4,"total":23}
```

```bash
.venv/bin/calix-telco-bench score \
  --generations outputs/mopd-student-v1/generations.jsonl \
  --judgments outputs/mopd-student-v1/judgments.jsonl \
  --judge-protocol prompts/JUDGE_PROTOCOL.md \
  --judge-provider anthropic \
  --judge-model claude-sonnet-4-5 \
  --output outputs/mopd-student-v1/score-summary.json
```

The scorer rejects unknown/duplicate route-sample pairs, out-of-range values,
and inconsistent totals. The score output hashes the generated and judgment
JSONL inputs, the scorer implementation, and supplied judge protocol/prompt
files. It reports coverage, per-route mean, micro mean (example-weighted), and
macro route mean (route-weighted). Report all of them, plus missing judgments
and generation failures. A rubric result is never reference-grounded accuracy.

## Publication checklist

Retain the Git commit; dataset/prompt hashes; model and adapter identity; exact
command; raw generations and judgments; judge identity and prompt/version;
coverage; and aggregation output. Before making public accuracy or
generalization claims, validate reference-based scoring for routes with
available `response` fields or add executable checks, and add blinded SME
review and judge–human agreement. Track the remaining
provenance, privacy, split, and quality gates in the data card.

## Package layout

```text
src/calix_telco_bench/
  contracts.py   immutable release identities and validation
  dataset.py     verified route-record iterator
  generation.py  MOPD-compatible Hugging Face + PEFT generation
  scoring.py     strict rubric aggregation
  cli.py         command-line interface
tests/           offline release-contract checks
```
