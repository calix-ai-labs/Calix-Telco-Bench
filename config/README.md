# Calix Telco-Bench configurations

This directory contains versioned, publishable Calix Telco-Bench evaluation
configurations. Run the default release configuration from the repository root:

```bash
.venv/bin/calix-telco-bench run-config --config config/default.yaml --target cpt_base --dry-run
```

All paths inside these files are relative to the repository root, so invoke the
CLI from that directory. Keep credentials in `.env` or the environment; do not
commit them to a configuration file.

`default.yaml` is the standard release configuration. `checkpoint-screen.yaml`
and `route-expert-v2.yaml` are alternate, environment-variable-based
reproduction configurations. They contain no endpoint, credential, or fixed
local artifact path; supply an approved artifact bundle through
`CALIX_TELCO_BENCH_RELEASE_ROOT`.

For a provider-neutral starting point, copy
`example-openai-compatible.yaml` to `config/local/`, export the three variables
below, and run a dry validation from the repository root:

```bash
export CALIX_TELCO_BENCH_BASE_URL=https://your-provider.example/v1
export CALIX_TELCO_BENCH_MODEL=your-model-id
export CALIX_TELCO_BENCH_API_KEY=your-secret-token
.venv/bin/calix-telco-bench run-config \
  --config config/local/example-openai-compatible.yaml \
  --target example_model \
  --dry-run
```

The sample uses the repository's benchmark data. Do not redistribute the data
until the release gates in `DATA_CARD.md` have been completed.

Local experiment configurations belong in `config/local/`. That directory is
ignored by Git because it may contain private endpoints, local model paths, or
other environment-specific details; do not add such files to a public release.
