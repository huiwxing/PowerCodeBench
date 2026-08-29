# PowerCodeBench reviewer runbook

This runbook separates the repository's supported claim-verification path
from fresh model execution. Run commands from the repository root.

## 1. Rebuild the reported claims (supported path)

The public artifact contains the frozen item, transcript, event, judgement,
and measurement records used by the verifier. The following path recomputes
the reported numerical values underlying tables, figures, contrasts, and
audits from those records; it does not call a model provider, load model
weights, or execute generated benchmark answers.

Requirements are a clean checkout, `sha256sum`, and Python 3.11. No Python
packages or network access are needed.

```bash
sha256sum --quiet -c checksums.sha256
python3 reproduce.py
```

For a quick smoke or a focused diagnostic:

```bash
python3 reproduce.py --group 1
python3 reproduce.py --help
```

`ARTIFACT_INDEX.md` maps each paper claim to its input and generating script.
`docs/PROVENANCE_BOUNDARIES.md` identifies evidence families represented by a
reduced record or deterministic rerun rather than by bulk historical logs.
Python 3.11 is intentional: Python 3.12 and later can change a few last-bit
floating-point results through the implementation of built-in `sum()`.

## 2. Inspect or import a method module

The repository copies are importable with `code/` on `PYTHONPATH`. This smoke
checks a CPU-side probing module without starting inference:

```bash
PYTHONPATH=code python3 -c "from probing.probe_framework import ProbeGenerator; print(ProbeGenerator.__name__)"
```

See `code/README.md` for the component-to-file map and each component's
runtime dependencies. This smoke checks the source layout without starting
model inference.

## 3. Runtime environments

### Reference-solution runtime

To execute the trusted reference programs in `benchmark.json`, create a small
environment from the three exact compatibility pins:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import sys, numpy, pandas, pandapower; print(sys.version.split()[0], numpy.__version__, pandas.__version__, pandapower.__version__)"
```

The expected package versions are `numpy 2.2.6`, `pandas 2.3.3`, and
`pandapower 3.4.0`, on Python 3.11. Treat a mismatch as a failed version gate.
Reference programs are ordinary unrestricted Python: execute only records
from a trusted checkout after the checksum check above. The helper in
`code/backend/utils.py` limits time and redirects common display side effects,
but it is an evaluation harness, not a security boundary for hostile code.

### Full method environment (two stages)

This environment is not needed for `reproduce.py`. For method exploration,
the repository records a two-stage setup so that installing vLLM does not
silently select the wrong torch/CUDA build:

```bash
conda env create -f environment/environment.yaml
conda activate powercodebench
python -m pip install vllm==0.15.1
```

Before a GPU run, verify the critical frozen versions rather than assuming
that a current package resolver reproduced the historical stack. This gate
has been exercised against the frozen experiment environment and matches its
recorded versions:

```bash
python - <<'PY'
from importlib.metadata import version

expected = {
    "accelerate": "1.13.0",
    "numpy": "2.2.6",
    "openai": "2.26.0",
    "pandapower": "3.4.0",
    "pandas": "2.3.3",
    "scikit-learn": "1.6.0",
    "torch": "2.9.1+cu129",
    "transformers": "4.57.6",
    "vllm": "0.15.1",
}
found = {name: version(name) for name in expected}
for name in expected:
    print(f"{name}: {found[name]} (expected {expected[name]})")
if found != expected:
    raise SystemExit("critical version mismatch")
PY
```

`environment/requirements.lock.txt` is the authoritative 281-package snapshot
of the experiment environment. It records the resolved environment for audit;
because CUDA/torch artifacts are platform-specific, it is not presented as a
portable `pip install -r` input.

## 4. Fresh open-weight and API runs

The repository also exposes the core backend, runner, prompt, probing, demand,
and intervention modules. Fresh runs require an operator-supplied execution
configuration in addition to the frozen environment:

- open-weight inference requires compatible GPUs, model access, and the
  verified torch/vLLM stack;
- multi-node jobs require scheduler and cache paths appropriate to the local
  site;
- provider runs require the corresponding service credentials.

Before launching a new run, explicitly record the model, backend, paths,
conditions, seeds, and resource topology rather than relying on the defaults
in `code/backend/probe_runner.py`. Verification of the paper's reported
results uses the frozen records indexed by `ARTIFACT_INDEX.md` and
`docs/PROVENANCE_BOUNDARIES.md`, as described in Section 1.

Credential variables recognized by the published backends (names only) are:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `DEEPSEEK_API_KEY`
- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` for gated model downloads

No credential variable is needed for checksum validation, `reproduce.py`, or
CPU-side inspection of the frozen evidence.
