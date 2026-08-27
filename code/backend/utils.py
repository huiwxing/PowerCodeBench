# --------------------------------------------------------------------------
# Repository copy of the pipeline module utils.py, unmodified apart from this
# header. Runs CPU-only against the archived corpus/spec files in this
# repository. See code/README.md for the module map.
# --------------------------------------------------------------------------
from io import StringIO
from unittest.mock import patch, MagicMock
import traceback
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import os
import sys
import re
import math
import json 

SANDBOX_DIR = Path("sandbox_output")


@contextmanager
def _silence_process_fds(enabled: bool = True):
    """Temporarily silence writes that bypass Python's sys.stdout/sys.stderr."""
    if not enabled or os.name != "posix":
        yield
        return

    saved_stdout_fd = saved_stderr_fd = devnull_fd = None
    try:
        try:
            sys.__stdout__.flush()
            sys.__stderr__.flush()
        except Exception:
            pass
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        if saved_stdout_fd is not None:
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)
        if saved_stderr_fd is not None:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


def execute_code_safely_subprocess(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Subprocess-isolated variant of execute_code_safely (same result schema).

    Required when the executed code drives native extensions that can crash
    the interpreter (e.g. the OpenDSS C++ engine): a segfault kills only the
    child, and is reported as error_type="NativeCrash" instead of taking the
    whole runner down.
    """
    import re as _re
    import subprocess
    import tempfile

    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    result = {"success": False, "error": None, "output": "", "error_type": None}

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", dir=str(SANDBOX_DIR), delete=False
    ) as f:
        f.write(code)
        script_path = f.name
    env = dict(os.environ)
    env.update(
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(SANDBOX_DIR),
            env=env,
        )
        result["output"] = proc.stdout
        if proc.returncode == 0:
            result["success"] = True
        elif proc.returncode < 0:
            result["error"] = f"terminated by signal {-proc.returncode}"
            result["error_type"] = "NativeCrash"
        else:
            stderr_lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
            last = stderr_lines[-1] if stderr_lines else ""
            m = _re.match(r"([A-Za-z_][\w.]*?(?:Error|Exception|Warning|Exit|Interrupt))\b", last)
            result["error"] = last or f"exit code {proc.returncode}"
            result["error_type"] = m.group(1).split(".")[-1] if m else "RuntimeError"
    except subprocess.TimeoutExpired:
        result["error"] = f"Code execution exceeded {timeout} seconds"
        result["error_type"] = "TimeoutError"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
    return result


def execute_code_safely(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code with display/save side effects mocked out and a wall-clock timeout.

    Returns a dict with success / error / output / error_type fields.
    """
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()

    @contextmanager
    def timeout_handler(seconds):
        """SIGALRM-based timeout (POSIX only)."""

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution exceeded {seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    output_buffer = StringIO()
    error_buffer = StringIO()
    sys.stdout = output_buffer
    sys.stderr = error_buffer

    # Suppress noisy library logging during generated-code execution. The
    # benchmark result should come from explicit print() output, not warnings
    # emitted by pandapower internals.
    import logging
    import warnings

    suppress_exec_logs = os.environ.get(
        "BENCHMARK_SUPPRESS_EXEC_LOGS", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    original_logging_disable = logging.root.manager.disable
    if suppress_exec_logs:
        logging.disable(logging.CRITICAL)

    result = {
        'success': False,
        'error': None,
        'output': '',
        'error_type': None
    }

    mock_show = MagicMock()
    mock_savefig = MagicMock()
    mock_figure = MagicMock()
    mock_open = MagicMock()
    mock_webbrowser = MagicMock()

    try:
        with _silence_process_fds(suppress_exec_logs):
            with warnings.catch_warnings():
                if suppress_exec_logs:
                    warnings.simplefilter("ignore")
                with timeout_handler(timeout):
                    with patch('matplotlib.pyplot.show', mock_show), \
                            patch('matplotlib.pyplot.savefig', mock_savefig), \
                            patch('matplotlib.pyplot.figure', mock_figure), \
                            patch('webbrowser.open', mock_open), \
                            patch('webbrowser.open_new_tab', mock_open), \
                            patch('webbrowser.open_new', mock_open):
                        namespace = {}

                        exec("import matplotlib", namespace)
                        exec("import matplotlib.pyplot as plt", namespace)
                        exec("matplotlib.use('Agg')", namespace)
                        os.chdir(SANDBOX_DIR)

                        code = re.sub(r'\n\s*exit\(.*?\)', '\n# exit() removed', code)
                        code = re.sub(r'\n\s*sys\.exit\(.*?\)', '\n# sys.exit() removed', code)
                        code = re.sub(r"output_path\s*=\s*['\"][^'\"]*['\"]", "output_path='./output'", code)

                        exec(code, namespace)

                        result['success'] = True
                        result['output'] = output_buffer.getvalue()

    except TimeoutError as e:
        result['error'] = str(e)
        result['error_type'] = 'TimeoutError'
        result['output'] = error_buffer.getvalue()

    except SyntaxError as e:
        result['error'] = str(e)
        result['error_type'] = 'SyntaxError'
        result['output'] = error_buffer.getvalue()

    except Exception as e:
        result['error'] = str(e)
        result['error_type'] = type(e).__name__
        result['output'] = error_buffer.getvalue()
        result['traceback'] = traceback.format_exc()

    finally:
        os.chdir(original_cwd)
        logging.disable(original_logging_disable)

        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return result

def parse_error(exec_result: Dict, code: str = None) -> str:
    """Extract compact error info from execution result."""
    import re
    if exec_result.get("success"):
        return ""
    tb = exec_result.get('traceback', '')
    lines = tb.strip().split('\n')
 
    line_num = None
    for line in lines:
        if 'File "<string>"' in line or "File '<string>'" in line:
            match = re.search(r'line (\d+)', line)
            if match:
                line_num = int(match.group(1))
                break
 
    code_line = None
    if line_num and code:
        code_lines = code.split('\n')
        if 1 <= line_num <= len(code_lines):
            code_line = code_lines[line_num - 1].strip()
 
    parts = []
    if line_num:
        parts.append(f"Line {line_num}")
    if code_line:
        parts.append(code_line)
    parts.append(f"{exec_result.get('error_type', 'Error')}: {exec_result.get('error', '')}")
    return ' | '.join(parts)


def _clean_code(raw: str) -> Optional[str]:
    """Extract executable Python from LLM output."""
    if not raw:
        return None

    code = raw.strip()

    # Extract from markdown code block
    match = re.search(r'```(?:python|py)?\s*\n(.*?)```', code, re.DOTALL)
    if match:
        code = match.group(1).strip()

    # Remove trailing ```
    code = re.sub(r'\s*```\s*$', '', code)

    # Skip if obviously not code
    if not code or len(code) < 10:
        return None

    # Check for at least one code-like line
    code_indicators = ['import ', 'from ', 'def ', 'class ', '= ', 'print(', 'pp.', 'net.']
    if not any(ind in code for ind in code_indicators):
        return None

    return code


def clean_generated_code(code: str) -> str:
    """Extract executable Python from raw LLM output.

    Tries in order: <CODE></CODE> tags, Python list-of-strings literal,
    markdown fenced block, then heuristic line-prefix matching.
    """
    if not code:
        return code

    import re
    import ast
    from textwrap import dedent

    CODE_SEPARATOR_1 = "<CODE>"
    CODE_SEPARATOR_2 = "</CODE>"

    stripped = code.strip()

    if CODE_SEPARATOR_1 in stripped and CODE_SEPARATOR_2 in stripped:
        pattern = re.escape(CODE_SEPARATOR_1) + r'\s*(.*?)\s*' + re.escape(CODE_SEPARATOR_2)
        match = re.search(pattern, stripped, re.DOTALL)
        if match:
            stripped = match.group(1).strip()
            return dedent(stripped).strip()

    if stripped.startswith('[') and stripped.endswith(']'):
        try:
            code_list = ast.literal_eval(stripped)
            if isinstance(code_list, list):
                stripped = '\n'.join(str(item) for item in code_list if item)
        except Exception:
            pass

    pattern = r'```(?:python|py)?\s*\n(.*?)(?:```|$)'
    matches = re.search(pattern, stripped, re.DOTALL | re.IGNORECASE)
    if matches:
        stripped = matches.group(1).strip()

    stripped = re.sub(r'\s*```\s*$', '', stripped)

    code_patterns = [
        r'^import\s+',
        r'^from\s+\w+\s+import',
        r'^def\s+',
        r'^class\s+',
        r'^@\w+',
        r'^\w+\s*=\s*',
        r'^if\s+',
        r'^for\s+',
        r'^while\s+',
        r'^try:',
        r'^with\s+',
        r'^return\s+',
        r'^print\(',
        r'^\w+\.',
    ]

    code_lines = []
    code_started = False
    lines = stripped.split('\n')

    for line in lines:
        if code_started:
            code_lines.append(line)
            continue

        stripped_line = line.lstrip()
        if stripped_line and any(re.match(pattern, stripped_line) for pattern in code_patterns):
            code_started = True
            code_lines.append(line)

    result = '\n'.join(code_lines).strip()
    result = dedent(result).strip()

    return result



def match_ground_truth(
    code_output: str,
    ground_truth: Any,
    gt_type: str,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Compare executed code output against ground truth.

    Args:
        code_output: raw stdout string from code execution
        ground_truth: expected value from benchmark
        gt_type: one of "float", "int", "bool", "dict"
        rtol: relative tolerance for float comparison
        atol: absolute tolerance for float comparison

    Returns:
        {
            "match": bool,
            "parsed_value": Any,     # what we parsed from output
            "error": str or None,    # parsing/comparison error
            "abs_error": float,      # for float type, absolute difference
            "rel_error": float,      # for float type, relative difference
        }
    """
    result = {
        "match": False,
        "parsed_value": None,
        "error": None,
        "abs_error": None,
        "rel_error": None,
    }

    if code_output is None or code_output.strip() == "":
        result["error"] = "empty_output"
        return result

    output = code_output.strip()
    # Take only the last non-empty line (the print(result) output)
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    if not lines:
        result["error"] = "empty_output"
        return result
    last_line = lines[-1]

    try:
        if gt_type == "float":
            return _match_float(last_line, ground_truth, rtol, atol)
        elif gt_type == "int":
            return _match_int(last_line, ground_truth)
        elif gt_type == "bool":
            return _match_bool(last_line, ground_truth)
        elif gt_type == "dict":
            return _match_dict(last_line, ground_truth, rtol, atol)
        else:
            result["error"] = f"unknown_gt_type:{gt_type}"
            return result
    except Exception as e:
        result["error"] = f"comparison_error:{type(e).__name__}:{e}"
        return result


def _match_float(output: str, gt: float, rtol: float, atol: float) -> Dict:
    """Float matching with combined relative + absolute tolerance."""
    result = {"match": False, "parsed_value": None, "error": None,
              "abs_error": None, "rel_error": None}
    try:
        parsed = float(output)
    except ValueError:
        result["error"] = f"cannot_parse_float:{output[:50]}"
        return result

    result["parsed_value"] = parsed

    if math.isnan(parsed) or math.isinf(parsed):
        result["error"] = "nan_or_inf_output"
        return result

    abs_err = abs(parsed - gt)
    rel_err = abs_err / max(abs(gt), 1e-6)
    result["abs_error"] = round(abs_err, 8)
    result["rel_error"] = round(rel_err, 8)

    # Match if EITHER absolute or relative tolerance is satisfied
    result["match"] = (abs_err < atol) or (rel_err < rtol)
    return result


def _match_int(output: str, gt: int) -> Dict:
    result = {"match": False, "parsed_value": None, "error": None,
              "abs_error": None, "rel_error": None}
    try:
        # Handle "3.0" style output
        parsed = int(round(float(output)))
    except ValueError:
        result["error"] = f"cannot_parse_int:{output[:50]}"
        return result

    result["parsed_value"] = parsed
    result["match"] = (parsed == gt)
    return result


def _match_bool(output: str, gt: bool) -> Dict:
    result = {"match": False, "parsed_value": None, "error": None,
              "abs_error": None, "rel_error": None}
    lower = output.strip().lower()
    if lower in ("true", "1"):
        parsed = True
    elif lower in ("false", "0"):
        parsed = False
    else:
        result["error"] = f"cannot_parse_bool:{output[:50]}"
        return result

    result["parsed_value"] = parsed
    result["match"] = (parsed == gt)
    return result


def _match_dict(output: str, gt: dict, rtol: float, atol: float) -> Dict:
    """Dict matching — all keys must match, float values within tolerance."""
    result = {"match": False, "parsed_value": None, "error": None,
              "abs_error": None, "rel_error": None}
    try:
        import ast
        parsed = ast.literal_eval(output)
    except Exception:
        try:
            parsed = json.loads(output)
        except Exception:
            result["error"] = f"cannot_parse_dict:{output[:80]}"
            return result

    result["parsed_value"] = parsed

    if not isinstance(parsed, dict):
        result["error"] = "output_not_dict"
        return result

    # Compare key sets
    gt_keys = set(str(k) for k in gt.keys())
    parsed_keys = set(str(k) for k in parsed.keys())
    if gt_keys != parsed_keys:
        result["error"] = f"key_mismatch:expected={len(gt_keys)},got={len(parsed_keys)}"
        return result

    # Compare values
    max_rel_err = 0.0
    for k, gt_val in gt.items():
        p_val = parsed.get(k) or parsed.get(str(k)) or parsed.get(int(k))
        if p_val is None:
            result["error"] = f"missing_key:{k}"
            return result
        abs_err = abs(float(p_val) - float(gt_val))
        rel_err = abs_err / max(abs(float(gt_val)), 1e-6)
        max_rel_err = max(max_rel_err, rel_err)
        if not ((abs_err < atol) or (rel_err < rtol)):
            result["error"] = f"value_mismatch:key={k}"
            result["rel_error"] = round(rel_err, 8)
            return result

    result["match"] = True
    result["rel_error"] = round(max_rel_err, 8)
    return result

# ---------------------------------------------------------------------------
# Prompt building for benchmark items
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BENCHMARK = """You are a Power System Python code generator for the {library} library.
Given a natural language task description, write complete, executable Python code that:
1. Loads the specified network
2. Applies any requested modifications
3. Runs the specified analysis
4. Prints the final numeric result using print()

Output ONLY executable Python code. Include all necessary imports.
The last print() statement should output ONLY the requested numeric value."""


def benchmark_library_name() -> str:
    """Library named in the benchmark system prompts (default: pandapower).

    The injected conditions (B/C/X/R/Rsem) already build their system prompt from the
    backend's LibrarySpec, i.e. they name the library under test. The bare baseline and
    the fix rounds used a hard-coded "pandapower" string, which on a non-pandapower
    backend tells the model to answer in the wrong library. BENCHMARK_LIBRARY is the
    backend-facing slot for that string; unset (main line) it resolves to pandapower and
    the prompts are byte-identical to before.
    """
    return (os.environ.get("BENCHMARK_LIBRARY") or "pandapower").strip() or "pandapower"


GIVEN_SETUP_CLAUSE = (
    "The network for this task is fully defined by the accompanying "
    "[Given Network Setup Code]: reuse that code verbatim as the starting point and do "
    "not substitute a different network. Request: "
)


def render_benchmark_task_text(item: Dict) -> str:
    """Task text of a benchmark item, plus its given network setup code when it has one.

    Items carrying a non-empty ``network_setup_code`` (E2 B1' self-contained items: the
    network is self-built, so the build code must travel WITH the task or the ground
    truth is unreachable in principle) render as

        <pointer clause><frozen request>

        [Given Network Setup Code]
        ```python
        <setup code>
        ```

    Items without the field -- every pandapower benchmark item -- render exactly as
    before, so default behaviour is unchanged.

    The pointer clause lives here rather than in the stored query because it is prompt
    boilerplate, not task content: storing it moved the zero-shot TF-IDF demand ranking
    on ~100% of E2 items (top-10 sets, mean Jaccard 0.56/0.68), i.e. it would have
    steered condition C's injection and desynchronised C's frozen demand export from
    Rsem's runtime SBERT retrieval. Keeping it in the renderer leaves every selector
    input byte-identical to the B2 freeze.
    """
    query = item.get("natural_language_query", "")
    setup = (item.get("network_setup_code") or "").strip()
    if not setup:
        return query
    return (f"{GIVEN_SETUP_CLAUSE}{query}\n\n"
            f"[Given Network Setup Code]\n```python\n{setup}\n```")


def build_benchmark_prompt(nl_query: str) -> List[Dict]:
    """Build messages for a benchmark item."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BENCHMARK.format(library=benchmark_library_name())},
        {"role": "user", "content": nl_query},
    ]

# ---------------------------------------------------------------------------
# Fix-round prompt building
# ---------------------------------------------------------------------------
 
SYSTEM_PROMPT_FIX = """You are an expert {library} code debugger.
Fix the buggy code based on the error message and any provided suggestions.
Make minimal changes while keeping the original logic and structure intact.
Output ONLY the corrected Python code. Include all necessary imports."""
 
 
def build_fix_prompt(
    buggy_code: str,
    error_msg: str,
    original_query: str = "",
    retrieved_docs: str = "",
) -> List[Dict]:
    """
    Build messages for a code-fix round.
 
    Args:
        buggy_code: the code that failed
        error_msg: compact error description
        original_query: the original NL task (optional, for context)
        retrieved_docs: error-specific docs from ErrorDocumentRetriever
    """
    parts = []
    if original_query:
        parts.append(f"Original task: {original_query}\n")
    parts.append(f"Buggy code:\n```python\n{buggy_code}\n```\n")
    parts.append(f"Error: {error_msg}\n")
    if retrieved_docs:
        parts.append(f"Relevant documentation:\n{retrieved_docs}\n")
    parts.append("Fix the code with minimal changes. Return only the corrected Python code.")
 
    return [
        {"role": "system", "content": SYSTEM_PROMPT_FIX.format(library=benchmark_library_name())},
        {"role": "user", "content": "\n".join(parts)},
    ]
