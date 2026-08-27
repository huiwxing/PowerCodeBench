# Verbatim prompt templates

Every prompt string the runs sent to a model is built by code in this
repository. This file quotes each template verbatim from its frozen source,
names the file and line it comes from, and — where a prompt is assembled
programmatically rather than stored as a single format string — quotes the
assembling function and shows a fully rendered example produced by running
that code on a released benchmark item.

Contents:

1. [Baseline generation prompt (condition A)](#1-baseline-generation-prompt-condition-a)
2. [Proactive conditions B / C / X / R / Rsem](#2-proactive-conditions-b--c--x--r--rsem)
3. [Reactive fix rounds FX / FD / FDR / FDRS](#3-reactive-fix-rounds-fx--fd--fdr--fdrs)
4. [BM25 tokeniser (condition R)](#4-bm25-tokeniser-condition-r)
5. [Reproducing the rendered examples](#5-reproducing-the-rendered-examples)
6. [What this file does not contain](#6-what-this-file-does-not-contain)

The supplementary material points here from Sec. S3 (fix-round template) and
Sec. S6 (system prompt, per-condition user turns, fix-round template).

Quoted example blocks use four-backtick fences, because the templates
themselves contain three-backtick `python` fences.

---

## 1. Baseline generation prompt (condition A)

### 1.1 System prompt

Source: `code/backend/utils.py:528` (`SYSTEM_PROMPT_BENCHMARK`). The
`{library}` slot is filled by `benchmark_library_name()`
(`code/backend/utils.py:539`); an unset `BENCHMARK_LIBRARY` is the pandapower
main line and resolves to `pandapower`, and the E2 transfer pilot sets
`BENCHMARK_LIBRARY=opendssdirect|pypsa`.

```text
You are a Power System Python code generator for the {library} library.
Given a natural language task description, write complete, executable Python code that:
1. Loads the specified network
2. Applies any requested modifications
3. Runs the specified analysis
4. Prints the final numeric result using print()

Output ONLY executable Python code. Include all necessary imports.
The last print() statement should output ONLY the requested numeric value.
```

### 1.2 User turn

Source: `code/backend/utils.py:591` (`build_benchmark_prompt`). The user turn
is the item's rendered task text and nothing else:

```python
def build_benchmark_prompt(nl_query: str) -> List[Dict]:
    """Build messages for a benchmark item."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BENCHMARK.format(library=benchmark_library_name())},
        {"role": "user", "content": nl_query},
    ]
```

For every pandapower benchmark item the rendered task text is exactly the
item's `natural_language_query`. Items carrying a non-empty
`network_setup_code` — only the E2 B1' self-contained items — additionally
receive the pointer clause `GIVEN_SETUP_CLAUSE`
(`code/backend/utils.py:552`) and the setup code, assembled by
`render_benchmark_task_text` (`code/backend/utils.py:559`):

````text
The network for this task is fully defined by the accompanying [Given Network Setup Code]: reuse that code verbatim as the starting point and do not substitute a different network. Request: {frozen query}

[Given Network Setup Code]
```python
{setup code}
```
````

### 1.3 Rendered example

Item `2be35441346a` of `benchmark.json`:

````text
---SYSTEM---
You are a Power System Python code generator for the pandapower library.
Given a natural language task description, write complete, executable Python code that:
1. Loads the specified network
2. Applies any requested modifications
3. Runs the specified analysis
4. Prints the final numeric result using print()

Output ONLY executable Python code. Include all necessary imports.
The last print() statement should output ONLY the requested numeric value.
---USER---
On the 5-bus PJM case, disconnect line 1 (take it out of service), disconnect line 0 (take it out of service), run a DC power flow and report the voltage magnitude (p.u.) at bus 4.
````

---

## 2. Proactive conditions B / C / X / R / Rsem

All proactive conditions are assembled by
`ProactiveInjector.build_messages` (`code/intervention/proactive.py:1103`).
Condition A returns `build_benchmark_prompt` unchanged; B, C, X, R and
Rsem/RsemB all return the same two-message shape

```python
[
    {"role": "system", "content": self.system_prompt},
    {"role": "user", "content": user_msg},
]
```

and differ only in the injection block spliced into the user turn.

### 2.1 System prompt

`self.system_prompt` is `LibraryKnowledgeBase.system_prompt_code()`
(`code/intervention/library_spec.py:190`), which returns the library
artifact's own `runtime.system_prompt_code` when present and otherwise a
generic fallback. For the pandapower main line the value is stored in
`dataset/pandapower_library_knowledge.json` (`runtime.system_prompt_code`) and
reads:

```text
You are a Power System Python code generator for the pandapower library. Given a natural language task description, write complete, executable Python code that:
1. Loads the specified network
2. Applies any requested modifications
3. Runs the specified analysis or computation
4. Prints the final numeric result using print()

Output ONLY executable Python code. Include all necessary imports. The last print() statement should output ONLY the requested numeric value.
```

Fallback, used only when a library artifact carries no `system_prompt_code`
(`code/intervention/library_spec.py:190-198`):

```python
    def system_prompt_code(self) -> str:
        if self.system_prompt:
            return self.system_prompt
        return (
            f"You are a Python code generator for the {self.library_name} library. "
            "Given a natural language task description, write complete, executable Python code. "
            "Output ONLY executable Python code. Include all necessary imports. "
            "The last print() statement should output ONLY the requested numeric value."
        )
```

### 2.2 Shared user-turn envelope

Every injected condition wraps its injection in the same envelope
(`code/intervention/proactive.py:1152`, `:1189`, `:1225`, `:1254`, `:1283`):

```python
            user_msg = (f"Task: {query}\n\n{injection}\n\n"
                        "Directly provide fully executable code, including library imports.")
```

`query` is `render_benchmark_task_text(item)`
(`code/intervention/proactive.py:1133`). When a condition's selection comes
back empty the injector falls back to `build_benchmark_prompt(query)`, i.e. to
the condition-A prompt.

### 2.3 Per-condition injection blocks

**B — demand-only function names** (`code/intervention/proactive.py:1138-1155`):

```python
            func_list = "\n".join(f"  - {self.library_name}.{fn}" for fn in top_funcs)
            injection = (
                "[Predicted API Functions for This Task]\n"
                f"The following {self.library_name} functions are likely needed:\n"
                f"{func_list}"
            )
```

**C — targeted injection, the main condition**
(`code/intervention/proactive.py:1157-1195`). The injection is the boundary
cards (when the query crosses a known API boundary) followed by the selected
layer snippets:

```python
            injection_parts = []
            if boundary_cards:
                injection_parts.append("\n\n".join(text for _, text, _ in boundary_cards))
            if selected:
                injection_body = self._format_layer_blocks(selected)
                injection_parts.append(
                    "[API Reference – Knowledge Supplement]\n"
                    "Layer-specific documentation selected because the task is predicted "
                    "to need these APIs and the selected layers carry model-side or "
                    "task-side interface risk:\n\n"
                    f"{injection_body}"
                )
            # [empty-selection fallback to build_benchmark_prompt omitted]
            injection = "\n\n".join(injection_parts)
```

**X — demand-only upper bound** (`code/intervention/proactive.py:1196-1229`),
same shape with its own header and a larger token budget:

```python
                injection_parts.append(
                    "[API Reference – Demand-Only Upper Bound]\n"
                    f"Documentation for demand-predicted {self.library_name} functions "
                    "without model-specific filtering:\n\n"
                    f"{injection_body}"
                )
```

**R — vanilla BM25 RAG baseline** (`code/intervention/proactive.py:1231-1258`).
No boundary cards, no query anchors, no model-side risk gating:

```python
            injection_body = self._format_layer_blocks(selected)
            injection = (
                f"[API Reference -- BM25 RAG Baseline]\n"
                f"Documentation for BM25-retrieved {self.library_name} functions "
                f"over the raw API specification:\n\n{injection_body}"
            )
```

**Rsem / RsemB — dense (SBERT) RAG baseline**
(`code/intervention/proactive.py:1260-1288`). Rsem uses condition R's nominal
budget, RsemB the calibrated `token_budget_RsemB`:

```python
            injection_body = self._format_layer_blocks(selected)
            injection = (
                f"[API Reference -- SBERT Dense-RAG Baseline]\n"
                f"Documentation for dense-retrieved {self.library_name} functions "
                f"over the raw API specification:\n\n{injection_body}"
            )
```

### 2.4 The two block renderers

`_format_layer_blocks` (`code/intervention/proactive.py:807`) joins the
selected snippets with a blank line; each snippet is produced by
`_layer_snippet_block` (`code/intervention/proactive.py:596`), whose header
format string is

```python
            header = (
                f"[{self.library_name}.{func_name} | {layer}: "
                f"{self.LAYER_LABELS.get(layer, 'knowledge supplement')}]\n"
            )
```

with the layer labels (`code/intervention/proactive.py:52`):

```python
    LAYER_LABELS = {
        "L0": "name and one-line purpose",
        "L1": "signature",
        "L2": "parameter, return, and semantic contract",
        "L3": "compact usage example",
    }
```

Boundary cards are rendered by `BoundaryContract.render`
(`code/intervention/library_spec.py:81`), selected by
`LibraryKnowledgeBase.boundary_card_entries`
(`code/intervention/library_spec.py:296`) via
`ProactiveInjector._boundary_card_entries`
(`code/intervention/proactive.py:860`):

```python
    def render(self, library_name: str = "") -> str:
        prefix = f"{library_name} " if library_name else ""
        lines = [f"[Boundary Card: {prefix}{self.title}]"]
        lines.extend(f"- {line}" for line in self.contract if line)
        return "\n".join(lines)
```

The card bodies themselves are data, not code: they live in
`dataset/pandapower_library_knowledge.json` under
`runtime.boundary_contracts`.

### 2.5 Rendered examples

Both blocks below are the exact strings returned by `build_messages` for item
`2be35441346a` under the released artifacts; Sec. 5 gives the command.

**Condition B:**

````text
---SYSTEM---
You are a Power System Python code generator for the pandapower library. Given a natural language task description, write complete, executable Python code that:
1. Loads the specified network
2. Applies any requested modifications
3. Runs the specified analysis or computation
4. Prints the final numeric result using print()

Output ONLY executable Python code. Include all necessary imports. The last print() statement should output ONLY the requested numeric value.
---USER---
Task: On the 5-bus PJM case, disconnect line 1 (take it out of service), disconnect line 0 (take it out of service), run a DC power flow and report the voltage magnitude (p.u.) at bus 4.

[Predicted API Functions for This Task]
The following pandapower functions are likely needed:
  - pandapower.case5
  - pandapower.rundcpp
  - pandapower.runpp

Directly provide fully executable code, including library imports.
````

**Condition C** (boundary card + selected L2/L3 snippets):

````text
---SYSTEM---
You are a Power System Python code generator for the pandapower library. Given a natural language task description, write complete, executable Python code that:
1. Loads the specified network
2. Applies any requested modifications
3. Runs the specified analysis or computation
4. Prints the final numeric result using print()

Output ONLY executable Python code. Include all necessary imports. The last print() statement should output ONLY the requested numeric value.
---USER---
Task: On the 5-bus PJM case, disconnect line 1 (take it out of service), disconnect line 0 (take it out of service), run a DC power flow and report the voltage magnitude (p.u.) at bus 4.

[Boundary Card: pandapower DataFrame/Table Contract]
- pandapower element and result tables are pandas DataFrames under `net.<table>` and `net.res_<table>`; use `.loc[index, 'column']` to update element values and `['column']` to read result columns.
- Element references such as bus, line, load, generator, and transformer IDs are pandapower table index labels; `.loc`/`.at` and constructor arguments such as `bus=` should use those labels directly, not a 1-based to 0-based conversion.
- Do not invent aggregate attributes such as `net.bus_min_vm_pu`, `net.bus_max_vm_pu`, or `net.res_line.loading`.
- Common element columns: `net.bus['min_vm_pu'/'max_vm_pu']`, `net.load['scaling']`, `net.gen['p_mw'/'min_p_mw'/'max_p_mw'/'in_service']`, `net.line['in_service'/'parallel']`, `net.trafo['parallel']`.
- After `runpp`/`runopp`, read common results with `net.res_bus['vm_pu']`, `net.res_line['loading_percent']`, `net.res_trafo['loading_percent']`; use `.idxmax()`/`.idxmin()` on those Series when an index is requested.

[API Reference – Knowledge Supplement]
Layer-specific documentation selected because the task is predicted to need these APIs and the selected layers carry model-side or task-side interface risk:

[pandapower.case5 | L3: compact usage example]
Example usage of `case5`:
```python
import pandapower as pp
net = pp.networks.case5()
```

[pandapower.rundcpp | L3: compact usage example]
Example usage of `rundcpp`:
```python
import pandapower as pp
net = pp.create_empty_network()
b1 = pp.create_bus(net, vn_kv=20.0)
b2 = pp.create_bus(net, vn_kv=20.0)
pp.create_ext_grid(net, bus=b1)
pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1.0, std_type="NAYY 4x50 SE")
pp.create_load(net, bus=b2, p_mw=0.1)
pp.rundcpp(net)
```

[pandapower.case5 | L2: parameter, return, and semantic contract]
Purpose: This is the 5 bus example from F.Li and R.Bo, "Small Test Systems for Power System Economic Studies" Its data origin is `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
Call contract: `pandapower.networks.case5()`
Return/side effect: **net** - Returns the required ieee network case5

[pandapower.runpp | L2: parameter, return, and semantic contract]
Purpose: Runs a power flow
Call contract: `pandapower.runpp(net, calculate_voltage_angles, voltage_depend_loads, consider_line_temperature, run_control, algorithm, ...)`
Key parameters:
- `net` (pandapowerNet): The pandapower format network [required]
- `calculate_voltage_angles` (str or bool): consider voltage angles in loadflow calculation If True, voltage angles of ext_grids and transformer shifts are considered in t... [default=True]
- `voltage_depend_loads` (bool): consideration of voltage-dependent loads. If False, net.load.const_z_percent and net.load.const_i_percent are not considered, i... [default=True]
- `consider_line_temperature` (bool): adjustment of line impedance based on provided line temperature. If True, net.line must contain a column "temperature_degree_ce... [default=False]
- `run_control` [default=False]
- `algorithm` (str, "nr"): algorithm that is used to solve the power flow problem. The following algorithms are available: - "nr" Newton-Raphson (pypower... [default=nr]

[pandapower.rundcpp | L2: parameter, return, and semantic contract]
Purpose: Runs PANDAPOWER DC Flow
Call contract: `pandapower.rundcpp(net, trafo_model, trafo_loading, recycle, check_connectivity, switch_rx_ratio, ...)`
Key parameters:
- `net` (pandapowerNet): The pandapower format network [required]
- `trafo_model` (str, "t"): transformer equivalent circuit model pandapower provides two equivalent circuit models for the transformer: - "t" - transformer... [default=t]
- `trafo_loading` (str, "current"): mode of calculation for transformer loading Transformer loading can be calculated relative to the rated current or the rated po... [default=current]
- `recycle`
- `check_connectivity` (bool): Perform an extra connectivity test after the conversion from pandapower to PYPOWER If true, an extra connectivity test based on... [default=True]
- `switch_rx_ratio` (float): rx_ratio of bus-bus-switches. If the impedance of switches defined in net.switch.z_ohm is zero, buses connected by a closed bus... [default=2]

Directly provide fully executable code, including library imports.
````

**Condition X** (same envelope, demand-only selection, larger budget):

````text
---SYSTEM---
You are a Power System Python code generator for the pandapower library. Given a natural language task description, write complete, executable Python code that:
1. Loads the specified network
2. Applies any requested modifications
3. Runs the specified analysis or computation
4. Prints the final numeric result using print()

Output ONLY executable Python code. Include all necessary imports. The last print() statement should output ONLY the requested numeric value.
---USER---
Task: On the 5-bus PJM case, disconnect line 1 (take it out of service), disconnect line 0 (take it out of service), run a DC power flow and report the voltage magnitude (p.u.) at bus 4.

[Boundary Card: pandapower DataFrame/Table Contract]
- pandapower element and result tables are pandas DataFrames under `net.<table>` and `net.res_<table>`; use `.loc[index, 'column']` to update element values and `['column']` to read result columns.
- Element references such as bus, line, load, generator, and transformer IDs are pandapower table index labels; `.loc`/`.at` and constructor arguments such as `bus=` should use those labels directly, not a 1-based to 0-based conversion.
- Do not invent aggregate attributes such as `net.bus_min_vm_pu`, `net.bus_max_vm_pu`, or `net.res_line.loading`.
- Common element columns: `net.bus['min_vm_pu'/'max_vm_pu']`, `net.load['scaling']`, `net.gen['p_mw'/'min_p_mw'/'max_p_mw'/'in_service']`, `net.line['in_service'/'parallel']`, `net.trafo['parallel']`.
- After `runpp`/`runopp`, read common results with `net.res_bus['vm_pu']`, `net.res_line['loading_percent']`, `net.res_trafo['loading_percent']`; use `.idxmax()`/`.idxmin()` on those Series when an index is requested.

[API Reference – Demand-Only Upper Bound]
Documentation for demand-predicted pandapower functions without model-specific filtering:

[pandapower.case5 | L1: signature]
Signature: `pandapower.networks.case5(**kwargs)`

[pandapower.case5 | L3: compact usage example]
Example usage of `case5`:
```python
import pandapower as pp
net = pp.networks.case5()
```

[pandapower.rundcpp | L1: signature]
Signature: `pandapower.rundcpp(net, trafo_model='t', trafo_loading='current', recycle=None, check_connectivity=True, switch_rx_ratio=2, trafo3w_losses='hv', **kwargs)`

[pandapower.rundcpp | L3: compact usage example]
Example usage of `rundcpp`:
```python
import pandapower as pp
net = pp.create_empty_network()
b1 = pp.create_bus(net, vn_kv=20.0)
b2 = pp.create_bus(net, vn_kv=20.0)
pp.create_ext_grid(net, bus=b1)
pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1.0, std_type="NAYY 4x50 SE")
pp.create_load(net, bus=b2, p_mw=0.1)
pp.rundcpp(net)
```

[pandapower.rundcpp | L0: name and one-line purpose]
- `rundcpp`: Runs PANDAPOWER DC Flow.

[pandapower.case5 | L0: name and one-line purpose]
- `case5`: This is the 5 bus example from F.

[pandapower.runpp | L3: compact usage example]
Example usage of `runpp`:
```python
import pandapower as pp
net = pp.create_empty_network()
b1 = pp.create_bus(net, vn_kv=20.0)
b2 = pp.create_bus(net, vn_kv=20.0)
pp.create_ext_grid(net, bus=b1)
pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1.0, std_type="NAYY 4x50 SE")
pp.create_load(net, bus=b2, p_mw=0.1, q_mvar=0.05)
pp.runpp(net)
```

[pandapower.runpp | L1: signature]
Signature: `pandapower.runpp(net, algorithm='nr', calculate_voltage_angles=True, init='auto', max_iteration='auto', tolerance_mva=1e-08, trafo_model='t', trafo_loading='current', enforce_q_lims=False, check_connectivity=True, voltage_depend_loads=True, consider_line_temperature=False, run_control=False, distributed_slack=False, tdpf=False, tdpf_delay_s=None, **kwargs)`

[pandapower.runpp | L0: name and one-line purpose]
- `runpp`: Runs a power flow.

[pandapower.case5 | L2: parameter, return, and semantic contract]
Description: This is the 5 bus example from F.Li and R.Bo, "Small Test Systems for Power System Economic Studies" Its data origin is `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
Parameters:
  - `kwargs` (dict)
Returns: **net** - Returns the required ieee network case5

[pandapower.rundcpp | L2: parameter, return, and semantic contract]
Description: Runs PANDAPOWER DC Flow
Parameters:
  - `net` (pandapowerNet) : The pandapower format network **required**
  - `trafo_model` (str, "t") : transformer equivalent circuit model pandapower provides two equivalent circuit models for the transformer: - "t" - transformer is modeled as equivalent with the T-model. This is consistent with PowerFactory and is also more accurate than the PI-model. We recommend using this transformer model. - "pi" - transformer is modeled as equivalent PI-model. This is consistent with Sincal, but the method is questionable since the transformer is physically T-shaped. We therefore recommend the use of the T-model. [default: t]
  - `trafo_loading` (str, "current") : mode of calculation for transformer loading Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer. - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current. - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer. [default: current]
  - `recycle`
  - `check_connectivity` (bool) : Perform an extra connectivity test after the conversion from pandapower to PYPOWER If true, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is perfomed. If check finds unsupplied buses, they are put out of service in the PYPOWER matrix [default: True]
  - `switch_rx_ratio` (float) : rx_ratio of bus-bus-switches. If the impedance of switches defined in net.switch.z_ohm is zero, buses connected by a closed bus-bus switch are fused to model an ideal bus. Closed bus-bus switches, whose impedance z_ohm is not zero, are modelled as branches with resistance and reactance according to net.switch.z_ohm and switch_rx_ratio. [default: 2]
  - `trafo3w_losses` (str, "hv") : defines where open loop losses of three-winding transformers are considered. Valid options are "hv", "mv", "lv" for HV/MV/LV side or "star" for the star point. [default: hv]
  - `kwargs` (dict) : options to use for PYPOWER.runpf

[pandapower.runpp | L2: parameter, return, and semantic contract]
Description: Runs a power flow
Parameters:
  - `net` (pandapowerNet) : The pandapower format network **required**
  - `algorithm` (str, "nr") : algorithm that is used to solve the power flow problem. The following algorithms are available: - "nr" Newton-Raphson (pypower implementation with numba accelerations) - "iwamoto_nr" Newton-Raphson with Iwamoto multiplier (maybe slower than NR but more robust) - "bfsw" backward/forward sweep (specially suited for radial and weakly-meshed networks) - "gs" gauss-seidel (pypower implementation) - "fdbx" fast-decoupled (pypower implementation) - "fdxb" fast-decoupled (pypower implementation) [default: nr]
  - `calculate_voltage_angles` (str or bool) : consider voltage angles in loadflow calculation If True, voltage angles of ext_grids and transformer shifts are considered in the loadflow calculation. Considering the voltage angles is only necessary in meshed networks that are usually found in higher voltage levels. calculate_voltage_angles in "auto" mode defaults to: - True, if the network voltage level is above 70 kV - False otherwise The network voltage level is defined as the maximum rated voltage of any bus in the network that is connected to a line. [default: True]
  - `init` (str, "auto") : initialization method of the loadflow pandapower supports four methods for initializing the loadflow: - "auto" - init defaults to "dc" if calculate_voltage_angles is True or "flat" otherwise - "flat"- flat start with voltage of 1.0pu and angle of 0° at all PQ-buses and 0° for PV buses as initial solution, the slack bus is initialized with the values provided in net["ext_grid"] - "dc" - initial DC loadflow before the AC loadflow. The results of the DC loadflow are used as initial solution for the AC loadflow. Note that the DC loadflow only calculates voltage angles at PQ and PV buses, voltage magnitudes are still flat started. - "results" - voltage vector of last loadflow from net.res_bus is used as initial solution. This can be useful to accelerate convergence in iterative loadflows like time series calculations. Considering the voltage angles might lead to non-convergence of the power flow in flat start. That is why in "auto" mode, init defaults to "dc" if calculate_voltage_angles is True or "flat" otherwise [default: auto]
  - `max_iteration` (int, "auto") : maximum number of iterations carried out in the power flow algorithm. In "auto" mode, the default value depends on the power flow solver: - 10 for "nr" - 100 for "bfsw" - 1000 for "gs" - 30 for "fdbx" - 30 for "fdxb" - 30 for "nr" with "tdpf" [default: auto]
  - `tolerance_mva` (float, 1e-8) : loadflow termination condition referring to P / Q mismatch of node power in MVA [default: 1e-08]
  - `trafo_model` (str, "t") : transformer equivalent circuit model pandapower provides two equivalent circuit models for the transformer: - "t" - transformer is modeled as equivalent with the T-model. - "pi" - transformer is modeled as equivalent PI-model. This is not recommended, since it is less exact than the T-model. It is only recommended for valdiation with other software that uses the pi-model. [default: t]
  - `trafo_loading` (str, "current") : mode of calculation for transformer loading Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer. - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current. - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer. [default: current]
  - `enforce_q_lims` (bool) : respect generator reactive power limits If True, the reactive power limits in net.gen.max_q_mvar/min_q_mvar are respected in the loadflow. This is done by running a second loadflow if reactive power limits are violated at any generator, so that the runtime for the loadflow will increase if reactive power has to be curtailed. [default: False]
  - `check_connectivity` (bool) : Perform an extra connectivity test after the conversion from pandapower to PYPOWER If True, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is perfomed. If check finds unsupplied buses, they are set out of service in the ppc [default: True]
  - `voltage_depend_loads` (bool) : consideration of voltage-dependent loads. If False, net.load.const_z_percent and net.load.const_i_percent are not considered, i.e. net.load.p_mw and net.load.q_mvar are considered as constant-power loads. [default: True]
  - `consider_line_temperature` (bool) : adjustment of line impedance based on provided line temperature. If True, net.line must contain a column "temperature_degree_celsius". The temperature dependency coefficient alpha must be provided in the net.line.alpha column, otherwise the default value of 0.004 is used [default: False]
  - `run_control` [default: False]
  - `distributed_slack` (bool) : Distribute slack power according to contribution factor weights for external grids and generators. [default: False]
  - `tdpf` (bool) : Temperature Dependent Power Flow (TDPF). If True, line temperature is calculated based on the TDPF parameters in net.line table. [default: False]
  - `tdpf_delay_s` (float) : TDPF parameter, specifies the time delay in s to consider thermal inertia of conductors.
  - `kwargs` (dict)

Directly provide fully executable code, including library imports.
````

Conditions R and Rsem/RsemB reuse this envelope and the same snippet renderer;
only the retrieval that picks the functions differs (BM25 over the raw API
spec, or SBERT over the same text), so their user turns differ from C only in
the header line quoted in Sec. 2.3 and in which functions appear.

---

## 3. Reactive fix rounds FX / FD / FDR / FDRS

### 3.1 System prompt (unchanged across rounds and conditions)

Source: `code/backend/utils.py:602` (`SYSTEM_PROMPT_FIX`):

```text
You are an expert {library} code debugger.
Fix the buggy code based on the error message and any provided suggestions.
Make minimal changes while keeping the original logic and structure intact.
Output ONLY the corrected Python code. Include all necessary imports.
```

### 3.2 User turn

Source: `code/backend/utils.py:608` (`build_fix_prompt`). This is the
structured task / previous-code / failure / relevant-doc / instruction layout
described in SM Sec. S3. Parts are joined by newlines; the task and
relevant-doc blocks are omitted when empty:

```python
    parts = []
    if original_query:
        parts.append(f"Original task: {original_query}\n")
    parts.append(f"Buggy code:\n```python\n{buggy_code}\n```\n")
    parts.append(f"Error: {error_msg}\n")
    if retrieved_docs:
        parts.append(f"Relevant documentation:\n{retrieved_docs}\n")
    parts.append("Fix the code with minimal changes. Return only the corrected Python code.")
```

i.e. the fully expanded user template is

````text
Original task: {original_query}

Buggy code:
```python
{buggy_code}
```

Error: {error_msg}

Relevant documentation:
{retrieved_docs}

Fix the code with minimal changes. Return only the corrected Python code.
````

### 3.3 What each fix condition puts in `{retrieved_docs}`

`retrieved_docs` is the string returned by
`ReactiveInjector.get_injection` (`code/intervention/reactive.py:669`),
dispatched by `get_injection_details` (`code/intervention/reactive.py:700`).
The condition table is `code/intervention/reactive.py:597`:

```python
    CONDITION_LAYERS: Dict[str, int] = {
        "FX": 0,   # no injection — pass-through to build_fix_prompt with empty docs
        "FR": 2,   # L0 + L1 (name + signature)
        "FD": -1,  # error-class-specific depth via ERROR_CLASS_LAYERS (see get_injection)
        "FDR": -2, # demand-routed basic/API-doc/boundary decision
        "FS": -3,  # semantic diagnosis only for executed-but-wrong value bugs
        "FDRS": -4,# FDR for runtime/API bugs, semantic diagnosis for value bugs
        "FE": 4,   # L0 + L1 + L2 + L3 (all layers including examples)
    }
```

**FX** returns `""`, so the relevant-doc block disappears and the prompt is the
plain error-feedback baseline.

**FD** injects documentation for the functions the error parser implicates, to
an error-class-specific depth (`code/intervention/reactive.py:130`):

```python
ERROR_CLASS_LAYERS: Dict[str, int] = {
    "AttributeError":        3,   # wrong method name → name + signature + description
    "TypeError":             3,   # wrong args/kwargs → name + signature + description
    "NameError":             2,   # undefined name → name + signature is enough
    "SyntaxError":           0,   # syntax issue → doc injection doesn't help → FX fallback
    "ImportError":           1,   # wrong import → name only
    "ValueError":            3,   # bad parameter value → name + signature + description
    "KeyError":              4,   # wrong key access pattern → full docs + example needed
    "IndexError":            1,   # indexing error → name only
    "RuntimeError":          3,
    "LoadflowNotConverged":  4,   # wrong ConstControl variable/data → full docs + example
    "default":               3,   # unknown error class → full description (safe default)
}
```

with the header (`code/intervention/reactive.py:812`):

```python
        header = (
            "[Relevant API Documentation]\n"
            f"{self.kb.library_name} documentation for functions implicated in the {error_class} "
            f"(layers: {', '.join(layers_to_include)}):\n\n"
        )
```

**FDR** routes each failure to a basic fix (empty docs), an API-doc block, or a
boundary-contract card (`code/intervention/reactive.py:835`). Its two headers
are (`code/intervention/reactive.py:880` and `:894`):

```python
                docs = (
                    "[Fix-Demand API Documentation]\n"
                    f"Route: {plan.route}. Reason: {plan.reason}.\n"
                    f"{self.kb.library_name} documentation for implicated functions "
                    f"(layers: {', '.join(plan.layers)}):\n\n"
                    + "\n\n".join(blocks)
                    + boundary_docs
                )
```

```python
                docs = (
                    "[Fix-Demand Boundary Contract]\n"
                    f"Route: {plan.route}. Reason: {plan.reason}.\n\n"
                    + "\n\n".join(cards)
                )
```

(when an API-doc route also matches a boundary card, the card is appended
under the sub-header `"\n\n[Fix-Demand Boundary Contract]\n"`,
`code/intervention/reactive.py:875`).

**FDRS** is FDR for runtime/API failures plus a semantic diagnosis path for
executed-but-wrong values (`code/intervention/reactive.py:1021`). The semantic
block is rendered by `SemanticFailureDiagnoser.render`
(`code/intervention/semantic.py:240`) and has two shapes. Output-format
failures:

```python
            return "\n".join([
                "[Value-Level Output Format Repair]",
                (
                    "The previous code executed without raising an exception, but "
                    "the printed payload could not be parsed as the expected scalar."
                ),
                "Repair instruction:",
                (
                    "- The last line must print exactly one numeric scalar value — "
                    "no labels, no f-strings wrapping the value, no multi-line text, "
                    "no DataFrame/Series/array dump."
                ),
                "Keep the rest of the analysis logic unchanged.",
            ])
```

Value failures with a usable execution trace:

```python
        lines = [
            "[Trace-Grounded Contract Context]",
            (
                "The previous code executed and printed a numeric value, but it did "
                "not match the requested result. Compare the original task with this "
                "implementation trace and the library contracts below. No reference "
                "answer is used."
            ),
        ]
        if plan.signals:
            lines.append("Structural trace signals:")
            lines.extend(f"- {signal}" for signal in plan.signals if signal)

        obligations = self._repair_obligations(plan)
        if obligations:
            lines.append("High-confidence trace invariants:")
            lines.extend(f"- {line}" for line in obligations)
```

No reference answer or ground truth enters either path.

### 3.4 Rendered examples

The four blocks below are the exact `build_fix_prompt` outputs for one
`AttributeError` failure; Sec. 5 gives the command.

**FX** (no injection):

````text
---SYSTEM---
You are an expert pandapower code debugger.
Fix the buggy code based on the error message and any provided suggestions.
Make minimal changes while keeping the original logic and structure intact.
Output ONLY the corrected Python code. Include all necessary imports.
---USER---
Original task: On the 5-bus PJM case, run a power flow and report the maximum line loading percentage.

Buggy code:
```python
import pandapower as pp
import pandapower.networks as pn
net = pn.case5()
pp.runpp(net)
print(net.res_line.loading.max())

```

Error: 'DataFrame' object has no attribute 'loading'

Fix the code with minimal changes. Return only the corrected Python code.
````

**FD** (error-class depth L0+L1+L2 for `AttributeError`; the block is long, so
the documentation body is shown truncated at the first function):

````text
---SYSTEM---
You are an expert pandapower code debugger.
Fix the buggy code based on the error message and any provided suggestions.
Make minimal changes while keeping the original logic and structure intact.
Output ONLY the corrected Python code. Include all necessary imports.
---USER---
Original task: On the 5-bus PJM case, run a power flow and report the maximum line loading percentage.

Buggy code:
```python
import pandapower as pp
import pandapower.networks as pn
net = pn.case5()
pp.runpp(net)
print(net.res_line.loading.max())

```

Error: 'DataFrame' object has no attribute 'loading'

Relevant documentation:
[Relevant API Documentation]
pandapower documentation for functions implicated in the AttributeError (layers: L0, L1, L2):

- `runpp`: Runs a power flow.
Signature: `pandapower.runpp(net, algorithm='nr', calculate_voltage_angles=True, init='auto', max_iteration='auto', tolerance_mva=1e-08, trafo_model='t', trafo_loading='current', enforce_q_lims=False, check_connectivity=True, voltage_depend_loads=True, consider_line_temperature=False, run_control=False, distributed_slack=False, tdpf=False, tdpf_delay_s=None, **kwargs)`
Description: Runs a power flow
Parameters:
  - `net` (pandapowerNet) : The pandapower format network **required**
  - `algorithm` (str, "nr") : algorithm that is used to solve the power flow problem. The following algorithms are available: - "nr" Newton-Raphson (pypower implementation with numba accelerations) - "iwamoto_nr" Newton-Raphson with Iwamoto multiplier (maybe slower than NR but more robust) - "bfsw" backward/forward sweep (specially suited for radial and weakly-meshed networks) - "gs" gauss-seidel (pypower implementation) - "fdbx" fast-decoupled (pypower implementation) - "fdxb" fast-decoupled (pypower implementation) [default: nr]
  [... remaining parameter lines and the L0/L1/L2 blocks of the other implicated functions ...]

Fix the code with minimal changes. Return only the corrected Python code.
````

**FDR** (this failure routes to `boundary_contract`):

````text
---SYSTEM---
You are an expert pandapower code debugger.
Fix the buggy code based on the error message and any provided suggestions.
Make minimal changes while keeping the original logic and structure intact.
Output ONLY the corrected Python code. Include all necessary imports.
---USER---
Original task: On the 5-bus PJM case, run a power flow and report the maximum line loading percentage.

Buggy code:
```python
import pandapower as pp
import pandapower.networks as pn
net = pn.case5()
pp.runpp(net)
print(net.res_line.loading.max())

```

Error: 'DataFrame' object has no attribute 'loading'

Relevant documentation:
[Fix-Demand Boundary Contract]
Route: boundary_contract. Reason: attribute failure matches a known result-object boundary.

[Boundary Card: pandapower DataFrame/Table Contract]
- pandapower element and result tables are pandas DataFrames under `net.<table>` and `net.res_<table>`; use `.loc[index, 'column']` to update element values and `['column']` to read result columns.
- Element references such as bus, line, load, generator, and transformer IDs are pandapower table index labels; `.loc`/`.at` and constructor arguments such as `bus=` should use those labels directly, not a 1-based to 0-based conversion.
- Do not invent aggregate attributes such as `net.bus_min_vm_pu`, `net.bus_max_vm_pu`, or `net.res_line.loading`.
- Common element columns: `net.bus['min_vm_pu'/'max_vm_pu']`, `net.load['scaling']`, `net.gen['p_mw'/'min_p_mw'/'max_p_mw'/'in_service']`, `net.line['in_service'/'parallel']`, `net.trafo['parallel']`.
- After `runpp`/`runopp`, read common results with `net.res_bus['vm_pu']`, `net.res_line['loading_percent']`, `net.res_trafo['loading_percent']`; use `.idxmax()`/`.idxmin()` on those Series when an index is requested.

Fix the code with minimal changes. Return only the corrected Python code.
````

**FDRS** on the same failure routes identically to FDR, because the failure is
a raised exception rather than an executed-but-wrong value; the semantic path
above is what FDRS adds over FDR.

---

## 4. BM25 tokeniser (condition R)

Source: `code/intervention/rag_baseline.py:32-39`, quoted verbatim:

```python
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _tokenize(text: str) -> List[str]:
    """Lower-cased word/identifier tokens. Preserves snake_case as a single token."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
```

The same tokeniser is applied to the corpus and to the query
(`code/intervention/rag_baseline.py:108` and `:130`). The indexed text per
documented function is built by `_function_text`
(`code/intervention/rag_baseline.py:42`), which deliberately excludes the
`examples` field:

```python
def _function_text(entry: Dict) -> str:
    """Concatenate the searchable fields of a documented function entry.

    The raw spec contains: name, full_path, signature, description, parameters,
    return_info, examples. Examples are excluded because they are typically
    noisy and biased toward construction patterns; including them would inflate
    BM25 hits on construction queries unrealistically.
    """
    parts: List[str] = []
    parts.append(str(entry.get("name", "")))
    parts.append(str(entry.get("full_path", "")))
    parts.append(str(entry.get("category", "")))
    parts.append(str(entry.get("signature", "")))
    parts.append(str(entry.get("description", "")))
    for p in entry.get("parameters") or []:
        parts.append(str(p.get("name", "")))
        parts.append(str(p.get("description", "")))
    for r in entry.get("return_info") or []:
        parts.append(str(r.get("description", "")))
    return " ".join(parts)
```

Retrieval is `rank_bm25.BM25Okapi` at library defaults
(`code/intervention/rag_baseline.py:84`, `:114`), and `score()` min-max
normalises the raw scores (`code/intervention/rag_baseline.py:123`). This
matches SM Sec. S6 ("Vanilla BM25 (condition R) configuration").

---

## 5. Reproducing the rendered examples

The proactive examples in Sec. 2.5 were produced with the released artifacts
only:

```python
import json, sys, tempfile
sys.path.insert(0, "code")
from intervention.proactive import ProactiveInjector

profile = json.load(open("results/supplementary_evidence/probe_profiles.json"))
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(profile["models"]["Qwen3-480B"]["profile"], tmp); tmp.close()

inj = ProactiveInjector.from_paths(
    knowledge_profile_path=tmp.name,
    docs_path="dataset/pandapower_docs.json",
    demand_suite_dir="results/demand_suites/suite_4428033",
    demand_model="hybrid_tfidf",
)
item = {b["id"]: b for b in json.load(open("benchmark.json"))}["2be35441346a"]
for role_msg in inj.build_messages(item, "C"):
    print(role_msg["role"].upper()); print(role_msg["content"])
```

Conditions R and Rsem additionally need `bm25_docs_path` / `sbert_docs_path`
(and the optional `rank_bm25` / `sentence-transformers` dependencies).

The fix-round examples in Sec. 3.4 were produced with:

```python
import sys
sys.path.insert(0, "code")
from intervention.reactive import ReactiveInjector
from backend.utils import build_fix_prompt

buggy = ("import pandapower as pp\n"
         "import pandapower.networks as pn\n"
         "net = pn.case5()\n"
         "pp.runpp(net)\n"
         "print(net.res_line.loading.max())\n")
query = "On the 5-bus PJM case, run a power flow and report the maximum line loading percentage."
err_type, err_msg = "AttributeError", "'DataFrame' object has no attribute 'loading'"

for cond in ["FX", "FD", "FDR", "FDRS"]:
    inj = ReactiveInjector.from_paths("dataset/pandapower_docs.json", fix_condition=cond)
    docs = inj.get_injection(error_type=err_type, error_msg=err_msg,
                             generated_code=buggy, original_query=query)
    for role_msg in build_fix_prompt(buggy, err_msg, query, docs):
        print(cond, role_msg["role"].upper()); print(role_msg["content"])
```

---

## 6. What this file does not contain

Templates, the assembling code, and the data the templates read (API-spec
corpus, boundary cards, layer snippets, demand exports, probe profiles) are
all in this repository, so any item's prompt under any condition can be
re-rendered exactly, as Sec. 5 shows.

The per-item prompt *strings* of the historical runs were not retained: the
frozen run trees stored the assembled prompts alongside the multi-gigabyte
generated-code trees, which are not released. What is released instead is the
selection that determines each prompt —
`results/raw/main_experiment/primary_outcomes_compact.json`
(`injection_log_compact`) holds the per-model, per-item function × layer
projection of the original `injection_log.json`, and
`results/supplementary_evidence/proactive_prompt_components.json` holds the
per-item snippet and boundary-card token vectors with the source file's
SHA-256. Prompt and completion token totals per item are in the same compact
run files. See `docs/PROVENANCE_BOUNDARIES.md` for the general policy.
