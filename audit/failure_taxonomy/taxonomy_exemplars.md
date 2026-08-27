# Failure taxonomy — shortlisted exemplars

Representative samples for each failure category: up to five, and fewer where a
category is sparse. Each excerpt is the first five non-blank lines of the
offending `generated_code`. See `README.md` in this directory for the labelling
round these exemplars come from.


## C1 — API contract / hallucinated callable

### s001 | Qwen2.5-Coder-0.5B (tier1, condA_FD, task=time_series)

*Hallucinated pp.read_power_system + network.load_from_csv/min_bus_voltage (none exist).*

```python
import pandapower as pp
import numpy as np
import random
# Load the network
network = pp.read_power_system('path_to_your_network_file.pypower')
```

### s022 | Qwen2.5-Coder-1.5B (tier1, condC_FDRS, task=time_series)

*pp.disconnect — no module-level disconnect function in pandapower.*

```python
import pandapower as pp
import pandas as pd
from pandapower.timeseries import DFData, OutputWriter, run_timeseries
from pandapower.control import ConstControl
# Load the network
```

### s072 | gpt-oss-120b (tier3, condC_FDRS, task=short_circuit_3ph)

*pn.create_kerber_dorf — typo of create_kerber_dorfnetz; AttributeError.*

```python
import pandapower as pp
import pandapower.shortcircuit as sc
import pandapower.networks as pn
# 1. Load the Kerber Dorf LV network
net = pn.create_kerber_dorf()
```

### s070 | Llama-3.1-70B (tier3, condA_FX, task=contingency)

*pp.contingency_analysis hallucinated; correct API is pandapower.contingency.run_contingency.*

```python
import pandapower as pp
import pandapower.networks as pn
# Load the Kerber village (Dorf) LV network
net = pn.case9()
# Run the power flow
```

### s033 | Qwen2.5-Coder-7B (tier2, condA_FDR, task=time_series)

*pts.DataSequence does not exist; real timeseries data class is DFData.*

```python
import pandapower as pp
import pandapower.timeseries as pts
import numpy as np
# Load the simple 4-bus system
net = pp.networks.create_cigre_network_mv(with_der=False)
```


## C2 — Parameter or result-table misuse

### s091 | Qwen3-Coder-480B (tier4, condA_FX, task=short_circuit_2ph)

*net.res_bus_sc['ikss_2ph'] — wrong column; pandapower returns only 'ikss_ka'.*

```python
import pandapower as pp
import pandapower.networks as nw
import pandapower.shortcircuit as sc
# Load the 33-bus Baran-Wu network
net = nw.case33bw()
```

### s044 | Llama-3.1-8B (tier2, condA_FX, task=time_series)

*KeyError 'p0_mw' — net.load column is 'p_mw'.*

```python
import pandas as pd
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as ppplot
```

### s097 | Qwen3-Coder-Next (tier4, condC_FDRS, task=state_estimation)

*pp.create_measurement positional args wrong; 0.01 read as bus index.*

```python
import pandapower as pp
import numpy as np
# Set random seed for reproducibility
np.random.seed(3442)
# Load the 9-bus WSCC case
```

### s057 | gpt-oss-120b (tier3, condA_FDR, task=time_series)

*Wrong element index: 'line 1' interpreted 1-indexed, disabled line 0 instead of line 1.*

```python
import pandapower as pp
import pandapower.networks as pn
import numpy as np
# Load the simple 4‑bus system
net = pn.simple_four_bus_system()
```

### s042 | Qwen2.5-Coder-32B (tier2, condA_FX, task=time_series)

*ConstControl omitted profile_name kwarg → KeyError: None inside run_timeseries.*

```python
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandapower.timeseries as ts
from pandapower.control import ConstControl
```


## C3 — Workflow / domain-logic sequencing

### s023 | Qwen2.5-Coder-0.5B (tier1, condC_FDRS, task=power_flow)

*Skipped pp.runpp(net) before reading net.res_bus['vm_pu']; result empty.*

```python
import pandapower as pp
# Load the specified network
net = pp.networks.create_cigre_network_lv()
# Apply any requested modifications
# (This part is not provided in the task description)
```

### s050 | Qwen2.5-Coder-14B (tier2, condC_FDRS, task=short_circuit_3ph)

*Disabled all generators BEFORE runpp; LoadflowNotConverged.*

```python
import pandapower as pp
import pandapower.shortcircuit as sc
# Load the IEEE 24-bus RTS network
net = pp.networks.case24_ieee_rts()
# Ensure ext_grid has necessary short-circuit parameters
```

### s088 | Llama-3.1-405B (tier4, condA_FDRS, task=time_series)

*Wrong network (example_simple) + wrong line disconnected → islanded topology fails ts.run_timeseries.*

```python
import pandapower as pp
import pandapower.timeseries as ts
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.control import ConstControl
```

### s046 | Qwen2.5-Coder-14B (tier2, condC_FDRS, task=state_estimation)

*Skipped capacitive shunt creation + power-injection measurements; SE under-observed.*

```python
import pandapower as pp
import pandapower.networks as pn
import pandapower.estimation as pe
import numpy as np
# Load the 9-bus WSCC case
```

### s048 | Qwen2.5-Coder-7B (tier2, condC_FDRS, task=time_series)

*Used deterministic ramp instead of seeded np.random.uniform load profile.*

```python
import pandapower as pp
import pandas as pd
from pandapower.timeseries import DFData, OutputWriter, run_timeseries
from pandapower.control import ConstControl
# Load the simple 4-bus system
```


## C4 — Numerical value or scalar extraction

### s089 | Qwen3-Coder-480B (tier4, condA_FDRS, task=contingency)

*Iterated contingency_results dict for vm_pu instead of net.res_bus['min_vm_pu'].min().*

```python
import pandapower as pp
import pandapower.contingency as ct
import numpy as np
# Load the 145-bus case network
net = pp.networks.case145()
```


## C5 — Formatting / execution environment

### s082 | Qwen3-Coder-Next (tier4, condA_FDR, task=state_estimation)

*SyntaxError: unclosed '{' at line 122 — truncated template output.*

```python
import numpy as np
import pandapower as pp
import pandapower.networks as nw
# Set random seed for reproducibility
np.random.seed(3442)
```

### s077 | Qwen3-Coder-Next (tier4, condA_FD, task=state_estimation)

*SyntaxError: unterminated string literal at line 132.*

```python
import numpy as np
import pandapower as pp
import pandapower.networks as nw
# Set random seed for reproducibility
np.random.seed(3442)
```

### s078 | Llama-3.1-405B (tier4, condA_FD, task=time_series)

*Printed 'Minimum bus voltage (p.u.): 1.02' — parser cannot extract bare float.*

```python
import pandapower as pp
import numpy as np
# Load the network
net = pp.networks.example_simple()
# Disconnect line 1
```

### s079 | Qwen3-Coder-480B (tier4, condA_FD, task=comparison)

*Printed 'Change in maximum line loading: -8.26%' — formatted prefix breaks scalar parse.*

```python
import pandapower as pp
import pandapower.networks as nw
# Load the simple MV open ring network
net = nw.simple_mv_open_ring_net()
# Run power flow analysis
```

### s100 | Qwen3-Coder-Next (tier4, condC_FDRS, task=pf_then_sc)

*Hard 75s execution timeout on case300 SC computation.*

```python
import pandapower as pp
import pandapower.shortcircuit as sc
# Load the 300-bus test case
net = pp.networks.case300()
# Execute power flow to determine the operating point
```
