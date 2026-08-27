# -*- coding: utf-8 -*-
"""Canonical Baran-Wu IEEE 69-bus radial distribution feeder (frozen artifact).

Why this file exists
--------------------
E4 layer-2b' query A02 asks for "the canonical Baran-Wu IEEE 69-bus radial
distribution feeder, with bus numbers referring to the published 1-69 labels and
with the exact base-case data frozen with the benchmark".  pandapower 3.4.0 has
no built-in 69-bus feeder (``pandapower.networks`` ships case4gs/5/6ww/9/14/30/
case_ieee30/33bw/39/57/89pegase/118/145/... but nothing 69-bus), so the network
data is pinned here and frozen together with the query set.  This is an
artifact-pinning requirement stated by the engineer-approved README; it is NOT a
change to the approved engineering request.

Data provenance
---------------
M. E. Baran and F. F. Wu, "Optimal capacitor placement on radial distribution
systems", IEEE Trans. Power Delivery 4(1):725-734, 1989 -- the 69-bus
(sometimes "70-bus" incl. the source node) feeder, in the widely reproduced
base-case tabulation: 69 buses, 68 series branches, nominal 12.66 kV, branch
resistance/reactance in ohms, spot loads in kW/kvar attached to the receiving
("to") bus of each branch.  Tie branches used in reconfiguration studies are NOT
part of the base case and are not created here (the base case is strictly
radial, as the query requires).

Totals implied by the table below (checked by ``sanity_check``):
    sum P = 3801.89 kW,  sum Q = 2694.10 kvar
Some papers quote 3802.19 kW / 2694.60 kvar for a near-identical variant of the
same feeder (+0.30 kW / +0.50 kvar); the tabulation frozen here is internally
consistent and is the one the benchmark is defined against.

Reference base-case results reproduced by this implementation (slack 1.0 p.u.
at bus label 1) -- these are the standard published check values for the feeder:
    total active loss ~ 225 kW,  min bus voltage ~ 0.9092 p.u. at bus label 65.

Bus labelling convention (E4 layer-2b' README rule)
---------------------------------------------------
Published labels are 1..69.  pandapower table indices are 0-based; this builder
creates bus label ``n`` at pandapower bus index ``n - 1`` and also writes the
label into ``net.bus['name']`` as ``"Bus <n>"``.  Use ``bus_index(label)`` to
convert.  No other mapping is implied anywhere in the benchmark.

Ratings note
------------
The published data set specifies only R, X and the spot loads.  Line current
ratings are not published; ``MAX_I_KA`` below is an implementation constant
(uniform, non-binding: the base case and query A02 draw <= ~0.19 kA at the
feeder head).  Shunt capacitance is zero, as in the published data.

Frozen: 2026-07-30 with ``e4_layer2b_set.json``.  Everything above the
END-FROZEN-BUILDER marker is embedded verbatim in the reference_code of A02, so
the query set is self-contained and the sha256 of this file pins the network.
"""
import pandapower as pp

VN_KV = 12.66
MAX_I_KA = 0.4
SLACK_LABEL = 1

# (from_label, to_label, r_ohm, x_ohm, p_kw_at_to_bus, q_kvar_at_to_bus)
BARAN_WU_69_BRANCHES = [
    (1, 2, 0.0005, 0.0012, 0.00, 0.00),
    (2, 3, 0.0005, 0.0012, 0.00, 0.00),
    (3, 4, 0.0015, 0.0036, 0.00, 0.00),
    (4, 5, 0.0251, 0.0294, 0.00, 0.00),
    (5, 6, 0.3660, 0.1864, 2.60, 2.20),
    (6, 7, 0.3811, 0.1941, 40.40, 30.00),
    (7, 8, 0.0922, 0.0470, 75.00, 54.00),
    (8, 9, 0.0493, 0.0251, 30.00, 22.00),
    (9, 10, 0.8190, 0.2707, 28.00, 19.00),
    (10, 11, 0.1872, 0.0619, 145.00, 104.00),
    (11, 12, 0.7114, 0.2351, 145.00, 104.00),
    (12, 13, 1.0300, 0.3400, 8.00, 5.00),
    (13, 14, 1.0440, 0.3450, 8.00, 5.50),
    (14, 15, 1.0580, 0.3496, 0.00, 0.00),
    (15, 16, 0.1966, 0.0650, 45.50, 30.00),
    (16, 17, 0.3744, 0.1238, 60.00, 35.00),
    (17, 18, 0.0047, 0.0016, 60.00, 35.00),
    (18, 19, 0.3276, 0.1083, 0.00, 0.00),
    (19, 20, 0.2106, 0.0690, 1.00, 0.60),
    (20, 21, 0.3416, 0.1129, 114.00, 81.00),
    (21, 22, 0.0140, 0.0046, 5.00, 3.50),
    (22, 23, 0.1591, 0.0526, 0.00, 0.00),
    (23, 24, 0.3463, 0.1145, 28.00, 20.00),
    (24, 25, 0.7488, 0.2475, 0.00, 0.00),
    (25, 26, 0.3089, 0.1021, 14.00, 10.00),
    (26, 27, 0.1732, 0.0572, 14.00, 10.00),
    (3, 28, 0.0044, 0.0108, 26.00, 18.60),
    (28, 29, 0.0640, 0.1565, 26.00, 18.60),
    (29, 30, 0.3978, 0.1315, 0.00, 0.00),
    (30, 31, 0.0702, 0.0232, 0.00, 0.00),
    (31, 32, 0.3510, 0.1160, 0.00, 0.00),
    (32, 33, 0.8390, 0.2816, 14.00, 10.00),
    (33, 34, 1.7080, 0.5646, 19.50, 14.00),
    (34, 35, 1.4740, 0.4873, 6.00, 4.00),
    (3, 36, 0.0044, 0.0108, 26.00, 18.55),
    (36, 37, 0.0640, 0.1565, 26.00, 18.55),
    (37, 38, 0.1053, 0.1230, 0.00, 0.00),
    (38, 39, 0.0304, 0.0355, 24.00, 17.00),
    (39, 40, 0.0018, 0.0021, 24.00, 17.00),
    (40, 41, 0.7283, 0.8509, 1.20, 1.00),
    (41, 42, 0.3100, 0.3623, 0.00, 0.00),
    (42, 43, 0.0410, 0.0478, 6.00, 4.30),
    (43, 44, 0.0092, 0.0116, 0.00, 0.00),
    (44, 45, 0.1089, 0.1373, 39.22, 26.30),
    (45, 46, 0.0009, 0.0012, 39.22, 26.30),
    (4, 47, 0.0034, 0.0084, 0.00, 0.00),
    (47, 48, 0.0851, 0.2083, 79.00, 56.40),
    (48, 49, 0.2898, 0.7091, 384.70, 274.50),
    (49, 50, 0.0822, 0.2011, 384.70, 274.50),
    (8, 51, 0.0928, 0.0473, 40.50, 28.30),
    (51, 52, 0.3319, 0.1114, 3.60, 2.70),
    (9, 53, 0.1740, 0.0886, 4.35, 3.50),
    (53, 54, 0.2030, 0.1034, 26.40, 19.00),
    (54, 55, 0.2842, 0.1447, 24.00, 17.20),
    (55, 56, 0.2813, 0.1433, 0.00, 0.00),
    (56, 57, 1.5900, 0.5337, 0.00, 0.00),
    (57, 58, 0.7837, 0.2630, 0.00, 0.00),
    (58, 59, 0.3042, 0.1006, 100.00, 72.00),
    (59, 60, 0.3861, 0.1172, 0.00, 0.00),
    (60, 61, 0.5075, 0.2585, 1244.00, 888.00),
    (61, 62, 0.0974, 0.0496, 32.00, 23.00),
    (62, 63, 0.1450, 0.0738, 0.00, 0.00),
    (63, 64, 0.7105, 0.3619, 227.00, 162.00),
    (64, 65, 1.0410, 0.5302, 59.00, 42.00),
    (11, 66, 0.2012, 0.0611, 18.00, 13.00),
    (66, 67, 0.0047, 0.0014, 18.00, 13.00),
    (12, 68, 0.7394, 0.2444, 28.00, 20.00),
    (68, 69, 0.0047, 0.0016, 28.00, 20.00),
]


def bus_index(label):
    """Published 1-based bus label -> 0-based pandapower bus table index."""
    return label - 1


def create_ieee69_baran_wu():
    """Build the canonical Baran-Wu IEEE 69-bus radial feeder (base case)."""
    net = pp.create_empty_network(name="IEEE 69-bus Baran-Wu radial feeder")
    for label in range(1, 70):
        pp.create_bus(net, vn_kv=VN_KV, name="Bus %d" % label, index=label - 1)
    pp.create_ext_grid(net, bus=SLACK_LABEL - 1, vm_pu=1.0, va_degree=0.0,
                       name="Substation")
    for f, t, r, x, p_kw, q_kvar in BARAN_WU_69_BRANCHES:
        pp.create_line_from_parameters(
            net, from_bus=f - 1, to_bus=t - 1, length_km=1.0,
            r_ohm_per_km=r, x_ohm_per_km=x, c_nf_per_km=0.0,
            max_i_ka=MAX_I_KA, name="Line %d-%d" % (f, t))
        if p_kw != 0.0 or q_kvar != 0.0:
            pp.create_load(net, bus=t - 1, p_mw=p_kw / 1000.0,
                           q_mvar=q_kvar / 1000.0, name="Load %d" % t)
    return net
# ---- END FROZEN BUILDER ----


def sanity_check(verbose=True):
    """Structural + electrical sanity checks against the published base case."""
    import networkx as nx
    net = create_ieee69_baran_wu()
    out = {}
    out["n_bus"] = len(net.bus)
    out["n_branch"] = len(net.line)
    out["n_load"] = len(net.load)
    g = nx.Graph()
    g.add_nodes_from(net.bus.index)
    g.add_edges_from(zip(net.line.from_bus, net.line.to_bus))
    out["connected"] = bool(nx.is_connected(g))
    out["n_cycles"] = g.number_of_edges() - g.number_of_nodes() + 1
    out["radial"] = out["connected"] and out["n_cycles"] == 0
    out["total_p_kw"] = round(float(net.load.p_mw.sum()) * 1000.0, 2)
    out["total_q_kvar"] = round(float(net.load.q_mvar.sum()) * 1000.0, 2)
    pp.runpp(net)
    out["converged"] = bool(net.converged)
    out["loss_kw"] = round(float(net.res_line.pl_mw.sum()) * 1000.0, 3)
    out["min_vm_pu"] = round(float(net.res_bus.vm_pu.min()), 5)
    out["min_vm_bus_label"] = int(net.res_bus.vm_pu.idxmin()) + 1
    out["slack_p_mw"] = round(float(net.res_ext_grid.p_mw.iloc[0]), 6)
    out["max_line_i_ka"] = round(float(net.res_line.i_ka.max()), 5)
    checks = {
        "69 buses": out["n_bus"] == 69,
        "68 branches": out["n_branch"] == 68,
        "radial (connected, no loop)": out["radial"],
        "power flow converged": out["converged"],
        "total P = 3801.89 kW": abs(out["total_p_kw"] - 3801.89) < 0.01,
        "total Q = 2694.10 kvar": abs(out["total_q_kvar"] - 2694.10) < 0.01,
        "loss ~225 kW (lit. check value)": 220.0 < out["loss_kw"] < 230.0,
        "min V ~0.9092 pu (lit. check value)": abs(out["min_vm_pu"] - 0.9092) < 0.001,
        "min V at bus label 65": out["min_vm_bus_label"] == 65,
        "ratings non-binding": out["max_line_i_ka"] < MAX_I_KA,
    }
    if verbose:
        for k, v in out.items():
            print("  %-24s %s" % (k, v))
        print("  --")
        for k, v in checks.items():
            print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    out["all_pass"] = all(checks.values())
    out["checks"] = checks
    return out


if __name__ == "__main__":
    r = sanity_check()
    print("\nSANITY:", "PASS" if r["all_pass"] else "FAIL")
