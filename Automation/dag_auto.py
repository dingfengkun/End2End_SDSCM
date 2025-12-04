

# ================================
# Name-only DAG Generator (No Data)
# ================================
# - Always acyclic (forward edges in a random topo order or layered stages)
# - Supports max_parents, edge probability OR degree priors
# - Optional: hard_requires / hard_forbids
# - Scores DAGs structurally
# - Returns parent_indices ready for your JSON

import numpy as np
from math import log
import matplotlib.pyplot as plt
from Automation.sdscm import plot_dag_from_sample_space
import json

def generate_random_topo_order(n, rng):
    order = np.arange(n)
    rng.shuffle(order)
    return order.tolist()


def assign_layers(order, n_layers):
    n = len(order)
    bounds = np.linspace(0, n, n_layers + 1, dtype=int)
    layer_of = {}
    for L in range(n_layers):
        for idx in order[bounds[L]:bounds[L+1]]:
            layer_of[idx] = L
    return layer_of

def sample_dag_no_data(
    var_names,
    *,
    edge_prob=0.25,
    max_parents=2,
    layered=True,
    n_layers=3,
    degree_prior=None,   # None | "poisson" | "scale_free"
    hard_requires=None,  # list of (u_name, v_name)
    hard_forbids=None,   # list of (u_name, v_name)
    root_vars=None,               # vars that must be roots (no incoming edges)
    top_vars=None,                # vars that must appear at the top of topo order
    # NEW:
    important_vars=None,          # vars that must not be isolated
    min_incident_degree=1,        # minimum incident degree per important var
    important_role="either",      # "either" | "parent" | "child"
    fail_on_impossible=True,
    seed=0
):
    """
    Returns: parents_dict, edges, topo_order_indices
    parents_dict: {child_idx: [parent_idx, ...]}
    edges: [(u_name, v_name), ...]
    """
    hard_requires = hard_requires or []
    hard_forbids  = set(hard_forbids or [])
    root_vars     = set(root_vars or [])
    top_vars      = list(dict.fromkeys(top_vars or []))  # keep order, dedup
    important_vars = list(dict.fromkeys(important_vars or []))

    # Validate names exist
    unknown = [v for v in list(root_vars) + top_vars + important_vars if v not in var_names]
    if unknown:
        raise ValueError(f"Unknown variables in root/top/important: {unknown}")

    name_to_idx = {n:i for i,n in enumerate(var_names)}
    rng = np.random.default_rng(seed)
    n = len(var_names)

    # ----- 1) Build topo order with top_vars first -----
    rest = [i for i,nm in enumerate(var_names) if nm not in top_vars]
    rng.shuffle(rest)
    order = [name_to_idx[nm] for nm in top_vars] + rest
    pos = {node: i for i, node in enumerate(order)}

    # Enforce required edges to be forward by swapping within order tail if needed
    changed = True
    iters = 0
    while changed and iters < 5 * n and hard_requires:
        changed = False; iters += 1
        for (u_name, v_name) in hard_requires:
            u, v = name_to_idx[u_name], name_to_idx[v_name]
            if pos[u] >= pos[v]:
                iu, iv = pos[u], pos[v]
                if var_names[v] in top_vars and var_names[u] not in top_vars:
                    order[iu], order[max(0, iv-1)] = order[max(0, iv-1)], order[iu]
                else:
                    order[iu], order[iv] = order[iv], order[iu]
                pos = {node: i for i, node in enumerate(order)}
                changed = True

    pos = {node: i for i, node in enumerate(order)}

    # ----- 2) Layers (top_vars in layer 0 if layered) -----
    layer_of = None
    if layered:
        bounds = np.linspace(0, n, n_layers + 1, dtype=int)
        layer_of = {}
        for idx in order:
            layer_of[idx] = 0 if var_names[idx] in top_vars else None
        for L in range(n_layers):
            start, end = bounds[L], bounds[L+1]
            cnt = 0
            for idx in order:
                if layer_of[idx] is None and cnt < (end - start):
                    layer_of[idx] = L
                    cnt += 1

    # ----- 3) Forbid incoming edges to root_vars -----
    for rv in root_vars:
        for u in var_names:
            if u != rv:
                hard_forbids.add((u, rv))

    # ----- 4) Seed parents with required edges -----
    parents = {j: [] for j in range(n)}
    for (u_name, v_name) in hard_requires:
        if (u_name, v_name) in hard_forbids:
            raise ValueError(f"Conflict: required edge {u_name}->{v_name} is forbidden.")
        u, v = name_to_idx[u_name], name_to_idx[v_name]
        if pos[u] < pos[v]:
            parents[v].append(u)
        else:
            raise ValueError(f"Required edge {u_name}->{v_name} violates topological order.")

    # ----- 5) Random forward edges under constraints -----
    for v in range(n):
        vname = var_names[v]
        candidates = []
        for u in range(n):
            if pos[u] < pos[v]:
                if layered and layer_of and (layer_of[u] >= layer_of[v]):
                    continue
                if (var_names[u], vname) in hard_forbids:
                    continue
                if u in parents[v]:
                    continue
                candidates.append(u)
        chosen = [u for u in candidates if rng.random() < edge_prob]
        rng.shuffle(chosen)

        req_for_v = {name_to_idx[u] for (u, w) in hard_requires if w == vname}
        merged = list(req_for_v) + [u for u in chosen if u not in req_for_v]
        parents[v] = merged[:max_parents]

    # ----- 6) Ensure important variables are connected -----
    # helper closures
    def indeg(j): return len(parents[j])
    def outneighbors(i):
        # scan children (cheap enough for small d)
        return [v for v in range(n) if i in parents[v]]
    def degree(i): return indeg(i) + len(outneighbors(i))

    def can_add_u_to_v(u, v):
        """Check if we can add u->v without breaking constraints."""
        if u == v: return False
        # forward only
        if pos[u] >= pos[v]: return False
        # layered forward
        if layered and layer_of and (layer_of[u] >= layer_of[v]): return False
        # forbids
        if (var_names[u], var_names[v]) in hard_forbids: return False
        # no duplicate
        if u in parents[v]: return False
        # respect max_parents
        if len(parents[v]) >= max_parents: return False
        return True

    imp_role = important_role.lower()
    for name in important_vars:
        i = name_to_idx[name]
        target_deg = max(0, int(min_incident_degree))
        # Try to add edges until degree >= target
        guard = 0
        while degree(i) < target_deg and guard < 3 * n:
            guard += 1
            progressed = False

            # Option A: make i a PARENT of some later node v
            if imp_role in ("either", "parent") and name not in root_vars:
                # (root_vars can also be parents; we only forbid incoming to roots, not outgoing)
                pass
            if imp_role in ("either", "parent"):
                for v in order[pos[i]+1:]:
                    if can_add_u_to_v(i, v):
                        parents[v].append(i)
                        progressed = True
                        break

            # Option B: make i a CHILD of some earlier node u (unless i is root)
            if not progressed and imp_role in ("either", "child"):
                if name not in root_vars:
                    for u in order[:pos[i]]:
                        if can_add_u_to_v(u, i):
                            parents[i].append(u)
                            progressed = True
                            break

            if not progressed:
                # cannot satisfy further under constraints
                if fail_on_impossible:
                    reason = []
                    if name in root_vars and imp_role in ("either","child"):
                        reason.append("node is root (no incoming edges allowed)")
                    if pos[i] == n-1 and imp_role in ("either","parent"):
                        reason.append("node is last in order (no later children)")
                    raise ValueError(
                        f"Cannot satisfy importance for '{name}' (needed degree >= {target_deg}). "
                        f"Constraints/layering/max_parents may block all additions. "
                        f"{' | '.join(reason) if reason else ''}"
                    )
                else:
                    break  # give up quietly

    # ----- finalize -----
    edges = [(var_names[u], var_names[v]) for v in range(n) for u in parents[v]]
    return parents, edges, order

def _safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, None))

def _empirical_pmf(values, kmax):
    hist = np.bincount(values, minlength=kmax+1).astype(float)
    pmf = hist / max(1, hist.sum())
    return pmf

def _poisson_pmf(kmax, lam):
    ks = np.arange(kmax+1)
    # unnormalized Poisson
    logp = ks * _safe_log(lam) - lam - np.array([np.sum(_safe_log(np.arange(1, k+1))) for k in ks])
    p = np.exp(logp - np.max(logp))
    p /= p.sum()
    return p

def _powerlaw_pmf(kmax, alpha=2.0):
    ks = np.arange(kmax+1)
    # shift by +1 to avoid k=0 singularity; allow zeros but keep tail heavy
    weights = (ks + 1.0) ** (-alpha)
    p = weights / weights.sum()
    return p

def _kl_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def _allowed_forward_pairs(n, topo_order=None, layered=False, n_layers=3):
    # approximate number of *eligible* edges (for density normalization)
    if not layered:
        return n*(n-1)//2
    # layered: count pairs only from lower to higher layer
    bounds = np.linspace(0, n, n_layers + 1, dtype=int)
    total = 0
    for L in range(n_layers):
        size_L = bounds[L+1] - bounds[L]
        for R in range(L+1, n_layers):
            size_R = bounds[R+1] - bounds[R]
            total += size_L * size_R
    return int(total)

def improved_structural_score(
    var_names,
    edges,                # list[(u_name, v_name)]
    topo_order,           # list[var_name] (perm)
    *,
    layered=False,
    n_layers=3,
    root_vars=None,
    # density / degree targets
    target_density=0.25,          # desired edge density among *eligible* pairs
    indegree_prior=("poisson", 1.0),  # ("poisson", lam) or ("powerlaw", alpha) or ("none", None)
    # weights (tune to taste)
    w_density=1.0,         # how strongly to hit the density target
    w_degree=1.0,          # how strongly to match indegree prior
    w_coverage=0.5,        # bonus for % nodes with indegree>=1 (excluding roots)
    w_balance=0.25,        # penalize extreme indegree variance (unless powerlaw)
    w_vstruct=0.1,         # bonus per v-structure
    w_layer=0.05,          # tiny bonus for forward layer edges (if layered)
):
    n = len(var_names)
    name_to_idx = {n:i for i,n in enumerate(var_names)}
    pos = {name: i for i, name in enumerate(topo_order)}

    # indegree array
    indeg = np.zeros(n, dtype=int)
    parent_sets = [[] for _ in range(n)]
    for u, v in edges:
        j = name_to_idx[v]
        i = name_to_idx[u]
        indeg[j] += 1
        parent_sets[j].append(i)

    m = len(edges)

    # 1) Density term: prefer edge count close to m* = target_density * eligible_pairs
    eligible = _allowed_forward_pairs(n, topo_order, layered, n_layers)
    m_star = target_density * max(1, eligible)
    density_penalty = (m - m_star)**2 / max(1.0, eligible)  # scale-invariant-ish
    score_density = - w_density * density_penalty

    # 2) Degree prior term: KL(empirical indegree || target)
    kmax = int(np.max(indeg)) if m > 0 else 0
    p_emp = _empirical_pmf(indeg, kmax=kmax)
    prior_type, param = indegree_prior
    if prior_type == "poisson":
        p_tar = _poisson_pmf(kmax, lam=float(param))
        kl = _kl_divergence(p_emp, p_tar)
    elif prior_type == "powerlaw":
        p_tar = _powerlaw_pmf(kmax, alpha=float(param))
        kl = _kl_divergence(p_emp, p_tar)
    else:
        kl = 0.0
    score_degree = - w_degree * kl

    # 3) Coverage: fraction of non-roots with >=1 parent
    roots = set(root_vars or [])
    nonroot_mask = np.array([var_names[i] not in roots for i in range(n)], dtype=bool)
    covered = np.sum((indeg >= 1) & nonroot_mask)
    total_nonroots = int(np.sum(nonroot_mask))
    coverage = covered / total_nonroots if total_nonroots > 0 else 1.0
    score_cov = w_coverage * coverage

    # 4) Balance: penalize high indegree variance unless you *want* powerlaw
    if prior_type == "powerlaw":
        score_bal = 0.0
    else:
        var_in = float(np.var(indeg)) if n > 1 else 0.0
        score_bal = - w_balance * var_in

    # 5) V-structure count: i->k<-j where i and j have no edge between them
    # Build quick adjacency for parent-parent check
    adj = {name_to_idx[u]: set() for u,_ in edges}
    for u,v in edges:
        adj.setdefault(name_to_idx[u], set()).add(name_to_idx[v])
    vstruct = 0
    for k in range(n):
        ps = parent_sets[k]
        L = len(ps)
        if L >= 2:
            # count unordered parent pairs with no direct edge between them (either direction)
            for a in range(L):
                for b in range(a+1, L):
                    i, j = ps[a], ps[b]
                    # no i->j and no j->i
                    if (j not in adj.get(i, set())) and (i not in adj.get(j, set())):
                        vstruct += 1
    score_v = w_vstruct * vstruct

    # 6) Layer bonus (tiny): reward crossing layers in forward direction
    if layered:
        # approximate layers from topo slices
        bounds = np.linspace(0, n, n_layers + 1, dtype=int)
        layer_of = {}
        for L in range(n_layers):
            for name in topo_order[bounds[L]:bounds[L+1]]:
                layer_of[name_to_idx[name]] = L
        layer_bonus = 0
        for u, v in edges:
            ui, vi = name_to_idx[u], name_to_idx[v]
            if layer_of.get(ui, 0) < layer_of.get(vi, 0):
                layer_bonus += 1
        score_layer = w_layer * layer_bonus
    else:
        score_layer = 0.0

    total = score_density + score_degree + score_cov + score_bal + score_v + score_layer
    return float(total)

def sample_best_dags_no_data(
    var_names,
    *,
    n_trials=50,
    edge_prob=0.25,
    max_parents=2,
    layered=True,
    n_layers=3,
    degree_prior=None,     # optional: wire back in if you like
    hard_requires=None,
    hard_forbids=None,
    root_vars=None,        # NEW
    top_vars=None,     # NEW
    important_vars = None,
    min_incident_degree = 1,
    important_role = "either",
    fail_on_impossible = True,
    seed=2,
    topk=1
):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        parents, edges, order = sample_dag_no_data(
            var_names,
            edge_prob=edge_prob,
            max_parents=max_parents,
            layered=layered,
            n_layers=n_layers,
            degree_prior=degree_prior,
            hard_requires=hard_requires,
            hard_forbids=hard_forbids,
            root_vars=root_vars,
            top_vars=top_vars,
            important_vars=important_vars,          # vars that must not be isolated
            min_incident_degree=min_incident_degree,        # minimum incident degree per important var
            important_role=important_role,      # "either" | "parent" | "child"
            fail_on_impossible=fail_on_impossible,
            seed=int(rng.integers(0, 1_000_000)),
        )
        score = improved_structural_score(
          var_names,
          edges,
          topo_order=[var_names[i] for i in order],         # list of names
          layered=layered,
          n_layers=n_layers,
          root_vars=root_vars,
          target_density=0.4,                 # <- set your desired density here
          indegree_prior=("none",None),     # or ("powerlaw", 2.2) or ("none", None)
          w_density=1.0,
          w_degree=1.0,
          w_coverage=1.0,
          w_balance=0.2,
          w_vstruct=0.15,
          w_layer=0.05,
        )
        parent_indices = [parents[j] for j in range(len(var_names))]
        results.append({
            "score": float(score),
            "edges": edges,
            "topo_order": [var_names[i] for i in order],
            "parent_indices": parent_indices,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topk]




def fill_DAG(json_framework, dag_edges, exogenous_dict, output_path=None):
    """
    Fill the DAG edges and exogenous values in JSON framework.

    Args:
        json_framework: dict or file path of JSON
        dag_edges: dict, variable -> list of parent variable names
        exogenous_dict: variable -> bool indicating if exogenous
        output_path: optional save path
    """

    # Load JSON if file path
    if isinstance(json_framework, str):
        with open(json_framework, "r") as f:
            config = json.load(f)
    else:
        config = json_framework

    nodes = config["setup_sequence_sample_space"]

    # Check exogenous list length
    if len(exogenous_dict) != len(nodes):
        raise ValueError("exogenous_dict length must match number of variables.")

    # Build mapping: variable_name -> index
    name_to_idx = {node["variable_name"]: i for i, node in enumerate(nodes)}

    # Fill DAG
    for i, node in enumerate(nodes):
        var = node["variable_name"]

        # Convert parent names -> indices
        if var in dag_edges:
            parent_indices = dag_edges[var]
        else:
            parent_indices = []
        if var in exogenous_dict:
            exogenous = exogenous_dict[var]
        else:
            exogenous = False

        node["parent_indices"] = parent_indices
        node["exogenous"] = exogenous

    # Save if needed
    if output_path:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

    return config

# fill_DAG("titanic_framework.json",parents_idx_by_name, exogenous_by_name , "test.json")


