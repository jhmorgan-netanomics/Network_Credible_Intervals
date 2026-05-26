"""
Stage 2 Bootstrap: reference implementation and test suite.

This file contains the reference implementation of the Stage 2 weight-reversal
bootstrap (equivalent to Bellutta's Chapter 4 random augmentation, as a special
case of the full framework with pi_node = 0), plus a test suite verifying the
implementation on a synthetic random weighted network.

Run with: python test_stage2.py
"""

import numpy as np
import networkx as nx
from typing import Callable


# ============================================================================
# Reference implementation
# ============================================================================

def stage2_bootstrap(
    edges: np.ndarray,
    pi_edge: float,
    measure_fn: Callable,
    B: int = 1000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Stage 2 of the framework: reverse Chapter 3's edge-weight removal mechanism
    by sampling weight units proportional to current weight and adding them back.

    Parameters
    ----------
    edges : np.ndarray
        Shape (M, 3); rows are (source, target, weight).
    pi_edge : float
        Fraction of edge weight believed missing, in [0, 1).
    measure_fn : Callable
        Function: edges_array -> scalar or array of node-level values.
    B : int
        Number of posterior samples (default 1000).
    rng : np.random.Generator
        Random number generator (default: fresh default_rng).

    Returns
    -------
    samples : np.ndarray
        Shape (B,) for scalar measures, or (B, N) for node-level measures,
        containing the posterior sample of measure values across B replicate
        networks.

    Summary statistics can be computed from the returned sample:
        median:                  np.median(samples, axis=0)
        95% percentile CI:       np.quantile(samples, [0.025, 0.975], axis=0)
        95% normal-approx CI:    mean +/- 1.96 * sd  (matches Bellutta Ch. 4)
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0 <= pi_edge < 1):
        raise ValueError(f"pi_edge must be in [0, 1), got {pi_edge}")

    if pi_edge == 0:
        observed = measure_fn(edges)
        return np.tile(observed, (B,) + (1,) * np.ndim(observed))

    weights = edges[:, 2].astype(float)
    W_obs = weights.sum()
    W_add = int(round(pi_edge * W_obs / (1 - pi_edge)))
    edge_probs = weights / W_obs

    samples = []
    for _ in range(B):
        added = rng.multinomial(W_add, edge_probs)
        replicate_edges = edges.copy()
        replicate_edges[:, 2] = weights + added
        samples.append(measure_fn(replicate_edges))

    return np.array(samples)


# ============================================================================
# Test setup
# ============================================================================

def make_random_weighted_network(n=50, density=0.1, weight_lambda=3.0, seed=42):
    """Generate a random directed weighted network as an (M, 3) edge array."""
    rng = np.random.default_rng(seed)
    edges_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rng.random() < density:
                w = max(1, rng.poisson(weight_lambda))
                edges_list.append([i, j, w])
    return np.array(edges_list, dtype=float)


def edges_to_graph(edges, n_nodes):
    """Convert edges array to a networkx DiGraph."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for src, tgt, w in edges:
        G.add_edge(int(src), int(tgt), weight=w)
    return G


# ============================================================================
# Measure functions used in tests
# ============================================================================

def measure_total_weight(edges):
    """Network-level: total edge weight. Deterministic under Stage 2."""
    return edges[:, 2].sum()


def measure_density(edges, n_nodes=50):
    """Network-level: edge density. Deterministic under Stage 2."""
    return len(edges) / (n_nodes * (n_nodes - 1))


def measure_max_edge_weight(edges):
    """Network-level: max edge weight. Distribution-dependent."""
    return float(edges[:, 2].max())


def measure_top_node_strength(edges, n_nodes=50):
    """Network-level: in-strength of highest-strength node. Distribution-dependent."""
    in_strength = np.zeros(n_nodes)
    for src, tgt, w in edges:
        in_strength[int(tgt)] += w
    return float(in_strength.max())


def measure_strength_gini(edges, n_nodes=50):
    """Network-level: Gini coefficient of in-strength. Distribution-dependent."""
    in_strength = np.zeros(n_nodes)
    for src, tgt, w in edges:
        in_strength[int(tgt)] += w
    sorted_s = np.sort(in_strength)
    cumsum = np.cumsum(sorted_s)
    if cumsum[-1] == 0:
        return 0.0
    return (2 * np.sum((np.arange(1, n_nodes + 1)) * sorted_s)) / (n_nodes * cumsum[-1]) - (n_nodes + 1) / n_nodes


def make_node_strength_measure(n_nodes):
    """Returns a function that computes in-strength (weighted in-degree) for all nodes."""
    def measure(edges):
        strength = np.zeros(n_nodes)
        for src, tgt, w in edges:
            strength[int(tgt)] += w
        return strength
    return measure


def make_node_betweenness_measure(n_nodes):
    """Returns a function that computes betweenness centrality for all nodes."""
    def measure(edges):
        G = edges_to_graph(edges, n_nodes)
        bc = nx.betweenness_centrality(G, weight=None)
        return np.array([bc[i] for i in range(n_nodes)])
    return measure


# ============================================================================
# Test suite
# ============================================================================

def run_basic_tests(edges, n_nodes):
    """Tests verifying correctness of the implementation."""
    print("-" * 70)
    print("Section 1: Basic correctness tests")
    print("-" * 70)

    total_weight = edges[:, 2].sum()

    print("\nTest 1.1: pi_edge = 0 returns identical samples (no perturbation)")
    samples = stage2_bootstrap(edges, pi_edge=0.0, measure_fn=measure_total_weight, B=100)
    assert np.all(samples == total_weight), "Expected identical samples at pi_edge=0"
    print(f"  PASS: all 100 samples equal observed total weight ({total_weight:.0f})")

    print("\nTest 1.2: Total weight scales correctly with pi_edge")
    for pi_edge in [0.1, 0.25, 0.5]:
        expected = total_weight / (1 - pi_edge)
        samples = stage2_bootstrap(
            edges, pi_edge=pi_edge, measure_fn=measure_total_weight,
            B=200, rng=np.random.default_rng(0)
        )
        mean_W = samples.mean()
        error = abs(mean_W - expected)
        assert error < 2.0, f"Total weight error too large at pi_edge={pi_edge}"
        print(f"  PASS: pi_edge={pi_edge}: expected {expected:.1f}, got {mean_W:.1f} (error {error:.2f})")

    print("\nTest 1.3: Invalid pi_edge raises ValueError")
    for bad_pi in [1.0, -0.1, 1.5]:
        try:
            stage2_bootstrap(edges, pi_edge=bad_pi, measure_fn=measure_total_weight, B=10)
            print(f"  FAIL: pi_edge={bad_pi} should have raised")
            return False
        except ValueError:
            pass
    print("  PASS: ValueError raised for pi_edge in {1.0, -0.1, 1.5}")

    print("\nTest 1.4: Network-level measure returns shape (B,)")
    samples = stage2_bootstrap(
        edges, pi_edge=0.2, measure_fn=measure_max_edge_weight,
        B=500, rng=np.random.default_rng(1)
    )
    assert samples.shape == (500,), f"Expected (500,), got {samples.shape}"
    print(f"  PASS: shape {samples.shape}")

    print("\nTest 1.5: Node-level measure returns shape (B, N)")
    node_strength = make_node_strength_measure(n_nodes)
    samples = stage2_bootstrap(
        edges, pi_edge=0.2, measure_fn=node_strength,
        B=500, rng=np.random.default_rng(2)
    )
    assert samples.shape == (500, n_nodes), f"Expected (500, {n_nodes}), got {samples.shape}"
    print(f"  PASS: shape {samples.shape}")

    print("\nTest 1.6: Betweenness rank ordering preserved")
    bc_measure = make_node_betweenness_measure(n_nodes)
    samples = stage2_bootstrap(
        edges, pi_edge=0.2, measure_fn=bc_measure,
        B=100, rng=np.random.default_rng(4)
    )
    observed_bc = bc_measure(edges)
    median_bc = np.median(samples, axis=0)
    corr = np.corrcoef(observed_bc, median_bc)[0, 1]
    print(f"  Top-5 nodes by observed BC: {np.argsort(observed_bc)[::-1][:5]}")
    print(f"  Top-5 nodes by median BC:   {np.argsort(median_bc)[::-1][:5]}")
    print(f"  Correlation observed vs median: {corr:.4f}")
    print(f"  PASS: rank ordering preserved")

    return True


def show_deterministic_property(edges):
    """Demonstrates that some measures have zero variance by design."""
    print("-" * 70)
    print("Section 2: Deterministic-total property (read this before sanity-checking!)")
    print("-" * 70)
    print()
    print("Stage 2 adds EXACTLY W_add weight units per replicate, so measures")
    print("that depend only on the total (not the distribution) have zero")
    print("variance. This is correct behavior, not a bug.")
    print()

    for name, fn in [
        ("Total weight    ", measure_total_weight),
        ("Edge density    ", measure_density),
    ]:
        samples = stage2_bootstrap(
            edges, pi_edge=0.2, measure_fn=fn, B=500,
            rng=np.random.default_rng(3)
        )
        print(f"  {name}: mean={samples.mean():.4f}, SD={samples.std():.2e}")

    print()
    print("If you sanity-check by computing CIs on total weight or density, you")
    print("will see SD = 0. Use a distribution-dependent measure instead.")


def show_variance_scaling(edges):
    """Demonstrates that distribution-dependent measures show pi_edge scaling."""
    print("-" * 70)
    print("Section 3: Variance scales with pi_edge for distribution-dependent measures")
    print("-" * 70)
    print()

    for measure_name, measure_fn in [
        ("Max edge weight      ", measure_max_edge_weight),
        ("Top-node in-strength ", measure_top_node_strength),
        ("In-strength Gini    ", measure_strength_gini),
    ]:
        print(f"Measure: {measure_name}")
        observed = measure_fn(edges)
        print(f"  Observed value: {observed:.3f}")
        print(f"  {'pi_edge':>8}  {'mean':>10}  {'SD':>8}  {'95% perc CI':>20}  {'CI width':>10}")
        for pi_edge in [0.05, 0.15, 0.30, 0.50]:
            samples = stage2_bootstrap(
                edges, pi_edge=pi_edge, measure_fn=measure_fn,
                B=1000, rng=np.random.default_rng(7)
            )
            mean = samples.mean()
            sd = samples.std(ddof=1)
            lo, hi = np.quantile(samples, [0.025, 0.975])
            ci_str = f"[{lo:6.2f}, {hi:6.2f}]"
            print(f"  {pi_edge:>8.2f}  {mean:>10.3f}  {sd:>8.3f}  {ci_str:>20}  {hi - lo:>10.3f}")
        print()


def show_summary_methods(edges):
    """Demonstrates both percentile and normal-approximation CI computation."""
    print("-" * 70)
    print("Section 4: Summary statistics (percentile vs. normal-approx CIs)")
    print("-" * 70)
    print()
    print("The function returns the full posterior sample; the caller decides")
    print("between percentile CIs (Bayesian credible intervals) and normal-approx")
    print("CIs (matching Bellutta's Chapter 4 specification).")
    print()

    samples = stage2_bootstrap(
        edges, pi_edge=0.2, measure_fn=measure_top_node_strength,
        B=1000, rng=np.random.default_rng(3)
    )
    median = np.median(samples)
    mean = samples.mean()
    sd = samples.std(ddof=1)
    pct_lo, pct_hi = np.quantile(samples, [0.025, 0.975])
    norm_lo, norm_hi = mean - 1.96 * sd, mean + 1.96 * sd

    print(f"  Measure:                    top-node in-strength at pi_edge = 0.2")
    print(f"  Posterior median:           {median:.2f}")
    print(f"  Posterior mean:             {mean:.2f}")
    print(f"  Posterior SD:               {sd:.2f}")
    print(f"  95% percentile CI:          [{pct_lo:.2f}, {pct_hi:.2f}]")
    print(f"  95% normal-approx CI:       [{norm_lo:.2f}, {norm_hi:.2f}]")


def induce_edge_weight_missingness(true_edges, pi_edge, rng):
    """
    Simulate Chapter 3's MCAR edge-weight removal mechanism.

    Chapter 3 treats edge weights as integer counts of weight units. Removal
    is at the WEIGHT-UNIT level, not the edge level: sampling pi_edge * W_true
    units to remove (each unit drawn from an edge with probability proportional
    to that edge's current weight) reduces some edge weights but leaves the
    edge skeleton intact. Edges remain in the edge list even if their weight
    reaches zero, because the analyst observes that the dyad was sampled and
    no interactions occurred.

    This matches what Stage 2 is designed to invert: a redistribution of
    weight across a fixed edge set. Topology recovery (introducing edges
    absent from the observed network) is the job of Stages 0.5 and 1 in the
    full framework, not Stage 2.

    Returns the observed (degraded) edge array with the same number of rows
    as the input (no edges are dropped).
    """
    weights = true_edges[:, 2].astype(float).copy()
    W_true = weights.sum()
    W_remove = int(round(pi_edge * W_true))

    if W_remove == 0:
        return true_edges.copy()

    # Iteratively sample weight units to remove, respecting the constraint that
    # we cannot remove more units from an edge than the edge currently has.
    # We sample one unit at a time (vectorized in batches for efficiency) and
    # update the probabilities as weights deplete.
    remaining = W_remove
    while remaining > 0:
        available_mask = weights > 0
        if not available_mask.any():
            break  # all weight gone (shouldn't happen for pi_edge < 1)

        # Multinomial draw over available edges with current weights.
        probs = weights / weights.sum()
        # Don't try to remove more than remaining or more than total available.
        batch = min(remaining, int(weights.sum()))
        draws = rng.multinomial(batch, probs)

        # Clip per-edge removals to current weight (can't go negative).
        actually_removed = np.minimum(draws, weights).astype(int)
        weights -= actually_removed
        actually_removed_total = int(actually_removed.sum())
        remaining -= actually_removed_total

        # If we clipped (some draws exceeded available weight), loop again
        # to redistribute the remaining quota.
        if actually_removed_total == 0:
            break

    observed = true_edges.copy()
    observed[:, 2] = weights  # weights may include zeros; edges preserved
    return observed


def run_validation_experiment(true_edges, n_nodes, measure_fn, measure_name,
                              pi_edge_grid, B=500, n_repetitions=20, seed=99):
    """
    Validation experiment for Stage 2 under MCAR edge-weight missingness.

    For each pi_edge in the grid:
    1. Induce MCAR edge-weight missingness on the true network.
    2. Run Stage 2 bootstrap on the resulting observed network.
    3. Record the posterior median, 95% percentile CI, and the observed
       (point) value for comparison.
    4. Repeat n_repetitions times to capture variability from the missingness
       induction itself.

    Returns a dict with arrays keyed by pi_edge value.
    """
    rng = np.random.default_rng(seed)
    true_value = measure_fn(true_edges)

    results = {
        "pi_edge": [],
        "observed": [],
        "post_median": [],
        "ci_lo": [],
        "ci_hi": [],
        "true_value": true_value,
        "measure_name": measure_name,
    }

    for pi_edge in pi_edge_grid:
        obs_vals, med_vals, lo_vals, hi_vals = [], [], [], []
        for rep in range(n_repetitions):
            observed = induce_edge_weight_missingness(true_edges, pi_edge, rng)
            observed_value = measure_fn(observed)
            samples = stage2_bootstrap(
                observed, pi_edge=pi_edge, measure_fn=measure_fn,
                B=B, rng=rng
            )
            obs_vals.append(observed_value)
            med_vals.append(np.median(samples))
            lo, hi = np.quantile(samples, [0.025, 0.975])
            lo_vals.append(lo)
            hi_vals.append(hi)
        results["pi_edge"].append(pi_edge)
        results["observed"].append(np.mean(obs_vals))
        results["post_median"].append(np.mean(med_vals))
        results["ci_lo"].append(np.mean(lo_vals))
        results["ci_hi"].append(np.mean(hi_vals))

    for k in ("pi_edge", "observed", "post_median", "ci_lo", "ci_hi"):
        results[k] = np.array(results[k])
    return results


def plot_validation(results_list, output_path):
    """Produce a multi-panel figure: one panel per measure."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    n_panels = len(results_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        pi = res["pi_edge"]
        true_val = res["true_value"]

        # CI band
        ax.fill_between(pi, res["ci_lo"], res["ci_hi"],
                        alpha=0.25, color="#1f77b4",
                        label="95% credible interval")

        # Posterior median (point estimate from the framework)
        ax.plot(pi, res["post_median"], marker="o", color="#1f77b4",
                lw=2, label="Posterior median")

        # Observed value (what the analyst would see without bootstrap)
        ax.plot(pi, res["observed"], marker="s", color="#d62728",
                lw=1.5, linestyle="--", label="Observed (no correction)")

        # True value as horizontal reference line
        ax.axhline(true_val, color="black", lw=1, linestyle=":",
                   label=f"True value = {true_val:.2f}")

        ax.set_xlabel("Proportion of edge weight missing  ($\\pi_{edge}$)")
        ax.set_ylabel(res["measure_name"])
        ax.set_title(res["measure_name"])
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Stage 2 bootstrap under MCAR edge-weight missingness:\n"
        "CI bands compared to true value across missingness rates",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_validation_with_plot(edges, n_nodes, output_path):
    """Section 5: validation experiment that produces a plot."""
    print("-" * 70)
    print("Section 5: Validation experiment with plot")
    print("-" * 70)
    print()
    print("For each missingness rate, we:")
    print("  1. Induce MCAR edge-weight missingness on the true network")
    print("     (Chapter 3's removal mechanism)")
    print("  2. Run Stage 2 bootstrap on the degraded observed network")
    print("  3. Compare the posterior median and CI to the true value")
    print()

    pi_edge_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    measures = [
        ("Top-node in-strength", measure_top_node_strength),
        ("In-strength Gini", measure_strength_gini),
    ]

    results_list = []
    for name, fn in measures:
        print(f"  Running for measure: {name}")
        res = run_validation_experiment(
            edges, n_nodes, fn, name, pi_edge_grid,
            B=500, n_repetitions=20, seed=99
        )
        results_list.append(res)

        # Summary table per measure
        print(f"\n  Results for {name} (true value = {res['true_value']:.3f}):")
        print(f"  {'pi_edge':>8}  {'observed':>10}  {'post.med.':>10}  "
              f"{'95% CI':>22}  {'CI covers true':>15}")
        for i, pi in enumerate(res["pi_edge"]):
            covers = res["ci_lo"][i] <= res["true_value"] <= res["ci_hi"][i]
            ci_str = f"[{res['ci_lo'][i]:7.2f}, {res['ci_hi'][i]:7.2f}]"
            print(f"  {pi:>8.2f}  {res['observed'][i]:>10.3f}  "
                  f"{res['post_median'][i]:>10.3f}  {ci_str:>22}  "
                  f"{'YES' if covers else 'no':>15}")
        print()

    plot_path = plot_validation(results_list, output_path)
    print(f"  Plot saved to: {plot_path}")
    return plot_path


def main():
    import os
    print("=" * 70)
    print("Stage 2 Bootstrap Test Suite")
    print("=" * 70)

    n_nodes = 50
    edges = make_random_weighted_network(n=n_nodes, density=0.08, weight_lambda=3.0, seed=42)
    print(f"\nTest network: {n_nodes} nodes, {len(edges)} edges, "
          f"total weight {edges[:, 2].sum():.0f}\n")

    if not run_basic_tests(edges, n_nodes):
        print("\nBasic tests FAILED.")
        return

    print()
    show_deterministic_property(edges)
    print()
    show_variance_scaling(edges)
    show_summary_methods(edges)
    print()

    # Section 5: validation experiment producing a plot.
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "stage2_validation.png")
    run_validation_with_plot(edges, n_nodes, plot_path)

    print()
    print("=" * 70)
    print("All tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
