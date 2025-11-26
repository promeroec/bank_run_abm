"""
Analysis Module for Bank Run ABM
=================================

Provides statistical analysis functions for examining:
- Coordination vs herding behavior
- Convergence properties
- Statistical summaries of simulation results
"""

import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from scipy import stats

if TYPE_CHECKING:
    from .bank_run_model import SimulationResults, BankRunSimulation, SARSAAgent


@dataclass
class ConvergenceAnalysis:
    """Results from convergence analysis."""
    converged: bool
    convergence_episode: Optional[int]
    final_q_change: float
    final_policy_stability: float
    half_life: Optional[int]


@dataclass
class CoordinationAnalysis:
    """Results from coordination vs herding analysis."""
    coordination_index: float
    herding_index: float
    independence_deviation: float
    classification: str  # 'coordinating', 'herding', 'independent'


@dataclass
class StatisticalSummary:
    """Comprehensive statistical summary."""
    # Central tendencies
    mean_run_rate: float
    median_run_rate: float
    mode_queue_size: float
    
    # Dispersion
    std_run_rate: float
    iqr_queue_size: float
    cv_consumption: float  # Coefficient of variation
    
    # Distribution shape
    skewness_queue: float
    kurtosis_queue: float
    
    # Confidence intervals (95%)
    ci_run_rate: tuple
    ci_queue_size: tuple
    
    # Time series properties
    autocorr_runs: float
    trend_consumption: float


def compute_coordination_index(results: 'SimulationResults',
                                n_agents: int,
                                run_threshold: int) -> float:
    """
    Compute coordination index.
    
    Measures how well agents coordinate to avoid bank runs
    compared to independent behavior baseline.
    
    Index > 0: Better than random (coordination)
    Index < 0: Worse than random (herding)
    Index ≈ 0: Near independent behavior
    
    Args:
        results: SimulationResults object
        n_agents: Population size
        run_threshold: Run threshold η
        
    Returns:
        Coordination index in [-1, 1]
    """
    observed_run_rate = results.run_frequency
    
    # Compute expected run rate under independence
    theta = results.bank_visit_frequency
    if theta <= 0 or theta >= 1:
        return 0.0
        
    expected_run_rate = _binomial_tail_prob(n_agents, run_threshold, theta)
    
    if expected_run_rate <= 0:
        return 1.0 if observed_run_rate <= 0 else -1.0
    
    # Positive = coordination (fewer runs than expected)
    # Negative = herding (more runs than expected)
    deviation = (expected_run_rate - observed_run_rate) / expected_run_rate
    
    return np.clip(deviation, -1.0, 1.0)


def compute_herding_index(results: 'SimulationResults') -> float:
    """
    Compute herding index based on queue size dynamics.
    
    Measures the tendency for queue sizes to cluster near
    extremes (0 or full) vs uniform distribution.
    
    Index close to 1: Strong herding
    Index close to 0: Independent behavior
    
    Args:
        results: SimulationResults object
        
    Returns:
        Herding index in [0, 1]
    """
    queue_sizes = np.array(results.episode_queue_sizes)
    
    if len(queue_sizes) == 0:
        return 0.0
    
    # Normalize queue sizes
    max_queue = np.max(queue_sizes) if np.max(queue_sizes) > 0 else 1
    normalized = queue_sizes / max_queue
    
    # Compute bimodality coefficient
    # High bimodality suggests herding (queues cluster at 0 or near threshold)
    n = len(normalized)
    skewness = stats.skew(normalized)
    kurtosis = stats.kurtosis(normalized)
    
    # Bimodality coefficient (Sarle's bimodality coefficient)
    bc = (skewness ** 2 + 1) / (kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    
    # Transform to [0, 1] range
    herding_index = min(1.0, max(0.0, bc))
    
    return herding_index


def analyze_convergence(results: 'SimulationResults',
                        threshold: float = 0.01,
                        window: int = 100) -> ConvergenceAnalysis:
    """
    Analyze learning convergence.
    
    Determines if and when the learning algorithm converged,
    based on Q-value changes and policy stability.
    
    Args:
        results: SimulationResults object
        threshold: Convergence threshold for Q-value changes
        window: Window size for stability assessment
        
    Returns:
        ConvergenceAnalysis object
    """
    q_changes = np.array(results.q_convergence)
    policy_changes = np.array(results.policy_changes)
    
    if len(q_changes) == 0:
        return ConvergenceAnalysis(
            converged=False,
            convergence_episode=None,
            final_q_change=0.0,
            final_policy_stability=0.0,
            half_life=None
        )
    
    # Find convergence point
    converged = False
    convergence_episode = None
    
    for i in range(window, len(q_changes)):
        recent_changes = q_changes[i-window:i]
        if np.mean(recent_changes) < threshold:
            converged = True
            convergence_episode = i
            break
    
    # Compute final statistics
    final_window = min(window, len(q_changes))
    final_q_change = np.mean(q_changes[-final_window:])
    
    if len(policy_changes) > 0:
        final_policy_stability = 1.0 - np.mean(policy_changes[-final_window:]) / max(1, np.max(policy_changes))
    else:
        final_policy_stability = 1.0
    
    # Compute half-life (episodes to halve initial Q-change rate)
    half_life = None
    if len(q_changes) > 1:
        initial_change = q_changes[0]
        half_target = initial_change / 2
        for i, change in enumerate(q_changes):
            if change < half_target:
                half_life = i
                break
    
    return ConvergenceAnalysis(
        converged=converged,
        convergence_episode=convergence_episode,
        final_q_change=final_q_change,
        final_policy_stability=final_policy_stability,
        half_life=half_life
    )


def statistical_summary(results: 'SimulationResults') -> StatisticalSummary:
    """
    Compute comprehensive statistical summary.
    
    Args:
        results: SimulationResults object
        
    Returns:
        StatisticalSummary object
    """
    runs = np.array(results.episode_runs, dtype=float)
    queues = np.array(results.episode_queue_sizes)
    consumption = np.array(results.episode_consumptions)
    
    n = len(runs)
    
    # Central tendencies
    mean_run_rate = np.mean(runs)
    median_run_rate = np.median(runs)
    
    if len(queues) > 0:
        mode_result = stats.mode(np.round(queues).astype(int), keepdims=True)
        mode_queue = float(mode_result.mode[0])
    else:
        mode_queue = 0.0
    
    # Dispersion
    std_run_rate = np.std(runs)
    iqr_queue = float(stats.iqr(queues)) if len(queues) > 0 else 0.0
    
    mean_cons = np.mean(consumption) if len(consumption) > 0 else 0.0
    std_cons = np.std(consumption) if len(consumption) > 0 else 0.0
    cv_consumption = std_cons / mean_cons if mean_cons > 0 else 0.0
    
    # Distribution shape
    skewness_queue = float(stats.skew(queues)) if len(queues) > 2 else 0.0
    kurtosis_queue = float(stats.kurtosis(queues)) if len(queues) > 3 else 0.0
    
    # Confidence intervals (95%)
    if n > 1:
        se_run = std_run_rate / np.sqrt(n)
        ci_run = (mean_run_rate - 1.96 * se_run, mean_run_rate + 1.96 * se_run)
        
        mean_queue = np.mean(queues) if len(queues) > 0 else 0.0
        se_queue = np.std(queues) / np.sqrt(len(queues)) if len(queues) > 0 else 0.0
        ci_queue = (mean_queue - 1.96 * se_queue, mean_queue + 1.96 * se_queue)
    else:
        ci_run = (mean_run_rate, mean_run_rate)
        ci_queue = (np.mean(queues) if len(queues) > 0 else 0.0,) * 2
    
    # Time series properties
    if len(runs) > 1:
        autocorr_runs = np.corrcoef(runs[:-1], runs[1:])[0, 1]
        if np.isnan(autocorr_runs):
            autocorr_runs = 0.0
    else:
        autocorr_runs = 0.0
    
    if len(consumption) > 1:
        x = np.arange(len(consumption))
        slope, _, _, _, _ = stats.linregress(x, consumption)
        trend_consumption = slope
    else:
        trend_consumption = 0.0
    
    return StatisticalSummary(
        mean_run_rate=mean_run_rate,
        median_run_rate=median_run_rate,
        mode_queue_size=mode_queue,
        std_run_rate=std_run_rate,
        iqr_queue_size=iqr_queue,
        cv_consumption=cv_consumption,
        skewness_queue=skewness_queue,
        kurtosis_queue=kurtosis_queue,
        ci_run_rate=ci_run,
        ci_queue_size=ci_queue,
        autocorr_runs=autocorr_runs,
        trend_consumption=trend_consumption
    )


def analyze_coordination_vs_herding(results: 'SimulationResults',
                                     n_agents: int,
                                     run_threshold: int) -> CoordinationAnalysis:
    """
    Comprehensive analysis of coordination vs herding behavior.
    
    Classifies the emergent behavior as:
    - 'coordinating': Fewer runs than independent baseline
    - 'herding': More runs than independent baseline
    - 'independent': Near-random behavior
    
    Args:
        results: SimulationResults object
        n_agents: Population size
        run_threshold: Run threshold η
        
    Returns:
        CoordinationAnalysis object
    """
    coordination_idx = compute_coordination_index(results, n_agents, run_threshold)
    herding_idx = compute_herding_index(results)
    
    # Compute deviation from independence
    theta = results.bank_visit_frequency
    expected_run_rate = _binomial_tail_prob(n_agents, run_threshold, theta)
    independence_dev = abs(results.run_frequency - expected_run_rate)
    
    # Classify behavior
    if coordination_idx > 0.2:
        classification = 'coordinating'
    elif coordination_idx < -0.2:
        classification = 'herding'
    else:
        classification = 'independent'
    
    return CoordinationAnalysis(
        coordination_index=coordination_idx,
        herding_index=herding_idx,
        independence_deviation=independence_dev,
        classification=classification
    )


def compare_experiments(results_list: List['SimulationResults'],
                        labels: List[str]) -> Dict[str, Any]:
    """
    Compare results across multiple experiments.
    
    Args:
        results_list: List of SimulationResults objects
        labels: Labels for each experiment
        
    Returns:
        Dictionary with comparison statistics
    """
    comparison = {
        'labels': labels,
        'run_frequencies': [r.run_frequency for r in results_list],
        'avg_queue_sizes': [r.avg_queue_size for r in results_list],
        'avg_consumptions': [r.avg_consumption for r in results_list],
        'bank_visit_frequencies': [r.bank_visit_frequency for r in results_list],
    }
    
    # Pairwise statistical tests
    n_exp = len(results_list)
    if n_exp > 1:
        # Run rate differences (chi-squared test)
        run_counts = np.array([[r.n_runs, r.n_episodes - r.n_runs] 
                               for r in results_list])
        chi2, p_value = stats.chi2_contingency(run_counts)[:2]
        comparison['chi2_test'] = {'statistic': chi2, 'p_value': p_value}
        
        # Queue size differences (Kruskal-Wallis test)
        queue_data = [r.episode_queue_sizes for r in results_list]
        if all(len(q) > 0 for q in queue_data):
            h_stat, p_kw = stats.kruskal(*queue_data)
            comparison['kruskal_wallis'] = {'statistic': h_stat, 'p_value': p_kw}
    
    return comparison


def _binomial_tail_prob(n: int, k: int, p: float) -> float:
    """
    Compute P(X >= k) for X ~ Binomial(n, p).
    
    Args:
        n: Number of trials
        k: Threshold
        p: Success probability
        
    Returns:
        Tail probability
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    
    return 1 - stats.binom.cdf(k - 1, n, p)


def monte_carlo_confidence_interval(simulation: 'BankRunSimulation',
                                     n_episodes: int = 1000,
                                     n_bootstrap: int = 100,
                                     confidence: float = 0.95) -> Dict[str, tuple]:
    """
    Compute confidence intervals via Monte Carlo bootstrap.
    
    Args:
        simulation: BankRunSimulation object
        n_episodes: Episodes per bootstrap sample
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Dictionary with confidence intervals for key metrics
    """
    run_rates = []
    queue_sizes = []
    consumptions = []
    
    for _ in range(n_bootstrap):
        results = simulation.run(n_episodes, track_convergence=False)
        run_rates.append(results.run_frequency)
        queue_sizes.append(results.avg_queue_size)
        consumptions.append(results.avg_consumption)
    
    alpha = 1 - confidence
    
    def ci(data):
        lower = np.percentile(data, 100 * alpha / 2)
        upper = np.percentile(data, 100 * (1 - alpha / 2))
        return (lower, upper)
    
    return {
        'run_rate': ci(run_rates),
        'queue_size': ci(queue_sizes),
        'consumption': ci(consumptions)
    }


def sensitivity_analysis(base_params: Dict[str, Any],
                          param_name: str,
                          param_values: List[Any],
                          n_episodes: int = 500,
                          n_runs: int = 5) -> Dict[str, List]:
    """
    Perform sensitivity analysis on a single parameter.
    
    Args:
        base_params: Base parameter dictionary
        param_name: Name of parameter to vary
        param_values: List of values to test
        n_episodes: Episodes per run
        n_runs: Runs for averaging
        
    Returns:
        Dictionary with sensitivity results
    """
    from .bank_run_model import BankRunSimulation, ModelParameters
    
    results = {
        'param_values': param_values,
        'run_rate_mean': [],
        'run_rate_std': [],
        'queue_size_mean': [],
        'queue_size_std': []
    }
    
    for value in param_values:
        params = {**base_params, param_name: value}
        model_params = ModelParameters(**params)
        
        run_rates = []
        queue_sizes = []
        
        for _ in range(n_runs):
            sim = BankRunSimulation(model_params)
            res = sim.run(n_episodes, track_convergence=False)
            run_rates.append(res.run_frequency)
            queue_sizes.append(res.avg_queue_size)
        
        results['run_rate_mean'].append(np.mean(run_rates))
        results['run_rate_std'].append(np.std(run_rates))
        results['queue_size_mean'].append(np.mean(queue_sizes))
        results['queue_size_std'].append(np.std(queue_sizes))
    
    return results
