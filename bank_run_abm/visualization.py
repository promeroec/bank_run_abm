"""
Visualization Module for Bank Run ABM
======================================

Provides plotting functions to replicate figures from the paper:
- Figure 2: Learning dynamics, investment trajectories, policy heatmaps
- Figure 3: Threshold and population sweep results

Requires matplotlib and optionally seaborn for enhanced styling.
"""

import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING

# Lazy imports to avoid issues if matplotlib not installed
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.figure


def _get_pyplot():
    """Lazy import of matplotlib.pyplot."""
    import matplotlib.pyplot as plt
    return plt


def _setup_style():
    """Setup consistent plot styling."""
    plt = _get_pyplot()
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass  # Use default style


def plot_learning_dynamics(results: 'SimulationResults',
                           figsize: tuple = (12, 8),
                           save_path: Optional[str] = None) -> 'matplotlib.figure.Figure':
    """
    Plot learning dynamics (Figure 2a from paper).
    
    Shows convergence of Q-matrix and policy changes over episodes.
    Uses logarithmic scales as in the original paper.
    
    Args:
        results: SimulationResults object with q_convergence and policy_changes
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    plt = _get_pyplot()
    _setup_style()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    episodes = np.arange(1, len(results.q_convergence) + 1)
    
    # Smooth the data for visualization
    window = max(1, len(episodes) // 100)
    
    def smooth(data, w):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    # Top plot: Q-value changes
    ax1 = axes[0]
    ax1.semilogy(episodes, results.q_convergence, 'b-', alpha=0.3, 
                 label='Original Δ(Q)')
    if len(episodes) > window:
        smoothed_q = smooth(results.q_convergence, window)
        ax1.semilogy(episodes[:len(smoothed_q)], smoothed_q, 'g-', 
                     linewidth=2, label='Smoothed Δ(Q)')
    ax1.set_ylabel('Total change in Q-values', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.set_title('Learning Dynamics', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Policy changes
    ax2 = axes[1]
    policy_changes = np.array(results.policy_changes) + 1  # Avoid log(0)
    ax2.semilogy(episodes, policy_changes, 'b-', alpha=0.3,
                 label=r'Original $\sum_{s} \pi_t(s) \neq \pi_{t-1}(s)$')
    if len(episodes) > window:
        smoothed_policy = smooth(results.policy_changes, window) + 1
        ax2.semilogy(episodes[:len(smoothed_policy)], smoothed_policy, 'g-',
                     linewidth=2, label='Smoothed')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Number of policy changes', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_investment_trajectory(cash_history: List[float],
                                deposit_history: List[float],
                                consumption_history: List[float],
                                figsize: tuple = (10, 6),
                                save_path: Optional[str] = None) -> 'matplotlib.figure.Figure':
    """
    Plot optimal investment trajectory (Figure 2b from paper).
    
    Shows cash, bank deposits, consumption, and deposit/withdrawal
    amounts over time for a single agent.
    
    Args:
        cash_history: List of cash amounts over time
        deposit_history: List of deposit amounts over time
        consumption_history: List of consumption amounts over time
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    plt = _get_pyplot()
    _setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ticks = np.arange(len(consumption_history))
    
    if cash_history:
        ax.plot(ticks[:len(cash_history)], cash_history, 'b-', 
                linewidth=2, label='Cash')
    if deposit_history:
        ax.plot(ticks[:len(deposit_history)], deposit_history, 'r-', 
                linewidth=2, label='Bank account')
    ax.plot(ticks, consumption_history, 'g-', 
            linewidth=2, label='Consumption')
    
    # Compute deposit/withdrawal amounts
    if deposit_history and len(deposit_history) > 1:
        transactions = np.diff(deposit_history)
        ax.bar(ticks[1:len(transactions)+1], transactions, alpha=0.3,
               color='purple', label='Deposit/Withdrawal')
    
    ax.set_xlabel('Episode tick', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Optimal Investment Scheme', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_policy_heatmap(heatmap_data: Dict,
                        figsize: tuple = (8, 6),
                        save_path: Optional[str] = None) -> 'matplotlib.figure.Figure':
    """
    Plot policy as heatmap (Figure 2c from paper).
    
    Shows probability of withdrawing as a function of agent's assets.
    Blue = less likely to go to bank, Red = more likely.
    
    Args:
        heatmap_data: Dictionary with 'heatmap', 'cash_bins', 'deposit_bins'
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    plt = _get_pyplot()
    _setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    heatmap = heatmap_data['heatmap']
    
    # Create heatmap with custom colormap (blue to red)
    im = ax.imshow(heatmap, cmap='RdBu_r', aspect='auto',
                   origin='lower', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability of withdrawal', fontsize=11)
    
    ax.set_xlabel('Log of cash', fontsize=12)
    ax.set_ylabel('Log of account', fontsize=12)
    ax.set_title('Mind of an Agent: Probability of Going to Bank', fontsize=14)
    
    # Set tick labels
    n_cash = len(heatmap_data.get('cash_bins', range(heatmap.shape[1])))
    n_deposit = len(heatmap_data.get('deposit_bins', range(heatmap.shape[0])))
    
    ax.set_xticks(np.linspace(0, n_cash-1, 5))
    ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(0, 5, 5)])
    ax.set_yticks(np.linspace(0, n_deposit-1, 5))
    ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(-1, 5, 5)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_threshold_sweep_results(results: Dict[int, Dict],
                                  n_agents: int = 20,
                                  figsize: tuple = (10, 10),
                                  save_path: Optional[str] = None) -> 'matplotlib.figure.Figure':
    """
    Plot threshold sweep results (Figure 3a from paper).
    
    Shows probability of going to bank and probability of bank run
    as functions of run threshold η.
    
    Args:
        results: Dictionary from run_threshold_sweep()
        n_agents: Population size
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    plt = _get_pyplot()
    _setup_style()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    thresholds = sorted(results.keys())
    
    # Top plot: Probability of going to bank
    ax1 = axes[0]
    means = [results[t]['bank_visit_mean'] for t in thresholds]
    stds = [results[t]['bank_visit_std'] for t in thresholds]
    
    ax1.errorbar(thresholds, means, yerr=stds, fmt='o-', capsize=5,
                 color='blue', label='Observed')
    ax1.set_ylabel('Probability of going to bank', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Influence of Run Threshold η (N = {n_agents})', fontsize=14)
    
    # Bottom plot: Probability of run
    ax2 = axes[1]
    run_means = [results[t]['run_frequency_mean'] for t in thresholds]
    run_stds = [results[t]['run_frequency_std'] for t in thresholds]
    predicted = [results[t]['predicted_run_freq'] for t in thresholds]
    
    ax2.errorbar(thresholds, run_means, yerr=run_stds, fmt='o-', capsize=5,
                 color='blue', label='Observed')
    ax2.plot(thresholds, predicted, 's--', color='green', 
             markersize=8, label='Predicted')
    ax2.set_xlabel('Run threshold', fontsize=12)
    ax2.set_ylabel('Probability of having a run', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_population_sweep_results(results: Dict[int, Dict],
                                   figsize: tuple = (10, 10),
                                   save_path: Optional[str] = None) -> 'matplotlib.figure.Figure':
    """
    Plot population sweep results (Figure 3b from paper).
    
    Shows probability of going to bank and probability of bank run
    as functions of population size N (with η = N/2).
    
    Args:
        results: Dictionary from run_population_sweep()
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    plt = _get_pyplot()
    _setup_style()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    pop_sizes = sorted(results.keys())
    
    # Top plot: Probability of going to bank
    ax1 = axes[0]
    means = [results[n]['bank_visit_mean'] for n in pop_sizes]
    stds = [results[n]['bank_visit_std'] for n in pop_sizes]
    
    ax1.errorbar(range(len(pop_sizes)), means, yerr=stds, fmt='o-', 
                 capsize=5, color='blue', label='Observed')
    ax1.set_xticks(range(len(pop_sizes)))
    ax1.set_xticklabels(pop_sizes)
    ax1.set_ylabel('Probability of going to bank', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(r'Influence of Population Size (η = 0.5N)', fontsize=14)
    
    # Bottom plot: Probability of run
    ax2 = axes[1]
    run_means = [results[n]['run_frequency_mean'] for n in pop_sizes]
    run_stds = [results[n]['run_frequency_std'] for n in pop_sizes]
    predicted = [results[n]['predicted_run_freq'] for n in pop_sizes]
    
    ax2.errorbar(range(len(pop_sizes)), run_means, yerr=run_stds, 
                 fmt='o-', capsize=5, color='blue', label='Observed')
    ax2.plot(range(len(pop_sizes)), predicted, 's--', color='green',
             markersize=8, label='Predicted')
    ax2.set_xticks(range(len(pop_sizes)))
    ax2.set_xticklabels(pop_sizes)
    ax2.set_xlabel('Population size', fontsize=12)
    ax2.set_ylabel('Probability of having a run', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_run_frequency_comparison(observed: List[float],
                                   predicted: List[float],
                                   labels: List[str],
                                   title: str = "Run Frequency: Observed vs Predicted",
                                   figsize: tuple = (10, 6),
                                   save_path: Optional[str] = None) -> 'matplotlib.figure.Figure':
    """
    Create bar chart comparing observed and predicted run frequencies.
    
    Args:
        observed: List of observed run frequencies
        predicted: List of predicted run frequencies
        labels: Labels for x-axis
        title: Plot title
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    plt = _get_pyplot()
    _setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, observed, width, label='Observed', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, predicted, width, label='Predicted',
                   color='forestgreen', alpha=0.8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Run Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_episode_dynamics(results: 'SimulationResults',
                          window: int = 100,
                          figsize: tuple = (12, 8),
                          save_path: Optional[str] = None) -> 'matplotlib.figure.Figure':
    """
    Plot dynamics over episodes: runs, queue sizes, consumption.
    
    Args:
        results: SimulationResults object
        window: Window size for moving average
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    plt = _get_pyplot()
    _setup_style()
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    episodes = np.arange(len(results.episode_runs))
    
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    # Run occurrence (moving average)
    ax1 = axes[0]
    runs = np.array(results.episode_runs, dtype=float)
    if len(runs) > window:
        ma_runs = moving_avg(runs, window)
        ax1.plot(episodes[:len(ma_runs)], ma_runs, 'r-', linewidth=2)
    ax1.set_ylabel('Run Rate\n(Moving Avg)', fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Episode Dynamics', fontsize=14)
    
    # Queue sizes
    ax2 = axes[1]
    ax2.plot(episodes, results.episode_queue_sizes, 'b-', alpha=0.3)
    if len(results.episode_queue_sizes) > window:
        ma_queue = moving_avg(results.episode_queue_sizes, window)
        ax2.plot(episodes[:len(ma_queue)], ma_queue, 'b-', linewidth=2)
    ax2.set_ylabel('Avg Queue Size', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Consumption
    ax3 = axes[2]
    ax3.plot(episodes, results.episode_consumptions, 'g-', alpha=0.3)
    if len(results.episode_consumptions) > window:
        ma_cons = moving_avg(results.episode_consumptions, window)
        ax3.plot(episodes[:len(ma_cons)], ma_cons, 'g-', linewidth=2)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Avg Consumption', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig
