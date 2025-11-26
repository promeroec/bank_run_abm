"""
Bank Run Agent-Based Model
==========================

A Python implementation of the agent-based computational model
for studying bank runs, based on Romero & Latek (2008).

This package extends the Diamond-Dybvig (1983) framework with:
- Heterogeneous agents with varying discount rates
- SARSA(λ) reinforcement learning for policy optimization
- Observable queue dynamics
- Multi-period investment and consumption decisions

Quick Start:
    from bank_run_abm import BankRunSimulation, ModelParameters
    
    sim = BankRunSimulation(n_agents=20, run_threshold=10)
    results = sim.run(n_episodes=1000)
    print(f"Run frequency: {results.run_frequency:.2%}")

For experiments replicating paper results:
    from bank_run_abm import run_threshold_sweep, run_population_sweep
    
    # Figure 3a: Threshold sweep
    threshold_results = run_threshold_sweep(n_agents=20)
    
    # Figure 3b: Population sweep  
    population_results = run_population_sweep()
"""

__version__ = "1.0.0"
__author__ = "Pedro P. Romero"

from .bank_run_model import (
    # Core classes
    ModelParameters,
    SARSAAgent,
    Bank,
    BankRunSimulation,
    SimulationResults,
    
    # Data classes and enums
    Action,
    AgentState,
    
    # Experiment runners
    run_single_agent_experiment,
    run_threshold_sweep,
    run_population_sweep,
    
    # Visualization helpers
    create_learning_plot_data,
    create_investment_plot_data,
    create_policy_heatmap_data,
)

from .visualization import (
    plot_learning_dynamics,
    plot_investment_trajectory,
    plot_policy_heatmap,
    plot_threshold_sweep_results,
    plot_population_sweep_results,
    plot_run_frequency_comparison,
)

from .analysis import (
    compute_coordination_index,
    compute_herding_index,
    analyze_convergence,
    statistical_summary,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core classes
    "ModelParameters",
    "SARSAAgent", 
    "Bank",
    "BankRunSimulation",
    "SimulationResults",
    
    # Data classes
    "Action",
    "AgentState",
    
    # Experiment runners
    "run_single_agent_experiment",
    "run_threshold_sweep",
    "run_population_sweep",
    
    # Visualization
    "plot_learning_dynamics",
    "plot_investment_trajectory",
    "plot_policy_heatmap",
    "plot_threshold_sweep_results",
    "plot_population_sweep_results",
    "plot_run_frequency_comparison",
    
    # Analysis
    "compute_coordination_index",
    "compute_herding_index",
    "analyze_convergence",
    "statistical_summary",
]
