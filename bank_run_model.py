"""
Bank Runs in an Agent-based Computational Model
================================================

Implementation based on Romero & Latek (2008)
"Bank Runs in an Agent-based Computational Model"

This module implements an agent-based computational model that extends 
the Diamond-Dybvig (1983) framework with:
- Heterogeneous discount rates and utility functions
- Multi-period investment and consumption decisions
- Observable queue size at decision time
- SARSA(λ) reinforcement learning for policy optimization

Author: Pedro P. Romero
Python Implementation for GitHub/Colab

Usage:
    from bank_run_model import BankRunSimulation, run_experiment
    
    sim = BankRunSimulation(n_agents=20, run_threshold=10)
    results = sim.run(n_episodes=1000)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import defaultdict


# ==============================================================================
# CONFIGURATION AND DATA CLASSES
# ==============================================================================

@dataclass
class ModelParameters:
    """
    Model parameters as specified in Table 1 of the paper.
    
    Attributes:
        n_agents: Population size N (default: 20, range: 10-1000)
        upper_deposit: Upper deposit amount T (default: 200)
        run_threshold: Run threshold η (default: N/2, range: 2 to N-2)
        interest_rate: Return on investment R (default: 0.1)
        discount_rate: Base discount rate γ (default: 0.95)
        episode_length: Episode length H (default: 100)
        initial_endowment: Initial endowment w₀ (default: 10)
        learning_rate: Learning rate α for SARSA (default: 0.2)
        epsilon: Exploration rate ε for ε-greedy (default: 0.1)
        lambda_trace: Eligibility trace decay λ (default: 0.9)
        discount_rate_std: Std dev for heterogeneous discount rates (default: 0.02)
    """
    n_agents: int = 20
    upper_deposit: float = 200.0
    run_threshold: Optional[int] = None  # If None, defaults to n_agents // 2
    interest_rate: float = 0.1
    discount_rate: float = 0.95
    episode_length: int = 100
    initial_endowment: float = 10.0
    learning_rate: float = 0.2
    epsilon: float = 0.1
    lambda_trace: float = 0.9
    discount_rate_std: float = 0.02
    
    def __post_init__(self):
        if self.run_threshold is None:
            self.run_threshold = self.n_agents // 2


class Action(Enum):
    """Possible actions for agents."""
    STAY_HOME = 0
    GO_TO_BANK_WITHDRAW = 1
    GO_TO_BANK_DEPOSIT = 2


@dataclass
class AgentState:
    """
    State representation for an agent.
    
    Attributes:
        cash: Current cash holdings
        deposits: Current bank deposits
        queue_size: Observed queue size when making decision
        run_threshold: Known bank capacity η
    """
    cash: float
    deposits: float
    queue_size: int
    run_threshold: int
    
    def discretize(self, n_cash_bins: int = 10, n_deposit_bins: int = 10, 
                   max_cash: float = 50.0, max_deposit: float = 200.0) -> Tuple:
        """
        Discretize continuous state for Q-table lookup.
        
        Returns:
            Tuple representing discretized state
        """
        cash_bin = min(int(self.cash / max_cash * n_cash_bins), n_cash_bins - 1)
        deposit_bin = min(int(self.deposits / max_deposit * n_deposit_bins), n_deposit_bins - 1)
        queue_ratio = min(self.queue_size / max(self.run_threshold, 1), 1.0)
        queue_bin = int(queue_ratio * 5)  # 5 bins for queue ratio
        
        return (max(0, cash_bin), max(0, deposit_bin), queue_bin)


@dataclass
class SimulationResults:
    """Container for simulation results."""
    n_runs: int = 0
    n_episodes: int = 0
    run_frequency: float = 0.0
    avg_queue_size: float = 0.0
    avg_consumption: float = 0.0
    avg_deposits: float = 0.0
    avg_cash: float = 0.0
    bank_visit_frequency: float = 0.0
    episode_runs: List[bool] = field(default_factory=list)
    episode_queue_sizes: List[float] = field(default_factory=list)
    episode_consumptions: List[float] = field(default_factory=list)
    q_convergence: List[float] = field(default_factory=list)
    policy_changes: List[int] = field(default_factory=list)


# ==============================================================================
# SARSA(λ) LEARNING AGENT
# ==============================================================================

class SARSAAgent:
    """
    Agent using SARSA(λ) reinforcement learning.
    
    Implements the learning algorithm described in Section 3.2 of the paper.
    Agents maximize expected discounted stream of consumption utility.
    """
    
    def __init__(self, agent_id: int, params: ModelParameters, 
                 n_cash_bins: int = 10, n_deposit_bins: int = 10):
        """
        Initialize a SARSA agent.
        
        Args:
            agent_id: Unique identifier for the agent
            params: Model parameters
            n_cash_bins: Number of bins for cash discretization
            n_deposit_bins: Number of bins for deposit discretization
        """
        self.agent_id = agent_id
        self.params = params
        self.n_cash_bins = n_cash_bins
        self.n_deposit_bins = n_deposit_bins
        
        # Heterogeneous discount rate
        self.discount_rate = np.clip(
            np.random.normal(params.discount_rate, params.discount_rate_std),
            0.5, 0.99
        )
        
        # Current assets
        self.cash = params.initial_endowment
        self.deposits = 0.0
        
        # Q-table: maps (state, action) -> value
        self.q_table: Dict[Tuple, Dict[Action, float]] = defaultdict(
            lambda: {a: 0.0 for a in Action}
        )
        
        # Eligibility traces for SARSA(λ)
        self.eligibility: Dict[Tuple, Dict[Action, float]] = defaultdict(
            lambda: {a: 0.0 for a in Action}
        )
        
        # Current state and action for learning
        self.current_state: Optional[AgentState] = None
        self.current_action: Optional[Action] = None
        self.previous_state: Optional[AgentState] = None
        self.previous_action: Optional[Action] = None
        
        # Decision history for analysis
        self.went_to_bank = False
        self.consumption_history: List[float] = []
        self.in_queue = False
        
    def reset_episode(self):
        """Reset agent state for a new episode."""
        self.cash = self.params.initial_endowment
        self.deposits = 0.0
        self.current_state = None
        self.current_action = None
        self.previous_state = None
        self.previous_action = None
        self.went_to_bank = False
        self.in_queue = False
        self.consumption_history = []
        # Reset eligibility traces
        self.eligibility = defaultdict(lambda: {a: 0.0 for a in Action})
        
    def get_state(self, queue_size: int) -> AgentState:
        """
        Create current state representation.
        
        Args:
            queue_size: Current observed queue size
            
        Returns:
            AgentState object
        """
        return AgentState(
            cash=self.cash,
            deposits=self.deposits,
            queue_size=queue_size,
            run_threshold=self.params.run_threshold
        )
    
    def select_action(self, state: AgentState) -> Action:
        """
        Select action using ε-greedy policy.
        
        As specified in the paper:
        a(s) = argmax_a Q(s,a) with probability 1-ε
               random action with probability ε
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state_key = state.discretize(self.n_cash_bins, self.n_deposit_bins)
        
        if np.random.random() < self.params.epsilon:
            # Exploration: random action
            valid_actions = self._get_valid_actions(state)
            return np.random.choice(valid_actions)
        else:
            # Exploitation: greedy action
            q_values = self.q_table[state_key]
            valid_actions = self._get_valid_actions(state)
            
            # Filter Q-values to valid actions only
            valid_q = {a: q_values[a] for a in valid_actions}
            return max(valid_q, key=valid_q.get)
    
    def _get_valid_actions(self, state: AgentState) -> List[Action]:
        """
        Get list of valid actions given current state.
        
        Args:
            state: Current state
            
        Returns:
            List of valid actions
        """
        valid = [Action.STAY_HOME]
        
        if state.deposits > 0:
            valid.append(Action.GO_TO_BANK_WITHDRAW)
        
        if state.cash > 0:
            valid.append(Action.GO_TO_BANK_DEPOSIT)
            
        return valid
    
    def utility(self, consumption: float) -> float:
        """
        Compute utility from consumption.
        
        Uses log utility for risk aversion.
        
        Args:
            consumption: Amount consumed
            
        Returns:
            Utility value
        """
        if consumption <= 0:
            return -10.0  # Penalty for zero/negative consumption
        return np.log(consumption + 1)
    
    def decide(self, queue_size: int) -> Tuple[Action, float]:
        """
        Make investment/consumption decision.
        
        This is Step 1 of the SARSA(λ) algorithm in the paper.
        
        Args:
            queue_size: Current observed queue size
            
        Returns:
            Tuple of (action, transaction_amount)
        """
        self.current_state = self.get_state(queue_size)
        self.current_action = self.select_action(self.current_state)
        
        # Determine transaction amount
        if self.current_action == Action.GO_TO_BANK_DEPOSIT:
            # Deposit some fraction of cash
            amount = self.cash * np.random.uniform(0.3, 0.7)
            self.went_to_bank = True
            self.in_queue = True
        elif self.current_action == Action.GO_TO_BANK_WITHDRAW:
            # Withdraw some fraction of deposits
            amount = self.deposits * np.random.uniform(0.3, 0.7)
            self.went_to_bank = True
            self.in_queue = True
        else:
            amount = 0.0
            self.went_to_bank = False
            self.in_queue = False
            
        return self.current_action, amount
    
    def consume(self) -> float:
        """
        Determine consumption amount.
        
        Returns:
            Amount consumed
        """
        # Consume a fraction of available cash
        if self.cash > 0:
            consumption = self.cash * np.random.uniform(0.1, 0.3)
            self.cash -= consumption
        else:
            consumption = 0.0
            
        self.consumption_history.append(consumption)
        return consumption
    
    def receive_reward(self, reward: float, next_queue_size: int):
        """
        Process reward and update Q-values using SARSA(λ).
        
        Implements Steps 2-5 of the algorithm in the paper.
        
        Args:
            reward: Reward received (utility from consumption)
            next_queue_size: Queue size for next state
        """
        if self.previous_state is None or self.previous_action is None:
            self.previous_state = self.current_state
            self.previous_action = self.current_action
            return
            
        # Step 2: Observe reward and next state
        next_state = self.get_state(next_queue_size)
        
        # Step 3: Choose next action
        next_action = self.select_action(next_state)
        
        # Get discretized states
        prev_state_key = self.previous_state.discretize(self.n_cash_bins, self.n_deposit_bins)
        next_state_key = next_state.discretize(self.n_cash_bins, self.n_deposit_bins)
        
        # Step 4: Compute TD error
        # δ = r + γQ(s', a') - Q(s, a)
        delta = (reward + 
                self.discount_rate * self.q_table[next_state_key][next_action] -
                self.q_table[prev_state_key][self.previous_action])
        
        # Step 5: Update eligibility and Q-values
        # e(s, a) += 1 for current state-action
        self.eligibility[prev_state_key][self.previous_action] += 1.0
        
        # Update all state-action pairs
        for state_key in list(self.q_table.keys()):
            for action in Action:
                if self.eligibility[state_key][action] > 0.001:
                    # Q(s,a) += αδe(s,a)
                    self.q_table[state_key][action] += (
                        self.params.learning_rate * delta * 
                        self.eligibility[state_key][action]
                    )
                    # e(s,a) *= γλ
                    self.eligibility[state_key][action] *= (
                        self.discount_rate * self.params.lambda_trace
                    )
        
        # Update for next iteration
        self.previous_state = self.current_state
        self.previous_action = self.current_action
        self.current_state = next_state
        self.current_action = next_action
    
    def get_q_table_norm(self) -> float:
        """
        Get L2 norm of Q-table for convergence tracking.
        
        Returns:
            L2 norm of Q-values
        """
        total = 0.0
        for state_key, actions in self.q_table.items():
            for action, value in actions.items():
                total += value ** 2
        return np.sqrt(total)
    
    def get_policy_hash(self) -> int:
        """
        Get hash of current policy for change detection.
        
        Returns:
            Hash value representing current policy
        """
        policy_tuple = tuple(
            (state_key, max(actions, key=actions.get))
            for state_key, actions in sorted(self.q_table.items())
        )
        return hash(policy_tuple)


# ==============================================================================
# BANK
# ==============================================================================

class Bank:
    """
    Bank entity that manages deposits, withdrawals, and run detection.
    
    Implements the banking mechanics from the Diamond-Dybvig model
    with extensions for sequential service constraint.
    """
    
    def __init__(self, params: ModelParameters):
        """
        Initialize bank.
        
        Args:
            params: Model parameters
        """
        self.params = params
        self.total_deposits = 0.0
        self.queue: List[Tuple[SARSAAgent, Action, float]] = []
        
    def reset(self):
        """Reset bank state for new episode."""
        self.total_deposits = 0.0
        self.queue = []
        
    def add_to_queue(self, agent: SARSAAgent, action: Action, amount: float):
        """
        Add agent to the queue.
        
        Args:
            agent: Agent joining queue
            action: Action (deposit or withdraw)
            amount: Transaction amount
        """
        self.queue.append((agent, action, amount))
        
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self.queue)
    
    def is_run(self) -> bool:
        """
        Check if a bank run occurs.
        
        A run happens if queue size >= run threshold η.
        
        Returns:
            True if run occurs
        """
        return len(self.queue) >= self.params.run_threshold
    
    def process_queue(self) -> Tuple[bool, List[SARSAAgent]]:
        """
        Process the queue and serve agents.
        
        If run occurs:
        - Reshuffle queue
        - Serve only η agents
        - Reject remaining orders
        
        Otherwise:
        - Serve all agents sequentially
        - Pay interest on deposits
        
        Returns:
            Tuple of (run_occurred, list of rejected agents)
        """
        run_occurred = self.is_run()
        rejected_agents = []
        
        if run_occurred:
            # Reshuffle queue randomly
            random.shuffle(self.queue)
            
            # Serve only η agents
            served = self.queue[:self.params.run_threshold]
            rejected = self.queue[self.params.run_threshold:]
            
            # Process served agents
            for agent, action, amount in served:
                self._process_transaction(agent, action, amount)
                
            # Handle rejected agents - they lose their deposits
            for agent, action, amount in rejected:
                rejected_agents.append(agent)
                # Penalty: reduce deposits
                agent.deposits *= 0.5  # Lose half of deposits in a run
        else:
            # Serve all agents and pay interest
            for agent, action, amount in self.queue:
                self._process_transaction(agent, action, amount)
                
            # Pay interest on all deposits
            self._pay_interest()
            
        # Clear queue
        self.queue = []
        
        return run_occurred, rejected_agents
    
    def _process_transaction(self, agent: SARSAAgent, action: Action, amount: float):
        """
        Process a single transaction.
        
        Args:
            agent: Agent making transaction
            action: Deposit or withdraw
            amount: Transaction amount
        """
        if action == Action.GO_TO_BANK_DEPOSIT:
            # Deposit: move cash to bank
            actual_amount = min(amount, agent.cash)
            actual_amount = min(actual_amount, 
                              self.params.upper_deposit - agent.deposits)
            agent.cash -= actual_amount
            agent.deposits += actual_amount
            self.total_deposits += actual_amount
            
        elif action == Action.GO_TO_BANK_WITHDRAW:
            # Withdraw: move deposits to cash
            actual_amount = min(amount, agent.deposits)
            actual_amount = min(actual_amount, self.total_deposits)
            agent.deposits -= actual_amount
            agent.cash += actual_amount
            self.total_deposits -= actual_amount
    
    def _pay_interest(self):
        """Pay interest on all deposits."""
        interest_factor = 1 + self.params.interest_rate
        self.total_deposits *= interest_factor
        

# ==============================================================================
# SIMULATION ENVIRONMENT
# ==============================================================================

class BankRunSimulation:
    """
    Main simulation environment coordinating agents and bank.
    
    Implements the simulation loop as described in Figure 1 of the paper.
    """
    
    def __init__(self, params: Optional[ModelParameters] = None, **kwargs):
        """
        Initialize simulation.
        
        Args:
            params: ModelParameters object (optional)
            **kwargs: Override individual parameters
        """
        if params is None:
            params = ModelParameters(**kwargs)
        else:
            # Allow overriding parameters
            for key, value in kwargs.items():
                if hasattr(params, key):
                    setattr(params, key, value)
                    
        self.params = params
        
        # Initialize agents
        self.agents = [
            SARSAAgent(i, params) 
            for i in range(params.n_agents)
        ]
        
        # Initialize bank
        self.bank = Bank(params)
        
        # Statistics tracking
        self.episode_count = 0
        self.run_count = 0
        self.results_history: List[Dict] = []
        
    def reset(self):
        """Reset simulation for new episode."""
        for agent in self.agents:
            agent.reset_episode()
        self.bank.reset()
        
    def run_episode(self) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Follows the flowchart in Figure 1:
        1. For each client: activate, observe queue, decide to go to bank
        2. Bank activates, checks for run
        3. If run: reshuffle, reject some orders
        4. Else: serve clients
        5. Calculate new assets
        6. Learn reward
        
        Returns:
            Dictionary with episode statistics
        """
        self.reset()
        
        episode_stats = {
            'run_occurred': False,
            'queue_sizes': [],
            'consumptions': [],
            'bank_visits': 0,
            'total_utility': 0.0
        }
        
        for t in range(self.params.episode_length):
            # Step 1: Agents sequentially decide whether to go to bank
            random.shuffle(self.agents)  # Random activation order
            
            for agent in self.agents:
                # Agent observes current queue size
                queue_size = self.bank.get_queue_size()
                
                # Agent makes decision
                action, amount = agent.decide(queue_size)
                
                # If going to bank, add to queue
                if action in [Action.GO_TO_BANK_DEPOSIT, Action.GO_TO_BANK_WITHDRAW]:
                    self.bank.add_to_queue(agent, action, amount)
                    episode_stats['bank_visits'] += 1
                    
            episode_stats['queue_sizes'].append(self.bank.get_queue_size())
            
            # Step 2-4: Bank activates and processes queue
            run_occurred, rejected = self.bank.process_queue()
            
            if run_occurred:
                episode_stats['run_occurred'] = True
                
            # Step 5-6: Agents consume and learn
            for agent in self.agents:
                # Consumption
                consumption = agent.consume()
                episode_stats['consumptions'].append(consumption)
                
                # Calculate reward (utility from consumption)
                reward = agent.utility(consumption)
                episode_stats['total_utility'] += reward
                
                # Apply penalty if rejected
                if agent in rejected:
                    reward -= 5.0  # Penalty for rejection
                    
                # Learn from experience
                agent.receive_reward(reward, self.bank.get_queue_size())
                
            # Pay interest on deposits for all agents
            for agent in self.agents:
                agent.deposits *= (1 + self.params.interest_rate)
                
        self.episode_count += 1
        if episode_stats['run_occurred']:
            self.run_count += 1
            
        return episode_stats
    
    def run(self, n_episodes: int = 1000, 
            track_convergence: bool = True,
            verbose: bool = False) -> SimulationResults:
        """
        Run multiple episodes and collect results.
        
        Args:
            n_episodes: Number of episodes to run
            track_convergence: Whether to track Q-table convergence
            verbose: Whether to print progress
            
        Returns:
            SimulationResults object
        """
        results = SimulationResults()
        results.n_episodes = n_episodes
        
        prev_q_norms = [agent.get_q_table_norm() for agent in self.agents]
        prev_policies = [agent.get_policy_hash() for agent in self.agents]
        
        for ep in range(n_episodes):
            ep_stats = self.run_episode()
            
            results.episode_runs.append(ep_stats['run_occurred'])
            results.episode_queue_sizes.append(
                np.mean(ep_stats['queue_sizes']) if ep_stats['queue_sizes'] else 0
            )
            results.episode_consumptions.append(
                np.mean(ep_stats['consumptions']) if ep_stats['consumptions'] else 0
            )
            
            # Track convergence
            if track_convergence:
                current_q_norms = [agent.get_q_table_norm() for agent in self.agents]
                q_change = sum(abs(c - p) for c, p in zip(current_q_norms, prev_q_norms))
                results.q_convergence.append(q_change)
                prev_q_norms = current_q_norms
                
                current_policies = [agent.get_policy_hash() for agent in self.agents]
                policy_changes = sum(1 for c, p in zip(current_policies, prev_policies) if c != p)
                results.policy_changes.append(policy_changes)
                prev_policies = current_policies
            
            if verbose and (ep + 1) % 100 == 0:
                run_rate = sum(results.episode_runs[-100:]) / 100
                print(f"Episode {ep + 1}/{n_episodes}, "
                      f"Run rate (last 100): {run_rate:.2%}")
        
        # Compute aggregate statistics
        results.n_runs = sum(results.episode_runs)
        results.run_frequency = results.n_runs / n_episodes
        results.avg_queue_size = np.mean(results.episode_queue_sizes)
        results.avg_consumption = np.mean(results.episode_consumptions)
        
        # Final state statistics
        results.avg_deposits = np.mean([a.deposits for a in self.agents])
        results.avg_cash = np.mean([a.cash for a in self.agents])
        
        total_visits = sum(1 for a in self.agents if a.went_to_bank)
        results.bank_visit_frequency = total_visits / len(self.agents)
        
        return results
    
    def get_agent_policies(self) -> List[Dict]:
        """
        Extract learned policies from all agents.
        
        Returns:
            List of policy dictionaries
        """
        policies = []
        for agent in self.agents:
            policy = {}
            for state_key, actions in agent.q_table.items():
                best_action = max(actions, key=actions.get)
                policy[state_key] = {
                    'best_action': best_action.name,
                    'q_values': {a.name: v for a, v in actions.items()}
                }
            policies.append({
                'agent_id': agent.agent_id,
                'discount_rate': agent.discount_rate,
                'policy': policy
            })
        return policies


# ==============================================================================
# EXPERIMENT RUNNERS
# ==============================================================================

def run_single_agent_experiment(params: Optional[ModelParameters] = None,
                                 n_episodes: int = 10000) -> SimulationResults:
    """
    Run single-agent dynamics experiment (Section 4.1 of paper).
    
    Validates the learning algorithm in a Markovian environment.
    
    Args:
        params: Model parameters
        n_episodes: Number of episodes
        
    Returns:
        SimulationResults
    """
    if params is None:
        params = ModelParameters(n_agents=1, run_threshold=2)
    else:
        params.n_agents = 1
        params.run_threshold = 2  # Effectively no runs with 1 agent
        
    sim = BankRunSimulation(params)
    return sim.run(n_episodes, track_convergence=True)


def run_threshold_sweep(n_agents: int = 20,
                        thresholds: Optional[List[int]] = None,
                        n_episodes: int = 1000,
                        n_runs: int = 10) -> Dict[int, Dict]:
    """
    Run experiments varying run threshold η (Figure 3a of paper).
    
    Args:
        n_agents: Population size
        thresholds: List of threshold values to test
        n_episodes: Episodes per run
        n_runs: Number of runs for averaging
        
    Returns:
        Dictionary mapping thresholds to results
    """
    if thresholds is None:
        thresholds = list(range(2, n_agents, 2))
        
    results = {}
    
    for eta in thresholds:
        run_results = []
        bank_visit_rates = []
        
        for _ in range(n_runs):
            params = ModelParameters(n_agents=n_agents, run_threshold=eta)
            sim = BankRunSimulation(params)
            res = sim.run(n_episodes, track_convergence=False)
            run_results.append(res.run_frequency)
            bank_visit_rates.append(res.bank_visit_frequency)
            
        results[eta] = {
            'run_frequency_mean': np.mean(run_results),
            'run_frequency_std': np.std(run_results),
            'bank_visit_mean': np.mean(bank_visit_rates),
            'bank_visit_std': np.std(bank_visit_rates),
            'predicted_run_freq': _compute_predicted_run_freq(
                n_agents, eta, np.mean(bank_visit_rates)
            )
        }
        
    return results


def run_population_sweep(population_sizes: Optional[List[int]] = None,
                          n_episodes: int = 1000,
                          n_runs: int = 10) -> Dict[int, Dict]:
    """
    Run experiments varying population size N (Figure 3b of paper).
    
    Uses η = N/2 as per paper.
    
    Args:
        population_sizes: List of population sizes to test
        n_episodes: Episodes per run
        n_runs: Number of runs for averaging
        
    Returns:
        Dictionary mapping population sizes to results
    """
    if population_sizes is None:
        population_sizes = [8, 16, 32, 64, 128, 256]
        
    results = {}
    
    for n in population_sizes:
        eta = n // 2
        run_results = []
        bank_visit_rates = []
        
        for _ in range(n_runs):
            params = ModelParameters(n_agents=n, run_threshold=eta)
            sim = BankRunSimulation(params)
            res = sim.run(n_episodes, track_convergence=False)
            run_results.append(res.run_frequency)
            bank_visit_rates.append(res.bank_visit_frequency)
            
        results[n] = {
            'run_frequency_mean': np.mean(run_results),
            'run_frequency_std': np.std(run_results),
            'bank_visit_mean': np.mean(bank_visit_rates),
            'bank_visit_std': np.std(bank_visit_rates),
            'predicted_run_freq': _compute_predicted_run_freq(
                n, eta, np.mean(bank_visit_rates)
            )
        }
        
    return results


def _compute_predicted_run_freq(n: int, threshold: int, theta: float) -> float:
    """
    Compute predicted run frequency under independence assumption.
    
    From Equation (3) in the paper:
    P(Run) = Σ_{i=T}^{N} C(N,i) θ^i (1-θ)^{N-i}
    
    Args:
        n: Population size
        threshold: Run threshold
        theta: Probability of going to bank
        
    Returns:
        Predicted probability of run
    """
    from scipy.stats import binom
    
    if theta <= 0 or theta >= 1:
        return 0.0
        
    # P(X >= threshold) where X ~ Binomial(n, theta)
    return 1 - binom.cdf(threshold - 1, n, theta)


# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def create_learning_plot_data(results: SimulationResults) -> Dict:
    """
    Prepare data for learning dynamics plot (Figure 2a).
    
    Args:
        results: Simulation results
        
    Returns:
        Dictionary with plot data
    """
    return {
        'episodes': list(range(len(results.q_convergence))),
        'q_changes': results.q_convergence,
        'policy_changes': results.policy_changes,
        'smoothed_q': _smooth(results.q_convergence, window=100),
        'smoothed_policy': _smooth(results.policy_changes, window=100)
    }


def create_investment_plot_data(agent: SARSAAgent) -> Dict:
    """
    Prepare data for investment trajectory plot (Figure 2b).
    
    Args:
        agent: Trained agent
        
    Returns:
        Dictionary with investment trajectory data
    """
    return {
        'cash_history': [],  # Would need to track during simulation
        'deposit_history': [],
        'consumption_history': agent.consumption_history
    }


def create_policy_heatmap_data(agent: SARSAAgent) -> Dict:
    """
    Prepare data for policy visualization (Figure 2c).
    
    Args:
        agent: Trained agent
        
    Returns:
        Dictionary with heatmap data
    """
    n_cash = agent.n_cash_bins
    n_deposit = agent.n_deposit_bins
    
    heatmap = np.zeros((n_deposit, n_cash))
    
    for cash_bin in range(n_cash):
        for deposit_bin in range(n_deposit):
            state_key = (cash_bin, deposit_bin, 2)  # Middle queue level
            if state_key in agent.q_table:
                actions = agent.q_table[state_key]
                # Probability of going to bank
                go_to_bank_q = max(
                    actions.get(Action.GO_TO_BANK_DEPOSIT, -np.inf),
                    actions.get(Action.GO_TO_BANK_WITHDRAW, -np.inf)
                )
                stay_q = actions.get(Action.STAY_HOME, 0)
                
                # Softmax probability
                if go_to_bank_q > -np.inf:
                    exp_go = np.exp(min(go_to_bank_q, 50))
                    exp_stay = np.exp(min(stay_q, 50))
                    prob_go = exp_go / (exp_go + exp_stay)
                else:
                    prob_go = 0.0
                    
                heatmap[deposit_bin, cash_bin] = prob_go
                
    return {
        'heatmap': heatmap,
        'cash_bins': list(range(n_cash)),
        'deposit_bins': list(range(n_deposit))
    }


def _smooth(data: List[float], window: int = 100) -> List[float]:
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return [
        np.mean(data[max(0, i-window):i+1]) 
        for i in range(len(data))
    ]


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("Bank Run Agent-Based Model")
    print("=" * 50)
    
    # Quick demonstration
    print("\nRunning demonstration simulation...")
    
    params = ModelParameters(
        n_agents=20,
        run_threshold=10,
        episode_length=50
    )
    
    sim = BankRunSimulation(params)
    results = sim.run(n_episodes=500, verbose=True)
    
    print("\n" + "=" * 50)
    print("Results Summary:")
    print(f"  Total episodes: {results.n_episodes}")
    print(f"  Bank runs: {results.n_runs}")
    print(f"  Run frequency: {results.run_frequency:.2%}")
    print(f"  Avg queue size: {results.avg_queue_size:.2f}")
    print(f"  Avg consumption: {results.avg_consumption:.4f}")
    print(f"  Bank visit frequency: {results.bank_visit_frequency:.2%}")
