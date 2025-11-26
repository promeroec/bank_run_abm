"""
Unit tests for bank_run_abm package.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bank_run_model import (
    ModelParameters,
    SARSAAgent,
    Bank,
    BankRunSimulation,
    Action,
    AgentState,
    run_single_agent_experiment,
    run_threshold_sweep,
    run_population_sweep,
)


class TestModelParameters:
    """Tests for ModelParameters dataclass."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = ModelParameters()
        assert params.n_agents == 20
        assert params.upper_deposit == 200.0
        assert params.run_threshold == 10  # n_agents // 2
        assert params.interest_rate == 0.1
        assert params.discount_rate == 0.95
        
    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = ModelParameters(n_agents=50, run_threshold=25)
        assert params.n_agents == 50
        assert params.run_threshold == 25
        
    def test_auto_threshold(self):
        """Test automatic threshold calculation."""
        params = ModelParameters(n_agents=30)
        assert params.run_threshold == 15  # 30 // 2


class TestAgentState:
    """Tests for AgentState class."""
    
    def test_state_creation(self):
        """Test state creation."""
        state = AgentState(cash=10.0, deposits=50.0, queue_size=5, run_threshold=10)
        assert state.cash == 10.0
        assert state.deposits == 50.0
        assert state.queue_size == 5
        
    def test_state_discretization(self):
        """Test state discretization for Q-table."""
        state = AgentState(cash=25.0, deposits=100.0, queue_size=5, run_threshold=10)
        discrete = state.discretize(n_cash_bins=10, n_deposit_bins=10)
        
        assert isinstance(discrete, tuple)
        assert len(discrete) == 3
        assert all(isinstance(x, int) for x in discrete)


class TestSARSAAgent:
    """Tests for SARSA learning agent."""
    
    def test_agent_creation(self):
        """Test agent initialization."""
        params = ModelParameters()
        agent = SARSAAgent(agent_id=0, params=params)
        
        assert agent.agent_id == 0
        assert agent.cash == params.initial_endowment
        assert agent.deposits == 0.0
        assert 0.5 <= agent.discount_rate <= 0.99
        
    def test_agent_reset(self):
        """Test agent state reset."""
        params = ModelParameters()
        agent = SARSAAgent(agent_id=0, params=params)
        
        # Modify state
        agent.cash = 100.0
        agent.deposits = 50.0
        
        # Reset
        agent.reset_episode()
        
        assert agent.cash == params.initial_endowment
        assert agent.deposits == 0.0
        
    def test_action_selection(self):
        """Test action selection."""
        params = ModelParameters()
        agent = SARSAAgent(agent_id=0, params=params)
        state = agent.get_state(queue_size=5)
        
        action = agent.select_action(state)
        assert isinstance(action, Action)
        
    def test_utility_function(self):
        """Test utility function."""
        params = ModelParameters()
        agent = SARSAAgent(agent_id=0, params=params)
        
        # Positive consumption should give positive utility
        assert agent.utility(10.0) > 0
        
        # Higher consumption should give higher utility
        assert agent.utility(20.0) > agent.utility(10.0)
        
        # Zero/negative consumption should be penalized
        assert agent.utility(0.0) < 0


class TestBank:
    """Tests for Bank class."""
    
    def test_bank_creation(self):
        """Test bank initialization."""
        params = ModelParameters()
        bank = Bank(params)
        
        assert bank.total_deposits == 0.0
        assert len(bank.queue) == 0
        
    def test_queue_operations(self):
        """Test queue management."""
        params = ModelParameters(n_agents=5, run_threshold=3)
        bank = Bank(params)
        agent = SARSAAgent(agent_id=0, params=params)
        
        # Add to queue
        bank.add_to_queue(agent, Action.GO_TO_BANK_DEPOSIT, 5.0)
        assert bank.get_queue_size() == 1
        
        # Not a run yet
        assert not bank.is_run()
        
    def test_run_detection(self):
        """Test bank run detection."""
        params = ModelParameters(n_agents=5, run_threshold=3)
        bank = Bank(params)
        
        # Add enough agents to trigger run
        for i in range(3):
            agent = SARSAAgent(agent_id=i, params=params)
            bank.add_to_queue(agent, Action.GO_TO_BANK_WITHDRAW, 5.0)
            
        assert bank.is_run()


class TestBankRunSimulation:
    """Tests for main simulation class."""
    
    def test_simulation_creation(self):
        """Test simulation initialization."""
        sim = BankRunSimulation(n_agents=10, run_threshold=5)
        
        assert len(sim.agents) == 10
        assert sim.params.run_threshold == 5
        
    def test_single_episode(self):
        """Test running a single episode."""
        sim = BankRunSimulation(n_agents=10, run_threshold=5, episode_length=20)
        
        stats = sim.run_episode()
        
        assert 'run_occurred' in stats
        assert 'queue_sizes' in stats
        assert isinstance(stats['run_occurred'], bool)
        
    def test_multi_episode_run(self):
        """Test running multiple episodes."""
        sim = BankRunSimulation(n_agents=10, run_threshold=5, episode_length=20)
        
        results = sim.run(n_episodes=50, track_convergence=True)
        
        assert results.n_episodes == 50
        assert 0 <= results.run_frequency <= 1
        assert len(results.episode_runs) == 50
        assert len(results.q_convergence) == 50
        
    def test_results_structure(self):
        """Test results structure."""
        sim = BankRunSimulation(n_agents=10, run_threshold=5)
        results = sim.run(n_episodes=20)
        
        # Check all expected fields
        assert hasattr(results, 'n_runs')
        assert hasattr(results, 'n_episodes')
        assert hasattr(results, 'run_frequency')
        assert hasattr(results, 'avg_queue_size')
        assert hasattr(results, 'avg_consumption')


class TestExperimentRunners:
    """Tests for experiment runner functions."""
    
    def test_single_agent_experiment(self):
        """Test single-agent experiment."""
        results = run_single_agent_experiment(n_episodes=100)
        
        assert results.n_episodes == 100
        # Single agent should have very few runs
        assert results.run_frequency < 0.5
        
    def test_threshold_sweep(self):
        """Test threshold sweep experiment."""
        results = run_threshold_sweep(
            n_agents=10,
            thresholds=[3, 5, 7],
            n_episodes=50,
            n_runs=2
        )
        
        assert len(results) == 3
        assert all(t in results for t in [3, 5, 7])
        assert all('run_frequency_mean' in r for r in results.values())
        
    def test_population_sweep(self):
        """Test population sweep experiment."""
        results = run_population_sweep(
            population_sizes=[8, 16],
            n_episodes=50,
            n_runs=2
        )
        
        assert len(results) == 2
        assert 8 in results
        assert 16 in results


class TestIntegration:
    """Integration tests."""
    
    def test_full_simulation_workflow(self):
        """Test complete simulation workflow."""
        # Create parameters
        params = ModelParameters(
            n_agents=15,
            run_threshold=8,
            episode_length=30,
            interest_rate=0.1
        )
        
        # Create and run simulation
        sim = BankRunSimulation(params)
        results = sim.run(n_episodes=100, track_convergence=True)
        
        # Verify results
        assert results.n_episodes == 100
        assert 0 <= results.run_frequency <= 1
        assert results.avg_queue_size >= 0
        
        # Check convergence tracking
        assert len(results.q_convergence) == 100
        assert len(results.policy_changes) == 100
        
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        np.random.seed(42)
        sim1 = BankRunSimulation(n_agents=10, run_threshold=5)
        results1 = sim1.run(n_episodes=50)
        
        np.random.seed(42)
        sim2 = BankRunSimulation(n_agents=10, run_threshold=5)
        results2 = sim2.run(n_episodes=50)
        
        # Results should be identical with same seed
        assert results1.n_runs == results2.n_runs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
