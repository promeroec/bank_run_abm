# bank_run_abm
ABM of Diamond and Dybvig with SARSA.
## Overview

This program extends the classic **Diamond-Dybvig (1983)** bank run model using agent-based computational methods. We relax assumptions about depositors by assuming they:

- Have **heterogeneous discount rates** (vs. two representative types)
- Integrate **private information** (financial standing, cash requirements)
- Observe **bank system information** (queue size, recent withdrawals)
- Use **reinforcement learning** (SARSA(λ)) to optimize behavior

## Key Features

- **SARSA(λ) Learning**: Agents use temporal-difference reinforcement learning to develop optimal deposit/withdrawal policies
- **Heterogeneous Agents**: Each agent has individual discount rates and learning parameters
- **Observable Queue Dynamics**: Agents can observe queue size before deciding to go to the bank
- **Coordination vs. Herding Analysis**: Tools to measure emergent collective behavior
- **Comprehensive Visualization**: Reproduce all figures from the original paper
