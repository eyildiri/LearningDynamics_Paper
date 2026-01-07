# Learning Dynamics in Social Dilemmas: Reinforcement Learning Explains Conditional Cooperation

This repository contains an implementation and replication of the research paper **"Reinforcement Learning Explains Conditional Cooperation and Its Moody Cousin"** by Takahiro Ezaki, Yutaka Horita, Masanori Takezawa, and Naoki Masuda. The project explores how reinforcement learning, specifically the Bush-Mosteller model, can explain cooperative behaviors in social dilemmas such as the Public Goods Game (PGG) and the Prisoner's Dilemma Game (PDG).

## Project Structure

```
.
├── README.md                 # This file: Overview and project description
├── docs/
│   └── Paper.pdf             # The original research paper (PDF format)
├── src/
│   ├── agents.py             # Agent classes: AspirationBMAgent (for PDG) and AspirationBMPGGAgent (for PGG)
│   ├── BMmodel.py            # Core Bush-Mosteller model functions
│   ├── analysis_PDG.py       # Analysis utilities for PDG
│   ├── analysis_PGG.py       # Analysis utilities for PGG
│   ├── loop_pgg.py           # Simulation loop for PGG
│   ├── loop_pz.py            # Simulation loop for PDG
│   ├── pdg_env.py            # PDG environment: PettingZoo ParallelEnv 
│   └── pgg_env.py            # PGG environment: PettingZoo ParallelEnv 
└── results/
    ├── results_dynamic_aspiration_PGG.ipynb   # Jupyter notebook: Simulations and plots for dynamic aspiration in PGG 
    ├── results_dynamic_aspiration.ipynb       # Jupyter notebook: Simulations and plots for dynamic aspiration in PDG
    ├── results_PDG.ipynb                      # Jupyter notebook: PDG-specific results
    └── results_PGG.ipynb                      # Jupyter notebook: PGG-specific results
```

### Key Components
- **docs/**: Contains the full PDF of the original paper for reference.
- **src/**: Core implementation code.
  - Environments (`pgg_env.py`, `pdg_env.py`): Define the game rules, agent interactions, and reward calculations.
  - Agents (`agents.py`): Implement the learning agents with Bush-Mosteller updates, aspiration levels, and noise.
  - Model (`BMmodel.py`): Mathematical functions for reinforcement learning updates.
  - Analysis (`analysis_*.py`): Functions to compute metrics like cooperation fractions, conditional probabilities, and fits.
  - Loops (`loop_*.py`): Scripts to run simulations and collect data over multiple episodes/trials.
- **results/**: Jupyter notebooks for running simulations, analyzing data, and generating plots that replicate the paper's figures.