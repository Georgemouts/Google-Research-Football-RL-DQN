# Google-Research-Football-RL :soccer:
Creating a AI-agent that can play football in the google research football environment.Project based on [Kaggle Competition](https://www.kaggle.com/c/google-football/overview/prizes) .Thesis for CSE-UOI 



![Game representation](https://github.com/Georgemouts/Google-Research-Football-RL/blob/main/images/grf.gif)
![Game_representation2](https://github.com/Georgemouts/Google-Research-Football-RL/blob/main/images/grf2.gif)

# Setup the environment

- All the stuff you need to know about environment and python libriaries is - [here](https://github.com/google-research/football)

- Observations and Actions list is - [here](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md)

- Compile the GRF-Engine [instructions](https://github.com/google-research/football/blob/master/gfootball/doc/compile_engine.md#windows)

# Interact with GRF environment 

You can interact with GRF environment and learn how to use it through [Simple_Observations.ipynb](https://github.com/Georgemouts/Google-Research-Football-RL/blob/main/Simple_Observations.ipynb)

# Learning & Playing :open_book:

Implemented algorithm **Deep Q-Learning(DQN)**  based on [Deepmind's paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

We need : 
  - The replay buffer
  - EpisolonGreedy action selection

We dont need:
  - Target neural network

# Scenarios

1. Academy_empty_goal ( agent learns to score doing the minimum number of steps )
2. 1vs1 (agent learn to score with opponent )
