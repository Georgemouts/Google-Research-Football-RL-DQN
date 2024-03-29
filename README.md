# Google-Research-Football-RL :soccer:
Creating a AI-agent that can play football in the google research football environment.Project based on [Kaggle Competition](https://www.kaggle.com/c/google-football/overview/prizes) .Undergraduate Thesis for Computer Science and Engineering - University of Ioannina.

## **Read the paper** - [here](https://github.com/Georgemouts/Google-Research-Football-RL-DQN/blob/main/thesis.pdf)   :page_facing_up:



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
  - The Experience Replay Buffer
  - Epsilon Greedy action selection algorithm

We dont need:
  - Target neural network

# Scenarios

1. Academy_empty_goal (agent learns to score doing the minimum number of steps) [Empty_goal.ipynb](https://github.com/Georgemouts/Google-Research-Football-RL-DQN/blob/main/Empty_Goal.ipynb)
2. 1vs1 (agent learn to score with opponent) [1vs1.ipynb](https://github.com/Georgemouts/Google-Research-Football-RL-DQN/blob/main/1vs1_.ipynb)
