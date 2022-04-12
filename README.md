# Google-Research-Football-RL :soccer:
Creating a AI-agent that can play football in the google research football environment.Project based on [Kaggle Competition](https://www.kaggle.com/c/google-football/overview/prizes) .Bachelor's thesis for CSE-UOI 



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
  - Target neural network

# Reward System

The environment has +1 reward for when our team scores and -1 when the opponent teams scores.

Custom Rewards: 
  1. Positives:
   - When the agent has the ball in position x > 0.6 and shouts it , **Reward += 0.1**
   - When the agent steals the ball from the opponent , **Reward+= 0.05**
  
  2. Negatives:
    

