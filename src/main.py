import sys
sys.path.append('C:/Users/Giorgos/AppData/Local/Programs/Python/Python39/Lib/site-packages')
import gfootball 
from gfootball.env.football_env import FootballEnv
from kaggle_environments import make
from gfootball.env.config import Config
from gfootball.env.football_env import FootballEnv

#import dqn libraries
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import pandas as pd
import itertools
import random
from collections import deque
import matplotlib.pyplot as plt

#import env 
import gym
 

#Initialize Variables

BUFFERSIZE = 100000  #how many experiences will store |ReplayBufferSize=100,000
REWBUFFERSIZE = 100   #how many episode rewards will store|RewardBufferSize =100
MINREPLAYSIZE= 3000 # Episode is 3000 steps
GAMMA = 0.04
EPSILON =0.03
TARGET_UPDATE_FREQ = 25

EPSILON_START=0.5
EPSILON_END =0.01
EPSILON_DECAY=0.001

BATCH_SIZE = 3


#----------------------------------------------------------------
#Deep Q Network Model
class DQN(nn.Module):
  def __init__(self,env):
    super(DQN,self).__init__()
    input_dims = int(np.prod(env.observation_space.shape)) #neurons input layer = number of observations
    self.net =nn.Sequential(nn.Linear(input_dims,200), 
                            nn.Tanh(),
                            nn.Linear(200,env.action_space.n)) # neurons outpul layer = number of observations 
                            
  def forward(self,x):
    return self.net(x) #use the dqnetwork

  def act(self,obs): #returns the best acrtion/highest value action of the net 
    obs_t =torch.as_tensor(obs,dtype=torch.float32)
    q_values=self(obs_t.unsqueeze(0)) # make tensor a batch dimension

    max_q_index = torch.argmax(q_values,dim=1)[0]   # taking action with highest q value
    action = max_q_index.detach().item() # making tensor to integer which represents action 
    
    return action 

#-----------------------------------------------------------------------
#Agent
class Agent() : 
  def __init__(self,gamma,epsilon):
    self.gamma =gamma
    self.epsilon =epsilon
    
    self.ReplayBuffer = deque(maxlen=BUFFERSIZE) #Store experiences 
    self.RewBuffer = deque(maxlen=REWBUFFERSIZE) #Store rewards
    
    self.online_net = DQN(env)   
    self.target_net = DQN(env)  

    self.target_net.load_state_dict(self.online_net.state_dict())


  def transition(self,obs,new_obs,action,reward ,done): # should be tuple ? 
    self.obs =obs
    self.action=action
    self.reward =reward 
    self.done=done
    #self.info =info
    self.new_obs =new_obs
    TransitionTuple= (obs,new_obs,action,reward,done)
    #print("class",TransitionTuple)
    return TransitionTuple

  def learn(self,optimizer):
    self.optimizer=torch.optim.Adam(self.online_net.parameters(),lr =1e-3) 
    
#------------------------------------------------------------------------
#All Prints
class All_prints():
  
  def __init___(self,step):
    self.step=step
    #self.RewBuffer = RewBuffer
    self.reward=reward

  def printstats(self,step,RewBuffer,eps_reward):  #Kaleitai otan ginei done , diladi otan teleiosei ena paixnidi
    self.step=step
    self.RewBuffer=RewBuffer
    self.eps_reward=eps_reward
    print("-->Episode:",self.step%3000 + 1 ,"\t","Episode Reward:",self.eps_reward,"<--")
    print("Step",step)
    print("lista apo rewards mexri tora" ,self.RewBuffer)
    print("Avg reward", np.mean(self.RewBuffer))
    print("---------------------------------------------------")

  def print_who_scored(self, reward):
    self.reward=reward
    if(self.reward==-1):
      print("opponent team scored")
    elif(self.reward==1):
      print("our team scored !!!")
 
  def rew_graph(self,RewBuffer,step,num_of_eps):
      self.RewBuffer=RewBuffer
      self.step=step
      self.num_of_eps=num_of_eps

      episode=self.step%3000
      eps_list=list(range(1,self.num_of_eps+1))#pairnei to proto , den pairnei to teleytaio
      #print(agent.RewBuffer,eps_list)
      plt.plot(eps_list,self.RewBuffer)
      plt.xlabel('Episode')
      plt.ylabel('Rewards')
      plt.grid(True)
      plt.show()

#-----------------------------------------------------------------
#Create Environment -- Main
#if __name__ =='main': 

#TO DO : MAKE REPLAY BUFFER A CLASS
#TO DO:MAKE A CLASS FOR LEARNING

env = gym.make("GFootball-11_vs_11_kaggle-simple115v2-v0") #List with the 115 states 
eps_reward =0.0

obs = env.reset()

#CREATE OBJECTS 
agent=Agent(GAMMA,EPSILON) #Agent class
all_prints=All_prints() # print class

#Initialize ReplayBuffer # TO DO : MAKE IT A CLASS
for i in range(MINREPLAYSIZE): 
  action =env.action_space.sample() #random action 

  new_obs ,reward,done ,info = env.step(action) 

  transition = agent.transition(obs,new_obs,action,reward ,done) #obs ,action ,reward , done ,info , new_obs  PROSEKSE TO MALLON LATHOS
  agent.ReplayBuffer.append(transition)  #Fill Replay Buffer with transitions
  obs=new_obs

 
  if(done):  # if someone score  a goal reset the env 
    obs=env.reset()


# MAIN TRAIN LOOP
obs =env.reset()
c= 0
num_of_eps= 2     #GIVE NUMBER OF EPISODES

for step in range((3000 * num_of_eps) + num_of_eps ):# play 3000 steps = 1 match = 1 episode
  
  
  #epsilon greedy
  epsilon = np.interp(step,[0,EPSILON_DECAY],[EPSILON_START,EPSILON_END]) #Epsilon start->end with epsilon decays steps from 100% random actions->2% rnd actions
  rnd_sample = random.random()
  
  if rnd_sample <= epsilon: #random action |explore|
    action = env.action_space.sample()
  else:  
    action = agent.online_net.act(obs) #best action |exploit|     YPARXEI THEMA EDO PERA , POU GEMIZEI TO ONLINE NET ??
# return action 
  
  
  
  new_obs,reward,done,info = env.step(action)

  all_prints.print_who_scored(reward)

  transition = agent.transition(obs,new_obs,action,reward ,done) #fill Replaybuffer with transitions 

  agent.ReplayBuffer.append(transition) # To Replay Buffer gemizei kanonika
  obs=new_obs

  eps_reward = eps_reward+reward
  #print("eps_reward:",eps_reward,"rew:",reward,info,done)

  if (done) :
    
    obs=env.reset()
   
    agent.RewBuffer.append(eps_reward)

    #Print Resume when an episode ends 
    all_prints.printstats(step,agent.RewBuffer,eps_reward)
    #print(step)
    if(step == (3000 * num_of_eps)+num_of_eps -1):
       
      all_prints.rew_graph(agent.RewBuffer,step,num_of_eps)
    
    eps_reward =0.0 


# Start Gradient Step 
  transitions =random.sample(agent.ReplayBuffer , BATCH_SIZE) #sample batch_size number of random transitions from Replaybuffer ,
                                                              # Replay buffer have been filled earlier
 #Store convert -> return pytorch tensors 
  #Store observations as arrays
  obses = np.asarray([t[0] for t in transitions])
  new_obses = np.asarray([t[1]for t in transitions])
  actions = np.asarray([t[2] for t in transitions])
  rewards = np.asarray([t[3] for t in transitions])
  dones = np.asarray([t[4] for t in transitions])
  
  #Convert observation arrays to pytorch tensors
  obses_t=torch.as_tensor(obses,dtype=torch.float32)
  actions_t = torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1) #making batch dimension to one dimension
  rewards_t = torch.as_tensor(rewards,dtype= torch.float32).unsqueeze(-1)
  dones_t = torch.as_tensor(dones,dtype= torch.float32).unsqueeze(-1)
  new_obses_t = torch.as_tensor(new_obses,dtype= torch.float32)


  #Compute Targets
  
  target_q_values = agent.target_net(new_obses_t)# q values for each observation 
  max_target_q_values = target_q_values.max(dim=1,keepdim=True)[0] #take the maximum value in dim =1 , discard all the rest dimensions
                                                                  #max returns tuple , first element is highest values and second is the index to them 

  targets = rewards_t +GAMMA + (1-dones_t) * max_target_q_values #deepmind_atari_paper dqn learn with replay
                                                                #if its a terminal state: dones_t =1 -> targets= rewards_t

  #Compute Loss
  q_values = agent.online_net(obses_t)
  action_q_values =torch.gather(input=q_values,dim=1,index=actions_t)
  loss=nn.functional.smooth_l1_loss(action_q_values,targets)

  #Gradient Descent -> NA MPEI SE SYNARTHSH LEARN TOU AGENT 
  optimizer=torch.optim.Adam(agent.online_net.parameters(),lr =0.02)
  #print(optimizer)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  #Update the target network, copying all weights and biases in DQN
  if step% TARGET_UPDATE_FREQ == 0:   #Update target network based on online network
    agent.target_net.load_state_dict(agent.online_net.state_dict())

  
  



