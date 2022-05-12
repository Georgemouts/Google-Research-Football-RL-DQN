#from gfootball.env.football_env import FootballEnv

import gfootball.env as football_env

#import dqn libraries
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import pandas as pd
import itertools
#import utils
import random
from collections import deque
import matplotlib.pyplot as plt

import math
import os
print(os.path.abspath(football_env.__file__))


BUFFERSIZE = 10000  #how many experiences will store |ReplayBufferSize=100,000
REWBUFFERSIZE = 500   #how many episode rewards will store|RewardBufferSize =10,000
MINREPLAYSIZE= 1000 # Episode is 3000 steps
GAMMA = 0.99


EPSILON_START=1.0
EPSILON_END =0.01
EPSILON_DECAY=10000

TARGET_UPDATE_FREQ = 400 # target parameters will be equal to online parameters
BATCH_SIZE = 32



class DQN(nn.Module):
  def __init__(self,env):
    super().__init__()
    #input_dims = int(np.prod(env.observation_space.shape)) #neurons input layer = number of observations
    self.net =nn.Sequential(
                            nn.Linear(10,250), 
                            #nn.ReLU(),
                            nn.ReLU(),
                            nn.Linear(250,4)) # neurons outpul layer = number of actons
                            
  def forward(self,x):
    return self.net(x) #use the dqnetwork

  def act(self,obs): #returns the best acrtion/highest value action of the net 
    obs_t =torch.as_tensor(obs,dtype=torch.float32)
    q_values=self(obs_t.unsqueeze(0)) # make tensor a batch dimension

    max_q_index = torch.argmax(q_values,dim=1)[0]   # taking action with highest q value
    action = max_q_index.detach().item() # making tensor to integer which represents action 
    #print("QVALUES",q_values)
    #print("MAX_VALUE",max_q_index)

    return action 





env = football_env.create_environment(env_name ='academy_empty_goal',render=True,representation='simple115v2') #List with the 115 states 

replay_buffer =deque(maxlen=BUFFERSIZE)
rew_buffer = deque([0,0],maxlen=100)
episode_reward = 0.0


episode_reward =0.0
online_net=DQN(env)
target_net=DQN(env)

target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters() , lr =0.001)


Action_list=[4,5,6,12]

#Initialize replaybuffer 
obs = env.reset()

for i in range(MINREPLAYSIZE):
  action = random.randint(0,3) # random action 
  while((obs[0]<0.65)  and (Action_list[action]==12)):
    action = random.randint(0,2)

  new_obs ,reward,done ,info = env.step(Action_list[action])



  transition = (obs, action,reward ,done ,new_obs)
  replay_buffer.append(transition)
  obs = new_obs

  if (done ==1):
    obs = env.reset()

#Main Training loop
obs =env.reset()

for step in itertools.count():
    epsilon = np.interp(step,[0,EPSILON_DECAY],[EPSILON_START , EPSILON_END]) # from 100%random actions to 1% random actions
    rnd_sample = random.random()

    if rnd_sample<epsilon :
        action = random.randint(0,3) # random action 
        while((obs[0]<0.65)  and (Action_list[action]==12)):
            action = random.randint(0,2)
            #print("Epilego action=",action)
    else:
        action = online_net.act(obs)    #best action
        #while((obs[0]<0.65)  and (Action_list[action]==12)): #Den kanei shout ektos periohis
        #action = online_net.act(obs)
    
    new_obs ,reward,done ,_ = env.step(Action_list[action])
  #custom rewards - here 
    if(done ==1 and reward != 1): #if ball is out ,loses -2
        print("Ball is out reward:",reward)
        reward = reward -10
        print("ball is out -10","episode",i,"Ball Position",obs[4],obs[5],obs[6])
    if(reward==1 and  done ==1): #if agent scores , wins +5
        print("goal","episode",i,)
        reward = 10

    if((obs[0]<0.5)  and (Action_list[action]==12)): #an shoutarei prin th megali perioxh -2
      
        reward= reward -200
        done=1
        print("shout ektos periohis","episode",i)
      
    if((obs[0]>0.6) and (Action_list[action]==12)): #an shoutarei mesa ti megali periohi +0.1
      #reward= reward +0.1
      print("shout entos periohis Ball Position",obs[4],obs[5],obs[6],"episode",i,"step=",step)
    
    
    reward = reward - ( math.sqrt( ((0.935 - obs[4])**2) + (0 -obs[5])**2 ) *0.2)
  #---------------------

    new_obs ,reward,done ,_ = env.step(Action_list[action])

    transition = (obs, action ,reward ,done, new_obs)
  
    replay_buffer.append(transition) # gemizei me transition
 
    obs = new_obs
    episode_reward += reward


    if (done ==1):
        print("Number of steps" ,step,"Reward:",episode_reward)
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward =0.0

    #Start Gradient Step
    transitions = random.sample(replay_buffer,BATCH_SIZE) #trabao tyxaia transitions apo to replay buffer
  
  
    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

  #Compute Targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]

    targets = rewards_t *GAMMA *(1-dones_t) * max_target_q_values

  #Compute Loss
    q_values = online_net(obses_t)
    action_q_values =torch.gather(input = q_values,dim=1,index=actions_t) # take the q value of the action we did throught the transition

    loss =nn.functional.smooth_l1_loss(action_q_values , targets)

  #Gradient Descent 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  #Update Target Network
    if step % TARGET_UPDATE_FREQ ==0:
        target_net.load_state_dict(online_net.state_dict())


    if step %1000 == 0 and step!=0:
        print("Step", step)
        print("Avg Reward",np.mean(rew_buffer))

