#from gfootball.env.football_env import FootballEnv

from argparse import Action
import gfootball.env as football_env

#import dqn libraries
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import pandas as pd
import itertools
#import utils
import random

import matplotlib.pyplot as plt

import math
import os
print(os.path.abspath(football_env.__file__))




class DeepQNetwork(nn.Module):
  def __init__(self,lr,input_dims,fc1_dims,fc2_dims,n_actions):
    super(DeepQNetwork,self).__init__()
   # self.lr=lr
    self.input_dims=input_dims
    self.fc1_dims=fc1_dims
    self.fc2_dims=fc2_dims
    
    self.n_actions=n_actions
    
    self.fc1=nn.Linear(*self.input_dims,self.fc1_dims) #pass list of observations as input
    self.fc2=nn.Linear(self.fc1_dims,self.fc2_dims)
    self.fc3=nn.Linear(self.fc2_dims,self.n_actions)
    #self.fc4=nn.Linear(self.fc3_dims,self.n_actions) #output number action

    self.optimizer = optim.Adam(self.parameters(),lr=lr)
    self.loss=nn.MSELoss()
    self.device =T.device('cuda:0' if T.cuda.is_available() else 'cpu' )
    self.to(self.device)

  def forward(self,state):
   
    x=F.relu(self.fc1(state))
    x=F.relu(self.fc2(x))
    actions=self.fc3(x)
    #x=F.relu(self.fc3(x))
    return actions

class Agent():
  def __init__(self,gamma,epsilon, lr , input_dims , batch_size ,n_actions, max_mem_size = 10000  , eps_end=0.01 , eps_dec = 5e-4):
    self.gamma=gamma
    self.epsilon =epsilon
    self.lr=lr
    self.eps_min=eps_end
    self.eps_dec=eps_dec
   
    self.action_space =[i for i in range(n_actions)]
    self.mem_size = max_mem_size
    self.batch_size = batch_size
    #self.n_actions=n_actions
    self.mem_cntr =0 # keep track of the position of first available memory 

    self.Q_eval = DeepQNetwork(self.lr,n_actions=n_actions,input_dims= input_dims, fc1_dims=123, fc2_dims=123)

    self.state_memory = np.zeros((self.mem_size,*input_dims),dtype =np.float32)
    self.new_state_memory= np.zeros((self.mem_size , *input_dims),dtype=np.float32)

    self.action_memory=np.zeros(self.mem_size , dtype=np.int32) #discrete actions 
    self.reward_memory=np.zeros(self.mem_size,dtype= np.float32)
    self.terminal_memory= np.zeros(self.mem_size,dtype=bool)


  def store_transition(self,state,action,reward,new_state , done ):
    index = self.mem_cntr% self.mem_size

    self.state_memory[index]= state
    self.new_state_memory[index]= new_state
    self.reward_memory[index]= reward
    self.action_memory[index]= action  #which action is taken 
    self.terminal_memory[index]= done

    self.mem_cntr +=1
    

  def choose_action(self,observation):
    if np.random.random()> self.epsilon:
      
      state =T.tensor([observation]).to(self.Q_eval.device) #turn observation to tensor and send it to device for computations
      action_list = self.Q_eval.forward(state) #returns the values of each action
      action = T.argmax(action_list).item()
      #print("exploit:",action)
    else:     #
      
      action = np.random.choice(self.action_space)
      #print("explore:",action)
    return action

  def learn(self):    #fill batch size then learn 
    if self.mem_cntr < self.batch_size :
     
      return
    
    
    self.Q_eval.optimizer.zero_grad()

    #calculate the position of max memory / extract subset of max memories
    max_mem =min(self.mem_cntr , self.mem_size)
    


    batch=np.random.choice(max_mem,self.batch_size,replace=False) #We dont keep selecting the same memories more than once
     
    #batch = np.random.permutation(max_mem)[:self.batch_size]
    #mem = np.array(exp_buffer)[perm_batch]

    #batch=np.random.choice(max_mem,self.batch_size,replace=False)
    batch_index = np.arange(self.batch_size , dtype=np.int32)

    state_batch=T.tensor(self.state_memory[batch]).to(self.Q_eval.device) #make numpy array a pytorch tensor
    new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
    reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
    terminal_batch= T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

    action_batch = self.action_memory[batch] 
  
    q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch] #EXEI THEMA
    q_next = self.Q_eval.forward(new_state_batch)

    q_next[terminal_batch] = 0.0
    q_target = reward_batch +self.gamma * T.max(q_next,dim=1)[0] #max value of next state

    loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
    loss.backward()
    self.Q_eval.optimizer.step()

    self.epsilon = self.epsilon - self.eps_dec if (self.epsilon > self.eps_min)  else self.eps_min


  

class All_prints():
  
  def __init___(self,step):
    self.step=step
    #self.RewBuffer = RewBuffer
    self.reward=reward

  def printstats(self,step,rew_list,eps_reward,epsilon):  #Kaleitai otan ginei done , diladi otan teleiosei ena paixnidi
    self.step=step
    self.rew_list=rew_list
    self.eps_reward=eps_reward
    self.epsilon=epsilon
    print("-->Episode:",i%3000 + 1,"\t","Episode Reward:",eps_reward,"\t Epsilon",agent.epsilon,"<--")
    #print("Step",step)
    print("lista apo rewards mexri tora" ,self.rew_list)
    print("Avg reward", np.mean(self.rew_list))
    print("---------------------------------------------------")

  def print_who_scored(self, reward):
    self.reward=reward
    if(self.reward==1):
      print("our team scored !!!")
    elif(self.reward ==-1):
      print("opponent team scored")
    
    
 
  def rew_graph(self,rew_list,num_of_eps):
      self.rew_list=rew_list
      
      self.num_of_eps=num_of_eps
      
      eps_list=list(range(1,self.num_of_eps+1))#pairnei to proto , den pairnei to teleytaio
      
      plt.plot(eps_list,self.rew_list)
      plt.xlabel('Episode')
      plt.ylabel('Rewards')
      plt.grid(True)
      plt.show()

  def step_graph(self,step_list,num_of_eps):
      self.step_list=step_list
      
      self.num_of_eps=num_of_eps
      
      eps_list=list(range(1,self.num_of_eps+1))#pairnei to proto , den pairnei to teleytaio
      
      plt.plot(eps_list,self.step_list)
      plt.xlabel('Episodes')
      plt.ylabel('Steps_to_score')
      plt.grid(True)
      plt.show()


  def score_graph(self,score_list,num_of_eps):
      self.score_list=score_list
      
      self.num_of_eps=num_of_eps
      
      eps_list=list(range(1,self.num_of_eps+1))#pairnei to proto , den pairnei to teleytaio
      
      plt.plot(eps_list,self.score_list)
      plt.xlabel('Episode')
      plt.ylabel('Score')
      plt.grid(True)
      plt.show()





env = football_env.create_environment(env_name ='academy_empty_goal',render=False,representation='simple115v2')  #List with the 115 states 


#CUSTOMIZE ACTION LIST AND OBSERVATIONS
Action_list=[4,5,6,12,13,14,15,17,18]
#print(env.action_space.n)
#Create Objects


agent = Agent(gamma=0.99,epsilon=1.0 ,batch_size = 64 ,lr=0.00115 ,input_dims= [17], n_actions = len(Action_list) )# batch = best 256
all_prints = All_prints()
#cus_rew =Custom_Rewards()

scores,ep_history =[],[]

steps=0
terminal =0 # an skorarei se pano apo enan arithmo paixnidion stamatao tin ekpaideysi 
episode =0
shout =0


num_of_eps = 801
eps_rew=0
rew_list =[]
score_list = []
step_list =[]
goal_steps=[]


for i in range(num_of_eps) : 
  score =0 
  done=False 
  eps_rew=0
  observation =env.reset()
  #print("Pinakas apo observations",observation)
  act =0 #first action will be to move right 
  shout=0 #mporei na kanei shout 1 fora se kathe ep
  timer=0 # an klepsei tin mpala kai tin kratisei pano apo 4 steps stamata
  sprint=0 #an perasei ton antipalo kai kanei sprint tin proti fora  einai +5
  checkpoint_reward=[1,1,1,1,1]
 
  while not done:

    #CUSTOMIZE ACTIONS HERE 
    #An einai i mpala sto 0.5 kai exo katoxi kane shout diladi action 12
    
    #print("------------")
    #print("Ball X-Y-Z Axis",observation[88],observation[89],observation[90],"||","direct",observation[91],observation[92],observation[93],"Katoxi",observation[95],observation[96])
    #print("Player X-Axis Y-Axis",observation[2],observation[3] ,"episode",i) SOSTO
    #print("------------")

    #print(observation[0],observation[1],observation[2],observation[3],observation[4],observation[5])
    

   
     
      
 
    if(act ==0 ): # proti praksi ena bima deksia
      
      # print("action 5",Action_list[action])
      new_observation,reward,done,info = env.step(5)
      
      act=1
      #print(new_observation)
    #print("Sto Else",observation[94],observation[95],observation[96])
    

    #CUSTOM ACTIONS
    action = agent.choose_action(observation) # from 1-9s which is index to action list


    while((observation[0]<0.65)  and (Action_list[action]==12)): #Den kanei shout ektos periohis
      action = agent.choose_action(observation)
     
      #print("Player Position:",observation[0],observation[1])
      #print("Player Direction:",observation[2],observation[3])
      #print("Ball Position:",observation[4],observation[5],observation[6])
      #print("Ball Direction:",observation[7],observation[8],observation[9])

    if(shout==0):   #ama kanei shout na min kanei tipota meta 
      new_observation,reward,done,info = env.step(Action_list[action])
      
      if(Action_list[action]==12):
        #print("EKANE SHOUT")
        shout=1
    else:
      #print("MPIKE STO ELSE")
      new_observation,reward,done,info = env.step(0)
      action=0 #Ta parakato if den pianoun to Action_list[action]=12 alla gia Action_list[action]=4

#CUSTOM REWARDS-------------------------------
    if(done ==1 and reward ==1): #if agent scores , wins +40
      print("goal","episode",i,"step=",steps)
      reward += 50 

   
    if(done ==1 and reward == 0): # an bgei i mpala apo ton antipalo einai -15 , a bgei apo ton paikti einai -5 
      if(shout ==1 ):
        reward = reward - 5
        print("Ball is out by player - 5")
      else:
        reward = reward -15 #12
        print("ball is out -15","episode",i,"Ball Position",observation[8],observation[9],observation[10],"step=",steps)
        terminal =0



      terminal= terminal +1
      goal_steps.append(steps)


    if(observation[16]==1 ):  # an klepsei tin mpala kai tin kratisei pano apo 4 steps stamata
      timer=timer+1
      if(timer <= 2 ):
        reward = reward - 30
      if (timer>2 and timer <  4 ):
        reward = reward - 2
      if (timer>4 and timer <6 ):
        reward = reward - 6
      if(timer >= 8):
        done=1
        reward = reward -20
        print("klepsimo")

    if(observation[8]< 0.0 ):  #An i mpala paei piso apo to kentro telos paixnidiou
      print("Mpala piso apo kentro")
      reward = reward-20
      done =1 

    

    if(((observation[0] > observation [4] +0.3) and observation[14]== 1 )):  # an perasei ton antipalo einai +5
      print("perase ton antipalo")
      reward = reward + 5  

    if(observation[14]== 0): #an den exo stin katoxi tin mpala einai -5
      reward =reward-5
    if(observation[14] == 1): # an exo tin katoxi na einai +2
      reward =reward +2
    if((observation[0] > observation [4] +0.1) and (Action_list[action]==13) and (sprint ==0)and (observation[14]==1)): #An peraso ton antipalo kai kano sprint tin proti fora +5
      print("perase ton antipalo kai sprint NO REWARD")
      #reward =reward + 10 
    
    if((observation[4] > observation[0])and (Action_list[action]==13 or Action_list[action]==15)): # an einai apenanti o paiktis kai kano sprint -5
      reward = reward -5
    if((observation[4]-observation[0]< 0.1)and (observation[4]-observation[0]> 0) and (Action_list[action]==17)):  # an kanei dirbble konta ston antipalo prin erthei se ayton +5 
      #print("dribble")
      reward = reward + 10                                     #PROSOXI EDO

    


    if((observation[0]<0.65)  and (Action_list[action]==12)): #an shoutarei prin th megali perioxh -2
      
      reward= reward -200
      done=1
      print("shout ektos periohis","episode",i,"step=",steps)
      
    if((observation[0]>0.6) and (Action_list[action]==12)): #an shoutarei mesa ti megali periohi +10
      reward= reward +10
      print("shout entos periohis Ball Position",observation[8],observation[9],observation[10],"episode",i,"step=",steps)
      
    
    """if(observation[0]>0.5 and checkpoint_reward[0]==1):
      reward = reward + 1 
      checkpoint_reward[0]=0
    elif(observation[0]>0.55 and checkpoint_reward[1]==1):
      reward=reward+1
      checkpoint_reward[1]=0
    elif(observation[0]>0.6 and checkpoint_reward[2]==1):
      reward=reward+1
      checkpoint_reward[2]=0
    elif(observation[0]>0.65 and checkpoint_reward[3]==1):
      reward=reward+1
      checkpoint_reward[3]=0
    if(observation[0]> observation[4] ):
      reward = reward +5"""

    
    reward = reward - ( math.sqrt( ((0.935 - observation[8])**2) + (0 -observation[9])**2 ) *0.3) #oso pio makria einai toso perissotero xanei
    #print("Den exo mpala",Action_list[action])
    #print("DEN EXO MPALA",Action_list[action],action)

    #END OF CUSTOM REWARDS ----------------------------- 
 
    
   
    score+= reward/11

    #for prints
    
    #all_prints.print_who_scored(reward)
    

    agent.store_transition(observation,action,reward,new_observation,done)
    agent.learn()
    observation = new_observation

    scores.append(score)
    ep_history.append(agent.epsilon)

    avg_score= np.mean(scores)


    steps=steps+1
    

#---- BE CAREFUL OF THE WHILE !!! HERE IS EPIDOSE ENDING--------
  #print("Reward",eps_rew,"Episode",i,"Steps" , steps)
  eps_rew+=reward
  step_list.append(steps)
  steps=0

  val = info.values()
  list_val=list(val)
  score_list.append(list_val)

  rew_list.append(eps_rew)
  episode = episode +1 

  print("---Episode reward:", reward ,"score",list_val,"steps",steps,"episode=",i,"---")
  #terminate if 500 episodes are correct 
  if(terminal ==30):
    print("!!! END OF TRAINING 20 CONTINUOUS GOALS !!!")
    print("---Avg reward last:", np.mean(rew_list[-10:]),"Avg score last",np.mean(score_list[-10:]),"Avg steps",np.mean(step_list[-10:]),"episode=",i,"---")
    all_prints.rew_graph(rew_list[-i:],i)
    all_prints.step_graph(goal_steps[-20:],20)
    break

# PRINTS
  if (i % 10)== 0 :
      print("---Avg reward last:", np.mean(rew_list[-10:]),"Avg score last",np.mean(score_list[-10:]),"Avg steps",np.mean(step_list[-10:]),"episode=",i,"---")
      #print(score_list)

  if (((i % 1000)== 0) and i!=0) :
      all_prints.score_graph(score_list[-1000:],1000)# graph the last 1000 episodes
  if(((i%50)==0) and i!=0):
    all_prints.rew_graph(rew_list[-i:],i)
     
  
  #EPISODE PRINTS
  #all_prints.printstats(i,rew_list,eps_rew,agent.epsilon)
  
  #eps_rew=0 #GIA NA BGALO SYNOLIKO GRAFIMA TO AFAIRO AYTO



#print("Avg score last:", np.mean(rew_list[-10:]),"Avg score",np.mean(score_list),"Avg steps",np.mean(step_list[-10:]),"episode=",i)
#all_prints.score_graph(score_list[-1000:],1000)# graph the last 1000 episodes
      
a = len(goal_steps)
all_prints.step_graph(goal_steps[-a:],len(goal_steps))
