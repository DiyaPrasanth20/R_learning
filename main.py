import os 
import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


#Load Environment - state, action, reward, policy 

env = gym.make("CartPole-v0")

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset() #gives initial set of observations
    done = False
    score = 0

    while not done:
        env.render() #shows environment 
        action = env.action_space.sample() #generates random action (either 0 or 1)
        n_state, reward, done, info = env.step(action) 
        '''
        pass through random action 
        get next set of obserations [] len 4, reward, terminal state reached?, 

        '''
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()



