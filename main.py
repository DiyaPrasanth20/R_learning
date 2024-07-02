import os 
import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


#Load Environment - state, action, reward, policy 

# Load Environment - state, action, reward, policy
env = gym.make("CartPole-v1", render_mode='human')

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()[0]  # For newer gym versions, reset() returns a tuple
    done = False
    score = 0

    while not done:
        env.render()  # shows environment
        action = env.action_space.sample()  # generates random action (either 0 or 1)
        n_state, reward, done, info = env.step(action)[:4]  # Handle additional return values if any
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


env.action_space.sample()  # generates random action (either 0 or 1) 0 pushes cart to the left, 1 pushes cart to the right
env.observation_space.sample()  # generates random observation # [cart position, cart velocity, pole angle, pole angular velocity]
