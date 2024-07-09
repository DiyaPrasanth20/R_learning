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



#Train RL Model 
log_path = os.path.join('Training', 'Logs') 
env = gym.make("CartPole-v1")
#the environment is wrapped in a DummyVecEnv object, which is a vectorized environment that allows the model to train
env = DummyVecEnv([lambda: env])
#PPO = proximal policy optimization model, reinforcement learing method, used to train deep neural networks 
#to make decisions in environments
model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log=log_path) # MlpPolicy is a neural network policy that is used to map states to actions 
#MLP = multi-layer perceptron, a type of feedforward neural network, mapping eventually decides the action to take for the state 
#verbose parameter is set to 1, which means that the training output will be printed to the console (logging info)

model.learn(total_timesteps=20000) #total_timesteps is the number of steps the model will take in the environment during training





#Save and reload the model 
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_CartPole')
#saves it in this path 
model.save(PPO_path)
del model 




PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_CartPole')
env = gym.make("CartPole-v1", render_mode='human')
model = PPO.load(PPO_path, env=env)
#Reinforcement learning algorithms are chosen based on action space and observation space 



#Evaluate the model
print(evaluate_policy(model, env, n_eval_episodes=10, render=True)) #score of 200 is best 
#gives (average reward over 10 episodes, standard deviation of the reward over 10 episodes)




#Test the model
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()[0]  # For newer gym versions, reset() returns a tuple
    done = False
    score = 0

    while not done:
        env.render()  # shows environment
        action, _ = model.predict(obs) #second compoenent is the state value
        obs, reward, done, info = env.step(action)[:4]  # Handle additional return values if any
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


 

