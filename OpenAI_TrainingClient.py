# This whole source code followed in this tutorial 
# Link https://www.youtube.com/watch?v=UCgsv6tMReY

import gym
# from gym import wrappers 
import numpy as np

from sample_DDQN import DDQNAgent 
# from utils import plotLearning 

if __name__ == '__main()__': 
    env = gym.make('LunarLander-v2')
    ddqn_agent = DDQNAgent(alpha = 0.0005, gamma = 0.99, epsilon = 1.0, 
                            input_dims=8, batch_size = 64, n_actions = 4) 
    # ddqn_agent = DDQNAgent(input_dims = 8, n_actions = 4, batch_size = 64)
    n_games = 500 
    ddqn_scores = []
    eps_history = []

    for i in range(n_games): 
        done = False 
        score = 0 
        observation = env.reset()
        while not done: 
            action = ddqn_agent.choose_action()
            observation_, reward, done, info = env.step(action) 
            score += reward 
            ddqn_agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            ddqn_agent.learn()
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])

        # print('episode ',i,'score %.2f ' % score, 'average score %.2f' % avg_score)
        print('----------------------------------------')
        print('episode ', i)
        print('score %.2f' %score)
        print('average score %.2f' %avg_score)
        print('----------------------------------------')
        
        if i % 10 == 0 and i > 0: 
            ddqn_agent.save_model()
