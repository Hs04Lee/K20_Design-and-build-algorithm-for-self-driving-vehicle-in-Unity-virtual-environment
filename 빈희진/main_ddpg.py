import gym
import numpy as np
import torch as T
from ddpg_torch import Agent
from utils import plot_learning_curve
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
channel = EngineConfigurationChannel()

def finished(obs):
    for i in range(6,11):
        if obs[0][0,i]>=5:
            return False
    return True

def crushed(obs):
    for i in range(6,11):
        if obs[0][0,i]<3:
            return True
    return False

if __name__ == '__main__':
    env = UnityEnvironment(file_name = 'C:/Users/heejb/Desktop/pre-rup/Road1/Prototype 1', side_channels=[channel])
    agent = Agent(alpha=0.0001, beta=0.001, #알파, 베타, 타우는 hyperparameter, 손대지 말자. 
                    input_dims=[5], tau=0.001, # input_dims는 5겠지?
                    batch_size=64, fc1_dims=400, fc2_dims=300, #batch_size<- 미니 배치(전체 데이터 나누는 부분의 크기)의 크기. 데이터 뭉텅이. fc1과 fc2는 망을 말하는 것? 
                    n_actions=1) # n_actions : action의 차원?(트럭에서는 3인데 2로도 가능.)
    
    n_games = 2000

    best_score=-5000
    score_history = []
    for i in range(n_games):
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale = 10)
        env.reset()
        behavior_name = list(env.behavior_specs)[0]

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        prev_action=[0]
        observation= decision_steps.obs[0][0,6:]
        done = False
        score = 0
        agent.noise.reset()
        while not done:  # observation은 과거의 상태, observation_은 현재 상태, reward는 observation-> observation_ 에서의 보상.
            action = agent.choose_action(observation) #Planning
            env.set_actions(behavior_name, np.array([[action[0],150,150]]))
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name) # Action한 후 값을 가져옴. <- print(env)
            observation_ = decision_steps.obs[0][0,6:]
            action[0]=T.clamp(T.FloatTensor([action[0]]),min=-1,max=1)
            print(action)
            reward = decision_steps.reward+1+2-0.5*abs(action[0])
            if prev_action[0]*action[0]<0:
                reward-=1
            done= finished(decision_steps.obs) or crushed(decision_steps.obs)
            if finished(decision_steps.obs): reward+=4000
            if crushed(decision_steps.obs): reward-=4000
            agent.remember(observation, action, reward, observation_, done)
            agent.learn() # 교육시키기.
            score += reward #전체 score ㅅㅔㄱㅣ
            observation = observation_ # observation 덮어쓰기   
            prev_action=action
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if score > best_score:
            best_score = score
            agent.save_models()
        x = [i+1 for i in range(i+1)]
        plot_learning_curve(x, score_history, 'C:/Users/heejb/Desktop/pre-rup/ddpg.png')

        print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, 'C:/Users/heejb/Desktop/pre-rup/ddpg.png')
