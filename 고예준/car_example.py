import numpy as np


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name = 'Road1/Prototype 1', side_channels=[channel])

channel.set_configuration_parameters(time_scale = 3)


env.reset()

global decision_steps
global cur_obs


behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]
go_back_count=[0 for i in range(40)]
x,y,z,x1,y1,z1,s1,s2,s3,s4,s5 = cur_obs

#for i in range(3000):

while not ((abs(x-x1)<10) & (abs(z - z1)<10) & (s2 > 3) & (s4 > 3)):

    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    print("cur observations : ", decision_steps.obs[0][0,:])
    # Set the actions
    x,y,z,x1,y1,z1,s1,s2,s3,s4,s5 = cur_obs


    def go_back():
        for a in range(15):
            decision_steps, _ = env.get_steps(behavior_name)
            cur_obs = decision_steps.obs[0][0,:]
            env.set_actions(behavior_name, np.array([[-1,-150,-150]]))
            env.step()
        env.set_actions(behavior_name, np.array([[0,-150,-150]]))


    def go_back_2():
        for b in range(15):
            decision_steps, _ = env.get_steps(behavior_name)
            cur_obs = decision_steps.obs[0][0,:]
            env.set_actions(behavior_name, np.array([[1,-150,-150]]))
            env.step()
        env.set_actions(behavior_name, np.array([[0,-150,-150]]))


    def go_back_f():
        for i in range(30):
            decision_steps, _ = env.get_steps(behavior_name)
            cur_obs = decision_steps.obs[0][0,:]
            env.set_actions(behavior_name, np.array([[0,-150,-150]]))
            env.step()
        env.set_actions(behavior_name, np.array([[0,150,150]]))

        if decision_steps.obs[0][0,:][7]>decision_steps.obs[0][0,:][9]:

            while decision_steps.obs[0][0,:][7]>5.5:

                decision_steps, terminal_steps = env.get_steps(behavior_name)
                cur_obs = decision_steps.obs[0][0,:]
                env.set_actions(behavior_name,np.array([[1,150,150]]))
                env.step()
                #print("cur observations : ", decision_steps.obs[0][0,:])

        else:
            while decision_steps.obs[0][0,:][9]>5.5:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                cur_obs = decision_steps.obs[0][0,:]
                env.set_actions(behavior_name,np.array([[-1,150,150]]))
                env.step()



    if((s2 < 3) | (s4 < 3)):
        if(s2 > s4):
            env.set_actions(behavior_name, np.array([[1,30,30]]))
        else:
            env.set_actions(behavior_name, np.array([[-1,30,30]]))


    elif((s1 < 5) | (s3 < 5) | (s5 < 5)):
        if((s2 < 5) | (s4 < 5)):
            if(abs(s2 - s4) < 1):
                if(s1 > s3):
                    env.set_actions(behavior_name, np.array([[1,30,30]]))
                else:
                    env.set_actions(behavior_name, np.array([[-1,30,30]]))
            elif(s2 > s4):
                env.set_actions(behavior_name, np.array([[1,30,30]]))
            else:
                env.set_actions(behavior_name, np.array([[-1,30,30]]))
        elif((s1 < 5) & (s3 > 10)):
            env.set_actions(behavior_name, np.array([[1,30,30]]))
        elif((s1 > 10) & (s3 < 5)):
            env.set_actions(behavior_name, np.array([[-1,30,30]]))

        else:
            if(abs(s1-s3) < 1):
                if(s2 > s4):
                    go_back()
                    go_back_count=go_back_count[1:]
                    go_back_count.append(1)
                else:
                    go_back_2()
                    go_back_count=go_back_count[1:]
                    go_back_count.append(1)

            elif(s1 > s3):
                if sum(go_back_count)>=7:
                    go_back_f()
                    go_back_count=[0 for i in range(40)]

                else:
                    if(s2 > 6):
                        go_back()
                        go_back_count=go_back_count[1:]
                        go_back_count.append(1)
                    else:
                        go_back_2()
                        go_back_count=go_back_count[1:]
                        go_back_count.append(1)


            else:
                if sum(go_back_count)>=7:
                    go_back_f()
                    go_back_count=[0 for i in range(40)]
                else:
                    if(s4 > 6):
                        go_back_2()
                        go_back_count=go_back_count[1:]
                        go_back_count.append(1)
                    else:
                        go_back()
                        go_back_count=go_back_count[1:]
                        go_back_count.append(1)



    elif((s2 < 5) | (s4 < 5)):
        if(s2 > s4):
            if(s4 < 4):
                env.set_actions(behavior_name, np.array([[0.65,45,45]]))
            else:
                env.set_actions(behavior_name, np.array([[0.45,65,65]]))
        else:
            if(s2 < 4):
                env.set_actions(behavior_name, np.array([[-0.65,45,45]]))
            else:
                env.set_actions(behavior_name, np.array([[-0.45,65,65]]))

    elif((s1 < 7.5) | (s3 < 7.5) | (s5 < 7.5)):
        if((s2>4) & (s4>4)):
            if(s1 > s3):
                env.set_actions(behavior_name, np.array([[-0.45,45,45]]))
            else:
                env.set_actions(behavior_name, np.array([[0.45,45,45]]))
        else:
            if(s2 > s4):
                env.set_actions(behavior_name, np.array([[0.75,50,50]]))
            else:
                env.set_actions(behavior_name, np.array([[-0.75,50,50]]))


    elif((s1 < 19) | (s3 < 19) | (s5 < 19)):
        if((s1 < 10) | (s3 < 10) | (s5 < 10)):
            if(s2 > s4):
                env.set_actions(behavior_name, np.array([[0.45,85,85]]))
            else:
                env.set_actions(behavior_name, np.array([[-0.45,85,85]]))

        elif((s1 < 12.5) | (s3 < 12.5) | (s5 < 12.5)):
            if(s2 > s4):
                env.set_actions(behavior_name, np.array([[0.3,100,100]]))
            else:
                env.set_actions(behavior_name, np.array([[-0.3,100,100]]))
        elif((s1 < 15) | (s3 < 15) | (s5 < 15)):
            if(s2 > s4):
                env.set_actions(behavior_name, np.array([[0.2,150,150]]))
            else:
                env.set_actions(behavior_name, np.array([[-0.2,150,150]]))
        else:
            if(s1 > s3):
                env.set_actions(behavior_name, np.array([[0.1,150,150]]))
            elif(s1 < s3):
                env.set_actions(behavior_name, np.array([[-0.1,150,150]]))
            else:
                env.set_actions(behavior_name, np.array([[0,150,150]]))
    else:
        env.set_actions(behavior_name, np.array([[0,150,150]]))

    #go_back_count=go_back_count[1:]
    #go_back_count.append(0)

    print(go_back_count)
    print(sum(go_back_count))
    # Move the simulation forward
    go_back_count=go_back_count[1:]
    go_back_count.append(0)
    env.step()


for i in range(50):
    env.set_actions(behavior_name, np.array([[0,150,150]]))
    env.step()

env.close()