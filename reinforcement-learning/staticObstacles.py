import re
import numpy as np
import math
import random
import plotly.graph_objects as go
import operator

GRID_ROWS = 5
GRID_COLS = 10
WIN_STATE = (3, 9)
BLOCK_STATE = [(2,7), (3,7), (4,7)]
# START = (0, 0)

def genRandomState(grid_rows, grid_cols, win_state, block_states):
    x = random.randint(0,grid_rows-1)
    y = random.randint(0,grid_cols-1)
    
    while ((x,y) == win_state) or ((x,y) in block_states):
        x = random.randint(0,grid_rows-1)
        y = random.randint(0,grid_cols-1)
    return (x,y)

class State:
    def __init__(self, state):
        self.grid = np.zeros([GRID_ROWS, GRID_COLS])  ### Change this to average of all rewards 
        self.state = state
        self.blockPos()
        self.isEnd = False

    def blockPos(self):
        for state in BLOCK_STATE:
            self.grid[state] = float("nan")

    def generateReward(self):
        if self.state == WIN_STATE:
            return 20
        elif self.state in BLOCK_STATE:
            return -100
        else:
            return -1

    def getReward(self, state):
        if state == WIN_STATE:
            return 20
        else:
            return -1
    def totalReward(self, states):
        reward = 0
        for state in states:
            reward = reward + self.getReward((state[0],state[1]))
        return reward

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.isEnd = True

    ##### Action: up, down, left, right, stay
    def nextPos(self, action):
        # print("state in nxtpos:", self.state)
        if action == "up":
            nextState = (self.state[0] - 1, self.state[1])
        elif action == "down":
            nextState = (self.state[0] + 1, self.state[1])
        elif action == "left":
            nextState = (self.state[0], self.state[1] - 1)
        elif action == "right":
            nextState = (self.state[0], self.state[1] + 1)
        # acts = ["up", "down", "left", "right"]
        # if next state legal
        # if nextState not in BLOCK_STATE:
        if (nextState[0] >= 0) and (nextState[0] <= (GRID_ROWS -1)):
            if (nextState[1] >= 0) and (nextState[1] <= (GRID_COLS -1)):
                    return nextState, action
        return self.state, action ###### stay on same position

    def showBoard(self):
        for i in range(0, GRID_ROWS):
            out = '| '
            for j in range(0, GRID_COLS):
                if self.grid[i, j] == 0:
                    token = '0'
                if  np.isnan(self.grid[i, j])==1:
                    token = 'NA'
                if (i,j) == self.state:
                    token ='*'
                if (i,j) == WIN_STATE:
                    token ='W'
                out += token + ' | '
            print(out)


# Agent of player

class sarsaAgent:

    def __init__(self, start_state):
        self.actions = ["up", "down", "left", "right"]
        self.actions_lookup = {
            "up":0,
            "down":1,
            "left":2,
            "right":3
        }
        self.alpha = 0.9
        self.gamma = 0.9
        self.epsilon = 0.5
        self.State = State(start_state)
        

        # initial state reward
        self.state_values = {}
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                for k in range(0,4):
                    self.state_values[(i, j, k)] = 10  # set initial value to 0
        # print("i m in init")
        self.action = self.chooseAction()
        self.states = [(self.State.state[0],self.State.state[1],self.actions_lookup[self.action])]
        # print("State:", self.State.state)

    def chooseAction(self):
        # choose action with most expected value
        action = ""

        if np.random.uniform(0, 1) <= self.epsilon:
            pot_action = np.random.choice(self.actions)
            nxt_pos, action = self.State.nextPos(pot_action)
        else:
            # greedy action
            reward_acts = {}
            for a in self.actions:
                # if the action is deterministic
                # print('curr_state: ', self.State.state)
                nxt_pos, chosen_action = self.State.nextPos(a)
                # print("pos action and state: ", nxt_pos, chosen_action)
            
                if chosen_action not in list(reward_acts.keys()):
                    nxt_rewards = [self.state_values[(nxt_pos[0], nxt_pos[1], 0)],
                            self.state_values[(nxt_pos[0], nxt_pos[1], 1)],
                            self.state_values[(nxt_pos[0], nxt_pos[1], 2)],
                            self.state_values[(nxt_pos[0], nxt_pos[1], 3)]]
                    nxt_reward = max(nxt_rewards)
                    reward_acts[chosen_action] = nxt_reward   
            print(reward_acts, nxt_pos)
            max_act_val = reward_acts[max(reward_acts, key=reward_acts.get)]
            max_acts = [key for key,val in reward_acts.items() if val==max_act_val]
            
            action = random.choice(max_acts) 
            print("R,np,act: ",reward_acts, nxt_pos, action)
        # print("returned action:", action)
        return action

    def takeAction(self, action):
        position, _ = self.State.nextPos(action)
        return State(state=position)

    def reset(self):
        start_state = genRandomState(GRID_ROWS, GRID_COLS , WIN_STATE, BLOCK_STATE)
        # print("new start state: ", start_state)
        self.State = State(start_state)
        self.action = self.chooseAction()
        self.states = [(self.State.state[0],self.State.state[1],self.actions_lookup[self.action])]

    def start(self, state):
        self.State = State(state)
        self.action = self.chooseAction()
        self.states = [(self.State.state[0],self.State.state[1],self.actions_lookup[self.action])]

    def play(self, num_episodes=1, num_steps = 20):
        i = 0
        curr_steps = 0
        rewards = []
        while i < num_episodes:
            # to the end of game back propagate reward
            if (self.State.isEnd) or (curr_steps == num_steps):
                # back propagate
                reward = self.State.generateReward()
                # explicitly assign end state to reward values
                self.state_values[(self.State.state[0], self.State.state[1], self.actions_lookup[self.action])] = reward
                print("Game End Reward", reward)
                print("all states:", self.states, len(self.states))
                # for s in reversed(self.states): #### update total reward
                #     reward = self.state_values[s] + self.alpha * (reward - self.state_values[s])
                #     self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1
                curr_steps = 0
                rewards.append(self.evaluate((1,6)))
            else:
                # print("in else of play")
                # append trace
                curr_state = self.State.state
                curr_action = self.action
                curr_reward = self.State.generateReward()
                
                self.State = self.takeAction(curr_action) ## Update to next state
                nxt_state = self.State.state
                nxt_action = self.chooseAction()
                self.action = nxt_action
                self.states.append((nxt_state[0],nxt_state[1],self.actions_lookup[nxt_action]))
                # print("nxt state: ", nxt_state, nxt_action)
                # print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                
                self.state_values[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])] = \
                            self.state_values[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])] + \
                            self.alpha*(curr_reward + \
                            self.gamma*self.state_values[(nxt_state[0], nxt_state[1], self.actions_lookup[nxt_action])] - \
                            self.state_values[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])])
                # mark is end
                self.State.isEndFunc()
                curr_steps = curr_steps + 1
                # print("nxt state", self.State.state)
                
                # print("---------------------")
        return rewards

    def evaluate(self,state, num_steps = 20):
        self.start(state)
        curr_steps = 0
        while not ((self.State.isEnd) or (curr_steps == num_steps)):
            # print(self.State.isEnd, curr_steps)
            curr_action = self.action
            self.State = self.takeAction(curr_action) ## Update to next state
            nxt_state = self.State.state
            nxt_action = self.chooseAction()
            self.action = nxt_action
            self.states.append((nxt_state[0],nxt_state[1],self.actions_lookup[nxt_action]))
            # print("nxt state: ", nxt_state, nxt_action)
            # print("current position {} action {}".format(self.State.state, action))
            # by taking the action, it reaches the next state
            # mark is end
            self.State.isEndFunc()
            curr_steps = curr_steps + 1
        reward = self.State.totalReward(self.states)
        print("Eval: all states: ", self.states)
        print("Eval Rewards: ", reward)
        return reward
                     

    def showValues(self):
        for k in range(0, 4):

            print("%d th action: " %(k))
            for i in range(0, GRID_ROWS):
                print('----------------------------------')
                out = '|'
                for j in range(0, GRID_COLS):
                    # out += "(" + str(i) + "," + str(j) + "):" + str(self.state_values[(i, j, k)]).ljust(6) + '|'
                    out += str(self.state_values[(i, j, k)]).ljust(6) + '|'
                print(out)
        print('----------------------------------')

class qlearningAgent:

    def __init__(self, start_state):
        self.actions = ["up", "down", "left", "right"]
        self.actions_lookup = {
            "up":0,
            "down":1,
            "left":2,
            "right":3
        }
        self.alpha = 0.9
        self.gamma = 0.9
        self.epsilon = 0.
        self.State = State(start_state)
        

        # initial state reward
        self.state_values = {}
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                for k in range(0,4):
                    self.state_values[(i, j, k)] = 10  # set initial value to 0
        # print("i m in init")
        self.action = self.chooseAction()
        self.states = [(self.State.state[0],self.State.state[1],self.actions_lookup[self.action])]
        # print("State:", self.State.state)

    def chooseAction(self):
        # choose action with most expected value
        action = ""

        if np.random.uniform(0, 1) <= self.epsilon:
            pot_action = np.random.choice(self.actions)
            nxt_pos, action = self.State.nextPos(pot_action)
        else:
            # greedy action
            reward_acts = {}
            for a in self.actions:
                # if the action is deterministic
                # print('curr_state: ', self.State.state)
                nxt_pos, chosen_action = self.State.nextPos(a)
                # print("pos action and state: ", nxt_pos, chosen_action)
            
                if chosen_action not in list(reward_acts.keys()):
                    nxt_rewards = [self.state_values[(nxt_pos[0], nxt_pos[1], 0)],
                            self.state_values[(nxt_pos[0], nxt_pos[1], 1)],
                            self.state_values[(nxt_pos[0], nxt_pos[1], 2)],
                            self.state_values[(nxt_pos[0], nxt_pos[1], 3)]]
                    nxt_reward = max(nxt_rewards)
                    reward_acts[chosen_action] = nxt_reward   
            print(reward_acts, nxt_pos)
            max_act_val = reward_acts[max(reward_acts, key=reward_acts.get)]
            max_acts = [key for key,val in reward_acts.items() if val==max_act_val]
            
            action = random.choice(max_acts) 
            print("R,np,act: ",reward_acts, nxt_pos, action)
        # print("returned action:", action)
        return action

    def takeAction(self, action):
        position, _ = self.State.nextPos(action)
        return State(state=position)

    def reset(self):
        start_state = genRandomState(GRID_ROWS, GRID_COLS , WIN_STATE, BLOCK_STATE)
        # print("new start state: ", start_state)
        self.State = State(start_state)
        self.action = self.chooseAction()
        self.states = [(self.State.state[0],self.State.state[1],self.actions_lookup[self.action])]

    def start(self, state):
        self.State = State(state)
        self.action = self.chooseAction()
        self.states = [(self.State.state[0],self.State.state[1],self.actions_lookup[self.action])]

    def play(self, num_episodes=1, num_steps = 20):
        i = 0
        curr_steps = 0
        rewards = []
        while i < num_episodes:
            # to the end of game back propagate reward
            if (self.State.isEnd) or (curr_steps == num_steps):
                # back propagate
                reward = self.State.generateReward()
                # explicitly assign end state to reward values
                self.state_values[(self.State.state[0], self.State.state[1], self.actions_lookup[self.action])] = reward
                # print("Game End Reward", reward)
                # print("all states:", self.states, len(self.states))
                # for s in reversed(self.states): #### update total reward
                #     reward = self.state_values[s] + self.alpha * (reward - self.state_values[s])
                #     self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1
                curr_steps = 0
                rewards.append(self.evaluate((1,6)))
            else:
                # print("in else of play")
                # append trace
                curr_state = self.State.state
                curr_action = self.action
                curr_reward = self.State.generateReward()
                
                self.State = self.takeAction(curr_action) ## Update to next state
                nxt_state = self.State.state
                nxt_action = self.chooseAction()
                self.action = nxt_action
                self.states.append((nxt_state[0],nxt_state[1],self.actions_lookup[nxt_action]))
                # print("nxt state: ", nxt_state, nxt_action)
                # print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                
                self.state_values[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])] = \
                            self.state_values[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])] + \
                            self.alpha*(curr_reward + \
                            self.gamma*self.state_values[(nxt_state[0], nxt_state[1], self.actions_lookup[nxt_action])] - \
                            self.state_values[(curr_state[0], curr_state[1], self.actions_lookup[curr_action])])
                # mark is end
                self.State.isEndFunc()
                curr_steps = curr_steps + 1
                # print("nxt state", self.State.state)
                
                # print("---------------------")
        return rewards

    def evaluate(self,state, num_steps = 20):
        self.start(state)
        curr_steps = 0
        while not ((self.State.isEnd) or (curr_steps == num_steps)):
            # print(self.State.isEnd, curr_steps)
            curr_action = self.action
            self.State = self.takeAction(curr_action) ## Update to next state
            nxt_state = self.State.state
            nxt_action = self.chooseAction()
            self.action = nxt_action
            self.states.append((nxt_state[0],nxt_state[1],self.actions_lookup[nxt_action]))
            # print("nxt state: ", nxt_state, nxt_action)
            # print("current position {} action {}".format(self.State.state, action))
            # by taking the action, it reaches the next state
            # mark is end
            self.State.isEndFunc()
            curr_steps = curr_steps + 1
        reward = self.State.totalReward(self.states)
        # print("Eval: all states: ", self.states)
        # print("Eval Rewards: ", reward)
        return reward
                     

    def showValues(self):
        for k in range(0, 4):

            print("%d th action: " %(k))
            for i in range(0, GRID_ROWS):
                print('----------------------------------')
                out = '|'
                for j in range(0, GRID_COLS):
                    # out += "(" + str(i) + "," + str(j) + "):" + str(self.state_values[(i, j, k)]).ljust(6) + '|'
                    out += str(self.state_values[(i, j, k)]).ljust(6) + '|'
                print(out)
        print('----------------------------------')

if __name__ == "__main__":

    #### Sample Random valid Start State
    start_state = (0,0)#genRandomState(GRID_ROWS, GRID_COLS , WIN_STATE, BLOCK_STATE)
    ag = sarsaAgent(start_state)
    num_episodes = 1000
    rewards = ag.play(num_episodes=num_episodes)
    ag.showValues()

    q_agent = qlearningAgent(start_state)
    num_episodes = 1000
    rewards_q = q_agent.play(num_episodes=num_episodes)
    q_agent.showValues()

    eval_on = [(1,7),(3,8),(2,5),(1,6),(2,8),(1,8),(4,8),(3,5),(2,6),(2,9)]
    reward_sarsa = []
    reward_qlearn =[]

    for st in eval_on:
        reward_sarsa.append(ag.evaluate(st))
        reward_qlearn.append(q_agent.evaluate(st))
    x = np.linspace(1, 10, 10)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=reward_sarsa, mode='lines+markers', name="SARSA"))
    fig.add_trace(go.Scatter(x=x, y=reward_qlearn, mode='lines+markers', name="Q-learning"))
    fig.update_layout(
    title={'text':"Reward over 5 trials",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="nth trial",
    yaxis_title="Reward",
    legend_title="N")
    
    fig.show()
    fig.write_image("reward_over_5_trials.png")

    # fig_sarsa = go.Figure()
    # fig_sarsa.add_trace(go.Scatter(x=x, y=rewards, mode='markers', name="Sarsa"))
    # fig_sarsa.update_layout(
    # title={'text':"Reward for 1000 episodes SARSA (epsilon: 0.2)",
    #         'y':0.9,
    #         'x':0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'},
    # xaxis_title="nth episode",
    # yaxis_title="Reward",
    # legend_title="N")
    
    # fig_sarsa.show()
    # fig_sarsa.write_image("reward_over_1000episodes_sarsa_2_2.png")

    # q_agent = qLearningAgent(start_state)
    # num_episodes = 1000
    # rewards_q = q_agent.play(num_episodes=num_episodes)
    
    # x = np.linspace(1, num_episodes, num_episodes)
    # fig_qlearn = go.Figure()
    # fig_qlearn.add_trace(go.Scatter(x=x, y=rewards_q, mode='markers', name="Q Learning"))
    # fig_qlearn.update_layout(
    # title={'text':"Reward for 1000 episodes q learn",
    #         'y':0.9,
    #         'x':0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'},
    # xaxis_title="nth episode",
    # yaxis_title="Reward",
    # legend_title="N")
    
    # fig_qlearn.show()
    # fig_qlearn.write_image("reward_over_5000episodes_qlearn.png")
    # ag.evaluate(state=(1,9))
    # q_agent.evaluate(state=(3,8))

    # print(ag.showValues())