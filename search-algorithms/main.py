import pandas as pd
import numpy as np
from random import sample, randint, random
import plotly.graph_objects as go
import time as t

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def dist_mat(city_cord):
    dist_matrix = np.empty((len(city_cord),len(city_cord)))
    for i in range (0, len(city_cord)):
        for j in range (0, len(city_cord)):
            dist_matrix[i][j] = distance(city_cord[i],city_cord[j])
    return dist_matrix

def update_temperature(discount, temp):
    return discount*temp

def calculate_cost(dist_matrix, state):
    state.append(state[0])
    cost = 0.0
    i = 0
    for idx in range(0,len(state)-1):
        i = idx
        cost = cost + dist_matrix[state[idx]][state[idx+1]]
    return cost

def calculate_prob(delta_e, temperature):
    return np.exp(-delta_e/temperature)

def mutation(state):
    samp_a = randint(1,len(state)-1)
    samp_b = randint(1,len(state)-1)
    mutated_state = state.copy()
    mutated_state[samp_a] = state[samp_b]
    mutated_state[samp_b] = state[samp_a]
    return mutated_state, state

def simulated_annealing(city_cord, temperature, discount):
    tic = t.perf_counter()
    city_list = list(np.arange(len(city_cord)))
    dist_matrix = dist_mat(city_cord)
    # print('city list: %s', city_list)
    initial_state = sample(city_list,len(city_cord))
    # print("initial_state: %s", initial_state)
    curr_state = initial_state.copy()
    while (temperature>1):
        new_state, curr_state = mutation(curr_state.copy())
        cost_curr = calculate_cost(dist_matrix, curr_state.copy())
        cost_new = calculate_cost(dist_matrix, new_state.copy())
        # print(cost_curr, curr_state)
        if cost_new < cost_curr:
            curr_state = new_state.copy()
        else:
            rand_prob = random()
            delta_e = cost_new - cost_curr
            # print(calculate_prob(delta_e, temperature), rand_prob, delta_e, delta_e/temperature)
            if calculate_prob(delta_e, temperature) > rand_prob:
               curr_state = new_state.copy() 

        temperature = update_temperature(discount, temperature)
    toc = t.perf_counter()
    return curr_state, calculate_cost(dist_matrix, curr_state.copy()), np.around(toc-tic, decimals=3)

def beam_search(city_cord, k):
    tic = t.perf_counter()
    curr_arr = list(np.empty(k))
    curr_cost = list(np.empty(k))
    city_list = list(np.arange(len(city_cord)))
    dist_matrix = dist_mat(city_cord)
    for i in range(0, k):
        curr_arr[i]=sample(city_list,1)
        # print('ini_arr: %s', curr_arr)
    while (len(curr_arr[0])!=len(city_cord)):
        potential_states = []
        potential_costs = []
        for i in range(0,k):
            val_next_cities = [x for x in city_list if x not in curr_arr[i]]
            # print("valid next cities: ",val_next_cities, curr_arr[i])
            for j in range(0, len(val_next_cities)):
                new_state = curr_arr[i].copy()
                new_state.append(val_next_cities[j])
                # print(new_state)
                potential_states.append(new_state)
                potential_costs.append(calculate_cost(dist_matrix, new_state.copy()))
        # print(potential_costs, potential_states, np.array(potential_costs).argsort()[:k])
        least_cost_idx = np.array(potential_costs).argsort()[:k]
        for i in range(0,k):
            curr_arr[i]=potential_states[least_cost_idx[i]].copy()
            curr_cost[i]=potential_costs[least_cost_idx[i]].copy()
        # print("states and costs each iter: %s %s", curr_cost, curr_arr)
    toc = t.perf_counter()
    return curr_arr[0], curr_cost[0], np.around(toc-tic, decimals=3)

def evolutionary_algo (city_cord, n, k, generations):
    tic = t.perf_counter()
    curr_arr = list(np.empty(n))
    curr_costs = list(np.empty(n))
    city_list = list(np.arange(len(city_cord)))
    dist_matrix = dist_mat(city_cord)
    for i in range(0, n):
        curr_arr[i] = sample(city_list,len(city_list))
        curr_costs[i] = calculate_cost(dist_matrix, curr_arr[i].copy())

    for i in range (0, generations):
        pot_states = []
        pot_costs = []
        pot_states.extend(curr_arr)
        pot_costs.extend(curr_costs)
        for j in range (0,n):
            tmp_states = []
            tmp_costs = []
            for l in range (0,k):
                new_state,_ = mutation(curr_arr[j].copy())
                # print(new_state)
                while new_state in tmp_states:
                    new_state, _ = mutation(curr_arr[j].copy())
                tmp_states.append(new_state)
                tmp_costs.append(calculate_cost(dist_matrix, new_state.copy()))
            pot_states.extend(tmp_states)
            pot_costs.extend(tmp_costs)
        # print(pot_costs, pot_states, curr_arr, curr_costs)
        least_cost_idx = np.array(pot_costs).argsort()[:k]
        for i in range(0,k):
            curr_arr[i]=pot_states[least_cost_idx[i]].copy()
            curr_costs[i]=pot_costs[least_cost_idx[i]].copy()
        # print(curr_arr, curr_costs)
    toc = t.perf_counter()            
    return curr_arr[0], curr_costs[0], np.around(toc-tic, decimals=3)


if __name__=="__main__":

    data = np.array(pd.read_csv("hw2_data/25cities_A.csv"))
    
    # print("simulated annealing: %s", simulated_annealing(data.copy(), 10000, 0.99))

############################ Experiments for Simulated Annealing ##################################
    dict = {}
    num_iter = 20
    T = [10, 100, 1000, 10000, 100000]
    for time in T:
        costs = []
        durations = []
        for i in range (0,num_iter):
            _, cost, duration = simulated_annealing(data.copy(), time, 0.99)
            costs.append(cost)
            durations.append(duration)
        dict[time]={}
        dict[time]['costs'] = costs
        dict[time]['durations'] = durations
    # print(dict)

    #### Plot ####
    x = np.linspace(1, num_iter, num_iter)
    fig_costs = go.Figure()
    fig_durations = go.Figure()
    for key, val in dict.items():
        fig_costs.add_trace(go.Box(y=val['costs'], name=str(key)))
        fig_durations.add_trace(go.Box(y=val['durations'], name=str(key)))
        
    fig_costs.update_layout(
    title={'text':"Path costs for different starting temperature",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="nth trial",
    yaxis_title="Path Cost",
    legend_title="Starting Temperature")
    
    fig_costs.show()

    fig_durations.update_layout(
    title={'text':"Runtime for different starting temperature",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="nth trial",
    yaxis_title="duration in seconds",
    legend_title="Starting Temperature")
    
    fig_durations.show()

    

    fig_costs.write_image("images/SA_costs_wrt_temp_20trials_25cities_A.png")
    fig_durations.write_image("images/SA_runtime_wrt_temp_20trials_25cities_A.png")

    ################################## Experiments for Beam Search ########################################
    dict = {}
    num_iter = 20
    n_values = [4, 8, 10, 12, 14, 18]
    for n_val in n_values:
        costs = []
        durations = []
        for i in range (0,num_iter):
            _, cost, duration = beam_search(data.copy(), n_val)
            costs.append(cost)
            durations.append(duration)
        dict[n_val]={}
        dict[n_val]['costs'] = costs
        dict[n_val]['durations'] = durations
    # print(dict)

    #### Plot ####
    x = np.linspace(1, num_iter, num_iter)
    fig_costs = go.Figure()
    fig_durations = go.Figure()

    for key, val in dict.items():
        fig_costs.add_trace(go.Box(y=val['costs'], name=str(key)))
        fig_durations.add_trace(go.Box(y=val['durations'], name=str(key)))
    
    fig_costs.update_layout(
    title={'text':"Path costs for different k",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="nth trial",
    yaxis_title="Path Cost",
    legend_title="k")
    
    fig_costs.show()

    fig_durations.update_layout(
    title={'text':"Runtime for different k",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="nth trial",
    yaxis_title="duration in seconds",
    legend_title="k")
    
    fig_durations.show()

    fig_costs.write_image("images/BS_costs_wrt_k_20trials_25cities_A.png")
    fig_durations.write_image("images/BS_runtime_k_temp_20trials_25cities_A.png")

    ########################### Experiments for Evolutionary Algorithms ##################################

    dict = {}
    num_iter = 20
    n_values = [8, 10, 12, 14, 18]
    for n_val in n_values:
        costs = []
        durations = []
        for i in range (0,num_iter):
            _, cost, duration = evolutionary_algo(data.copy(), n_val, int(n_val/4), 1000)
            costs.append(cost)
            durations.append(duration)
        dict[n_val]={}
        dict[n_val]['costs'] = costs
        dict[n_val]['durations'] = durations
    # print(dict)

    #### Plot ####
    x = np.linspace(1, num_iter, num_iter)
    fig_costs = go.Figure()
    fig_durations = go.Figure()

    for key, val in dict.items():
        fig_costs.add_trace(go.Box(y=val['costs'], name=str(key)))
        fig_durations.add_trace(go.Box(y=val['durations'], name=str(key)))
        # fig_durations.add_trace(go.Scatter(x=x, y=val['durations'], mode='lines+markers', name=str(key)))
    
    fig_costs.update_layout(
    title={'text':"Path costs for different N",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="nth trial",
    yaxis_title="Path Cost",
    legend_title="N")
    
    fig_costs.show()

    fig_durations.update_layout(
    title={'text':"Runtime for different N",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    xaxis_title="nth trial",
    yaxis_title="duration in seconds",
    legend_title="N")
    
    fig_durations.show()

    fig_costs.write_image("images/EA_costs_wrt_n_20trials_25cities_A.png")
    fig_durations.write_image("images/EA_runtime_n_temp_20trials_25cities_A.png")
    
    # print("Beam Search: ",beam_search(data.copy(), 10))

    # print("evolutionary_algo: ", evolutionary_algo(data.copy(), 10, 3, 1000))
    

    ###### Plot Coordinates of cities ######
    # x, y = data.T
    # plt.scatter(x,y)
    # plt.xlabel("x cordinates of cities --->")
    # plt.ylabel("y cordinates of cities --->")
    # plt.show()
