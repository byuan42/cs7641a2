# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:39:21 2024

@author: boyua
"""
import pandas as pd
import numpy as np
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
#Read in 100 days worth of relevant columns of csv
city = pd.read_csv("C:/Users/boyua/Documents/GHCN/ghcnd/ghcnd_georgia_stations.csv",usecols=['Station','Date','Data type','Value']).groupby(['Station','Data type']).head(100)
#Filter for temp data types
temp_data = city[city['Data type'].isin(['TMAX','TMIN','TOBS'])]
#Time and value format transformations, as well as adding season and day labels and historical averages of values
temp_data['Value'] = temp_data['Value']/10
temp_data['Date'] = temp_data['Date'].astype(str)
temp_data['Time'] = pd.to_datetime(temp_data['Date'],errors='coerce').dropna().apply(lambda date: date.toordinal())
temp_data['Month'] = temp_data['Date'].str.slice(4,6)
temp_data['Station'] = temp_data['Station'].astype(str)
temp_data['Month and day'] = temp_data['Date'].str.slice(4,8)
temp_data['Data type and station'] = temp_data['Data type'] + temp_data['Station']
temp_data['Season'] = temp_data['Month'].apply(lambda month: 'Autumn' if month in ['09', '10', '11'] else 'Winter' if month in ['12', '01', '02'] else 'Spring' if month in ['03', '04', '05'] else 'Summer' if month in ['06', '07', '08'] else 'Unknown')
temp_data['Day of the Week'] = [i.weekday()+1 if pd.notna(i) else None for i in pd.to_datetime(temp_data['Date'],format = '%Y%m%d', errors = 'coerce')]
temp_data_yearly_averages = temp_data.groupby(['Month and day','Data type','Station'])['Value'].mean().reset_index()
temp_data = pd.merge(temp_data,temp_data_yearly_averages,on=['Month and day','Data type','Station'])
temp_data.rename(columns={'Value_x':'Value','Value_y':'Historical average'},inplace=True)
temp_data_yearly_std = temp_data.groupby(['Month and day','Data type','Station'])['Value'].std().reset_index()
temp_data = pd.merge(temp_data,temp_data_yearly_std,on=['Month and day','Data type','Station'])
temp_data.rename(columns={'Value_x':'Value','Value_y':'Historical standard deviation'},inplace=True)
temp_data['Normalized value'] = (temp_data['Value'] - temp_data['Historical average'])/temp_data['Historical standard deviation']
#Checking length of values in each data type
len(temp_data[temp_data['Data type']=='TMAX'])
len(temp_data[temp_data['Data type']=='TMIN'])
len(temp_data[temp_data['Data type']=='TOBS'])
temp_data_fitted = temp_data[['Date','Data type and station','Normalized value']].dropna()
# Define a fitness function
def fitness_fn(state):
    pattern_length = len(state)
    score = 0
    state = np.array(state, dtype=float)
    # Iterate through unique data types and stations
    for station in temp_data_fitted['Data type and station'].unique():
        station_data = temp_data_fitted[temp_data_fitted['Data type and station'] == station]
        station_array = station_data[['Date', 'Normalized value']].to_numpy()
        
        # Compare the pattern with the actual data for this station
        for i in range(len(station_array) - pattern_length):
            pattern = station_array[i:i + pattern_length, 1].astype(float)  # Extract the pattern from the data
            if np.allclose(pattern, state, atol=0.1):  # Check if the pattern matches
                score += 1  # Increase score for matching pattern
    
    return score if np.isfinite(score) else -np.inf
# Define the optimization problem
problem_length = 10  # Example length of the state vector
problem = mlrose.DiscreteOpt(length=problem_length, fitness_fn=mlrose.CustomFitness(fitness_fn), maximize=True, max_val=2)
# Randomized Hill Climbing
best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=1000, random_state=1, curve=True)
print("RHC Best state:", best_state_rhc)
print("RHC Best fitness:", best_fitness_rhc)

# Simulated Annealing
schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001)
best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=100, max_iters=1000, random_state=1, curve=True)
print("SA Best state:", best_state_sa)
print("SA Best fitness:", best_fitness_sa)

# Genetic Algorithm
best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=1000, random_state=1, curve=True)
print("GA Best state:", best_state_ga)
print("GA Best fitness:", best_fitness_ga)


# Extract the iterations and fitness values for each algorithm
rhc_iterations = fitness_curve_rhc[:, 1]
rhc_fitness = fitness_curve_rhc[:, 0]
sa_iterations = fitness_curve_sa[:, 1]
sa_fitness = fitness_curve_sa[:, 0]
ga_iterations = fitness_curve_ga[:, 1]
ga_fitness = fitness_curve_ga[:, 0]

# Plot the fitness curves
plt.figure(figsize=(10, 6))
plt.plot(rhc_iterations, rhc_fitness, label='Randomized Hill Climbing')
plt.plot(sa_iterations, sa_fitness, label='Simulated Annealing')
plt.plot(ga_iterations, ga_fitness, label='Genetic Algorithm')

plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Fitness Curves for RHC, SA, and GA')
plt.legend()
plt.grid(True)
plt.show()
# Define initial weights (example: for a neural network with 10 input features and 5 neurons in the hidden layer)
initial_weights = np.random.randn(15)  # Adjust the size according to your network architecture

# Define RHC, SA, and GA optimization functions (as provided earlier)
def randomized_hill_climbing(fitness_func, initial_state, max_iters):
    current_state = initial_state
    best_state = current_state
    best_fitness = fitness_func(current_state)
    fitness_curve = [(best_fitness, 0)]
    
    for iteration in range(1, max_iters + 1):
        neighbor = current_state + np.random.normal(0, 0.1, size=current_state.shape)
        neighbor_fitness = fitness_func(neighbor)
        
        if neighbor_fitness > best_fitness:
            best_fitness = neighbor_fitness
            best_state = neighbor
        
        fitness_curve.append((best_fitness, iteration))
        
        # Debugging output
        print(f"Iteration {iteration}: Best Fitness = {best_fitness}")
    
    print("RHC Fitness Curve:", fitness_curve)
    return best_state, best_fitness, fitness_curve

def simulated_annealing(fitness_func, initial_state, initial_temp, min_temp, alpha, max_iters):
    current_state = initial_state
    current_temp = initial_temp
    best_state = current_state
    best_fitness = fitness_func(current_state)
    fitness_curve = [(best_fitness, 0)]
    
    for iteration in range(1, max_iters + 1):
        if current_temp < min_temp:
            break
        
        neighbor = current_state + np.random.normal(0, 0.1, size=current_state.shape)
        neighbor_fitness = fitness_func(neighbor)
        delta_fitness = neighbor_fitness - best_fitness
        
        if delta_fitness > 0 or np.random.rand() < np.exp(delta_fitness / current_temp):
            current_state = neighbor
            best_fitness = neighbor_fitness
        
        fitness_curve.append((best_fitness, iteration))
        current_temp *= alpha
        
        # Debugging output
        print(f"Iteration {iteration}: Best Fitness = {best_fitness}")
    
    print("SA Fitness Curve:", fitness_curve)
    return best_state, best_fitness, fitness_curve

def genetic_algorithm(fitness_func, population_size, crossover_rate, mutation_rate, max_generations):
    def select_parents(population, fitnesses):
        fitnesses = np.where(np.isnan(fitnesses), -np.inf, fitnesses)  # Replace NaN values
        fitnesses = np.where(np.isinf(fitnesses), -np.inf, fitnesses)  # Replace infinite values
        fitness_sum = fitnesses.sum()
        if fitness_sum == 0 or np.isnan(fitness_sum) or np.isinf(fitness_sum):
            fitnesses = np.ones_like(fitnesses)  # If all fitnesses are bad, set equal probability
        else:
            fitnesses = fitnesses / fitness_sum  # Normalize fitness values to sum to 1
        if not np.isclose(fitnesses.sum(), 1.0):
            fitnesses = np.ones_like(fitnesses) / len(fitnesses)  # Ensure sum to 1 if normalization fails
        selected_indices = np.random.choice(range(len(population)), size=population_size, p=fitnesses).astype(int)
        return np.array(population)[selected_indices]
    
    def crossover(parent1, parent2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1, parent2
    
    def mutate(individual):
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] = individual[i] + np.random.normal(0, 0.1)
        return individual
    
    population = [np.random.randn(*initial_weights.shape) for _ in range(population_size)]
    best_weights = population[0]
    best_fitness = fitness_func(best_weights)
    fitness_curve = [(best_fitness, 0)]
    
    for generation in range(1, max_generations + 1):
        fitnesses = np.array([fitness_func(ind) for ind in population])
        
        # Replace NaN and infinite values with negative infinity
        fitnesses = np.where(np.isnan(fitnesses), -np.inf, fitnesses)
        fitnesses = np.where(np.isinf(fitnesses), -np.inf, fitnesses)
        
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best_weights = population[best_idx]
        
        selected_parents = select_parents(population, fitnesses)
        next_population = []
        
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_parents[i], selected_parents[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))
        
        population = next_population
        fitness_curve.append((best_fitness, generation))
        
        # Debugging output
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
    
    print("GA Fitness Curve:", fitness_curve)
    return best_weights, best_fitness, fitness_curve

# Execute the algorithms
rhc_best_weights, rhc_best_fitness, rhc_fitness_curve = randomized_hill_climbing(fitness_fn, initial_weights, max_iters=1000)
sa_best_weights, sa_best_fitness, sa_fitness_curve = simulated_annealing(fitness_fn, initial_weights, initial_temp=1000, min_temp=0.1, alpha=0.95, max_iters=20000)
ga_best_weights, ga_best_fitness, ga_fitness_curve = genetic_algorithm(fitness_fn, population_size=50, crossover_rate=0.8, mutation_rate=0.02, max_generations=1000)

# Extract fitness curves for plotting
rhc_iterations, rhc_fitness_values = zip(*rhc_fitness_curve)
sa_iterations, sa_fitness_values = zip(*sa_fitness_curve)
ga_iterations, ga_fitness_values = zip(*ga_fitness_curve)

# Plot the fitness curves
plt.figure(figsize=(10, 6))
plt.plot(rhc_iterations, rhc_fitness_values, label='Randomized Hill Climbing')
plt.plot(sa_iterations, sa_fitness_values, label='Simulated Annealing')
plt.plot(ga_iterations, ga_fitness_values, label='Genetic Algorithm')

plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Fitness Curves for RHC, SA, and GA')
plt.legend()
plt.grid(True)
plt.show()