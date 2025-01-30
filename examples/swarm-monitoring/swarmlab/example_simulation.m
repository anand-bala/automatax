close all;
% Example call with specific parameters
nb_agents = 2;
initial_positions = [0, 0, 0; 0, 50, 0]';
goal_positions = [50, 0, 0]';
map_name = 'param_map_5';

run_swarm_simulation(5.0, map_name, nb_agents, initial_pos=initial_positions, goal_pos=goal_positions)
