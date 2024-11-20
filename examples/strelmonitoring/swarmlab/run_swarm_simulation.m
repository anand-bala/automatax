function run_swarm_simulation(sim_time, map_name, num_agents, options)

arguments
  sim_time (1,1) {mustBeNumeric}
  map_name string
  num_agents (1,1) {mustBeNumeric}
  options.initial_pos (3,:) {mustBeNumeric}
  options.goal_pos (3,:) {mustBeNumeric}
  options.drone_type string = "point_mass"
  options.swarm_algo string = "vasarhelyi"
  options.d_ref (1,1) {mustBeNumeric} = 25
  options.v_ref (1,1) {mustBeNumeric} = 6
  options.orientation (1,1) {mustBeNumeric} = 45
  options.video logical = true
  options.debug_plot logical = true
  options.active_env logical = true
end

% Clear console and workspace
% close all;
% project_root = strcat(extractBefore(mfilename('fullpath'),mfilename),'../..');
% addpath(genpath(project_root));

nb_agents = num_agents;
pos0 = options.initial_pos;
goals = options.goal_pos;
drone_type = options.drone_type;
swarm_algo = options.swarm_algo;
video = options.video;
debug_plot = options.debug_plot;
d_ref = options.d_ref;
v_ref = options.v_ref;
orientation = options.orientation;

% Simulation options
DRONE_TYPE = drone_type;
DEBUG = debug_plot;
VIDEO = video;
SWARM_ALGORITHM = swarm_algo;
ACTIVE_ENVIRONMENT = options.active_env;

% Directory setup for results if debugging or recording video
if DEBUG || VIDEO
  results_dirname = 'results/swarm';
  date_string = string(datetime("now"),'yyyy-MM-dd-hh:mm:ss');
  results_dirname = fullfile(results_dirname, map_name);
  if ~exist(results_dirname, 'dir')
    mkdir(results_dirname)
  end
end

% Simulation parameters
p_sim.end_time = sim_time;

% Call parameter files
run('param_sim');
run('param_battery');
run('param_physics');
if strcmp(DRONE_TYPE, "fixed_wing") || strcmp(DRONE_TYPE, "quadcopter")
  run('param_drone');
elseif strcmp(DRONE_TYPE, "point_mass")
  run('param_drone');
end
run(map_name); % creates map: struct for map params

% Set swarming parameters
p_swarm.nb_agents = nb_agents;
p_swarm.d_ref = d_ref;
p_swarm.v_ref = v_ref;
p_swarm.u_ref = [-cosd(orientation), -sind(orientation), 0]';
p_swarm.Pos0 = pos0; % Initial positions for each agent
p_swarm.x_goal = goals;

% Add goal positions for each agent if provided
if exist('goals', 'var')
  p_swarm.Goals = goals;

end

% Set algorithm-specific swarm parameters
run('param_swarm');
wind = zeros(6,1);
% Initialize swarm object and variables
swarm = Swarm();
swarm.algorithm = SWARM_ALGORITHM;
for i = 1 : p_swarm.nb_agents
  swarm.add_drone(DRONE_TYPE, p_drone, p_battery, p_sim, p_physics, map);
end
swarm.set_pos(p_swarm.Pos0);

% Wind setup
wind_params.steady = wind;
% wind_params.gust = wind_gust;
% wind_params.level = wind_level;
% wind_params.gust_level = wind_gust_level;

x0 = [p_swarm.Pos0; zeros(3,p_swarm.nb_agents)];
x_history(1,:) = x0(:);
% Video and viewer setup
if VIDEO
  video_filename = strcat('swarm_simulation_', date_string);
  video_filepath = fullfile(results_dirname, video_filename);
  video = VideoWriterWithRate(video_filepath, p_sim.dt_video);
end
swarm_viewer = SwarmViewer(p_sim.dt_plot, false); % view centered on swarm
swarm_viewer.viewer_type = "agent";

%% Main simulation loop
disp('Type CTRL-C to exit');
for time = p_sim.start_time:p_sim.dt:p_sim.end_time
  % Get changes from wind parameters
  if wind_params.steady
    % Set wind if active
  end

  % Compute velocity commands from swarming algorithm
  [vel_c, collisions] = swarm.update_command(p_swarm, p_swarm.r_coll, p_sim.dt);

  % Update swarm states and plot the drones
  swarm.update_state(wind_params.steady, time);

  % Plot state variables for debugging
  if DEBUG
    swarm.plot_state(time, p_sim.end_time, ...
      1, p_sim.dt_plot, collisions, p_swarm.r_coll/2);
  end

  % Update video
  if VIDEO
    swarm_viewer.update(time, swarm, map);
    video.update(time, swarm_viewer.figure_handle);
  end
end

% Close video
if VIDEO
  video.close();
end
fontsize = 12;
close all;

if DEBUG && ~isempty(results_dirname)
  %% Plot offline viewer

  % SwarmViewerOffline(p_sim.dt_video, ...
  %     CENTER_VIEW_ON_SWARM, p_sim.dt, swarm, map);

  %% Analyse swarm state variables
  dronetrace = [];
  % Generate the filename with a timestamp
  filename = sprintf('%s/trace_%s.csv', results_dirname, date_string);
  % Open the log file for writing
  log_file = fopen(filename, 'w');
  if log_file == -1
    error('Failed to open log file. Check directory permissions and path.');
  end

  % Write header
  fprintf(log_file, 'Drone,Time,North,East,Altitude,Vx,Vy\n');
  time_history = p_sim.start_time:p_sim.dt:p_sim.end_time;
  pos_ned_history = swarm.get_pos_ned_history();
  pos_ned_history = pos_ned_history(2:end,:);
  vel_ned_history = swarm.get_vel_xyz_history();
  accel_history = [zeros(1, p_swarm.nb_agents * 3); diff(vel_ned_history, 1) / p_sim.dt];

  for t = 1:length(time_history)
    time = time_history(t);
    for drone_idx = 1:p_swarm.nb_agents
      % Collecting drone's position, velocity, attitude, and rates
      drone = swarm.drones(drone_idx);  % Assume drones are stored in swarm.drones
      % Position and velocity data
      pn = drone.pos_ned(1);   % North position (meters)
      pe = drone.pos_ned(2);   % East position (meters)
      h  = -drone.pos_ned(3);  % Altitude (meters)
      vx = drone.vel_xyz(1);   % Velocity along x-axis (meters/s)
      vy = drone.vel_xyz(2);   % Velocity along y-axis (meters/s)
      vz = drone.vel_xyz(3);   % Velocity along z-axis (meters/s)

      % Attitude (angles in degrees)
      phi   = 180 / pi * drone.attitude(1);  % Roll angle (degrees)
      theta = 180 / pi * drone.attitude(2);  % Pitch angle (degrees)
      psi   = 180 / pi * drone.attitude(3);  % Yaw angle (degrees)

      % Angular rates (degrees/s)
      p = 180 / pi * drone.rates(1);  % Rate along x-axis (degrees/s)
      q = 180 / pi * drone.rates(2);  % Rate along y-axis (degrees/s)
      r = 180 / pi * drone.rates(3);  % Rate along z-axis (degrees/s)

      % Append current drone's data to the trace
      dronetrace = [dronetrace; drone_idx, time, pn, pe, h, vx, vy];

      % Write each row of the drone trace to the CSV file
      fprintf(log_file, '%d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n', ...
        drone_idx, time, pn, pe, h, vx, vy);

      % Debug message to confirm each write
      fprintf('Logged data for Drone %d at time %.2f\n', drone_idx, time);
    end
  end

  % Close the CSV file
  fclose(log_file);

  % Save workspace
  wokspace_path = strcat(results_dirname,'/state_var');
  save(wokspace_path,'time_history','pos_ned_history','vel_ned_history', ...
    'accel_history', 'dronetrace');

  % Plot state variables
  agents_color = swarm.get_colors();
  lines_color = [];

  plot_state_offline(time_history', pos_ned_history, vel_ned_history, ...
    accel_history, agents_color, p_swarm, map, fontsize, lines_color, ...
    results_dirname);


  %% Analyse performance

  % Compute swarm performance
  [safety, order, union, alg_conn, safety_obs, min_d_obs] = ...
    compute_swarm_performance(pos_ned_history, vel_ned_history, ...
    p_swarm, results_dirname);

  % Plot performance
  [perf_handle] = plot_swarm_performance(time_history', safety, order, ...
    union, alg_conn, safety_obs, min_d_obs, p_swarm, fontsize, results_dirname);
end
disp('Simulation completed successfully');
end
