%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for the map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if DRONE_TYPE == "quadcopter" || DRONE_TYPE == "point_mass"
    map.width = 300; % the map is of size (width)x(width)
    map.max_height = 100; % maximum height of trees
elseif DRONE_TYPE == "fixed_wing"
    map.width = 4000; % the map is of size (width)x(width)
    map.max_height = 100; % maximum height of trees
end

if exist('ACTIVE_ENVIRONMENT', 'var')
    map.ACTIVE_ENVIRONMENT = ACTIVE_ENVIRONMENT; % for those functions calling map
end
    
if ~exist('ACTIVE_ENVIRONMENT', 'var') || ACTIVE_ENVIRONMENT == false     
    return 
end

if exist('ACTIVE_ENVIRONMENT', 'var') && ACTIVE_ENVIRONMENT == true
    map.bl_corner_north = 0;
    map.bl_corner_east = 0;

    map.nb_blocks = 7; % the number of blocks per row
    map.street_width_perc = 0.7; % percentage of block that is empty

    map.building_width = map.width/map.nb_blocks*(1-map.street_width_perc);
    map.street_width = map.width/map.nb_blocks*map.street_width_perc;
    map.building_shape = 'cylinder';
    map.ground_stations = [];
    func_name = mfilename;

     map.filename = sprintf('%s.json', func_name);
    map.ground_station_indices = [1, 16, 20, 32, 35, 44]
    % map.building_shape = 'parallelepiped';

    map.radius_of_influence = 60;
    map = create_shifted_buildings(map);

end

