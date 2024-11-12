function map = create_shifted_buildings(map)
% CREATE_SHIFTED_BUILDINGS - Create the city buildings in a shifted grid.
% The heights are fixed.

    for i = 1:map.nb_blocks
        buildings_north(i) = 0.5*map.width/map.nb_blocks*(2*(i-1)+1);
    end
    buildings_north = buildings_north';

    map.buildings_east = [];
    map.buildings_north = [];
    for i = 1:map.nb_blocks
        if mod(i,2) == 1
            map.buildings_north = [map.buildings_north; repmat(buildings_north(i),map.nb_blocks,1)];
            map.buildings_east = [map.buildings_east; buildings_north];
        else
            map.buildings_north = [map.buildings_north; repmat(buildings_north(i),map.nb_blocks-1,1)];
            map.buildings_east = [map.buildings_east; ...
                buildings_north(1:(end-1))+(map.building_width+map.street_width)/2 ];
        end
    end

    nb_buildings = length(map.buildings_east);
    map.buildings_heights = map.max_height*ones(nb_buildings,1);

    
    %  ground station positions
    map.ground_stations.north = map.buildings_north(map.ground_station_indices);
    map.ground_stations.east = map.buildings_east(map.ground_station_indices);
    map.ground_stations.heights = map.buildings_heights(map.ground_station_indices);
    
    json_data = struct();

% Ground station data in json
json_data.ground_stations = struct();
for i = 1:length(map.ground_station_indices)
    ground_id = map.ground_station_indices(i);
    json_data.ground_stations(i).id = ground_id;
    json_data.ground_stations(i).north = map.buildings_north(ground_id);
    json_data.ground_stations(i).east = map.buildings_east(ground_id);
end

% Building data in json
json_data.buildings = struct();
for i = 1:length(map.buildings_east)
    json_data.buildings(i).id = i;
    json_data.buildings(i).north = map.buildings_north(i);
    json_data.buildings(i).east = map.buildings_east(i);
end

%  map data in json
json_data.map_properties = struct();
json_data.map_properties.nb_blocks = map.nb_blocks;
json_data.map_properties.street_width_perc = map.street_width_perc;
json_data.map_properties.building_width = map.building_width;
json_data.map_properties.street_width = map.street_width;

json_text = jsonencode(json_data);

fid = fopen(map.filename, 'w');
if fid == -1
    error('Could not open file for writing: %s', map.filename);
end
fprintf(fid, '%s', json_text);
fclose(fid);

    
end