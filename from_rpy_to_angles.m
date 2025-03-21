function [theta, phi, T] = from_rpy_to_angles(rpy, l)
    % 这是中间长度+orientation得到EE的
    % Assuming the initial frame is
    % [0; 0; 1]
    % [0; 1; 0]
    % [1; 0; 0]
    % theta is the bending angle
    % phi is the directional angle
    z = [0;0;1];
    y = [0;1;0];
    x = [1;0;0];
    roll = rpy(1);
    pitch = rpy(2);
    yaw = rpy(3);

    T = [0;0;0];

    R_roll = [1 0 0; 0 cosd(roll) -sind(roll); 0 sind(roll) cosd(roll)];
    R_pitch = [cosd(pitch) 0 sind(pitch); 0 1 0; -sind(pitch) 0 cosd(pitch)];
    R_yaw = [cosd(yaw) -sind(yaw) 0; sind(yaw) cosd(yaw) 0; 0 0 1];
    R = R_yaw * R_pitch * R_roll;

    n = R * z;
    p = [dot(n,x)/norm(x); dot(n, y)/norm(y) ; 0];

    % angles are in rad
    theta = acos(dot(n,z)/norm(z));
    phi = acos(dot(n,x)/norm(x)/norm(p)) - pi/2;

    xy_proj = p/norm(p)*cot((pi-theta)/2);
    r = l/theta;
    T(1) = xy_proj(1);
    T(2) = xy_proj(2);
    T(3) = sin(theta) * r;
end
