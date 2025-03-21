r = 60/sqrt(3);
v1 = [r; 0 ;0 ;1];
v2 = [-r/2; r/2*sqrt(3); 0; 1];
v3 = [-r/2; -r/2*sqrt(3); 0; 1];
vs = [v1, v2,v3];

function [tri_x, tri_y, tri_z] = get_triangle(trs)
    len_base = 60 / sqrt(3);
    tri_x = [len_base -len_base /2  -len_base / 2];
    tri_y = [0 30 -30];
    tri_z = [0 0 0];
    M = [tri_x; tri_y; tri_z; 1 1 1];
    M1 = trs * M;
    tri_x = M1(1,:);
    tri_y = M1(2,:);
    tri_z = M1(3,:);
end

function draw_panel(trs)
    [tri_x, tri_y, tri_z] = get_triangle(trs);
    fill3(tri_x, tri_y, tri_z, 'y');
end

function init()
    figure;
    % axis([-100 250 -100 150 0 300]);
    axis equal;
    hold on;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    view(3);
    grid on;
    [base_tri_x, base_tri_y, base_tri_z] = get_triangle(eye(4));
    fill3(base_tri_x, base_tri_y, base_tri_z, 'y');
end

function R = compute_rotation_matrix(phi, beta)
    % 计算矩阵元素
    a = cos(beta) * cos(phi)^2 + sin(phi)^2;
    b = (-1 + cos(beta)) * cos(phi) * sin(phi);
    c = cos(phi) * sin(beta);
    d = sin(beta) * sin(phi);
    e = cos(beta);
    f = cos(beta) * sin(phi)^2 + cos(phi)^2;

    % 构造旋转矩阵
    R = [ a,   b,  c;
         b,   f,  d;
         -c,   -d,  e];
end

function draw_circle(start_pos, start_ang, center, beta)
    % draw the initial circle
    beta_vec = linspace(0,beta,1000);
    radi = start_pos - center;
    r = norm(radi);
    ini_x = r * cos(beta_vec);
    ini_y = r * sin(beta_vec);
    ini_z = zeros(1,1000);
    
    normal = cross(radi, start_ang);
    normal = transpose(normal / norm(normal));
    new_x = transpose(radi / norm(radi));
    new_y = cross(normal, new_x);
    Rot = [new_x, new_y, normal];

    ini_cir = [ini_x; ini_y; ini_z];
    new_cir = Rot * ini_cir;
    new1 = center(1) + new_cir(1,:);
    new2 = center(2) + new_cir(2,:);
    new3 = center(3) + new_cir(3,:);

    plot3(new1, new2, new3, 'r-', 'LineWidth',2);
    plot3([center(1), new1(1000)], [center(2), new2(1000)], [center(3), new3(1000)], 'g--', 'LineWidth',1);
    plot3([center(1), new1(1)], [center(2), new2(1)], [center(3), new3(1)], 'g--', 'LineWidth',1);
end

function ptr = calc_displacement(rho, beta, phi)
    ptr = 1/rho * [(1 - cos(beta))*cos(phi); (1- cos(beta))*sin(phi); sin(beta)];
end

function center1 = find_center(start, ends, st_dir, beta)
    d = ends - start;
    d_len = norm(d);
    norms = cross(d, st_dir);
    rad = cross (st_dir, norms);
    rad = rad / norm(norms) * (d_len / 2 / sin(beta /2));
    center1 = rad + start;
    % fill3([start(1) ends(1) center1(1)], [start(2) ends(2) center1(2)], [start(3) ends(3) center1(3)], 'y');
end

% input: start/end position, start direction vector, central angle
function draw_arm_sec(st, ends, st_dir, beta)
    center = find_center(st, ends,st_dir, beta);
    draw_circle(st, st_dir, center, beta);
end

function draw_rigid(base, ptr)
    plot3([base(1), ptr(1)], [base(2), ptr(2)], [base(3), ptr(3)], ...
        'b-', 'LineWidth',2);
end

function [ends, trs] = transformed_vector(rho, beta, phi, rigid_len)
    ptr = calc_displacement(rho, beta,phi);
    rot = compute_rotation_matrix(phi, beta);
    dir = rot * [0;0;1];
    ends = ptr + dir * rigid_len;
    trs = [rot, ends;0 0 0 1];
end

function betas = get_beta(l1, l2, l3, r)
    betas = 2 * sqrt(l1*l1 + l2*l2 + l3*l3 - l1*l2 - l1*l3 -l2*l3)/ 3 / r;
end

function phis = get_phi(l1, l2, l3)
    phis = atan2(3*(l2- l3),sqrt(3)*(l2 + l3 - 2 * l1));
end

function ls = get_lks(lc, r, beta, phi)
    ls = [lc;lc;lc] - r*beta*[cos(-phi); cos(2*pi/3-phi); cos(4*pi/3-phi)];
end

init();

l11=60;l12=80;l13=100;
l21=70;l22=50;l23=70;

lc1 = (l11 + l12 + l13) / 3;
lc2 = (l21 + l22 + l23) / 3;

% parameters
phi1 = get_phi(l11, l12, l13);
phi2 = get_phi(l21, l22, l23);
% angles
beta1 = get_beta(l11, l12, l13, r);
beta2 = get_beta(l21, l22, l23, r);
% rigit body length, unit: mm
rigid_len = 49; 

rho1 = beta1 / lc1;
rho2 = beta2 / lc2;

l1 = get_lks(lc1, r, beta1, phi1);
l2 = get_lks(lc2, r, beta2, phi2);

% constants
base = [0, 0, 0];
init_dir_ptr = [0; 0; 1];

% section 1
ptr1 = calc_displacement(rho1, beta1, phi1);
rot1 = compute_rotation_matrix(phi1, beta1);
dir1 = rot1 * init_dir_ptr;
base1 = ptr1 + dir1 * rigid_len;
trs1 = [rot1, base1; 0 0 0 1];
trs1_center = [rot1, ptr1 + dir1 * rigid_len / 2; 0 0 0 1];
trs1_end = [rot1, ptr1; 0 0 0 1];

%[base1, trs1] = transformed_vector(rho1,beta1, phi1, rigid_len);

% section 2
ptr2 = calc_displacement(rho2, beta2, phi2);
ptr2_extend = [ptr2; 1];
ptr2_transformed = trs1 * ptr2_extend;
ptr2 = ptr2_transformed(1:3);
rot2 = compute_rotation_matrix(phi2, beta2);
dir2 = rot2 * dir1;
base2 = ptr2 + dir2 * rigid_len / 2;
trs2 = [rot2, base2; 0 0 0 1];
trs2_end = [rot2, ptr2; 0 0 0 1];
%[base2, trs2] = transformed_vector(rho2, beta2, phi2, rigid_len);

draw_rigid(ptr1, base1);
draw_rigid(ptr2, base2);

draw_panel(trs1);
draw_panel(trs2);
draw_panel(trs1_end);
draw_panel(trs2_end);

draw_arm_sec(base, transpose(ptr1), init_dir_ptr, beta1);
draw_arm_sec(transpose(base1), transpose(ptr2), dir1, beta2);

t1 = trs1_end*vs;
draw_arm_sec(transpose(vs(1:3,1)), transpose(t1(1:3,1)), init_dir_ptr, beta1);
draw_arm_sec(transpose(vs(1:3,2)), transpose(t1(1:3,2)), init_dir_ptr, beta1);
draw_arm_sec(transpose(vs(1:3,3)), transpose(t1(1:3,3)), init_dir_ptr, beta1);

t1_fn = trs1*vs;
t2 = transpose(trs2_end*vs);
draw_arm_sec(transpose(t1_fn(1:3,1)), t2(1,1:3), dir1, beta2);
draw_arm_sec(transpose(t1_fn(1:3,2)), t2(2,1:3), dir1, beta2);
draw_arm_sec(transpose(t1_fn(1:3,3)), t2(3,1:3), dir1, beta2);