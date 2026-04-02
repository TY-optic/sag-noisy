function [subA, subB, truth_data] = generate_noisy_peaks_subaps_6dof(sigma_n_um, Lc_mm, rng_seed, pose6dof)
% GENERATE_NOISY_PEAKS_SUBAPS
% 生成带空间相关噪声的 peaks 母面形子孔径数据，并采用真实 6DoF 位姿生成 B 子孔径。
%
% 用法:
%   [subA, subB, truth_data] = generate_noisy_peaks_subaps(sigma_n_um, Lc_mm)
%   [subA, subB, truth_data] = generate_noisy_peaks_subaps(sigma_n_um, Lc_mm, rng_seed)
%   [subA, subB, truth_data] = generate_noisy_peaks_subaps(sigma_n_um, Lc_mm, rng_seed, pose6dof)
%
% 输入:
%   sigma_n_um : 噪声 RMS，单位 um
%   Lc_mm      : 高斯相关长度，单位 mm
%   rng_seed   : 随机种子（可选）
%   pose6dof   : 6DoF 真值结构体（可选），字段为
%                .tx_mm, .ty_mm, .tz_mm, .rx_deg, .ry_deg, .rz_deg
%                若未提供，则 tx,ty 由目标重叠率自动确定，tz/rx/ry/rz 采用默认值
%
% 输出:
%   subA, subB : 子孔径局部测量数据
%   truth_data : 真值数据与母面形数据

%% 1) 随机种子
if nargin > 2 && ~isempty(rng_seed)
    rng(rng_seed);
else
    rng('shuffle');
end

if nargin < 4
    pose6dof = struct();
end

%% 2) 全局参数
ds             = 0.5;     % mm
D_full         = 100;     % mm
R_full         = D_full / 2;
R_sub          = 30;      % mm
overlap_target = 0.50;
phi_layout_deg = 25;
target_pv_f1   = 2.5;     % mm
sigma_n        = sigma_n_um * 1e-3;   % mm

%% 3) 构造母面形
xv = -R_full : ds : R_full;
yv = -R_full : ds : R_full;
[Xm, Ym] = meshgrid(xv, yv);
Rm = hypot(Xm, Ym);
mask_full = (Rm <= R_full);

Xbar = 6 * Xm / D_full;
Ybar = 6 * Ym / D_full;
Z_base = peaks(Xbar, Ybar);
Z_base(~mask_full) = NaN;

coef0 = fit_plane_ls(Xm(mask_full), Ym(mask_full), Z_base(mask_full));
Z_base = Z_base - (coef0(1)*Xm + coef0(2)*Ym + coef0(3));

vmax = max(abs(Z_base(mask_full)));
if vmax <= 0
    vmax = 1;
end
Z_base(mask_full) = Z_base(mask_full) / vmax;

Z_clean = Z_base;
coef1 = fit_plane_ls(Xm(mask_full), Ym(mask_full), Z_clean(mask_full));
Z_clean(mask_full) = Z_clean(mask_full) - ...
    (coef1(1)*Xm(mask_full) + coef1(2)*Ym(mask_full) + coef1(3));

pv0 = max(Z_clean(mask_full)) - min(Z_clean(mask_full));
if pv0 > 0
    Z_clean(mask_full) = Z_clean(mask_full) * (target_pv_f1 / pv0);
end

%% 4) 构造空间相关噪声
noise_raw = randn(size(Xm));
noise_raw(~mask_full) = 0;

sigma_px = Lc_mm / ds;
ker_half = max(3, ceil(4*sigma_px));
[xk, yk] = meshgrid(-ker_half:ker_half, -ker_half:ker_half);
gk = exp(-(xk.^2 + yk.^2) / (2*sigma_px^2));
gk = gk / sum(gk(:));

noise_corr = conv2(noise_raw, gk, 'same');
noise_corr(~mask_full) = NaN;

tmp = noise_corr(mask_full);
tmp = tmp - mean(tmp, 'omitnan');
rms0 = sqrt(mean(tmp.^2, 'omitnan'));
if rms0 <= 0
    error('相关噪声 RMS 为零，无法归一化。');
end

Noise = nan(size(noise_corr));
Noise(mask_full) = sigma_n * tmp / rms0;

%% 5) 含噪母面形
Z_noisy = Z_clean;
Z_noisy(mask_full) = Z_clean(mask_full) + Noise(mask_full);

%% 6) 目标重叠布局与 6DoF 真值
sep = solve_circle_sep_from_overlap(R_sub, overlap_target);
phi_layout = deg2rad(phi_layout_deg);
u_layout = [cos(phi_layout); sin(phi_layout)];

cA = -0.5 * sep * u_layout;
cB_nom = 0.5 * sep * u_layout;

pose6dof = fill_default_pose6dof(pose6dof, cA, cB_nom);

tA = [cA(:); 0];
RA = eye(3);

tB = tA + [pose6dof.tx_mm; pose6dof.ty_mm; pose6dof.tz_mm];
RB = eul_zyx_to_rotm_deg(pose6dof.rx_deg, pose6dof.ry_deg, pose6dof.rz_deg);

%% 7) 构造插值器
Z_clean_grid = Z_clean.';
Z_noisy_grid = Z_noisy.';

F_clean = griddedInterpolant({xv, yv}, Z_clean_grid, 'cubic', 'none');
F_noisy = griddedInterpolant({xv, yv}, Z_noisy_grid, 'cubic', 'none');

%% 8) 采样两个子孔径
subA = sample_circular_subap_local_6dof(F_noisy, F_clean, tA, RA, R_sub, ds);
subB = sample_circular_subap_local_6dof(F_noisy, F_clean, tB, RB, R_sub, ds);

%% 9) 打包真值
truth_data = struct();

truth_data.ds = ds;
truth_data.D_full = D_full;
truth_data.R_full = R_full;
truth_data.R_sub = R_sub;
truth_data.overlap_target = overlap_target;
truth_data.phi_layout_deg = phi_layout_deg;

truth_data.Xm = Xm;
truth_data.Ym = Ym;
truth_data.mask = mask_full;
truth_data.Z_clean = Z_clean;
truth_data.Z_noisy = Z_noisy;
truth_data.Noise = Noise;

truth_data.cA = cA(:);
truth_data.cB_nom = cB_nom(:);

truth_data.A_pose = struct( ...
    't', tA, ...
    'R', RA, ...
    'eul_deg', [0, 0, 0]);

truth_data.B_pose = struct( ...
    't', tB, ...
    'R', RB, ...
    'eul_deg', [pose6dof.rx_deg, pose6dof.ry_deg, pose6dof.rz_deg]);

truth_data.true_tx = pose6dof.tx_mm;
truth_data.true_ty = pose6dof.ty_mm;
truth_data.true_tz = pose6dof.tz_mm;
truth_data.true_theta = deg2rad(pose6dof.rz_deg);
truth_data.true_rx_deg = pose6dof.rx_deg;
truth_data.true_ry_deg = pose6dof.ry_deg;
truth_data.true_rz_deg = pose6dof.rz_deg;
truth_data.true_t_mm = [pose6dof.tx_mm; pose6dof.ty_mm; pose6dof.tz_mm];
truth_data.true_eul_deg = [pose6dof.rx_deg; pose6dof.ry_deg; pose6dof.rz_deg];

truth_data.pv_clean = max(Z_clean(mask_full)) - min(Z_clean(mask_full));
truth_data.rmse_noise_actual = sqrt(mean(Noise(mask_full).^2, 'omitnan'));

end

%% =========================================================================
function pose = fill_default_pose6dof(pose, cA, cB_nom)

if ~isfield(pose, 'tx_mm') || isempty(pose.tx_mm)
    pose.tx_mm = cB_nom(1) - cA(1);
end
if ~isfield(pose, 'ty_mm') || isempty(pose.ty_mm)
    pose.ty_mm = cB_nom(2) - cA(2);
end
if ~isfield(pose, 'tz_mm') || isempty(pose.tz_mm)
    pose.tz_mm = 0.010;
end
if ~isfield(pose, 'rx_deg') || isempty(pose.rx_deg)
    pose.rx_deg = 0.020;
end
if ~isfield(pose, 'ry_deg') || isempty(pose.ry_deg)
    pose.ry_deg = -0.040;
end
if ~isfield(pose, 'rz_deg') || isempty(pose.rz_deg)
    pose.rz_deg = 2.000;
end

end

%% =========================================================================
function subap = sample_circular_subap_local_6dof(F_noisy, F_clean, t, R, Rsub, ds)

xv = -Rsub : ds : Rsub;
yv = -Rsub : ds : Rsub;
[Xl, Yl] = meshgrid(xv, yv);
mask = (Xl.^2 + Yl.^2 <= Rsub^2);

u = Xl(mask);
v = Yl(mask);

[w_noisy, valid] = solve_surface_intersection_along_local_z(F_noisy, u, v, t, R);
u = u(valid);
v = v(valid);
w_noisy = w_noisy(valid);

[w_clean, valid2] = solve_surface_intersection_along_local_z(F_clean, u, v, t, R);
u = u(valid2);
v = v(valid2);
w_noisy = w_noisy(valid2);
w_clean = w_clean(valid2);

Pw = local_to_world(u, v, w_noisy, t, R);

subap = struct();
subap.type = 'circle';
subap.Rsub = Rsub;
subap.ds = ds;

subap.x = u(:);
subap.y = v(:);
subap.z = w_noisy(:);
subap.z_clean = w_clean(:);

subap.world_x = Pw(:,1);
subap.world_y = Pw(:,2);
subap.world_z = Pw(:,3);

subap.pose_t = t(:);
subap.pose_R = R;
subap.euler_deg = rotm_to_eul_zyx_deg(R);

subap.c = t(1:2);
subap.theta = deg2rad(subap.euler_deg(3));
subap.n = numel(subap.z);

end

%% =========================================================================
function [w, valid] = solve_surface_intersection_along_local_z(Fsurf, u, v, t, R)

u = u(:);
v = v(:);

r11 = R(1,1); r12 = R(1,2); r13 = R(1,3);
r21 = R(2,1); r22 = R(2,2); r23 = R(2,3);
r31 = R(3,1); r32 = R(3,2); r33 = R(3,3);

if abs(r33) < 1e-10
    error('局部 z 轴与全局 XY 平面近乎平行，当前参数下无法稳定求交。');
end

x0 = t(1) + r11*u + r12*v;
y0 = t(2) + r21*u + r22*v;
z0 = Fsurf(x0, y0);

valid = isfinite(z0);
w = nan(size(u));

w(valid) = (z0(valid) - t(3) - r31*u(valid) - r32*v(valid)) / r33;

maxIter = 15;
tol = 1e-12;

for it = 1:maxIter
    idx = find(valid);
    if isempty(idx)
        break;
    end

    x = t(1) + r11*u(idx) + r12*v(idx) + r13*w(idx);
    y = t(2) + r21*u(idx) + r22*v(idx) + r23*w(idx);
    z = Fsurf(x, y);

    good = isfinite(z);
    valid(idx(~good)) = false;
    idx = idx(good);

    if isempty(idx)
        break;
    end

    w_new = (z(good) - t(3) - r31*u(idx) - r32*v(idx)) / r33;
    dw = max(abs(w_new - w(idx)));

    w(idx) = w_new;

    if dw < tol
        break;
    end
end

w = w(:);
valid = valid & isfinite(w);

end

%% =========================================================================
function Pw = local_to_world(u, v, w, t, R)
P = [u(:), v(:), w(:)].';
Pw = (R * P + t(:)).';
end

%% =========================================================================
function coef = fit_plane_ls(x, y, z)
A = [x(:), y(:), ones(numel(x),1)];
coef = A \ z(:);
end

%% =========================================================================
function d = solve_circle_sep_from_overlap(R, target_eta)
lo = 0;
hi = 2*R;
for it = 1:80
    mid = 0.5 * (lo + hi);
    eta_mid = circle_overlap_ratio(mid, R);
    if eta_mid > target_eta
        lo = mid;
    else
        hi = mid;
    end
end
d = 0.5 * (lo + hi);
end

%% =========================================================================
function eta = circle_overlap_ratio(d, R)
if d >= 2*R
    eta = 0;
    return;
end
if d <= 0
    eta = 1;
    return;
end
part1 = 2 * R^2 * acos(d/(2*R));
part2 = 0.5 * d * sqrt(4*R^2 - d^2);
eta = (part1 - part2) / (pi * R^2);
end

%% =========================================================================
function R = eul_zyx_to_rotm_deg(rx_deg, ry_deg, rz_deg)

rx = deg2rad(rx_deg);
ry = deg2rad(ry_deg);
rz = deg2rad(rz_deg);

Rx = [1, 0, 0;
      0, cos(rx), -sin(rx);
      0, sin(rx),  cos(rx)];

Ry = [ cos(ry), 0, sin(ry);
       0,       1, 0;
      -sin(ry), 0, cos(ry)];

Rz = [cos(rz), -sin(rz), 0;
      sin(rz),  cos(rz), 0;
      0,        0,       1];

R = Rz * Ry * Rx;
end

%% =========================================================================
function eul_deg = rotm_to_eul_zyx_deg(R)

sy = -R(3,1);
cy = sqrt(max(0, 1 - sy^2));

if cy > 1e-12
    rx = atan2(R(3,2), R(3,3));
    ry = asin(sy);
    rz = atan2(R(2,1), R(1,1));
else
    rx = atan2(-R(2,3), R(2,2));
    ry = asin(sy);
    rz = 0;
end

eul_deg = rad2deg([rx, ry, rz]);
end
