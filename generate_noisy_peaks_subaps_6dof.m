function [subA, subB, truth_data] = generate_noisy_peaks_subaps_6dof(sigma_n_um, Lc_mm, rng_seed, pose6dof)
%GENERATE_NOISY_PEAKS_SUBAPS_6DOF
% 双子孔径独立噪声版本：
% 1) 先从无噪声 peaks 母面形生成 clean 子孔径；
% 2) 再在两个子孔径各自的局部坐标系内独立生成空间相关噪声；
% 3) 将噪声沿各自局部 z 方向叠加到测量 sag 上。
%
% 该版本保留原函数名与调用接口，run_sag_noise_mc.m 无需改动。

%% 1) 输入与随机种子
if nargin > 2 && ~isempty(rng_seed)
    rng(rng_seed);
else
    rng('shuffle');
end
if nargin < 4 || isempty(pose6dof)
    pose6dof = struct();
end

%% 2) 全局参数
ds = 0.5;                 % mm
D_full = 100;             % mm
R_full = D_full / 2;
R_sub  = 30;              % mm

overlap_target = 0.50;
phi_layout_deg = 25;
target_pv_f1   = 2.5;     % mm

sigma_n = sigma_n_um * 1e-3;   % mm

%% 3) 构造 clean 母面形
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
Z_base = Z_base - (coef0(1) * Xm + coef0(2) * Ym + coef0(3));

vmax = max(abs(Z_base(mask_full)));
if vmax <= 0
    vmax = 1;
end
Z_base(mask_full) = Z_base(mask_full) / vmax;

Z_clean = Z_base;
coef1 = fit_plane_ls(Xm(mask_full), Ym(mask_full), Z_clean(mask_full));
Z_clean(mask_full) = Z_clean(mask_full) - ...
    (coef1(1) * Xm(mask_full) + coef1(2) * Ym(mask_full) + coef1(3));

pv0 = max(Z_clean(mask_full)) - min(Z_clean(mask_full));
if pv0 > 0
    Z_clean(mask_full) = Z_clean(mask_full) * (target_pv_f1 / pv0);
end

%% 4) 布局与 6DoF 真值
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

%% 5) clean 曲面插值器
Z_clean_grid = Z_clean.';
F_clean = griddedInterpolant({xv, yv}, Z_clean_grid, 'cubic', 'none');

%% 6) 先采样 clean 子孔径
subA_clean = sample_circular_subap_local_clean_6dof(F_clean, tA, RA, R_sub, ds);
subB_clean = sample_circular_subap_local_clean_6dof(F_clean, tB, RB, R_sub, ds);

%% 7) 在两个子孔径局部坐标系内独立生成相关噪声
[noiseA_grid, noiseA_mask, noiseA_stat, XnA, YnA] = ...
    generate_local_correlated_noise(R_sub, ds, sigma_n, Lc_mm);

[noiseB_grid, noiseB_mask, noiseB_stat, XnB, YnB] = ...
    generate_local_correlated_noise(R_sub, ds, sigma_n, Lc_mm);

%% 8) 将独立噪声叠加到两个子孔径
subA = attach_independent_noise_to_subap(subA_clean, noiseA_grid);
subB = attach_independent_noise_to_subap(subB_clean, noiseB_grid);

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

% 兼容字段：独立局部测量噪声无法对应为唯一的全局 noisy 母面形。
truth_data.Z_noisy = Z_clean;
truth_data.Noise   = zeros(size(Z_clean));

truth_data.noise_model = 'independent_local_correlated_measurement_noise';
truth_data.sigma_n_target_um = sigma_n_um;
truth_data.Lc_mm = Lc_mm;

truth_data.localNoiseA = struct( ...
    'X', XnA, 'Y', YnA, ...
    'mask', noiseA_mask, ...
    'noise', noiseA_grid, ...
    'rms_actual_mm', noiseA_stat.rms);

truth_data.localNoiseB = struct( ...
    'X', XnB, 'Y', YnB, ...
    'mask', noiseB_mask, ...
    'noise', noiseB_grid, ...
    'rms_actual_mm', noiseB_stat.rms);

truth_data.rmse_noise_actual_A = noiseA_stat.rms;
truth_data.rmse_noise_actual_B = noiseB_stat.rms;
truth_data.rmse_noise_actual   = sqrt(0.5 * (noiseA_stat.rms^2 + noiseB_stat.rms^2));
truth_data.rmse_noise_pair_actual = sqrt(noiseA_stat.rms^2 + noiseB_stat.rms^2);

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
function subap = sample_circular_subap_local_clean_6dof(F_clean, t, R, Rsub, ds)
xv = -Rsub : ds : Rsub;
yv = -Rsub : ds : Rsub;
[Xl, Yl] = meshgrid(xv, yv);
mask = (Xl.^2 + Yl.^2 <= Rsub^2);

u = Xl(mask);
v = Yl(mask);

[w_clean, valid] = solve_surface_intersection_along_local_z(F_clean, u, v, t, R);
u = u(valid);
v = v(valid);
w_clean = w_clean(valid);

Pw_clean = local_to_world(u, v, w_clean, t, R);

subap = struct();
subap.type = 'circle';
subap.Rsub = Rsub;
subap.ds = ds;

subap.x = u(:);
subap.y = v(:);
subap.z_clean = w_clean(:);
subap.z = w_clean(:);

subap.world_x_clean = Pw_clean(:,1);
subap.world_y_clean = Pw_clean(:,2);
subap.world_z_clean = Pw_clean(:,3);

subap.world_x = Pw_clean(:,1);
subap.world_y = Pw_clean(:,2);
subap.world_z = Pw_clean(:,3);

subap.pose_t = t(:);
subap.pose_R = R;
subap.euler_deg = rotm_to_eul_zyx_deg(R);

subap.c = t(1:2);
subap.theta = deg2rad(subap.euler_deg(3));
subap.n = numel(subap.z);
end

%% =========================================================================
function [noiseLocal, maskLocal, stats, Xloc, Yloc] = generate_local_correlated_noise(Rsub, ds, sigma_n, Lc_mm)
xv = -Rsub : ds : Rsub;
yv = -Rsub : ds : Rsub;
[Xloc, Yloc] = meshgrid(xv, yv);
maskLocal = (Xloc.^2 + Yloc.^2 <= Rsub^2);

noiseLocal = nan(size(Xloc));

if sigma_n <= 0
    noiseLocal(maskLocal) = 0;
    stats = struct('mean', 0, 'rms', 0, 'sigma_target', sigma_n, 'Lc_mm', Lc_mm);
    return;
end

sigma_px = max(Lc_mm / ds, 1e-6);
padPix   = max(8, ceil(4 * sigma_px));

ny = numel(yv) + 2 * padPix;
nx = numel(xv) + 2 * padPix;

noise_raw = randn(ny, nx);

if sigma_px < 0.25
    noise_corr_full = noise_raw;
else
    ker_half = max(3, ceil(4 * sigma_px));
    [xk, yk] = meshgrid(-ker_half:ker_half, -ker_half:ker_half);
    gk = exp(-(xk.^2 + yk.^2) / (2 * sigma_px^2));
    gk = gk / sum(gk(:));
    noise_corr_full = conv2(noise_raw, gk, 'same');
end

iy = (1 + padPix) : (padPix + numel(yv));
ix = (1 + padPix) : (padPix + numel(xv));
noise_crop = noise_corr_full(iy, ix);

tmp = noise_crop(maskLocal);
tmp = tmp - mean(tmp);

rms0 = sqrt(mean(tmp.^2));
if rms0 <= eps
    error('局部相关噪声 RMS 为零，无法归一化。');
end

noiseLocal(maskLocal) = sigma_n * tmp / rms0;

stats = struct();
stats.mean = mean(noiseLocal(maskLocal), 'omitnan');
stats.rms  = sqrt(mean(noiseLocal(maskLocal).^2, 'omitnan'));
stats.sigma_target = sigma_n;
stats.Lc_mm = Lc_mm;
end

%% =========================================================================
function subap = attach_independent_noise_to_subap(subapClean, noiseGrid)
ds   = subapClean.ds;
Rsub = subapClean.Rsub;

xv = -Rsub : ds : Rsub;
yv = -Rsub : ds : Rsub;

ix = round((subapClean.x(:) - xv(1)) / ds) + 1;
iy = round((subapClean.y(:) - yv(1)) / ds) + 1;

valid = ix >= 1 & ix <= numel(xv) & iy >= 1 & iy <= numel(yv);
if ~all(valid)
    error('局部噪声映射索引越界。');
end

lin = sub2ind(size(noiseGrid), iy, ix);
noise_vec = noiseGrid(lin);

if any(~isfinite(noise_vec))
    error('独立噪声映射失败：存在非有限噪声样本。');
end

z_noisy = subapClean.z_clean(:) + noise_vec(:);
Pw = local_to_world(subapClean.x(:), subapClean.y(:), z_noisy, ...
    subapClean.pose_t, subapClean.pose_R);

subap = subapClean;
subap.z = z_noisy;
subap.noise = noise_vec(:);

subap.world_x = Pw(:,1);
subap.world_y = Pw(:,2);
subap.world_z = Pw(:,3);

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

x0 = t(1) + r11 * u + r12 * v;
y0 = t(2) + r21 * u + r22 * v;
z0 = Fsurf(x0, y0);

valid = isfinite(z0);
w = nan(size(u));
w(valid) = (z0(valid) - t(3) - r31 * u(valid) - r32 * v(valid)) / r33;

maxIter = 15;
tol = 1e-12;

for it = 1:maxIter
    idx = find(valid);
    if isempty(idx)
        break;
    end

    x = t(1) + r11 * u(idx) + r12 * v(idx) + r13 * w(idx);
    y = t(2) + r21 * u(idx) + r22 * v(idx) + r23 * w(idx);
    z = Fsurf(x, y);

    good = isfinite(z);
    valid(idx(~good)) = false;
    idx = idx(good);

    if isempty(idx)
        break;
    end

    w_new = (z(good) - t(3) - r31 * u(idx) - r32 * v(idx)) / r33;
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
A = [x(:), y(:), ones(numel(x), 1)];
coef = A \ z(:);
end

%% =========================================================================
function d = solve_circle_sep_from_overlap(R, target_eta)
lo = 0;
hi = 2 * R;
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
if d >= 2 * R
    eta = 0;
    return;
end
if d <= 0
    eta = 1;
    return;
end
part1 = 2 * R^2 * acos(d / (2 * R));
part2 = 0.5 * d * sqrt(4 * R^2 - d^2);
eta = (part1 - part2) / (pi * R^2);
end

%% =========================================================================
function R = eul_zyx_to_rotm_deg(rx_deg, ry_deg, rz_deg)
rx = deg2rad(rx_deg);
ry = deg2rad(ry_deg);
rz = deg2rad(rz_deg);

Rx = [1, 0, 0; 0, cos(rx), -sin(rx); 0, sin(rx), cos(rx)];
Ry = [cos(ry), 0, sin(ry); 0, 1, 0; -sin(ry), 0, cos(ry)];
Rz = [cos(rz), -sin(rz), 0; sin(rz), cos(rz), 0; 0, 0, 1];

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