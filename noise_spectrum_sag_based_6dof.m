%% run_noise_spectrum_sag_based_6dof.m
% Sag-based 6DoF：噪声频谱敏感性
% 固定噪声 RMS 幅值 sigma_n = 0.5 um，扫描相关长度 Lc
%
% 依赖：
%   1) generate_noisy_peaks_subaps.m
%      或 generate_noisy_peaks_subaps_6dof.m
%   2) Optimization Toolbox（可选，用于 lsqnonlin；若无则自动回退 fminsearch）
%
% 输出：
%   figures/noise_spectrum_sag6dof/noise_spectrum_sag6dof.pdf
%   figures/noise_spectrum_sag6dof/noise_spectrum_sag6dof.png
%   figures/noise_spectrum_sag6dof/noise_spectrum_sag6dof_results.mat
%   figures/noise_spectrum_sag6dof/noise_spectrum_sag6dof_summary.csv

clear; clc; close all;

%% ========================= 基本设置 ======================================
cfg = struct();

% 生成器函数名：若你已将 6DoF 版重命名覆盖原函数，可改成 generate_noisy_peaks_subaps
cfg.generatorFcn = 'generate_noisy_peaks_subaps_6dof';

% 固定噪声幅值
cfg.noise.sigma_n_um = 1;

% 扫描相关长度
cfg.noise.LcList_mm = [0.2, 0.5, 1.0,  2.0,  4.0,  8.0];

% Monte Carlo
cfg.mc.N = 20;
cfg.mc.baseSeed = 20260331;

% 6DoF 真值位姿（与前面 intensity 脚本保持一致）
cfg.pose6dof = struct( ...
    'tx_mm',  [], ...
    'ty_mm',  [], ...
    'tz_mm',  0.010, ...
    'rx_deg', 0.020, ...
    'ry_deg', -0.040, ...
    'rz_deg', 0.200);

% nominal pose 采用真值邻域，隔离“搜索初始化”对频谱敏感性的干扰
cfg.nominal = struct();
cfg.nominal.centerMode = 'truth_plus_delta';
cfg.nominal.dtx_mm  =  0.002;
cfg.nominal.dty_mm  = -0.0015;
cfg.nominal.dtz_mm  =  0.0010;
cfg.nominal.drx_deg =  0.0010;
cfg.nominal.dry_deg = -0.0010;
cfg.nominal.drz_deg =  0.0050;

% 搜索参数
cfg.search = struct();
cfg.search.nSearchSamples  = 2500;
cfg.search.nRefineSamples  = 4500;
cfg.search.invalidResidual_mm = 0.05;

cfg.search.coarseYaw.tx_span_mm  = 0.015;
cfg.search.coarseYaw.ty_span_mm  = 0.015;
cfg.search.coarseYaw.rz_span_deg = 0.050;
cfg.search.coarseYaw.nTx = 5;
cfg.search.coarseYaw.nTy = 5;
cfg.search.coarseYaw.nRz = 7;

cfg.search.coarseTilt.rx_span_deg = 0.030;
cfg.search.coarseTilt.ry_span_deg = 0.030;
cfg.search.coarseTilt.nRx = 7;
cfg.search.coarseTilt.nRy = 7;

cfg.search.refine.useLsqnonlin = true;
cfg.search.refine.maxIter = 80;
cfg.search.refine.maxFunEvals = 3000;
cfg.search.refine.lb_delta = [-0.020, -0.020, -0.020, -0.050, -0.050, -0.100];
cfg.search.refine.ub_delta = [ 0.020,  0.020,  0.020,  0.050,  0.050,  0.100];

% 评价参数
cfg.eval = struct();
cfg.eval.inner_erode_px = 3;

% 输出目录
cfg.io.outDir = fullfile('figures', 'noise_spectrum_sag6dof');
if ~exist(cfg.io.outDir, 'dir')
    mkdir(cfg.io.outDir);
end

%% ========================= 主循环 ========================================
nLc = numel(cfg.noise.LcList_mm);
Nmc = cfg.mc.N;

allRes = cell(nLc, 1);

fprintf('\n===================== Sag-based 6DoF noise-spectrum =====================\n');
fprintf('sigma_n = %.4f um, N = %d\n', cfg.noise.sigma_n_um, Nmc);
fprintf('Generator = %s\n', cfg.generatorFcn);
fprintf('=========================================================================\n');

trialTemplate = struct( ...
    'rngSeed', [], ...
    'Lc_mm', [], ...
    'sigma_n_set_um', [], ...
    'sigma_n_actual_um', [], ...
    'pose_est', [], ...
    'pose_nominal', [], ...
    'reg', [], ...
    'eval', [], ...
    'solve_time_s', []);

for iLc = 1:nLc
    Lc_mm = cfg.noise.LcList_mm(iLc);

    fprintf('\n------------------------------------------------------------\n');
    fprintf('Lc %d/%d : L_c = %.4f mm\n', iLc, nLc, Lc_mm);
    fprintf('------------------------------------------------------------\n');

    trial = repmat(trialTemplate, Nmc, 1);

    for k = 1:Nmc
        rngSeed = cfg.mc.baseSeed + 1000*iLc + k;

        [subA_raw, subB_raw, truth] = feval( ...
            cfg.generatorFcn, ...
            cfg.noise.sigma_n_um, ...
            Lc_mm, ...
            rngSeed, ...
            cfg.pose6dof);

        A = build_ref_subap_world(subA_raw, truth.A_pose);
        B = build_mov_subap_local(subB_raw);

        nominalPose = make_nominal_pose(truth.B_pose, cfg.nominal);

        tStart = tic;
        reg = sag_register_6dof(A, B, nominalPose, cfg);
        solveTime = toc(tStart);

        evalRes = evaluate_registration(A, B, truth, reg.pose, cfg);

        trial(k).rngSeed = rngSeed;
        trial(k).Lc_mm = Lc_mm;
        trial(k).sigma_n_set_um = cfg.noise.sigma_n_um;
        trial(k).sigma_n_actual_um = truth.rmse_noise_actual * 1e3;

        trial(k).pose_est = reg.pose;
        trial(k).pose_nominal = nominalPose;
        trial(k).reg = reg;
        trial(k).eval = evalRes;
        trial(k).solve_time_s = solveTime;

        fprintf('  MC %02d/%02d | fused = %8.4f um | inner = %8.4f um | overlap = %8.4f um | e_t = %7.4f um | e_R = %8.6f deg | t = %.3f s\n', ...
            k, Nmc, ...
            evalRes.fused_RMSE_um, ...
            evalRes.inner_RMSE_um, ...
            evalRes.overlap_RMSE_um, ...
            evalRes.e_t_um, ...
            evalRes.e_R_deg, ...
            solveTime);
    end

    allRes{iLc} = summarize_one_Lc(trial);
end

%% ========================= 汇总表 ========================================
summaryTbl = build_summary_table(allRes);
disp(' ');
disp('===================== Summary table =====================');
disp(summaryTbl);

%% ========================= 保存数据 ======================================
save(fullfile(cfg.io.outDir, 'noise_spectrum_sag6dof_results.mat'), ...
    'cfg', 'allRes', 'summaryTbl');

writetable(summaryTbl, fullfile(cfg.io.outDir, 'noise_spectrum_sag6dof_summary.csv'));

%% ========================= 绘图 ==========================================
fig = figure('Color', 'w', 'Position', [120 120 900 650]);
ax = axes(fig); 
hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');

x = summaryTbl.Lc_set_mm;
y = summaryTbl.fused_RMSE_mean_um;
e = summaryTbl.fused_RMSE_std_um;

errorbar(ax, x, y, e, 'o-', ...
    'LineWidth', 1.6, ...
    'MarkerSize', 9, ...
    'CapSize', 6);

set(ax, 'XScale', 'log');
xlim(ax, [0.15, 10]);
xticks(ax, [0.2 0.5 1 2 5 10]);
xticklabels(ax, {'0.2','0.5','1','2','5','10'});
ax.XMinorGrid = 'off';

xlabel(ax, '$L_c\ (\mathrm{mm})$', 'Interpreter', 'latex');
ylabel(ax, 'Fused RMSE ($\mu$m)', 'Interpreter', 'latex');
title(ax, sprintf('Sag-based 6DoF noise-spectrum sensitivity ($\\sigma_n = %.2f\\,\\mu$m, N = %d)', ...
    cfg.noise.sigma_n_um, Nmc), 'Interpreter', 'latex');

set(ax, 'FontName', 'Times New Roman', 'FontSize', 16, 'LineWidth', 1.0);

exportgraphics(fig, fullfile(cfg.io.outDir, 'noise_spectrum_sag6dof.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(cfg.io.outDir, 'noise_spectrum_sag6dof.png'), ...
    'Resolution', 300);

%% ========================= 结束 ==========================================
fprintf('\nSaved to:\n  %s\n', cfg.io.outDir);


%% ========================================================================
function A = build_ref_subap_world(subA_raw, poseA)
A = struct();

A.x_local = subA_raw.x(:);
A.y_local = subA_raw.y(:);
A.z_local = subA_raw.z(:);

A.t = poseA.t(:);
A.R = poseA.R;

if isfield(subA_raw, 'world_x') && isfield(subA_raw, 'world_y') && isfield(subA_raw, 'world_z')
    A.xw = subA_raw.world_x(:);
    A.yw = subA_raw.world_y(:);
    A.zw = subA_raw.world_z(:);
else
    Pw = transform_local_pts_to_world([A.x_local, A.y_local, A.z_local], A.t, A.R);
    A.xw = Pw(:,1);
    A.yw = Pw(:,2);
    A.zw = Pw(:,3);
end

A.Fworld = scatteredInterpolant(A.xw, A.yw, A.zw, 'natural', 'none');
end

%% ========================================================================
function B = build_mov_subap_local(subB_raw)
B = struct();
B.x_local = subB_raw.x(:);
B.y_local = subB_raw.y(:);
B.z_local = subB_raw.z(:);
B.n = numel(B.z_local);
end

%% ========================================================================
function pose = make_nominal_pose(truthPose, nominalCfg)
pose = truthPose;

switch lower(nominalCfg.centerMode)
    case 'truth_plus_delta'
        pose.t = truthPose.t(:) + [nominalCfg.dtx_mm; nominalCfg.dty_mm; nominalCfg.dtz_mm];
        eul = truthPose.eul_deg(:) + [nominalCfg.drx_deg; nominalCfg.dry_deg; nominalCfg.drz_deg];
        pose.eul_deg = eul(:).';
        pose.R = eul_zyx_to_rotm_deg(eul(1), eul(2), eul(3));
    otherwise
        error('未知 nominal.centerMode: %s', nominalCfg.centerMode);
end
end

%% ========================================================================
function reg = sag_register_6dof(A, B, nominalPose, cfg)

idxSearch = make_sample_indices(B.n, cfg.search.nSearchSamples);
idxRefine = make_sample_indices(B.n, cfg.search.nRefineSamples);

bestPose = nominalPose;
bestObj = sag_obj_full_6dof(bestPose, A, B, idxSearch);

% ---------- Stage 1: (tx, ty, rz) ----------
txList = linspace(bestPose.t(1) - cfg.search.coarseYaw.tx_span_mm, ...
                  bestPose.t(1) + cfg.search.coarseYaw.tx_span_mm, ...
                  cfg.search.coarseYaw.nTx);
tyList = linspace(bestPose.t(2) - cfg.search.coarseYaw.ty_span_mm, ...
                  bestPose.t(2) + cfg.search.coarseYaw.ty_span_mm, ...
                  cfg.search.coarseYaw.nTy);
rz0 = bestPose.eul_deg(3);
rzList = linspace(rz0 - cfg.search.coarseYaw.rz_span_deg, ...
                  rz0 + cfg.search.coarseYaw.rz_span_deg, ...
                  cfg.search.coarseYaw.nRz);

rxFix = bestPose.eul_deg(1);
ryFix = bestPose.eul_deg(2);

for itx = 1:numel(txList)
    for ity = 1:numel(tyList)
        for irz = 1:numel(rzList)
            pose = bestPose;
            pose.t(1) = txList(itx);
            pose.t(2) = tyList(ity);
            pose.eul_deg = [rxFix, ryFix, rzList(irz)];
            pose.R = eul_zyx_to_rotm_deg(pose.eul_deg(1), pose.eul_deg(2), pose.eul_deg(3));

            pose.t(3) = estimate_best_tz(A, B, pose, idxSearch);

            obj = sag_obj_full_6dof(pose, A, B, idxSearch);
            if obj < bestObj
                bestObj = obj;
                bestPose = pose;
            end
        end
    end
end

% ---------- Stage 2: (rx, ry) ----------
rxList = linspace(bestPose.eul_deg(1) - cfg.search.coarseTilt.rx_span_deg, ...
                  bestPose.eul_deg(1) + cfg.search.coarseTilt.rx_span_deg, ...
                  cfg.search.coarseTilt.nRx);
ryList = linspace(bestPose.eul_deg(2) - cfg.search.coarseTilt.ry_span_deg, ...
                  bestPose.eul_deg(2) + cfg.search.coarseTilt.ry_span_deg, ...
                  cfg.search.coarseTilt.nRy);

rzFix = bestPose.eul_deg(3);
txFix = bestPose.t(1);
tyFix = bestPose.t(2);

for irx = 1:numel(rxList)
    for iry = 1:numel(ryList)
        pose = bestPose;
        pose.t(1) = txFix;
        pose.t(2) = tyFix;
        pose.eul_deg = [rxList(irx), ryList(iry), rzFix];
        pose.R = eul_zyx_to_rotm_deg(pose.eul_deg(1), pose.eul_deg(2), pose.eul_deg(3));

        pose.t(3) = estimate_best_tz(A, B, pose, idxSearch);

        obj = sag_obj_full_6dof(pose, A, B, idxSearch);
        if obj < bestObj
            bestObj = obj;
            bestPose = pose;
        end
    end
end

% ---------- Stage 3: 6DoF refine ----------
p0 = pose_to_vec(bestPose);

lb = p0 + [cfg.search.refine.lb_delta(1:3), cfg.search.refine.lb_delta(4:6)];
ub = p0 + [cfg.search.refine.ub_delta(1:3), cfg.search.refine.ub_delta(4:6)];

resFun = @(p) residual_vector_6dof_fixed(p, A, B, idxRefine, cfg);

useLsq = cfg.search.refine.useLsqnonlin && exist('lsqnonlin', 'file') == 2;
if useLsq
    opts = optimoptions('lsqnonlin', ...
        'Display', 'off', ...
        'FunctionTolerance', 1e-14, ...
        'StepTolerance', 1e-14, ...
        'OptimalityTolerance', 1e-14, ...
        'MaxFunctionEvaluations', cfg.search.refine.maxFunEvals, ...
        'MaxIterations', cfg.search.refine.maxIter);
    try
        pOpt = lsqnonlin(resFun, p0, lb, ub, opts);
    catch
        objFun = @(p) sum(resFun(project_to_box(p, lb, ub)).^2);
        opts2 = optimset('Display', 'off', ...
            'MaxFunEvals', cfg.search.refine.maxFunEvals, ...
            'MaxIter', cfg.search.refine.maxIter, ...
            'TolX', 1e-12, ...
            'TolFun', 1e-12);
        pOpt = fminsearch(objFun, p0, opts2);
        pOpt = project_to_box(pOpt, lb, ub);
    end
else
    objFun = @(p) sum(resFun(project_to_box(p, lb, ub)).^2);
    opts2 = optimset('Display', 'off', ...
        'MaxFunEvals', cfg.search.refine.maxFunEvals, ...
        'MaxIter', cfg.search.refine.maxIter, ...
        'TolX', 1e-12, ...
        'TolFun', 1e-12);
    pOpt = fminsearch(objFun, p0, opts2);
    pOpt = project_to_box(pOpt, lb, ub);
end

bestPose = vec_to_pose(pOpt);
bestPose.t(3) = estimate_best_tz(A, B, bestPose, idxRefine);
bestObj = sag_obj_full_6dof(bestPose, A, B, idxRefine);

reg = struct();
reg.pose = bestPose;
reg.obj = bestObj;
reg.idxSearch = idxSearch;
reg.idxRefine = idxRefine;
end

%% ========================================================================
function idx = make_sample_indices(n, nTarget)
if n <= nTarget
    idx = (1:n).';
else
    idx = unique(round(linspace(1, n, nTarget))).';
end
end

%% ========================================================================
function tz = estimate_best_tz(A, B, pose, idx)

ptsLocal = [B.x_local(idx), B.y_local(idx), B.z_local(idx)];
R = pose.R;

Pw0 = transform_local_pts_to_world(ptsLocal, [pose.t(1); pose.t(2); 0], R);
zA = A.Fworld(Pw0(:,1), Pw0(:,2));

valid = isfinite(zA);
if ~any(valid)
    tz = pose.t(3);
    return;
end

res0 = zA(valid) - Pw0(valid,3);
tz = mean(res0);
end

%% ========================================================================
function obj = sag_obj_full_6dof(pose, A, B, idx)

ptsLocal = [B.x_local(idx), B.y_local(idx), B.z_local(idx)];
Pw = transform_local_pts_to_world(ptsLocal, pose.t, pose.R);

zA = A.Fworld(Pw(:,1), Pw(:,2));
valid = isfinite(zA);

if nnz(valid) < max(20, round(0.15*numel(idx)))
    obj = inf;
    return;
end

res = zA(valid) - Pw(valid,3);
obj = sqrt(mean(res.^2));
end

%% ========================================================================
function r = residual_vector_6dof_fixed(p, A, B, idx, cfg)

pose = vec_to_pose(p);
pose.t(3) = estimate_best_tz(A, B, pose, idx);

ptsLocal = [B.x_local(idx), B.y_local(idx), B.z_local(idx)];
Pw = transform_local_pts_to_world(ptsLocal, pose.t, pose.R);

zA = A.Fworld(Pw(:,1), Pw(:,2));
r = cfg.search.invalidResidual_mm * ones(numel(idx), 1);

valid = isfinite(zA);
r(valid) = zA(valid) - Pw(valid,3);
end

%% ========================================================================
function out = evaluate_registration(A, B, truth, poseEst, cfg)

% A 世界面
FA = A.Fworld;

% B 估计世界面
PwB = transform_local_pts_to_world([B.x_local, B.y_local, B.z_local], poseEst.t, poseEst.R);
FB = scatteredInterpolant(PwB(:,1), PwB(:,2), PwB(:,3), 'natural', 'none');

X = truth.Xm;
Y = truth.Ym;
truthMask = truth.mask;

zA = FA(X, Y);
zB = FB(X, Y);

hasA = isfinite(zA);
hasB = isfinite(zB);

fused = nan(size(X));
fused(hasA & ~hasB) = zA(hasA & ~hasB);
fused(~hasA & hasB) = zB(~hasA & hasB);
fused(hasA & hasB)  = 0.5 * (zA(hasA & hasB) + zB(hasA & hasB));

fusedMask = truthMask & isfinite(fused);
overlapMask = truthMask & hasA & hasB;
innerMask = binary_erode(fusedMask, cfg.eval.inner_erode_px);

err = fused - truth.Z_clean;

out = struct();
out.fusedMask = fusedMask;
out.overlapMask = overlapMask;
out.innerMask = innerMask;
out.errMap = err;
out.fused = fused;

out.fused_RMSE_mm = calc_rmse(err(fusedMask));
out.inner_RMSE_mm = calc_rmse(err(innerMask));
out.overlap_RMSE_mm = calc_rmse((zA(overlapMask) - zB(overlapMask)));

out.fused_RMSE_um = 1e3 * out.fused_RMSE_mm;
out.inner_RMSE_um = 1e3 * out.inner_RMSE_mm;
out.overlap_RMSE_um = 1e3 * out.overlap_RMSE_mm;

out.e_t_um = 1e3 * norm(poseEst.t(:) - truth.B_pose.t(:));
out.e_R_deg = rotation_geodesic_error_deg(poseEst.R, truth.B_pose.R);
end

%% ========================================================================
function s = summarize_one_Lc(trial)

Lc_mm = trial(1).Lc_mm;
sigmaSet = trial(1).sigma_n_set_um;

sigmaActual = arrayfun(@(t) t.sigma_n_actual_um, trial).';
fused = arrayfun(@(t) t.eval.fused_RMSE_um, trial).';
inner = arrayfun(@(t) t.eval.inner_RMSE_um, trial).';
overlap = arrayfun(@(t) t.eval.overlap_RMSE_um, trial).';
et = arrayfun(@(t) t.eval.e_t_um, trial).';
eR = arrayfun(@(t) t.eval.e_R_deg, trial).';
ts = arrayfun(@(t) t.solve_time_s, trial).';

s = struct();
s.Lc_set_mm = Lc_mm;
s.sigma_n_set_um = sigmaSet;

s.Lc_actual_mean_mm = Lc_mm;
s.Lc_actual_std_mm = 0;

s.sigma_n_actual_mean_um = mean(sigmaActual);
s.sigma_n_actual_std_um = std(sigmaActual, 0);

s.fused_RMSE_mean_um = mean(fused);
s.fused_RMSE_std_um = std(fused, 0);

s.inner_RMSE_mean_um = mean(inner);
s.inner_RMSE_std_um = std(inner, 0);

s.overlap_RMSE_mean_um = mean(overlap);
s.overlap_RMSE_std_um = std(overlap, 0);

s.e_t_mean_um = mean(et);
s.e_t_std_um = std(et, 0);

s.e_R_mean_deg = mean(eR);
s.e_R_std_deg = std(eR, 0);

s.solve_time_mean_s = mean(ts);
s.solve_time_std_s = std(ts, 0);

s.trial = trial;
end

%% ========================================================================
function T = build_summary_table(allRes)

n = numel(allRes);

Lc_set_mm = zeros(n,1);
sigma_n_set_um = zeros(n,1);
Lc_actual_mean_mm = zeros(n,1);
Lc_actual_std_mm = zeros(n,1);
sigma_n_actual_mean_um = zeros(n,1);
sigma_n_actual_std_um = zeros(n,1);
fused_RMSE_mean_um = zeros(n,1);
fused_RMSE_std_um = zeros(n,1);
inner_RMSE_mean_um = zeros(n,1);
inner_RMSE_std_um = zeros(n,1);
overlap_RMSE_mean_um = zeros(n,1);
overlap_RMSE_std_um = zeros(n,1);
e_t_mean_um = zeros(n,1);
e_t_std_um = zeros(n,1);
e_R_mean_deg = zeros(n,1);
e_R_std_deg = zeros(n,1);
solve_time_mean_s = zeros(n,1);
solve_time_std_s = zeros(n,1);

for i = 1:n
    s = allRes{i};

    Lc_set_mm(i) = s.Lc_set_mm;
    sigma_n_set_um(i) = s.sigma_n_set_um;
    Lc_actual_mean_mm(i) = s.Lc_actual_mean_mm;
    Lc_actual_std_mm(i) = s.Lc_actual_std_mm;
    sigma_n_actual_mean_um(i) = s.sigma_n_actual_mean_um;
    sigma_n_actual_std_um(i) = s.sigma_n_actual_std_um;
    fused_RMSE_mean_um(i) = s.fused_RMSE_mean_um;
    fused_RMSE_std_um(i) = s.fused_RMSE_std_um;
    inner_RMSE_mean_um(i) = s.inner_RMSE_mean_um;
    inner_RMSE_std_um(i) = s.inner_RMSE_std_um;
    overlap_RMSE_mean_um(i) = s.overlap_RMSE_mean_um;
    overlap_RMSE_std_um(i) = s.overlap_RMSE_std_um;
    e_t_mean_um(i) = s.e_t_mean_um;
    e_t_std_um(i) = s.e_t_std_um;
    e_R_mean_deg(i) = s.e_R_mean_deg;
    e_R_std_deg(i) = s.e_R_std_deg;
    solve_time_mean_s(i) = s.solve_time_mean_s;
    solve_time_std_s(i) = s.solve_time_std_s;
end

T = table( ...
    Lc_set_mm, ...
    sigma_n_set_um, ...
    Lc_actual_mean_mm, ...
    Lc_actual_std_mm, ...
    sigma_n_actual_mean_um, ...
    sigma_n_actual_std_um, ...
    fused_RMSE_mean_um, ...
    fused_RMSE_std_um, ...
    inner_RMSE_mean_um, ...
    inner_RMSE_std_um, ...
    overlap_RMSE_mean_um, ...
    overlap_RMSE_std_um, ...
    e_t_mean_um, ...
    e_t_std_um, ...
    e_R_mean_deg, ...
    e_R_std_deg, ...
    solve_time_mean_s, ...
    solve_time_std_s);
end

%% ========================================================================
function Pworld = transform_local_pts_to_world(Plocal, t, R)
Pworld = (R * Plocal.' + t(:)).';
end

%% ========================================================================
function v = pose_to_vec(pose)
v = [pose.t(:).', pose.eul_deg(:).'];
end

%% ========================================================================
function pose = vec_to_pose(v)
pose = struct();
pose.t = [v(1); v(2); v(3)];
pose.eul_deg = [v(4), v(5), v(6)];
pose.R = eul_zyx_to_rotm_deg(v(4), v(5), v(6));
end

%% ========================================================================
function x = project_to_box(x, lb, ub)
x = min(max(x, lb), ub);
end

%% ========================================================================
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

%% ========================================================================
function e = rotation_geodesic_error_deg(R1, R2)
R = R1 * R2.';
c = 0.5 * (trace(R) - 1);
c = min(1, max(-1, c));
e = rad2deg(acos(c));
end

%% ========================================================================
function y = calc_rmse(x)
x = x(isfinite(x));
if isempty(x)
    y = NaN;
else
    y = sqrt(mean(x.^2));
end
end

%% ========================================================================
function mask2 = binary_erode(mask, r)

if r <= 0
    mask2 = mask;
    return;
end

ker = ones(2*r+1);
cnt = conv2(double(mask), ker, 'same');
mask2 = cnt == numel(ker);
end