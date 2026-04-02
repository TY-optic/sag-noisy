clear; clc; close all;

%% ========================= 用户参数 =====================================
% 本脚本与 generate_noisy_peaks_subaps_6dof.m 配套。
% 目标：在真实 6DoF 数据生成条件下，对 sag-based 方法执行 6DoF 搜索与精修，
%      统计 fig:noise_intensity 所需的 Monte Carlo 误差曲线。

sigmaList_um = [0, 2, 4, 6, 8, 10, 12, 14,16,18,20];
Lc_mm        = 1.5;
Nmc          = 20;
baseSeed     = 2026;

pose6dof = struct( ...
    'tx_mm', [], ...
    'ty_mm', [], ...
    'tz_mm',  0.010, ...
    'rx_deg', 0.020, ...
    'ry_deg', -0.040, ...
    'rz_deg', 0.200);

outDir = fullfile(pwd, 'figures', 'noise_intensity_sag_based_6dof');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% ========================= 搜索与评估设置 ===============================
cfg = struct();

cfg.nominal = struct();
% 若要测试纯搜索能力，可改为 'layout'.
cfg.nominal.centerMode = 'truth_plus_delta';   % 'truth_plus_delta' | 'layout'
cfg.nominal.dtx_mm  = 0.030;
cfg.nominal.dty_mm  = -0.020;
cfg.nominal.dtz_mm  = 0.004;
cfg.nominal.drx_deg = 0.010;
cfg.nominal.dry_deg = -0.015;
cfg.nominal.drz_deg = 0.080;

cfg.search = struct();
cfg.search.minOverlapFrac = 0.10;
cfg.search.minOverlapPts  = 150;
cfg.search.maxSearchPts   = 2500;
cfg.search.maxEvalPts     = 8000;
cfg.search.penaltyWeight  = 25;
cfg.search.invalidResidual_mm = 0.05;

cfg.search.coarseYaw = struct();
cfg.search.coarseYaw.nPerDim = 7;
cfg.search.coarseYaw.levels  = 3;
cfg.search.coarseYaw.shrink  = 0.45;
cfg.search.coarseYaw.tx_span_mm  = 0.25;
cfg.search.coarseYaw.ty_span_mm  = 0.25;
cfg.search.coarseYaw.rz_span_deg = 0.50;

cfg.search.coarseTilt = struct();
cfg.search.coarseTilt.nPerDim = 5;
cfg.search.coarseTilt.levels  = 2;
cfg.search.coarseTilt.shrink  = 0.45;
cfg.search.coarseTilt.rx_span_deg = 0.08;
cfg.search.coarseTilt.ry_span_deg = 0.08;

cfg.refine = struct();
cfg.refine.use_lsqnonlin = (exist('lsqnonlin', 'file') == 2);
cfg.refine.maxIter    = 150;
cfg.refine.maxFunEval = 2500;
cfg.refine.stepTol    = 1e-12;
cfg.refine.funTol     = 1e-15;
cfg.refine.optTol     = 1e-15;

cfg.stats = struct();
cfg.stats.innerErodeRadius = 2;

cfg.plot = struct();
cfg.plot.fontName   = 'Times New Roman';
cfg.plot.lineWidth  = 1.6;
cfg.plot.markerSize = 7;

set(groot, 'defaultAxesFontName', cfg.plot.fontName);
set(groot, 'defaultTextFontName', cfg.plot.fontName);

%% ========================= 主循环 =======================================
nSigma = numel(sigmaList_um);
allTrial = cell(nSigma, Nmc);

rmseMat_mm        = nan(nSigma, Nmc);
innerRmseMat_mm   = nan(nSigma, Nmc);
overlapRmseMat_mm = nan(nSigma, Nmc);
solveTimeMat_s    = nan(nSigma, Nmc);
etMat_mm          = nan(nSigma, Nmc);
eRMat_deg         = nan(nSigma, Nmc);
actualNoiseMat_um = nan(nSigma, Nmc);

fprintf('============================================================\n');
fprintf('Sag-based 6DoF 噪声强度 Monte Carlo 统计开始\n');
fprintf('Lc = %.3f mm, Nmc = %d\n', Lc_mm, Nmc);
fprintf('sigma_n_um = [%s]\n', num2str(sigmaList_um));
fprintf('============================================================\n');

for i = 1:nSigma
    sigma_n_um = sigmaList_um(i);
    fprintf('\n------------------------------------------------------------\n');
    fprintf('Sigma %d/%d : sigma_n = %.4f um\n', i, nSigma, sigma_n_um);
    fprintf('------------------------------------------------------------\n');

    for k = 1:Nmc
        rngSeed = baseSeed + 1000*i + k;

        [subA_raw, subB_raw, truth_raw] = generate_noisy_peaks_subaps_6dof( ...
            sigma_n_um, Lc_mm, rngSeed, pose6dof);

        [truth, A, B, truthPose, nominalPose] = prepare_case_6dof(subA_raw, subB_raw, truth_raw, cfg);

        tSolve = tic;
        reg = sag_register_6dof(A, B, nominalPose, cfg);
        solveTime = toc(tSolve);

        poseErr = evaluate_pose_error_6dof(reg, truthPose);
        [~, ~, ~, stats] = fuse_and_evaluate_6dof(truth, A, B, reg, cfg);

        rmseMat_mm(i, k)        = stats.fusedRMSE;
        innerRmseMat_mm(i, k)   = stats.innerRMSE;
        overlapRmseMat_mm(i, k) = reg.overlapRMSE;
        solveTimeMat_s(i, k)    = solveTime;
        etMat_mm(i, k)          = poseErr.et_mm;
        eRMat_deg(i, k)         = poseErr.eR_deg;
        actualNoiseMat_um(i, k) = 1e3 * truth_raw.rmse_noise_actual;

        T = struct();
        T.seed            = rngSeed;
        T.sigma_n_um      = sigma_n_um;
        T.Lc_mm           = Lc_mm;
        T.actualNoise_um  = actualNoiseMat_um(i, k);
        T.truthPose       = truthPose;
        T.nominalPose     = nominalPose;
        T.reg             = reg;
        T.poseErr         = poseErr;
        T.stats           = stats;
        allTrial{i, k}    = T;

        fprintf(['  MC %02d/%02d | actual noise = %.4f um | fused RMSE = %.4f um | ' ...
                 'e_t = %.4f um | e_R = %.4e deg | overlap RMSE = %.4f um\n'], ...
            k, Nmc, actualNoiseMat_um(i, k), 1e3*stats.fusedRMSE, ...
            1e3*poseErr.et_mm, poseErr.eR_deg, 1e3*reg.overlapRMSE);
    end
end

%% ========================= 统计汇总 =====================================
meanRmse_um      = 1e3 * mean(rmseMat_mm, 2, 'omitnan');
stdRmse_um       = 1e3 * std(rmseMat_mm, 0, 2, 'omitnan');
meanInnerRmse_um = 1e3 * mean(innerRmseMat_mm, 2, 'omitnan');
stdInnerRmse_um  = 1e3 * std(innerRmseMat_mm, 0, 2, 'omitnan');
meanOvRmse_um    = 1e3 * mean(overlapRmseMat_mm, 2, 'omitnan');
stdOvRmse_um     = 1e3 * std(overlapRmseMat_mm, 0, 2, 'omitnan');
meanEt_um        = 1e3 * mean(etMat_mm, 2, 'omitnan');
stdEt_um         = 1e3 * std(etMat_mm, 0, 2, 'omitnan');
meanER_deg       = mean(eRMat_deg, 2, 'omitnan');
stdER_deg        = std(eRMat_deg, 0, 2, 'omitnan');
meanSolve_s      = mean(solveTimeMat_s, 2, 'omitnan');
stdSolve_s       = std(solveTimeMat_s, 0, 2, 'omitnan');
meanNoise_um     = mean(actualNoiseMat_um, 2, 'omitnan');
stdNoise_um      = std(actualNoiseMat_um, 0, 2, 'omitnan');

summaryTbl = table( ...
    sigmaList_um(:), meanNoise_um, stdNoise_um, ...
    meanRmse_um, stdRmse_um, ...
    meanInnerRmse_um, stdInnerRmse_um, ...
    meanOvRmse_um, stdOvRmse_um, ...
    meanEt_um, stdEt_um, ...
    meanER_deg, stdER_deg, ...
    meanSolve_s, stdSolve_s, ...
    'VariableNames', { ...
    'sigma_n_set_um', 'sigma_n_actual_mean_um', 'sigma_n_actual_std_um', ...
    'fused_RMSE_mean_um', 'fused_RMSE_std_um', ...
    'inner_RMSE_mean_um', 'inner_RMSE_std_um', ...
    'overlap_RMSE_mean_um', 'overlap_RMSE_std_um', ...
    'e_t_mean_um', 'e_t_std_um', ...
    'e_R_mean_deg', 'e_R_std_deg', ...
    'solve_time_mean_s', 'solve_time_std_s'});

disp(' ');
disp('===================== Summary table =====================');
disp(summaryTbl);

writetable(summaryTbl, fullfile(outDir, 'noise_intensity_sag_based_6dof_summary.csv'));
save(fullfile(outDir, 'noise_intensity_sag_based_6dof_results.mat'), ...
    'cfg', 'sigmaList_um', 'Lc_mm', 'Nmc', 'baseSeed', 'pose6dof', 'allTrial', ...
    'rmseMat_mm', 'innerRmseMat_mm', 'overlapRmseMat_mm', ...
    'solveTimeMat_s', 'etMat_mm', 'eRMat_deg', 'actualNoiseMat_um', ...
    'summaryTbl');

%% ========================= 误差棒绘图 ===================================
fig = figure('Color', 'w', 'Position', [120 120 760 520]);
ax = axes(fig); %#ok<LAXES>
hold(ax, 'on');
box(ax, 'on');
grid(ax, 'on');

errorbar(ax, sigmaList_um, meanRmse_um, stdRmse_um, '-o', ...
    'LineWidth', cfg.plot.lineWidth, 'MarkerSize', cfg.plot.markerSize, ...
    'CapSize', 10);

xlabel(ax, '$\\sigma_n$ ($\\mu$m)', 'Interpreter', 'latex');
ylabel(ax, 'Fused RMSE ($\\mu$m)', 'Interpreter', 'latex');
title(ax, sprintf('Sag-based 6DoF noise-intensity robustness (L_c = %.2f mm, N = %d)', Lc_mm, Nmc), ...
    'Interpreter', 'none');
set(ax, 'FontSize', 12, 'LineWidth', 1.0);

exportgraphics(fig, fullfile(outDir, 'fig_noise_intensity_sag_based_6dof.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(outDir, 'fig_noise_intensity_sag_based_6dof.png'), 'Resolution', 300);

fprintf('\n结果已保存到: %s\n', outDir);

%% ========================================================================
%% 局部函数
%% ========================================================================
function [truth, A, B, truthPose, nominalPose] = prepare_case_6dof(subA_raw, subB_raw, truth_raw, cfg)

    truth = struct();
    truth.X    = double(truth_raw.Xm);
    truth.Y    = double(truth_raw.Ym);
    truth.Z    = double(truth_raw.Z_clean);
    truth.mask = logical(truth_raw.mask);

    A = build_measured_subap_world(subA_raw, cfg.search.maxSearchPts, cfg.search.maxEvalPts);
    B = build_measured_subap_local(subB_raw, cfg.search.maxSearchPts, cfg.search.maxEvalPts);

    truthPose = struct();
    truthPose.t_mm    = double(truth_raw.B_pose.t(:)).';
    truthPose.eul_deg = double(truth_raw.B_pose.eul_deg(:)).';
    truthPose.R       = double(truth_raw.B_pose.R);

    nominalPose = get_nominal_pose(truth_raw, cfg);
end

function A = build_measured_subap_world(sub_raw, maxSearchPts, maxEvalPts)
    xw = double(sub_raw.world_x(:));
    yw = double(sub_raw.world_y(:));
    zw = double(sub_raw.world_z(:));

    valid = isfinite(xw) & isfinite(yw) & isfinite(zw);
    xw = xw(valid); yw = yw(valid); zw = zw(valid);

    A = struct();
    A.xWorld  = xw;
    A.yWorld  = yw;
    A.zWorld  = zw;
    A.n       = numel(zw);
    A.Fworld  = scatteredInterpolant(xw, yw, zw, 'natural', 'none');
    A.idxEval = pick_subsample_indices(A.n, maxEvalPts);
end

function B = build_measured_subap_local(sub_raw, maxSearchPts, maxEvalPts)
    xL = double(sub_raw.x(:));
    yL = double(sub_raw.y(:));
    zL = double(sub_raw.z(:));

    valid = isfinite(xL) & isfinite(yL) & isfinite(zL);
    xL = xL(valid); yL = yL(valid); zL = zL(valid);

    B = struct();
    B.xLocal    = xL;
    B.yLocal    = yL;
    B.zLocal    = zL;
    B.n         = numel(zL);
    B.idxSearch = pick_subsample_indices(B.n, maxSearchPts);
    B.idxEval   = pick_subsample_indices(B.n, maxEvalPts);
end

function nominalPose = get_nominal_pose(truth_raw, cfg)
    mode = lower(cfg.nominal.centerMode);

    switch mode
        case 'truth_plus_delta'
            nominalPose.t_mm = double(truth_raw.B_pose.t(:)).' + ...
                [cfg.nominal.dtx_mm, cfg.nominal.dty_mm, cfg.nominal.dtz_mm];
            nominalPose.eul_deg = double(truth_raw.B_pose.eul_deg(:)).' + ...
                [cfg.nominal.drx_deg, cfg.nominal.dry_deg, cfg.nominal.drz_deg];

        case 'layout'
            nominalPose.t_mm = [truth_raw.cB_nom(:).', 0] + ...
                [cfg.nominal.dtx_mm, cfg.nominal.dty_mm, cfg.nominal.dtz_mm];
            nominalPose.eul_deg = [cfg.nominal.drx_deg, cfg.nominal.dry_deg, cfg.nominal.drz_deg];

        otherwise
            error('未知 nominal.centerMode: %s', cfg.nominal.centerMode);
    end

    nominalPose.R = eul_zyx_to_rotm_deg(nominalPose.eul_deg(1), nominalPose.eul_deg(2), nominalPose.eul_deg(3));
end

function idx = pick_subsample_indices(n, maxN)
    if isinf(maxN) || maxN >= n || maxN <= 0
        idx = (1:n).';
        return;
    end
    idx = round(linspace(1, n, maxN));
    idx = unique(max(min(idx, n), 1)).';
end

function reg = sag_register_6dof(A, B, nominalPose, cfg)

    % ---------- Stage 1: tx, ty, rz 分层搜索，rx/ry 固定在当前中心 ----------
    center = [nominalPose.t_mm(1), nominalPose.t_mm(2), deg2rad(nominalPose.eul_deg(3))];
    span   = [cfg.search.coarseYaw.tx_span_mm, ...
              cfg.search.coarseYaw.ty_span_mm, ...
              deg2rad(cfg.search.coarseYaw.rz_span_deg)];

    rx0 = deg2rad(nominalPose.eul_deg(1));
    ry0 = deg2rad(nominalPose.eul_deg(2));
    tz0 = nominalPose.t_mm(3);

    bestPose = [center(1), center(2), tz0, rx0, ry0, center(3)];
    bestScore = inf;
    bestRMSE  = inf;
    bestOut   = empty_obj_out();

    for lev = 1:cfg.search.coarseYaw.levels
        vx = center(1) + linspace(-span(1), span(1), cfg.search.coarseYaw.nPerDim);
        vy = center(2) + linspace(-span(2), span(2), cfg.search.coarseYaw.nPerDim);
        vz = center(3) + linspace(-span(3), span(3), cfg.search.coarseYaw.nPerDim);

        levBestScore = inf;
        levBestPose  = bestPose;
        levBestRMSE  = inf;
        levBestOut   = empty_obj_out();

        for i1 = 1:numel(vx)
            for i2 = 1:numel(vy)
                for i3 = 1:numel(vz)
                    p = [vx(i1), vy(i2), tz0, rx0, ry0, vz(i3)];
                    [score, rmse, out, pAdj] = sag_obj_with_closedform_tz(p, A, B, cfg, 'search');
                    if score < levBestScore
                        levBestScore = score;
                        levBestPose  = pAdj;
                        levBestRMSE  = rmse;
                        levBestOut   = out;
                    end
                end
            end
        end

        bestPose  = levBestPose;
        bestScore = levBestScore;
        bestRMSE  = levBestRMSE;
        bestOut   = levBestOut;

        center = [bestPose(1), bestPose(2), bestPose(6)];
        tz0    = bestPose(3);
        span   = span * cfg.search.coarseYaw.shrink;
    end

    % ---------- Stage 2: rx, ry 分层搜索，tz 仍然闭式更新 ----------
    centerTilt = [bestPose(4), bestPose(5)];
    spanTilt   = [deg2rad(cfg.search.coarseTilt.rx_span_deg), ...
                  deg2rad(cfg.search.coarseTilt.ry_span_deg)];

    for lev = 1:cfg.search.coarseTilt.levels
        vrx = centerTilt(1) + linspace(-spanTilt(1), spanTilt(1), cfg.search.coarseTilt.nPerDim);
        vry = centerTilt(2) + linspace(-spanTilt(2), spanTilt(2), cfg.search.coarseTilt.nPerDim);

        levBestScore = inf;
        levBestPose  = bestPose;
        levBestRMSE  = inf;
        levBestOut   = empty_obj_out();

        for i1 = 1:numel(vrx)
            for i2 = 1:numel(vry)
                p = bestPose;
                p(4) = vrx(i1);
                p(5) = vry(i2);
                [score, rmse, out, pAdj] = sag_obj_with_closedform_tz(p, A, B, cfg, 'search');
                if score < levBestScore
                    levBestScore = score;
                    levBestPose  = pAdj;
                    levBestRMSE  = rmse;
                    levBestOut   = out;
                end
            end
        end

        bestPose  = levBestPose;
        bestScore = levBestScore;
        bestRMSE  = levBestRMSE;
        bestOut   = levBestOut;

        centerTilt = [bestPose(4), bestPose(5)];
        spanTilt   = spanTilt * cfg.search.coarseTilt.shrink;
    end

    % ---------- Stage 3: 6DoF 联合精修 ----------
    if cfg.refine.use_lsqnonlin
        lb = [bestPose(1)-0.20, bestPose(2)-0.20, bestPose(3)-0.03, bestPose(4)-deg2rad(0.10), bestPose(5)-deg2rad(0.10), bestPose(6)-deg2rad(0.30)];
        ub = [bestPose(1)+0.20, bestPose(2)+0.20, bestPose(3)+0.03, bestPose(4)+deg2rad(0.10), bestPose(5)+deg2rad(0.10), bestPose(6)+deg2rad(0.30)];

        opts = optimoptions('lsqnonlin', ...
            'Display', 'off', ...
            'MaxIterations', cfg.refine.maxIter, ...
            'MaxFunctionEvaluations', cfg.refine.maxFunEval, ...
            'StepTolerance', cfg.refine.stepTol, ...
            'FunctionTolerance', cfg.refine.funTol, ...
            'OptimalityTolerance', cfg.refine.optTol);

        resFun = @(p) residual_vector_6dof(p, A, B, cfg, 'eval');
        pOpt = lsqnonlin(resFun, bestPose, lb, ub, opts);
        [scoreOpt, rmseOpt, outOpt] = sag_obj_full_6dof(pOpt, A, B, cfg, 'eval');

        if scoreOpt < bestScore
            bestPose  = pOpt;
            bestScore = scoreOpt;
            bestRMSE  = rmseOpt;
            bestOut   = outOpt;
        end
    else
        objFun = @(p) sag_obj_scalar_6dof(p, A, B, cfg);
        opts = optimset('Display', 'off', ...
            'MaxIter', cfg.refine.maxIter, ...
            'MaxFunEvals', cfg.refine.maxFunEval, ...
            'TolX', cfg.refine.stepTol, ...
            'TolFun', cfg.refine.funTol);
        pOpt = fminsearch(objFun, bestPose, opts);
        [scoreOpt, rmseOpt, outOpt] = sag_obj_full_6dof(pOpt, A, B, cfg, 'eval');

        if scoreOpt < bestScore
            bestPose  = pOpt;
            bestScore = scoreOpt;
            bestRMSE  = rmseOpt;
            bestOut   = outOpt;
        end
    end

    reg = struct();
    reg.p = bestPose;
    reg.t_mm = bestPose(1:3);
    reg.eul_deg = rad2deg(bestPose(4:6));
    reg.R = eul_zyx_to_rotm_rad(bestPose(4), bestPose(5), bestPose(6));
    reg.score = bestScore;
    reg.overlapRMSE = bestRMSE;
    reg.overlapRatio = bestOut.overlapRatio;
    reg.overlapN = bestOut.overlapN;
end

function [score, rmse, out, pAdj] = sag_obj_with_closedform_tz(p, A, B, cfg, mode)
    switch lower(mode)
        case 'search'
            idx = B.idxSearch;
        otherwise
            idx = B.idxEval;
    end

    xL = B.xLocal(idx);
    yL = B.yLocal(idx);
    zL = B.zLocal(idx);

    R = eul_zyx_to_rotm_rad(p(4), p(5), p(6));

    [xw0, yw0, zw0] = transform_local_points(xL, yL, zL, [p(1), p(2), 0], R);
    zAref = A.Fworld(xw0, yw0);
    ov = isfinite(zAref) & isfinite(zw0);

    nOv  = nnz(ov);
    nRef = numel(idx);
    overlapRatio = nOv / max(nRef, 1);
    nMin = max(cfg.search.minOverlapPts, round(cfg.search.minOverlapFrac * nRef));

    pAdj = p;
    if nOv < nMin
        score = 1e12 + (nMin - nOv)^2;
        rmse  = inf;
        out = empty_obj_out();
        out.overlapRatio = overlapRatio;
        out.overlapN = nOv;
        return;
    end

    tz = mean(zAref(ov) - zw0(ov));
    zw = zw0 + tz;
    res = zw(ov) - zAref(ov);
    rmse = sqrt(mean(res.^2));
    penalty = cfg.search.penaltyWeight * max(0, cfg.search.minOverlapFrac - overlapRatio)^2;
    score = rmse * (1 + penalty);

    pAdj(3) = tz;
    out = struct('overlapRatio', overlapRatio, 'overlapN', nOv);
end

function f = sag_obj_scalar_6dof(p, A, B, cfg)
    [f, ~, ~] = sag_obj_full_6dof(p, A, B, cfg, 'eval');
end

function [score, rmse, out] = sag_obj_full_6dof(p, A, B, cfg, mode)
    bundle = eval_residual_bundle_6dof(p, A, B, cfg, mode);
    resValid = bundle.resValid;
    nOv = bundle.nOv;
    nRef = bundle.nRef;
    overlapRatio = bundle.overlapRatio;
    nMin = max(cfg.search.minOverlapPts, round(cfg.search.minOverlapFrac * nRef));

    if nOv < nMin || isempty(resValid)
        score = 1e12 + (nMin - nOv)^2;
        rmse  = inf;
        out = empty_obj_out();
        out.overlapRatio = overlapRatio;
        out.overlapN = nOv;
        return;
    end

    rmse = sqrt(mean(resValid.^2));
    penalty = cfg.search.penaltyWeight * max(0, cfg.search.minOverlapFrac - overlapRatio)^2;
    score = rmse * (1 + penalty);
    out = struct('overlapRatio', overlapRatio, 'overlapN', nOv);
end

function res = residual_vector_6dof(p, A, B, cfg, mode)
    bundle = eval_residual_bundle_6dof(p, A, B, cfg, mode);
    res = bundle.resAll;
end

function bundle = eval_residual_bundle_6dof(p, A, B, cfg, mode)
    switch lower(mode)
        case 'search'
            idx = B.idxSearch;
        otherwise
            idx = B.idxEval;
    end

    xL = B.xLocal(idx);
    yL = B.yLocal(idx);
    zL = B.zLocal(idx);

    R = eul_zyx_to_rotm_rad(p(4), p(5), p(6));
    [xw, yw, zw] = transform_local_points(xL, yL, zL, p(1:3), R);

    zAref = A.Fworld(xw, yw);
    ov = isfinite(zAref) & isfinite(zw);

    resAll = cfg.search.invalidResidual_mm * ones(numel(idx), 1);
    if any(ov)
        resAll(ov) = zw(ov) - zAref(ov);
    end

    bundle = struct();
    bundle.resAll = resAll(:);
    bundle.resValid = resAll(ov);
    bundle.nOv = nnz(ov);
    bundle.nRef = numel(idx);
    bundle.overlapRatio = bundle.nOv / max(bundle.nRef, 1);
end

function nRef = get_nref(B, mode)
    switch lower(mode)
        case 'search'
            nRef = numel(B.idxSearch);
        otherwise
            nRef = numel(B.idxEval);
    end
end

function out = empty_obj_out()
    out = struct('overlapRatio', 0, 'overlapN', 0);
end

function [xw, yw, zw] = transform_local_points(xL, yL, zL, t_mm, R)
    P = [xL(:), yL(:), zL(:)].';
    Pw = (R * P + t_mm(:)).';
    xw = Pw(:,1);
    yw = Pw(:,2);
    zw = Pw(:,3);
end

function poseErr = evaluate_pose_error_6dof(reg, truthPose)
    dt = reg.t_mm(:) - truthPose.t_mm(:);
    et = norm(dt);

    Rerr = reg.R * truthPose.R.';
    val = (trace(Rerr) - 1) / 2;
    val = min(max(val, -1), 1);
    dang = acos(val);

    deul = reg.eul_deg(:) - truthPose.eul_deg(:);

    poseErr = struct();
    poseErr.dt_mm    = dt(:).';
    poseErr.et_mm    = et;
    poseErr.deul_deg = deul(:).';
    poseErr.eR_deg   = abs(rad2deg(dang));
end

function [fusedMap, errMap, unionMask, stats] = fuse_and_evaluate_6dof(truth, A, B, reg, cfg)
    xAw = A.xWorld;
    yAw = A.yWorld;
    zAw = A.zWorld;

    R = reg.R;
    [xBw, yBw, zBw] = transform_local_points(B.xLocal, B.yLocal, B.zLocal, reg.t_mm, R);

    FAw = scatteredInterpolant(xAw, yAw, zAw, 'natural', 'none');
    FBw = scatteredInterpolant(xBw, yBw, zBw, 'natural', 'none');

    ZAmap = FAw(truth.X, truth.Y);
    ZBmap = FBw(truth.X, truth.Y);
    ZAmap(~truth.mask) = NaN;
    ZBmap(~truth.mask) = NaN;

    vA = isfinite(ZAmap) & truth.mask;
    vB = isfinite(ZBmap) & truth.mask;

    unionMask   = vA | vB;
    overlapMask = vA & vB;
    onlyA       = vA & ~vB;
    onlyB       = ~vA & vB;

    fusedMap = nan(size(truth.Z));
    fusedMap(overlapMask) = 0.5 * (ZAmap(overlapMask) + ZBmap(overlapMask));
    fusedMap(onlyA) = ZAmap(onlyA);
    fusedMap(onlyB) = ZBmap(onlyB);

    errMap = nan(size(truth.Z));
    errMap(unionMask) = fusedMap(unionMask) - truth.Z(unionMask);

    ev = errMap(unionMask);
    ev = ev(isfinite(ev));

    stats = struct();
    stats.fusedRMSE = safe_rmse(ev);
    stats.fusedMAE  = safe_mae(ev);
    stats.fusedPV   = safe_pv(ev);
    stats.fusedSTD  = safe_std(ev);

    ovErr = errMap(overlapMask);
    ovErr = ovErr(isfinite(ovErr));
    stats.overlapOnlyRMSE = safe_rmse(ovErr);

    innerMask = erode_mask_disk(unionMask, cfg.stats.innerErodeRadius);
    innerErr = errMap(innerMask);
    innerErr = innerErr(isfinite(innerErr));
    stats.innerRMSE = safe_rmse(innerErr);

    if isempty(ev)
        stats.robustPV995 = NaN;
    else
        q1 = prctile(ev, 0.5);
        q2 = prctile(ev, 99.5);
        stats.robustPV995 = q2 - q1;
    end
end

function R = eul_zyx_to_rotm_deg(rx_deg, ry_deg, rz_deg)
    R = eul_zyx_to_rotm_rad(deg2rad(rx_deg), deg2rad(ry_deg), deg2rad(rz_deg));
end

function R = eul_zyx_to_rotm_rad(rx, ry, rz)
    Rx = [1, 0, 0; 0, cos(rx), -sin(rx); 0, sin(rx), cos(rx)];
    Ry = [cos(ry), 0, sin(ry); 0, 1, 0; -sin(ry), 0, cos(ry)];
    Rz = [cos(rz), -sin(rz), 0; sin(rz), cos(rz), 0; 0, 0, 1];
    R = Rz * Ry * Rx;
end

function v = safe_rmse(x)
    if isempty(x), v = NaN; else, v = sqrt(mean(x.^2)); end
end

function v = safe_mae(x)
    if isempty(x), v = NaN; else, v = mean(abs(x)); end
end

function v = safe_pv(x)
    if isempty(x), v = NaN; else, v = max(x) - min(x); end
end

function v = safe_std(x)
    if isempty(x), v = NaN; else, v = std(x, 1); end
end

function maskOut = erode_mask_disk(maskIn, radius)
    if radius <= 0
        maskOut = maskIn;
        return;
    end
    [xx, yy] = meshgrid(-radius:radius, -radius:radius);
    ker = (xx.^2 + yy.^2 <= radius^2);
    cnt = conv2(double(maskIn), double(ker), 'same');
    maskOut = (cnt == nnz(ker));
end
