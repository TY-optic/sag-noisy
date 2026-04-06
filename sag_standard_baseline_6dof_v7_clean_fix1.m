function results = sag_standard_baseline_6dof_v7_clean_fix1()
%SAG_STANDARD_BASELINE_6DOF_V7_CLEAN_FIX1
% 清理后的 sag-based 6DoF 噪声批处理基线。
%
% 运行方式：
%   results = sag_standard_baseline_6dof_v7_clean();
%
% 依赖：
%   generate_noisy_peaks_subaps_6dof.m

clc; close all;
tic;

assert(exist('generate_noisy_peaks_subaps_6dof.m','file') == 2, ...
    '未找到 generate_noisy_peaks_subaps_6dof.m');

%% ========================= 全局配置 =========================
cfg = struct();

% ---------- 固定重叠率 ----------
cfg.overlapTarget = 0.50;
cfg.phiDeg        = 25.0;

% ---------- 固定相对位姿中的非平面内参数 ----------
cfg.pose6dof = struct( ...
    'tx_mm',  0, ...
    'ty_mm',  0, ...
    'tz_mm',  0.010, ...
    'rx_deg', 0.020, ...
    'ry_deg', -0.040, ...
    'rz_deg', 0.200);

% ---------- Monte Carlo ----------
cfg.mc = struct();
cfg.mc.N = 10;
cfg.mc.seedBase = 20260410;

% ---------- 幅值噪声实验 ----------
cfg.noiseIntensity = struct();
cfg.noiseIntensity.sigmaList_um = [0.5, 1, 1.5, 2, 3, 4, 6, 8];
cfg.noiseIntensity.Lc_mm = 0.5;

% ---------- 频谱噪声实验 ----------
cfg.noiseSpectrum = struct();
cfg.noiseSpectrum.sigma_n_um = 5.0;
cfg.noiseSpectrum.LcList_mm  = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0];

% ---------- 预处理（仅用于 3DoF 粗搜索） ----------
cfg.pre = struct();
cfg.pre.smoothSigmaPix = 0.8;
cfg.pre.removePlane    = true;
cfg.pre.clipSigma      = 6.0;

% ---------- 3DoF 粗搜索 ----------
cfg.coarse = struct();
cfg.coarse.thetaDegList      = -1.0 : 0.05 : 1.0;
cfg.coarse.dList_mm          = [];      
cfg.coarse.phiDegList        = -180 : 3 : 180;
cfg.coarse.minOverlapN       = 1200;
cfg.coarse.maskInterpThresh  = 0.85;
cfg.coarse.overlapPenalty    = 0.15;
cfg.coarse.dPadMM            = 4.0;
cfg.coarse.dStepMM           = 0.5;

% ---------- 6DoF 单初值局部优化 ----------
cfg.refine6 = struct();
cfg.refine6.nRepeat    = 3;
cfg.refine6.maxIter    = 160;
cfg.refine6.maxFunEval = 4000;
cfg.refine6.stepTol    = 1e-10;
cfg.refine6.funTol     = 1e-10;

cfg.refine6.bound = struct();
cfg.refine6.bound.rxDeg = 0.12;
cfg.refine6.bound.ryDeg = 0.12;
cfg.refine6.bound.rzDeg = 0.60;
cfg.refine6.bound.txMM  = 1.00;
cfg.refine6.bound.tyMM  = 1.00;
cfg.refine6.bound.tzMM  = 0.03;

cfg.refine6.maskInterpThresh = 0.55;
cfg.refine6.minValidN        = 1200;
cfg.refine6.bigPenalty       = 1e3;

cfg.refine6.lambdaMaskPenalty = 2.0e-2;
cfg.refine6.lambdaRxRy        = 3.0e-3;
cfg.refine6.lambdaTz          = 3.0e-3;
cfg.refine6.lambdaRz          = 5.0e-4;

% ---------- 误差评价与绘图 ----------
cfg.plot = struct();
cfg.plot.colorPrctile = 99.7;
cfg.plot.useDisplayPrunedForStats = true;

cfg.plot.displayPrune = struct();
cfg.plot.displayPrune.enabled = true;
cfg.plot.displayPrune.singleSupportBandPix = 0;
cfg.plot.displayPrune.dualSupportBandPix   = 0;
cfg.plot.displayPrune.minNeighborCount     = 1;

cfg.plot.pvPrune = struct();
cfg.plot.pvPrune.enabled = true;
cfg.plot.pvPrune.singleSupportBandPix = 1;
cfg.plot.pvPrune.dualSupportBandPix   = 0;
cfg.plot.pvPrune.minNeighborCount     = 3;

cfg.plot.titlePVMode = 'robust995';

cfg.saveRepresentativeMap = true;

% ---------- 输出 ----------
outRoot = fullfile(pwd, 'sag_standard_baseline_outputs_v7_clean');
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

%% ========================= 固定重叠率对应真值平移 =========================
[subA_tmp, ~, ~] = generate_noisy_peaks_subaps_6dof( ...
    0.0, cfg.noiseIntensity.Lc_mm, cfg.mc.seedBase, cfg.pose6dof);
Rsub = double(subA_tmp.Rsub);

dTheory = solve_center_distance_from_overlap_local(Rsub, cfg.overlapTarget);
cfg.pose6dof.tx_mm = dTheory * cosd(cfg.phiDeg);
cfg.pose6dof.ty_mm = dTheory * sind(cfg.phiDeg);

dLo = max(0, dTheory - cfg.coarse.dPadMM);
dHi = min(2*Rsub - 1e-6, dTheory + cfg.coarse.dPadMM);
cfg.coarse.dList_mm = dLo : cfg.coarse.dStepMM : dHi;

fprintf('\n============================================================\n');
fprintf('Standard sag-based baseline v7 clean | noise batch\n');
fprintf('Fixed overlap = %.3f | dTheory = %.4f mm\n', cfg.overlapTarget, dTheory);
fprintf('Truth tx = %+10.6f mm | ty = %+10.6f mm\n', cfg.pose6dof.tx_mm, cfg.pose6dof.ty_mm);
fprintf('Coarse d search = [%.4f : %.2f : %.4f] mm\n', dLo, cfg.coarse.dStepMM, dHi);
fprintf('MC N = %d\n', cfg.mc.N);
fprintf('============================================================\n');

%% ========================= 幅值噪声实验 =========================
outDirIntensity = fullfile(outRoot, 'noise_intensity');
if ~exist(outDirIntensity, 'dir')
    mkdir(outDirIntensity);
end

[rawIntensityTbl, sumIntensityTbl] = run_noise_intensity_experiment_local(cfg, outDirIntensity);

writetable(rawIntensityTbl, fullfile(outDirIntensity, 'raw_results_intensity.csv'));
writetable(sumIntensityTbl, fullfile(outDirIntensity, 'summary_intensity.csv'));
save(fullfile(outDirIntensity, 'results_intensity.mat'), 'rawIntensityTbl', 'sumIntensityTbl', 'cfg');

plot_noise_summary_curves_local(sumIntensityTbl, ...
    'sigma_n_um', 'Noise amplitude \sigma_n (\mum)', outDirIntensity, 'intensity');

%% ========================= 频谱噪声实验 =========================
outDirSpectrum = fullfile(outRoot, 'noise_spectrum');
if ~exist(outDirSpectrum, 'dir')
    mkdir(outDirSpectrum);
end

[rawSpectrumTbl, sumSpectrumTbl] = run_noise_spectrum_experiment_local(cfg, outDirSpectrum);

writetable(rawSpectrumTbl, fullfile(outDirSpectrum, 'raw_results_spectrum.csv'));
writetable(sumSpectrumTbl, fullfile(outDirSpectrum, 'summary_spectrum.csv'));
save(fullfile(outDirSpectrum, 'results_spectrum.mat'), 'rawSpectrumTbl', 'sumSpectrumTbl', 'cfg');

plot_noise_summary_curves_local(sumSpectrumTbl, ...
    'Lc_mm', 'Noise correlation length L_c (mm)', outDirSpectrum, 'spectrum');

fprintf('\n全部结果已保存到：\n%s\n', outRoot);
toc;

results = struct();
results.cfg = cfg;
results.outRoot = outRoot;
results.rawIntensityTbl = rawIntensityTbl;
results.sumIntensityTbl = sumIntensityTbl;
results.rawSpectrumTbl = rawSpectrumTbl;
results.sumSpectrumTbl = sumSpectrumTbl;

end

%% ========================================================================
function [rawTbl, sumTbl] = run_noise_intensity_experiment_local(cfg, outDir)

    sigmaList = cfg.noiseIntensity.sigmaList_um(:).';
    Lc_mm = cfg.noiseIntensity.Lc_mm;

    rawRows = [];
    sumRows = [];

    for i = 1:numel(sigmaList)
        sigma_n_um = sigmaList(i);

        fprintf('\n================ Intensity case %d/%d ================\n', i, numel(sigmaList));
        fprintf('sigma_n = %.4f um | Lc = %.4f mm\n', sigma_n_um, Lc_mm);

        condName = sprintf('sigma_%0.3fum', sigma_n_um);
        condDir = fullfile(outDir, condName);
        if ~exist(condDir, 'dir')
            mkdir(condDir);
        end

        M = run_mc_condition_local(cfg, sigma_n_um, Lc_mm, condDir, i);

        rawRows = [rawRows; M.rawRows]; %#ok<AGROW>
        sumRows = [sumRows; M.sumRow];  %#ok<AGROW>
    end

    rawTbl = struct2table(rawRows);
    sumTbl = struct2table(sumRows);
end

function [rawTbl, sumTbl] = run_noise_spectrum_experiment_local(cfg, outDir)

    sigma_n_um = cfg.noiseSpectrum.sigma_n_um;
    LcList = cfg.noiseSpectrum.LcList_mm(:).';

    rawRows = [];
    sumRows = [];

    for i = 1:numel(LcList)
        Lc_mm = LcList(i);

        fprintf('\n================ Spectrum case %d/%d ================\n', i, numel(LcList));
        fprintf('sigma_n = %.4f um | Lc = %.4f mm\n', sigma_n_um, Lc_mm);

        condName = sprintf('Lc_%0.3fmm', Lc_mm);
        condDir = fullfile(outDir, condName);
        if ~exist(condDir, 'dir')
            mkdir(condDir);
        end

        M = run_mc_condition_local(cfg, sigma_n_um, Lc_mm, condDir, i);

        rawRows = [rawRows; M.rawRows]; %#ok<AGROW>
        sumRows = [sumRows; M.sumRow];  %#ok<AGROW>
    end

    rawTbl = struct2table(rawRows);
    sumTbl = struct2table(sumRows);
end

function M = run_mc_condition_local(cfg, sigma_n_um, Lc_mm, condDir, condIdx)

    Nmc = cfg.mc.N;
    rawRows = repmat(make_empty_raw_row_template_local(), Nmc, 1);

    repSaved = false;

    for imc = 1:Nmc
        rngSeed = cfg.mc.seedBase + condIdx*1000 + imc;

        [subA_raw, subB_raw, truth_raw] = generate_noisy_peaks_subaps_6dof( ...
            sigma_n_um, Lc_mm, rngSeed, cfg.pose6dof);

        R = estimate_sag_pose_standard_baseline_local(subA_raw, subB_raw, cfg);

        truthErr = evaluate_fused_vs_truth_peaks_local_v6(R, truth_raw, cfg.plot);
        poseErr  = compute_pose_error_metrics_local(R.p6_final, R.truthRel);
        poseComp = compute_pose_error_components_local(R.p6_final, R.truthRel);

        row = struct();
        row.sigma_n_um = sigma_n_um;
        row.Lc_mm = Lc_mm;
        row.mcIdx = imc;
        row.seed = rngSeed;

        row.e_t_um = poseErr.e_t_um;
        row.e_R_deg = poseErr.e_R_deg;

        row.dtx_um = poseComp.dtx_um;
        row.dty_um = poseComp.dty_um;
        row.dtz_um = poseComp.dtz_um;
        row.drx_deg = poseComp.drx_deg;
        row.dry_deg = poseComp.dry_deg;
        row.drz_deg = poseComp.drz_deg;

        row.RMSE_um = truthErr.rmse_um_raw;
        row.MAE_um  = truthErr.mae_um_raw;
        row.PV_raw_um = truthErr.pv_um_raw;
        row.PV_pruned_um = truthErr.pv_um_pruned;
        row.rPV995_um = truthErr.robustPV995_um;

        row.validN = truthErr.validN;
        row.displayN = truthErr.displayN;
        row.pvMaskN = truthErr.pvMaskN;

        row.runtime_s = R.runtime_sec;

        rawRows(imc) = row;

        fprintf('  MC %02d/%02d | e_t=%.3f um | e_R=%.4e deg | RMSE=%.3f um | rPV=%.3f um | t=%.2f s\n', ...
            imc, Nmc, row.e_t_um, row.e_R_deg, row.RMSE_um, row.rPV995_um, row.runtime_s);

        if cfg.saveRepresentativeMap && ~repSaved
            plot_single_error_map_local(truthErr, cfg.plot, condDir, ...
                sprintf('rep_sigma_%0.3fum_Lc_%0.3fmm', sigma_n_um, Lc_mm));
            repSaved = true;
        end
    end

    sumRow = summarize_rows_local(rawRows, sigma_n_um, Lc_mm);
    M = struct('rawRows', rawRows, 'sumRow', sumRow);
end

function row = make_empty_raw_row_template_local()

    row = struct( ...
        'sigma_n_um', NaN, ...
        'Lc_mm', NaN, ...
        'mcIdx', NaN, ...
        'seed', NaN, ...
        'e_t_um', NaN, ...
        'e_R_deg', NaN, ...
        'dtx_um', NaN, ...
        'dty_um', NaN, ...
        'dtz_um', NaN, ...
        'drx_deg', NaN, ...
        'dry_deg', NaN, ...
        'drz_deg', NaN, ...
        'RMSE_um', NaN, ...
        'MAE_um', NaN, ...
        'PV_raw_um', NaN, ...
        'PV_pruned_um', NaN, ...
        'rPV995_um', NaN, ...
        'validN', NaN, ...
        'displayN', NaN, ...
        'pvMaskN', NaN, ...
        'runtime_s', NaN);
end

function S = summarize_rows_local(rows, sigma_n_um, Lc_mm)

    S = struct();
    S.sigma_n_um = sigma_n_um;
    S.Lc_mm = Lc_mm;

    fMeanStd = { ...
        'e_t_um','e_R_deg', ...
        'dtx_um','dty_um','dtz_um','drx_deg','dry_deg','drz_deg', ...
        'RMSE_um','MAE_um','PV_raw_um','PV_pruned_um','rPV995_um','runtime_s'};

    for i = 1:numel(fMeanStd)
        fn = fMeanStd{i};
        vals = [rows.(fn)].';
        S.(['mean_' fn]) = mean(vals, 'omitnan');
        S.(['std_'  fn]) = std(vals, 0, 'omitnan');
    end

    vals = [rows.validN].';
    S.mean_validN = mean(vals, 'omitnan');
    S.std_validN  = std(vals, 0, 'omitnan');

    vals = [rows.displayN].';
    S.mean_displayN = mean(vals, 'omitnan');
    S.std_displayN  = std(vals, 0, 'omitnan');

    vals = [rows.pvMaskN].';
    S.mean_pvMaskN = mean(vals, 'omitnan');
    S.std_pvMaskN  = std(vals, 0, 'omitnan');
end

%% ========================================================================
function R = estimate_sag_pose_standard_baseline_local(subA_raw, subB_raw, cfg)

    t0 = tic;

    [A, poseA] = build_local_grid_subap_local(subA_raw, 'noisy');
    [B, poseB] = build_local_grid_subap_local(subB_raw, 'noisy');

    Aprep = preprocess_local_sag_local(A, cfg.pre);
    Bprep = preprocess_local_sag_local(B, cfg.pre);

    truthRel = true_relative_pose_local(poseA, poseB);
    truth2d = struct();
    truth2d.thetaDeg = rad2deg(truthRel.rz);
    truth2d.tx = truthRel.t(1);
    truth2d.ty = truthRel.t(2);

    coarse = coarse_search_best_local(Aprep, Bprep, cfg.coarse);

    p0 = [0; 0; deg2rad(coarse.thetaDeg); coarse.tx; coarse.ty; 0];

    modelA = build_surface_model_from_local_local(A);
    maskB = B.mask & isfinite(B.Z);
    PBcand = [B.X(maskB), B.Y(maskB), B.Z(maskB)];

    [p6_final, out6] = refine_pose_6dof_repeated_local(PBcand, p0, modelA, cfg.refine6);

    R = struct();
    R.A = A;
    R.B = B;
    R.Aprep = Aprep;
    R.Bprep = Bprep;
    R.truthRel = truthRel;
    R.truth2d = truth2d;
    R.coarse = coarse;
    R.p0 = p0;
    R.p6_final = p6_final;
    R.refine6 = out6;
    R.runtime_sec = toc(t0);
end

%% ========================================================================
function best = coarse_search_best_local(A, B, coarseCfg)

    best = struct();
    best.thetaDeg = NaN;
    best.tx = NaN;
    best.ty = NaN;
    best.obj = inf;
    best.overlapN = 0;

    for thetaDeg = coarseCfg.thetaDegList
        for dmm = coarseCfg.dList_mm
            for phiDeg = coarseCfg.phiDegList
                tx = dmm * cosd(phiDeg);
                ty = dmm * sind(phiDeg);

                [obj, info] = objective_se2_planar_local( ...
                    A, B, thetaDeg, tx, ty, ...
                    coarseCfg.minOverlapN, ...
                    coarseCfg.maskInterpThresh, ...
                    coarseCfg.overlapPenalty, 'linear');

                if obj < best.obj
                    best.thetaDeg = thetaDeg;
                    best.tx = tx;
                    best.ty = ty;
                    best.obj = obj;
                    best.overlapN = info.overlapN;
                end
            end
        end
    end

    if ~isfinite(best.obj)
        error('3DoF 粗搜索没有得到有效候选。');
    end
end

%% ========================================================================
function [obj, info] = objective_se2_planar_local(A, B, thetaDeg, tx, ty, minOverlapN, maskInterpThresh, overlapPenalty, interpMethod)
    if nargin < 9
        interpMethod = 'linear';
    end

    theta = deg2rad(thetaDeg);
    c = cos(theta);
    s = sin(theta);

    qx = A.X - tx;
    qy = A.Y - ty;

    xq =  c .* qx + s .* qy;
    yq = -s .* qx + c .* qy;

    if strcmpi(interpMethod, 'spline') || strcmpi(interpMethod, 'cubic')
        Z_target = B.Zfill;
    else
        Z_target = B.Z;
    end

    ZBw = interp2(B.X(1,:), B.Y(:,1), Z_target, xq, yq, interpMethod, NaN);
    MBw = interp2(B.X(1,:), B.Y(:,1), double(B.mask), xq, yq, 'linear', 0);

    overlap = A.mask & isfinite(A.Z) & isfinite(ZBw) & (MBw >= maskInterpThresh);
    nOv = nnz(overlap);

    info = struct();
    info.overlapN = nOv;

    if nOv < minOverlapN
        obj = inf;
        info.rmse = inf;
        return;
    end

    xa = A.X(overlap);
    ya = A.Y(overlap);
    ra = A.Z(overlap) - ZBw(overlap);

    M = [xa(:), ya(:), ones(nOv,1)];
    beta = M \ ra(:);
    rr = ra(:) - M * beta;

    medr = median(rr);
    mad0 = median(abs(rr - medr));
    sigma = 1.4826 * max(mad0, eps);
    cHuber = 1.5 * sigma;

    w = ones(size(rr));
    idx = abs(rr) > cHuber;
    w(idx) = cHuber ./ abs(rr(idx));

    rmseRobust = sqrt(sum(w .* (rr.^2)) / max(sum(w), eps));
    obj = rmseRobust + overlapPenalty / sqrt(double(nOv));

    info.rmse = rmseRobust;
    info.beta = beta;
end

%% ========================================================================
function [pBest, bestOut] = refine_pose_6dof_repeated_local(PBcand, pInit, modelA, cfg6)

    pCur = pInit(:);
    bestOut = struct();
    bestOut.J = inf;
    bestOut.info = struct('validN', 0);
    bestOut.history = [];

    for it = 1:cfg6.nRepeat
        [pNew, outOne] = refine_pose_6dof_single_local(PBcand, pCur, modelA, cfg6);
        pCur = pNew(:);

        bestOut.J = outOne.J;
        bestOut.info = outOne.info;
        bestOut.history = [bestOut.history; struct( ...
            'iter', it, ...
            'p', pCur(:).', ...
            'J', outOne.J, ...
            'validN', outOne.info.validN)];
    end

    pBest = pCur;
end

function [pBest, out] = refine_pose_6dof_single_local(PBcand, pCenter, modelA, cfg6)

    pCenter = pCenter(:);

    lb = pCenter;
    ub = pCenter;

    lb(1) = pCenter(1) - deg2rad(cfg6.bound.rxDeg); ub(1) = pCenter(1) + deg2rad(cfg6.bound.rxDeg);
    lb(2) = pCenter(2) - deg2rad(cfg6.bound.ryDeg); ub(2) = pCenter(2) + deg2rad(cfg6.bound.ryDeg);
    lb(3) = pCenter(3) - deg2rad(cfg6.bound.rzDeg); ub(3) = pCenter(3) + deg2rad(cfg6.bound.rzDeg);
    lb(4) = pCenter(4) - cfg6.bound.txMM;           ub(4) = pCenter(4) + cfg6.bound.txMM;
    lb(5) = pCenter(5) - cfg6.bound.tyMM;           ub(5) = pCenter(5) + cfg6.bound.tyMM;
    lb(6) = pCenter(6) - cfg6.bound.tzMM;           ub(6) = pCenter(6) + cfg6.bound.tzMM;

    obj = @(p) objective_pose_6dof_scalar_bounded_local(p, lb, ub, PBcand, modelA, cfg6);

    opts = optimset('Display','off', ...
        'MaxIter', cfg6.maxIter, ...
        'MaxFunEvals', cfg6.maxFunEval, ...
        'TolX', cfg6.stepTol, ...
        'TolFun', cfg6.funTol);

    pBest = fminsearch(obj, pCenter, opts);
    pBest = clip_box_local(pBest(:), lb(:), ub(:));

    [J, info] = objective_pose_6dof_scalar_local(pBest, PBcand, modelA, cfg6);

    out = struct();
    out.J = J;
    out.info = info;
end

function val = objective_pose_6dof_scalar_bounded_local(p, lb, ub, PBcand, modelA, cfg6)
    p = p(:);
    pClip = clip_box_local(p, lb, ub);
    penBox = cfg6.bigPenalty * sum((p - pClip).^2);
    [J0, ~] = objective_pose_6dof_scalar_local(pClip, PBcand, modelA, cfg6);
    val = J0 + penBox;
end

function [J, info] = objective_pose_6dof_scalar_local(p, PB, modelA, cfg6)

    [R, t] = posevec_to_rt_local(p);
    PA = apply_transform_local(PB, R, t);

    x = PA(:,1);
    y = PA(:,2);
    z = PA(:,3);

    zA = interp2(modelA.xVec, modelA.yVec, modelA.Zfill, x, y, 'spline', NaN);
    mv = interp2(modelA.xVec, modelA.yVec, double(modelA.mask), x, y, 'linear', 0);

    valid = isfinite(zA) & (mv >= cfg6.maskInterpThresh);
    nValid = nnz(valid);

    info = struct();
    info.validN = nValid;

    if nValid < cfg6.minValidN
        J = cfg6.bigPenalty + 10 * (cfg6.minValidN - nValid);
        info.rmse = inf;
        info.penMask = inf;
        info.penRxRy = inf;
        info.penTz = inf;
        info.penRz = inf;
        return;
    end

    dz = z(valid) - zA(valid);

    medr = median(dz);
    mad0 = median(abs(dz - medr));
    sigma = 1.4826 * max(mad0, eps);
    cHuber = 1.5 * sigma;

    w = ones(size(dz));
    idx = abs(dz) > cHuber;
    w(idx) = cHuber ./ abs(dz(idx));

    rmseRobust = sqrt(sum(w .* (dz.^2)) / max(sum(w), eps));

    penMask = cfg6.lambdaMaskPenalty / sqrt(double(nValid));
    penRxRy = cfg6.lambdaRxRy * norm(p(1:2));
    penTz   = cfg6.lambdaTz * abs(p(6));
    penRz   = cfg6.lambdaRz * abs(p(3));

    J = rmseRobust + penMask + penRxRy + penTz + penRz;

    info.rmse = rmseRobust;
    info.penMask = penMask;
    info.penRxRy = penRxRy;
    info.penTz = penTz;
    info.penRz = penRz;
end

%% ========================================================================
function truthErr = evaluate_fused_vs_truth_peaks_local_v6(R, truth_raw, plotCfg)

    Xg = double(truth_raw.Xm);
    Yg = double(truth_raw.Ym);
    Zg = double(truth_raw.Z_clean);
    Mg = logical(truth_raw.mask) & isfinite(Zg);

    A = R.A;
    MA = logical(A.mask) & isfinite(A.Z);
    PA_local = [A.X(MA), A.Y(MA), A.Z(MA)];
    PA_world = apply_pose_struct_local(PA_local, truth_raw.A_pose);

    B = R.B;
    MB = logical(B.mask) & isfinite(B.Z);
    PB_local = [B.X(MB), B.Y(MB), B.Z(MB)];

    [RBA, tBA] = posevec_to_rt_local(R.p6_final);
    PB_in_A = apply_transform_local(PB_local, RBA, tBA);
    PB_world = apply_pose_struct_local(PB_in_A, truth_raw.A_pose);

    FA = scatteredInterpolant(PA_world(:,1), PA_world(:,2), PA_world(:,3), 'natural', 'none');
    FB = scatteredInterpolant(PB_world(:,1), PB_world(:,2), PB_world(:,3), 'natural', 'none');

    ZA = FA(Xg, Yg);
    ZB = FB(Xg, Yg);

    mA = isfinite(ZA);
    mB = isfinite(ZB);

    Zfused = nan(size(Zg));
    both  = mA & mB;
    onlyA = mA & ~mB;
    onlyB = ~mA & mB;

    Zfused(both)  = 0.5 * (ZA(both) + ZB(both));
    Zfused(onlyA) = ZA(onlyA);
    Zfused(onlyB) = ZB(onlyB);

    validRaw = (mA | mB) & Mg & isfinite(Zfused) & isfinite(Zg);

    ErrRaw_um = nan(size(Zg));
    ErrRaw_um(validRaw) = 1e3 * (Zfused(validRaw) - Zg(validRaw));

    [displayMask, pruneInfoDisplay] = build_display_mask_by_support_local(validRaw, mA, mB, plotCfg.displayPrune);
    [pvMask, pruneInfoPV] = build_display_mask_by_support_local(validRaw, mA, mB, plotCfg.pvPrune);

    ErrDisplay_um = ErrRaw_um;

    statMask = validRaw;
    if plotCfg.useDisplayPrunedForStats
        statMask = displayMask;
    end

    ev_raw = ErrRaw_um(validRaw);
    ev_display = ErrDisplay_um(statMask);
    ev_pv = ErrRaw_um(pvMask);

    truthErr = struct();
    truthErr.rmse_um_raw = safe_rmse_local(ev_raw);
    truthErr.mae_um_raw  = safe_mae_local(ev_raw);
    truthErr.pv_um_raw   = safe_pv_local(ev_raw);

    truthErr.rmse_um_display = safe_rmse_local(ev_display);
    truthErr.mae_um_display  = safe_mae_local(ev_display);
    truthErr.pv_um_display   = safe_pv_local(ev_display);

    truthErr.pv_um_pruned = safe_pv_local(ev_pv);
    truthErr.robustPV995_um = safe_rpv_local(ev_pv, 0.3, 99.7);

    truthErr.validN = nnz(validRaw);
    truthErr.displayN = nnz(displayMask);
    truthErr.pvMaskN = nnz(pvMask);

    truthErr.errMap_um_raw = ErrRaw_um;
    truthErr.errMap_um_display = ErrDisplay_um;
    truthErr.validMaskRaw = validRaw;
    truthErr.displayMask = displayMask;
    truthErr.pvMask = pvMask;

    truthErr.pruneInfoDisplay = pruneInfoDisplay;
    truthErr.pruneInfoPV = pruneInfoPV;
    truthErr.Zfused = Zfused;

    tt = linspace(0, 2*pi, 721).';
    circA_local = [A.Rsub*cos(tt), A.Rsub*sin(tt), zeros(size(tt))];
    circA_world = apply_pose_struct_local(circA_local, truth_raw.A_pose);

    circB_local = [B.Rsub*cos(tt), B.Rsub*sin(tt), zeros(size(tt))];
    circB_in_A  = apply_transform_local(circB_local, RBA, tBA);
    circB_world = apply_pose_struct_local(circB_in_A, truth_raw.A_pose);

    truthErr.Xg = Xg;
    truthErr.Yg = Yg;
    truthErr.circA_world = circA_world;
    truthErr.circB_world = circB_world;
end

%% ========================================================================
function plot_single_error_map_local(truthErr, plotCfg, outDir, baseName)

    Xg = truthErr.Xg;
    Yg = truthErr.Yg;
    E  = truthErr.errMap_um_display;
    M  = truthErr.displayMask;
    circA = truthErr.circA_world;
    circB = truthErr.circB_world;

    [rmseTxt, pvTxt, pvLabel] = compose_metric_text_local(truthErr, plotCfg);

    fig = figure('Color','w', 'Position', [90 90 820 680], 'Visible', 'off');
    ax = axes(fig);

    imagesc(ax, Xg(1,:), Yg(:,1), E, 'AlphaData', double(M));
    axis(ax, 'image');
    set(ax, 'YDir', 'normal');
    box(ax, 'on');
    hold(ax, 'on');

    plot(ax, circA(:,1), circA(:,2), 'k-',  'LineWidth', 2.2);
    plot(ax, circB(:,1), circB(:,2), 'k--', 'LineWidth', 2.2);

    colormap(ax, bluewhitered_local(256));

    ev4color = abs(E(M));
    cmax = prctile(ev4color(:), plotCfg.colorPrctile);
    if ~isfinite(cmax) || cmax <= 0
        cmax = max(ev4color(:));
    end
    if ~isfinite(cmax) || cmax <= 0
        cmax = 1;
    end
    caxis(ax, [-cmax, cmax]);

    xmin = min([circA(:,1); circB(:,1)]);
    xmax = max([circA(:,1); circB(:,1)]);
    ymin = min([circA(:,2); circB(:,2)]);
    ymax = max([circA(:,2); circB(:,2)]);
    pad = 1.0;
    xlim(ax, [xmin-pad, xmax+pad]);
    ylim(ax, [ymin-pad, ymax+pad]);

    xlabel(ax, 'x (mm)');
    ylabel(ax, 'y (mm)');
    title(ax, { ...
        'Final fused surface vs original peaks', ...
        sprintf('RMSE=%.3f \\mum, %s=%.3f \\mum', rmseTxt, pvLabel, pvTxt)});

    cb = colorbar(ax);
    ylabel(cb, 'Error (\mum)');

    exportgraphics(fig, fullfile(outDir, [baseName '.png']), 'Resolution', 220);
    exportgraphics(fig, fullfile(outDir, [baseName '.pdf']), 'ContentType', 'vector');
    close(fig);
end

%% ========================================================================
function render_unified_colorbar_pngs_local(allRes, outRoot, plotCfg)

    pngDir = fullfile(outRoot, 'png_unified_colorbar');
    if ~exist(pngDir, 'dir')
        mkdir(pngDir);
    end

    nCase = numel(allRes);

    allVals = [];
    for k = 1:nCase
        S = allRes{k};
        E = S.truthErr.errMap_um_display;
        M = S.truthErr.displayMask;
        vals = abs(E(M));
        vals = vals(isfinite(vals));
        allVals = [allVals; vals(:)]; %#ok<AGROW>
    end

    if isempty(allVals)
        cmax = 1;
    else
        cmax = prctile(allVals, plotCfg.colorPrctile);
        if ~isfinite(cmax) || cmax <= 0
            cmax = max(allVals);
        end
        if ~isfinite(cmax) || cmax <= 0
            cmax = 1;
        end
    end

    save(fullfile(pngDir, 'unified_colorbar_value.mat'), 'cmax');

    for k = 1:nCase
        S = allRes{k};

        Xg = S.truthErr.Xg;
        Yg = S.truthErr.Yg;
        E  = S.truthErr.errMap_um_display;
        M  = S.truthErr.displayMask;

        circA = S.truthErr.circA_world;
        circB = S.truthErr.circB_world;

        [rmseTxt, pvTxt, pvLabel] = compose_metric_text_local(S.truthErr, plotCfg);

        fig = figure('Color','w', 'Position', [90 90 820 680], 'Visible', 'off');
        ax = axes(fig);

        imagesc(ax, Xg(1,:), Yg(:,1), E, 'AlphaData', double(M));
        axis(ax, 'image');
        set(ax, 'YDir', 'normal');
        box(ax, 'on');
        hold(ax, 'on');

        plot(ax, circA(:,1), circA(:,2), 'k-',  'LineWidth', 2.2);
        plot(ax, circB(:,1), circB(:,2), 'k--', 'LineWidth', 2.2);

        colormap(ax, bluewhitered_local(256));
        caxis(ax, [-cmax, cmax]);

        xmin = min([circA(:,1); circB(:,1)]);
        xmax = max([circA(:,1); circB(:,1)]);
        ymin = min([circA(:,2); circB(:,2)]);
        ymax = max([circA(:,2); circB(:,2)]);
        pad = 1.0;
        xlim(ax, [xmin-pad, xmax+pad]);
        ylim(ax, [ymin-pad, ymax+pad]);

        xlabel(ax, 'x (mm)');
        ylabel(ax, 'y (mm)');
        title(ax, { ...
            'Final fused surface vs original peaks', ...
            sprintf('RMSE=%.3f \\mum, %s=%.3f \\mum', rmseTxt, pvLabel, pvTxt)});

        cb = colorbar(ax);
        ylabel(cb, 'Error (\mum)');

        fname = sprintf('%02d_case_unified.png', k);
        exportgraphics(fig, fullfile(pngDir, fname), 'Resolution', 220);
        close(fig);
    end

    figAll = figure('Color','w', 'Position', [60 80 1800 520], 'Visible', 'off');
    tl = tiledlayout(figAll, 1, nCase, 'TileSpacing', 'compact', 'Padding', 'compact');

    for k = 1:nCase
        S = allRes{k};

        Xg = S.truthErr.Xg;
        Yg = S.truthErr.Yg;
        E  = S.truthErr.errMap_um_display;
        M  = S.truthErr.displayMask;

        circA = S.truthErr.circA_world;
        circB = S.truthErr.circB_world;

        [rmseTxt, pvTxt, pvLabel] = compose_metric_text_local(S.truthErr, plotCfg);

        ax = nexttile(tl);
        imagesc(ax, Xg(1,:), Yg(:,1), E, 'AlphaData', double(M));
        axis(ax, 'image');
        set(ax, 'YDir', 'normal');
        box(ax, 'on');
        hold(ax, 'on');

        plot(ax, circA(:,1), circA(:,2), 'k-',  'LineWidth', 2.0);
        plot(ax, circB(:,1), circB(:,2), 'k--', 'LineWidth', 2.0);

        colormap(ax, bluewhitered_local(256));
        caxis(ax, [-cmax, cmax]);

        xmin = min([circA(:,1); circB(:,1)]);
        xmax = max([circA(:,1); circB(:,1)]);
        ymin = min([circA(:,2); circB(:,2)]);
        ymax = max([circA(:,2); circB(:,2)]);
        pad = 1.0;
        xlim(ax, [xmin-pad, xmax+pad]);
        ylim(ax, [ymin-pad, ymax+pad]);

        title(ax, sprintf('RMSE=%.3f \\mum, %s=%.3f \\mum', rmseTxt, pvLabel, pvTxt));
        xlabel(ax, 'x (mm)');
        ylabel(ax, 'y (mm)');
    end

    cb = colorbar;
    cb.Layout.Tile = 'east';
    ylabel(cb, 'Error (\mum)');

    exportgraphics(figAll, fullfile(pngDir, 'all_cases_unified_colorbar.png'), 'Resolution', 220);
    close(figAll);
end


%% ========================================================================
function plot_noise_summary_curves_local(T, xVarName, xLabelText, outDir, tag)

    metricList = { ...
        'mean_e_t_um',      'std_e_t_um',      'e_t (\mum)'; ...
        'mean_e_R_deg',     'std_e_R_deg',     'e_R (deg)'; ...
        'mean_RMSE_um',     'std_RMSE_um',     'RMSE (\mum)'; ...
        'mean_MAE_um',      'std_MAE_um',      'MAE (\mum)'; ...
        'mean_PV_raw_um',   'std_PV_raw_um',   'PV raw (\mum)'; ...
        'mean_PV_pruned_um','std_PV_pruned_um','PV pruned (\mum)'; ...
        'mean_rPV995_um',   'std_rPV995_um',   'rPV995 (\mum)'; ...
        'mean_runtime_s',   'std_runtime_s',   'Runtime (s)'};

    x = T.(xVarName);

    for i = 1:size(metricList,1)
        yName  = metricList{i,1};
        sdName = metricList{i,2};
        yLabelText = metricList{i,3};

        y  = T.(yName);
        ys = T.(sdName);

        fig = figure('Color','w','Position',[100 100 760 560],'Visible','off');
        ax = axes(fig);

        errorbar(ax, x, y, ys, '-o', 'LineWidth', 1.6, 'MarkerSize', 6);
        grid(ax, 'on');
        box(ax, 'on');

        if numel(x) >= 2
            dx = max(diff(sort(x(:))));
            if ~isfinite(dx) || dx <= 0
                dx = max(abs(x(:)));
            end
            if ~isfinite(dx) || dx <= 0
                dx = 1;
            end
            xlim(ax, [min(x)-0.6*dx, max(x)+0.6*dx]);
        elseif numel(x) == 1
            dx = max(abs(x), 1);
            xlim(ax, [x-0.6*dx, x+0.6*dx]);
        end

        xlabel(ax, xLabelText);
        ylabel(ax, yLabelText);
        title(ax, sprintf('%s vs %s', yLabelText, xLabelText), 'Interpreter', 'none');

        exportgraphics(fig, fullfile(outDir, sprintf('%s_%s.png', tag, yName)), 'Resolution', 220);
        exportgraphics(fig, fullfile(outDir, sprintf('%s_%s.pdf', tag, yName)), 'ContentType','vector');
        close(fig);
    end
end

%% ========================================================================
function [local, pose] = build_local_grid_subap_local(sub_raw, sourceMode)
    ds   = double(sub_raw.ds);
    Rsub = double(sub_raw.Rsub);

    xVec = -Rsub : ds : Rsub;
    yVec = -Rsub : ds : Rsub;
    [X, Y] = meshgrid(xVec, yVec);

    Z = nan(size(X));
    mask = false(size(X));

    ix = round((double(sub_raw.x(:)) - xVec(1)) / ds) + 1;
    iy = round((double(sub_raw.y(:)) - yVec(1)) / ds) + 1;

    switch lower(sourceMode)
        case 'noisy'
            zsrc = double(sub_raw.z(:));
        case 'clean'
            zsrc = double(sub_raw.z_clean(:));
        otherwise
            error('sourceMode 仅支持 noisy / clean');
    end

    valid = ix >= 1 & ix <= numel(xVec) & iy >= 1 & iy <= numel(yVec) & isfinite(zsrc);
    ix = ix(valid);
    iy = iy(valid);
    zv = zsrc(valid);

    ind = sub2ind(size(Z), iy, ix);
    Z(ind) = zv;
    mask(ind) = true;

    local = struct();
    local.X = X;
    local.Y = Y;
    local.Z = Z;
    local.mask = mask;
    local.ds = ds;
    local.Rsub = Rsub;
    local.Zfill = fill_invalid_by_nearest_local(X, Y, Z, mask & isfinite(Z));

    pose = struct();
    pose.R_w = double(sub_raw.pose_R);
    pose.T_w = double(sub_raw.pose_t(:));
end

function local2 = preprocess_local_sag_local(local, pre)
    X = local.X;
    Y = local.Y;
    Z = local.Z;
    M = local.mask & isfinite(Z);

    Z2 = Z;

    if pre.smoothSigmaPix > 0
        Z2 = masked_gaussian_smooth_local(Z2, M, pre.smoothSigmaPix);
    end

    if pre.removePlane
        x = X(M);
        y = Y(M);
        z = Z2(M);

        A = [x(:), y(:), ones(nnz(M),1)];
        coef = A \ z(:);

        tmp = Z2;
        tmp(M) = z(:) - A * coef;
        Z2 = tmp;
    end

    if pre.clipSigma > 0
        z = Z2(M);
        mu = mean(z, 'omitnan');
        sd = std(z, 0, 'omitnan');
        if isfinite(sd) && sd > 0
            lo = mu - pre.clipSigma * sd;
            hi = mu + pre.clipSigma * sd;
            Z2(M) = min(max(Z2(M), lo), hi);
        end
    end

    Zfill = fill_invalid_by_nearest_local(X, Y, Z2, M);

    local2 = local;
    local2.Z = Z2;
    local2.Zfill = Zfill;
end

function Zs = masked_gaussian_smooth_local(Z, M, sigmaPix)
    if sigmaPix <= 0
        Zs = Z;
        return;
    end

    rad = max(1, ceil(4 * sigmaPix));
    [xx, yy] = meshgrid(-rad:rad, -rad:rad);
    K = exp(-(xx.^2 + yy.^2) / (2 * sigmaPix^2));
    K = K / sum(K(:));

    Z0 = Z;
    Z0(~M) = 0;

    num = conv2(Z0, K, 'same');
    den = conv2(double(M), K, 'same');

    Zs = Z;
    valid = M & (den > 1e-12);
    Zs(valid) = num(valid) ./ den(valid);
end

function modelA = build_surface_model_from_local_local(A)
    XA = double(A.X);
    YA = double(A.Y);
    ZA = double(A.Z);
    MA = logical(A.mask) & isfinite(ZA);

    if isfield(A, 'Zfill')
        Zfill = A.Zfill;
    else
        Zfill = fill_invalid_by_nearest_local(XA, YA, ZA, MA);
    end

    modelA = struct();
    modelA.X = XA;
    modelA.Y = YA;
    modelA.xVec = XA(1,:);
    modelA.yVec = YA(:,1);
    modelA.Zfill = Zfill;
    modelA.mask = MA;
end

function Zfill = fill_invalid_by_nearest_local(X, Y, Z, mask)
    Zfill = Z;
    miss = ~(logical(mask) & isfinite(Z));
    good = ~miss;

    if ~any(good(:))
        Zfill(:) = 0;
        return;
    end
    if ~any(miss(:))
        return;
    end

    F = scatteredInterpolant(X(good), Y(good), Z(good), 'nearest', 'nearest');
    Zfill(miss) = F(X(miss), Y(miss));
end

%% ========================================================================
function truthRel = true_relative_pose_local(SA, SB)
    RA = SA.R_w;
    RB = SB.R_w;
    TA = SA.T_w(:);
    TB = SB.T_w(:);

    R_true = RA.' * RB;
    t_true = RA.' * (TB - TA);
    [rx, ry, rz] = rotm_to_eul_zyx_local(R_true);

    truthRel = struct();
    truthRel.R  = R_true;
    truthRel.t  = t_true(:).';
    truthRel.rx = rx;
    truthRel.ry = ry;
    truthRel.rz = rz;
end

function [R, t] = posevec_to_rt_local(p)
    rx = p(1);
    ry = p(2);
    rz = p(3);

    R = eul_zyx_to_rotm_local(rx, ry, rz);
    t = [p(4); p(5); p(6)];
end

function P2 = apply_transform_local(P1, R, t)
    P2 = (R * P1.' + t(:)).';
end

function Pw = apply_pose_struct_local(P, poseS)
    Rw = double(poseS.R);
    tw = double(poseS.t(:));
    Pw = (Rw * P.' + tw).';
end

function R = eul_zyx_to_rotm_local(rx, ry, rz)
    cx = cos(rx); sx = sin(rx);
    cy = cos(ry); sy = sin(ry);
    cz = cos(rz); sz = sin(rz);

    R = [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx; ...
         sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx; ...
         -sy,   cy*sx,            cy*cx];
end

function [rx, ry, rz] = rotm_to_eul_zyx_local(R)
    ry = -asin(max(min(R(3,1), 1), -1));
    cy = cos(ry);

    if abs(cy) > 1e-12
        rx = atan2(R(3,2), R(3,3));
        rz = atan2(R(2,1), R(1,1));
    else
        rx = 0;
        rz = atan2(-R(1,2), R(2,2));
    end
end

function q = clip_box_local(q, lb, ub)
    q = min(max(q, lb), ub);
end

function cmap = bluewhitered_local(m)
    if nargin < 1
        m = 256;
    end
    n1 = floor(m/2);
    n2 = m - n1;

    c1 = [linspace(0.10,1.00,n1)', linspace(0.25,1.00,n1)', linspace(0.85,1.00,n1)'];
    c2 = [linspace(1.00,0.85,n2)', linspace(1.00,0.10,n2)', linspace(1.00,0.10,n2)'];

    cmap = [c1; c2];
end


function d = solve_center_distance_from_overlap_local(R, overlapRatio)
% 求解两个等半径圆在给定重叠面积比例（相对于单圆面积）下的中心距 d

    overlapRatio = max(1e-8, min(1-1e-8, overlapRatio));

    f = @(d) circle_overlap_ratio_local(R, d) - overlapRatio;

    dLo = 0;
    dHi = 2*R - 1e-9;

    if f(dLo) * f(dHi) > 0
        error('重叠率求解失败：无法在 [0, 2R) 上建立有根区间。');
    end

    d = fzero(f, [dLo, dHi]);
end

function ratio = circle_overlap_ratio_local(R, d)
% 两个等半径圆的重叠面积 / 单圆面积

    d = max(0, min(2*R, d));

    if d <= 0
        Aov = pi * R^2;
    elseif d >= 2*R
        Aov = 0;
    else
        Aov = 2*R^2 * acos(d/(2*R)) - 0.5 * d * sqrt(max(4*R^2 - d^2, 0));
    end

    ratio = Aov / (pi * R^2);
end


function [displayMask, info] = build_display_mask_by_support_local(validRaw, mA, mB, pruneCfg)

    displayMask = validRaw;

    info = struct();
    info.enabled = false;
    info.nRemoved_single = 0;
    info.nRemoved_dual = 0;
    info.nRemoved_sparse = 0;

    if nargin < 4 || isempty(pruneCfg)
        return;
    end
    if ~isfield(pruneCfg, 'enabled') || ~pruneCfg.enabled
        return;
    end

    info.enabled = true;

    if ~isfield(pruneCfg, 'singleSupportBandPix')
        pruneCfg.singleSupportBandPix = 1;
    end
    if ~isfield(pruneCfg, 'dualSupportBandPix')
        pruneCfg.dualSupportBandPix = 0;
    end
    if ~isfield(pruneCfg, 'minNeighborCount')
        pruneCfg.minNeighborCount = 0;
    end

    % ---------- 单孔径支撑区：删除边界带 ----------
    if pruneCfg.singleSupportBandPix > 0
        edgeA1 = build_edge_band_weight_local(mA, pruneCfg.singleSupportBandPix, 'linear') > 0;
        edgeB1 = build_edge_band_weight_local(mB, pruneCfg.singleSupportBandPix, 'linear') > 0;

        badSingle = (mA & ~mB & edgeA1) | (~mA & mB & edgeB1);
        badSingle = badSingle & validRaw;

        displayMask(badSingle) = false;
        info.nRemoved_single = nnz(badSingle);
    end

    % ---------- 双孔径共同支撑区：可选删除边界带 ----------
    if pruneCfg.dualSupportBandPix > 0
        edgeA2 = build_edge_band_weight_local(mA, pruneCfg.dualSupportBandPix, 'linear') > 0;
        edgeB2 = build_edge_band_weight_local(mB, pruneCfg.dualSupportBandPix, 'linear') > 0;

        badDual = (mA & mB & edgeA2 & edgeB2) & validRaw;
        displayMask(badDual) = false;
        info.nRemoved_dual = nnz(badDual);
    end

    % ---------- 删除孤立/稀疏像素 ----------
    if pruneCfg.minNeighborCount > 0
        K = ones(3,3);
        neigh = conv2(double(displayMask), K, 'same');
        badSparse = displayMask & (neigh <= pruneCfg.minNeighborCount);
        displayMask(badSparse) = false;
        info.nRemoved_sparse = nnz(badSparse);
    end
end

function w = build_edge_band_weight_local(mask, bandPix, blendMode)

    mask = logical(mask);
    w = zeros(size(mask));

    cur = mask;
    for k = 1:bandPix
        nxt = binary_erode_local(cur, 1);
        ring = cur & ~nxt;

        t = 1 - (k-1)/bandPix;
        switch lower(blendMode)
            case 'linear'
                wk = t;
            case 'cosine'
                wk = 0.5 * (1 - cos(pi * t));
            otherwise
                wk = t;
        end

        w(ring) = wk;
        cur = nxt;

        if ~any(cur(:))
            break;
        end
    end
end

function maskOut = binary_erode_local(maskIn, nIter)

    maskOut = logical(maskIn);
    K = ones(3,3);

    for it = 1:nIter
        cnt = conv2(double(maskOut), K, 'same');
        maskOut = maskOut & (cnt == 9);
    end
end

function y = safe_rmse_local(v)
    v = v(isfinite(v));
    if isempty(v)
        y = NaN;
    else
        y = sqrt(mean(v.^2));
    end
end

function y = safe_mae_local(v)
    v = v(isfinite(v));
    if isempty(v)
        y = NaN;
    else
        y = mean(abs(v));
    end
end

function y = safe_pv_local(v)
    v = v(isfinite(v));
    if isempty(v)
        y = NaN;
    else
        y = max(v) - min(v);
    end
end

function y = safe_rpv_local(v, pLo, pHi)
    v = v(isfinite(v));
    if isempty(v)
        y = NaN;
    else
        y = prctile(v, pHi) - prctile(v, pLo);
    end
end

function [rmseTxt, pvTxt, pvLabel] = compose_metric_text_local(truthErr, plotCfg)

    if plotCfg.useDisplayPrunedForStats
        rmseTxt = truthErr.rmse_um_display;
    else
        rmseTxt = truthErr.rmse_um_raw;
    end

    switch lower(plotCfg.titlePVMode)
        case 'raw'
            pvTxt = truthErr.pv_um_raw;
            pvLabel = 'PV(raw)';
        case 'robust995'
            pvTxt = truthErr.robustPV995_um;
            pvLabel = 'rPV';
        otherwise
            pvTxt = truthErr.pv_um_pruned;
            pvLabel = 'PV(pruned)';
    end
end


function poseErr = compute_pose_error_metrics_local(p, truthRel)
    dt = [p(4)-truthRel.t(1); p(5)-truthRel.t(2); p(6)-truthRel.t(3)];
    e_t = norm(dt);

    Rest = eul_zyx_to_rotm_local(p(1), p(2), p(3));
    Rerr = Rest * truthRel.R.';
    val = (trace(Rerr) - 1) / 2;
    val = min(max(val, -1), 1);
    e_R = rad2deg(acos(val));

    poseErr = struct();
    poseErr.dtx_um = 1e3 * dt(1);
    poseErr.dty_um = 1e3 * dt(2);
    poseErr.dtz_um = 1e3 * dt(3);
    poseErr.e_t_um = 1e3 * e_t;
    poseErr.e_R_deg = e_R;
end

function poseComp = compute_pose_error_components_local(p, truthRel)

    dt = [p(4)-truthRel.t(1); p(5)-truthRel.t(2); p(6)-truthRel.t(3)];

    Rest = eul_zyx_to_rotm_local(p(1), p(2), p(3));
    Rerr = Rest * truthRel.R.';
    [drx, dry, drz] = rotm_to_eul_zyx_local(Rerr);

    poseComp = struct();
    poseComp.dtx_um = 1e3 * dt(1);
    poseComp.dty_um = 1e3 * dt(2);
    poseComp.dtz_um = 1e3 * dt(3);

    poseComp.drx_deg = rad2deg(drx);
    poseComp.dry_deg = rad2deg(dry);
    poseComp.drz_deg = rad2deg(drz);
end
