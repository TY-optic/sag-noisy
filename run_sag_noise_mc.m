function summaryTbl = run_sag_noise_mc(cfg)
%RUN_SAG_NOISE_MC
% Sag-based 噪声 Monte Carlo 统计（含 oracle / registration-only 分离）
%
% 输出三类核心误差：
% 1) alg_fused_RMSE      : 含噪子孔径 + 估计位姿 的端到端误差
% 2) oracle_fused_RMSE   : 含噪子孔径 + 真值位姿 的噪声底
% 3) poseOnly_clean_RMSE : 理想无噪子孔径 + 估计位姿 的纯配准误差
%
% 同时输出：
% delta_fused_RMSE      = max(alg - oracle, 0)
% reg_only_energy_RMSE  = sqrt(max(alg^2 - oracle^2, 0))
%
% cfg.mode = 'intensity' | 'spectrum'

cfg = apply_run_defaults(cfg);

assert(exist(cfg.generatorFcn, 'file') == 2, ...
    '未找到数据生成器 %s', cfg.generatorFcn);
assert(exist('build_sag_feature_map', 'file') == 2, ...
    '未找到 build_sag_feature_map.m');

if ~exist(cfg.io.outDir, 'dir')
    mkdir(cfg.io.outDir);
end

set(groot, 'defaultAxesFontName', cfg.plot.fontName);
set(groot, 'defaultTextFontName', cfg.plot.fontName);

switch lower(cfg.mode)
    case 'intensity'
        xList = cfg.noise.sigmaList_um(:).';
        xLabel = '$\sigma_n$ ($\mu$m)';
        titleMain = sprintf('Sag-based noise-intensity robustness (L_c = %.2f mm, N = %d)', ...
            cfg.noise.Lc_mm, cfg.mc.N);
    case 'spectrum'
        xList = cfg.noise.LcList_mm(:).';
        xLabel = '$L_c$ (mm)';
        titleMain = sprintf('Sag-based noise-spectrum robustness (sigma_n = %.2f um, N = %d)', ...
            cfg.noise.sigma_n_um, cfg.mc.N);
    otherwise
        error('cfg.mode 仅支持 intensity / spectrum');
end

nX  = numel(xList);
Nmc = cfg.mc.N;

allTrial = cell(nX, Nmc);

algRmseMat_mm            = nan(nX, Nmc);
algInnerRmseMat_mm       = nan(nX, Nmc);
oracleRmseMat_mm         = nan(nX, Nmc);
oracleInnerRmseMat_mm    = nan(nX, Nmc);
deltaRmseMat_mm          = nan(nX, Nmc);
regEnergyRmseMat_mm      = nan(nX, Nmc);
poseOnlyRmseMat_mm       = nan(nX, Nmc);
poseOnlyInnerRmseMat_mm  = nan(nX, Nmc);

algOverlapRmseMat_mm       = nan(nX, Nmc);
oracleOverlapRmseMat_mm    = nan(nX, Nmc);
poseOnlyOverlapRmseMat_mm  = nan(nX, Nmc);

solveTimeMat_s           = nan(nX, Nmc);
etMat_mm                 = nan(nX, Nmc);
eRMat_deg                = nan(nX, Nmc);
actualNoiseMat_um        = nan(nX, Nmc);
exitFlagMat              = nan(nX, Nmc);
successMat               = false(nX, Nmc);

fprintf('============================================================\n');
fprintf('Sag-based Monte Carlo 统计开始（oracle / reg-only 分离版）\n');
fprintf('mode = %s\n', cfg.mode);
fprintf('Nmc  = %d\n', Nmc);
fprintf('Generator = %s\n', cfg.generatorFcn);
fprintf('============================================================\n');

for i = 1:nX
    switch lower(cfg.mode)
        case 'intensity'
            sigma_n_um = xList(i);
            Lc_mm = cfg.noise.Lc_mm;
            fprintf('\n------------------------------------------------------------\n');
            fprintf('Sigma %d/%d : sigma_n = %.4f um, Lc = %.4f mm\n', ...
                i, nX, sigma_n_um, Lc_mm);
            fprintf('------------------------------------------------------------\n');
        case 'spectrum'
            sigma_n_um = cfg.noise.sigma_n_um;
            Lc_mm = xList(i);
            fprintf('\n------------------------------------------------------------\n');
            fprintf('Lc %d/%d : sigma_n = %.4f um, Lc = %.4f mm\n', ...
                i, nX, sigma_n_um, Lc_mm);
            fprintf('------------------------------------------------------------\n');
    end

    for k = 1:Nmc
        rngSeed = cfg.mc.baseSeed + 1000*i + k;

        Ttrial = struct();
        Ttrial.seed = rngSeed;
        Ttrial.sigma_n_um = sigma_n_um;
        Ttrial.Lc_mm = Lc_mm;
        Ttrial.success = false;
        Ttrial.error_message = '';

        try
            [subA_raw, subB_raw, truth_raw] = feval(cfg.generatorFcn, ...
                sigma_n_um, Lc_mm, rngSeed, cfg.pose6dof);

            data = prepare_sag_case_from_generator(subA_raw, subB_raw, truth_raw, cfg);
            Ttrial.actualNoise_um = 1e3 * truth_raw.rmse_noise_actual;

            [ZAfeat, maskZA, auxA] = build_sag_feature_map(data.localA_noisy, cfg.step1);
            [ZBfeat, maskZB, auxB] = build_sag_feature_map(data.localB_noisy, cfg.step1);

            if nnz(maskZA) < cfg.step1.minOverlapPix || nnz(maskZB) < cfg.step1.minOverlapPix
                error('Sag 有效区域过小：A=%d, B=%d', nnz(maskZA), nnz(maskZB));
            end

            coarse2d = coarse_search_sag_se2( ...
                ZAfeat, maskZA, ZBfeat, maskZB, ...
                data.localA_noisy.X, data.localA_noisy.Y, ...
                data.localB_noisy.X, data.localB_noisy.Y, cfg.step1);

            if cfg.step1.doFineRefine
                fine2d = refine_search_sag_se2( ...
                    ZAfeat, maskZA, ZBfeat, maskZB, ...
                    data.localA_noisy.X, data.localA_noisy.Y, ...
                    data.localB_noisy.X, data.localB_noisy.Y, ...
                    coarse2d, cfg.step1);
            else
                fine2d = coarse_to_fine_passthrough(coarse2d);
            end

            modelA = build_surface_model_from_sag(data.localA_noisy, cfg.step2);
            p0 = build_initial_pose_from_step1(fine2d);
            [PBcand, metaCand] = build_candidate_points_from_step1( ...
                data.localA_noisy, data.localB_noisy, fine2d, p0, cfg.step2);

            if size(PBcand, 1) < cfg.step2.minCandidatePts
                error('候选点数不足：%d < %d', size(PBcand,1), cfg.step2.minCandidatePts);
            end

            tSolve = tic;
            [pEst, exitflag, solverOut] = refine_pose_height_residual(PBcand, p0, modelA, cfg.step2);
            solveTime = toc(tSolve);

            reg_est = sag_pose_to_reg(pEst);
            reg_est = attach_overlap_metrics_to_reg(reg_est, data.A_noisy, data.B_noisy, cfg.eval);

            reg_true = truth_relative_pose_to_reg(data.truthRel);
            poseErr = evaluate_pose_error_sag(reg_est, data.truthRel);

            [~, ~, ~, stats_alg] = fuse_and_evaluate_surface_pair(data, reg_est, cfg, 'noisy');
            [~, ~, ~, stats_oracle] = fuse_and_evaluate_surface_pair(data, reg_true, cfg, 'noisy');
            [~, ~, ~, stats_poseOnly] = fuse_and_evaluate_surface_pair(data, reg_est, cfg, 'clean');

            algRmseMat_mm(i, k)           = stats_alg.fusedRMSE;
            algInnerRmseMat_mm(i, k)      = stats_alg.innerRMSE;
            oracleRmseMat_mm(i, k)        = stats_oracle.fusedRMSE;
            oracleInnerRmseMat_mm(i, k)   = stats_oracle.innerRMSE;
            deltaRmseMat_mm(i, k)         = max(stats_alg.fusedRMSE - stats_oracle.fusedRMSE, 0);
            regEnergyRmseMat_mm(i, k)     = sqrt(max(stats_alg.fusedRMSE^2 - stats_oracle.fusedRMSE^2, 0));
            poseOnlyRmseMat_mm(i, k)      = stats_poseOnly.fusedRMSE;
            poseOnlyInnerRmseMat_mm(i, k) = stats_poseOnly.innerRMSE;

            algOverlapRmseMat_mm(i, k)      = stats_alg.overlapRMSE;
            oracleOverlapRmseMat_mm(i, k)   = stats_oracle.overlapRMSE;
            poseOnlyOverlapRmseMat_mm(i, k) = stats_poseOnly.overlapRMSE;

            solveTimeMat_s(i, k)         = solveTime;
            etMat_mm(i, k)               = poseErr.et_mm;
            eRMat_deg(i, k)              = poseErr.eR_deg;
            actualNoiseMat_um(i, k)      = Ttrial.actualNoise_um;
            exitFlagMat(i, k)            = exitflag;
            successMat(i, k)             = true;

            Ttrial.success = true;
            Ttrial.reg_est = reg_est;
            Ttrial.reg_true = reg_true;
            Ttrial.poseErr = poseErr;
            Ttrial.stats_alg = stats_alg;
            Ttrial.stats_oracle = stats_oracle;
            Ttrial.stats_poseOnly = stats_poseOnly;
            Ttrial.coarse2d = coarse2d;
            Ttrial.fine2d = fine2d;
            Ttrial.exitflag = exitflag;
            Ttrial.solverOut = solverOut;
            Ttrial.modelA = rmfield(modelA, {'Zfill','mask'});
            Ttrial.auxA = auxA;
            Ttrial.auxB = auxB;
            Ttrial.candidateMeta = metaCand;
            Ttrial.dataBrief = struct('truthRel', data.truthRel);

            allTrial{i, k} = Ttrial;

            fprintf([' MC %02d/%02d | actual noise = %.4f um | alg fused = %.4f um | ' ...
                     'oracle = %.4f um | delta = %.4f um | reg-energy = %.4f um | ' ...
                     'pose-only(clean) = %.4f um | e_t = %.4f um | e_R = %.4e deg\n'], ...
                    k, Nmc, actualNoiseMat_um(i, k), ...
                    1e3*stats_alg.fusedRMSE, 1e3*stats_oracle.fusedRMSE, ...
                    1e3*deltaRmseMat_mm(i, k), 1e3*regEnergyRmseMat_mm(i, k), ...
                    1e3*stats_poseOnly.fusedRMSE, ...
                    1e3*poseErr.et_mm, poseErr.eR_deg);

        catch ME
            Ttrial.error_message = getReport(ME, 'basic', 'hyperlinks', 'off');
            allTrial{i, k} = Ttrial;
            warning('Trial failed at i=%d, k=%d:\n%s', i, k, Ttrial.error_message);
        end
    end
end

switch lower(cfg.mode)
    case 'intensity'
        summaryTbl = make_summary_table_intensity(xList, successMat, actualNoiseMat_um, ...
            algRmseMat_mm, algInnerRmseMat_mm, ...
            oracleRmseMat_mm, oracleInnerRmseMat_mm, ...
            deltaRmseMat_mm, regEnergyRmseMat_mm, ...
            poseOnlyRmseMat_mm, poseOnlyInnerRmseMat_mm, ...
            algOverlapRmseMat_mm, oracleOverlapRmseMat_mm, poseOnlyOverlapRmseMat_mm, ...
            etMat_mm, eRMat_deg, solveTimeMat_s);
        csvName = 'noise_intensity_sag_based_6dof_v2_summary.csv';
        matName = 'noise_intensity_sag_based_6dof_v2_results.mat';
        figBase = 'fig_noise_intensity_sag_based_6dof_v2';
    case 'spectrum'
        summaryTbl = make_summary_table_spectrum(xList, successMat, actualNoiseMat_um, ...
            algRmseMat_mm, algInnerRmseMat_mm, ...
            oracleRmseMat_mm, oracleInnerRmseMat_mm, ...
            deltaRmseMat_mm, regEnergyRmseMat_mm, ...
            poseOnlyRmseMat_mm, poseOnlyInnerRmseMat_mm, ...
            algOverlapRmseMat_mm, oracleOverlapRmseMat_mm, poseOnlyOverlapRmseMat_mm, ...
            etMat_mm, eRMat_deg, solveTimeMat_s);
        csvName = 'noise_spectrum_sag_based_6dof_v2_summary.csv';
        matName = 'noise_spectrum_sag_based_6dof_v2_results.mat';
        figBase = 'fig_noise_spectrum_sag_based_6dof_v2';
end

disp(' ');
disp('===================== Summary table =====================');
disp(summaryTbl);

writetable(summaryTbl, fullfile(cfg.io.outDir, csvName));
save(fullfile(cfg.io.outDir, matName), ...
    'cfg', 'allTrial', 'summaryTbl', ...
    'algRmseMat_mm', 'algInnerRmseMat_mm', ...
    'oracleRmseMat_mm', 'oracleInnerRmseMat_mm', ...
    'deltaRmseMat_mm', 'regEnergyRmseMat_mm', ...
    'poseOnlyRmseMat_mm', 'poseOnlyInnerRmseMat_mm', ...
    'algOverlapRmseMat_mm', 'oracleOverlapRmseMat_mm', 'poseOnlyOverlapRmseMat_mm', ...
    'solveTimeMat_s', 'etMat_mm', 'eRMat_deg', ...
    'actualNoiseMat_um', 'exitFlagMat', 'successMat', '-v7.3');

plot_total_vs_oracle(xList, ...
    1e3*mean(algRmseMat_mm, 2, 'omitnan'),      1e3*std(algRmseMat_mm, 0, 2, 'omitnan'), ...
    1e3*mean(oracleRmseMat_mm, 2, 'omitnan'),   1e3*std(oracleRmseMat_mm, 0, 2, 'omitnan'), ...
    xLabel, titleMain, ...
    fullfile(cfg.io.outDir, [figBase '_alg_vs_oracle.pdf']), ...
    fullfile(cfg.io.outDir, [figBase '_alg_vs_oracle.png']), cfg.plot);

plot_registration_only_metrics(xList, ...
    1e3*mean(deltaRmseMat_mm, 2, 'omitnan'),        1e3*std(deltaRmseMat_mm, 0, 2, 'omitnan'), ...
    1e3*mean(regEnergyRmseMat_mm, 2, 'omitnan'),    1e3*std(regEnergyRmseMat_mm, 0, 2, 'omitnan'), ...
    1e3*mean(poseOnlyRmseMat_mm, 2, 'omitnan'),     1e3*std(poseOnlyRmseMat_mm, 0, 2, 'omitnan'), ...
    xLabel, titleMain, ...
    fullfile(cfg.io.outDir, [figBase '_registration_only.pdf']), ...
    fullfile(cfg.io.outDir, [figBase '_registration_only.png']), cfg.plot);

plot_pose_errors(xList, ...
    1e3*mean(etMat_mm, 2, 'omitnan'), 1e3*std(etMat_mm, 0, 2, 'omitnan'), ...
    mean(eRMat_deg, 2, 'omitnan'),    std(eRMat_deg, 0, 2, 'omitnan'), ...
    xLabel, titleMain, ...
    fullfile(cfg.io.outDir, [figBase '_pose_error.pdf']), ...
    fullfile(cfg.io.outDir, [figBase '_pose_error.png']), cfg.plot);

fprintf('\n结果已保存到: %s\n', cfg.io.outDir);
end

function cfg = apply_run_defaults(cfg)
    if ~isfield(cfg, 'mode') || isempty(cfg.mode)
        cfg.mode = 'intensity';
    end
    if ~isfield(cfg, 'generatorFcn') || isempty(cfg.generatorFcn)
        cfg.generatorFcn = 'generate_noisy_peaks_subaps_6dof';
    end
    if ~isfield(cfg, 'pose6dof') || isempty(cfg.pose6dof)
        cfg.pose6dof = struct( ...
            'tx_mm',  [], ...
            'ty_mm',  [], ...
            'tz_mm',   0.010, ...
            'rx_deg',  0.020, ...
            'ry_deg', -0.040, ...
            'rz_deg',  0.200);
    end
    if ~isfield(cfg, 'noise') || isempty(cfg.noise)
        cfg.noise = struct();
    end
    if ~isfield(cfg, 'mc') || isempty(cfg.mc)
        cfg.mc = struct();
    end
    if ~isfield(cfg, 'io') || isempty(cfg.io)
        cfg.io = struct();
    end
    if ~isfield(cfg, 'plot') || isempty(cfg.plot)
        cfg.plot = struct();
    end
    cfg = apply_step1_defaults(cfg);
    cfg = apply_step2_defaults(cfg);
    cfg = apply_eval_defaults(cfg);

    if ~isfield(cfg.noise, 'sigmaList_um') || isempty(cfg.noise.sigmaList_um)
        cfg.noise.sigmaList_um = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
    end
    if ~isfield(cfg.noise, 'Lc_mm') || isempty(cfg.noise.Lc_mm)
        cfg.noise.Lc_mm = 1.5;
    end
    if ~isfield(cfg.noise, 'sigma_n_um') || isempty(cfg.noise.sigma_n_um)
        cfg.noise.sigma_n_um = 1.0;
    end
    if ~isfield(cfg.noise, 'LcList_mm') || isempty(cfg.noise.LcList_mm)
        cfg.noise.LcList_mm = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0];
    end

    if ~isfield(cfg.mc, 'N') || isempty(cfg.mc.N)
        cfg.mc.N = 20;
    end
    if ~isfield(cfg.mc, 'baseSeed') || isempty(cfg.mc.baseSeed)
        cfg.mc.baseSeed = 20260401;
    end

    if ~isfield(cfg.io, 'outDir') || isempty(cfg.io.outDir)
        switch lower(cfg.mode)
            case 'intensity'
                cfg.io.outDir = fullfile(pwd, 'figures', 'noise_intensity_sag_based_6dof_v2');
            case 'spectrum'
                cfg.io.outDir = fullfile(pwd, 'figures', 'noise_spectrum_sag_based_6dof_v2');
        end
    end

    if ~isfield(cfg.plot, 'fontName') || isempty(cfg.plot.fontName)
        cfg.plot.fontName = 'Times New Roman';
    end
    if ~isfield(cfg.plot, 'lineWidth') || isempty(cfg.plot.lineWidth)
        cfg.plot.lineWidth = 1.6;
    end
    if ~isfield(cfg.plot, 'markerSize') || isempty(cfg.plot.markerSize)
        cfg.plot.markerSize = 7;
    end
end

function cfg = apply_step1_defaults(cfg)
    if ~isfield(cfg, 'step1') || isempty(cfg.step1)
        cfg.step1 = struct();
    end
    s1 = cfg.step1;

    if ~isfield(s1, 'marginPix') || isempty(s1.marginPix)
        s1.marginPix = 2;
    end
    if ~isfield(s1, 'removeBestFitPlane') || isempty(s1.removeBestFitPlane)
        s1.removeBestFitPlane = true;
    end
    if ~isfield(s1, 'smoothSigmaPix') || isempty(s1.smoothSigmaPix)
        s1.smoothSigmaPix = 0.0;
    end
    if ~isfield(s1, 'useZScore') || isempty(s1.useZScore)
        s1.useZScore = true;
    end
    if ~isfield(s1, 'clipSigma') || isempty(s1.clipSigma)
        s1.clipSigma = 6.0;
    end
    if ~isfield(s1, 'thetaSearchDeg') || isempty(s1.thetaSearchDeg)
        s1.thetaSearchDeg = -4.0 : 0.05 : 4.0;
    end
    
    % === 修复 2：提高边缘重叠检测的下限像素面积 ===
    if ~isfield(s1, 'minOverlapPix') || isempty(s1.minOverlapPix)
        s1.minOverlapPix = 3500; % 之前是 1200，增大以抑制边缘虚假峰值
    end
    % ==============================================

    if ~isfield(s1, 'doFineRefine') || isempty(s1.doFineRefine)
        s1.doFineRefine = true;
    end
    if ~isfield(s1, 'thetaFineSpanDeg') || isempty(s1.thetaFineSpanDeg)
        s1.thetaFineSpanDeg = 0.12;
    end
    if ~isfield(s1, 'thetaFineNum') || isempty(s1.thetaFineNum)
        s1.thetaFineNum = 13;
    end
    if ~isfield(s1, 'shiftFineSpanPx') || isempty(s1.shiftFineSpanPx)
        s1.shiftFineSpanPx = 1.2;
    end
    if ~isfield(s1, 'shiftFineNum') || isempty(s1.shiftFineNum)
        s1.shiftFineNum = 13;
    end
    if ~isfield(s1, 'maskInterpThresh') || isempty(s1.maskInterpThresh)
        s1.maskInterpThresh = 0.85;
    end

    cfg.step1 = s1;
end

function cfg = apply_step2_defaults(cfg)
    if ~isfield(cfg, 'step2') || isempty(cfg.step2)
        cfg.step2 = struct();
    end
    s2 = cfg.step2;

    if ~isfield(s2, 'minCandidatePts') || isempty(s2.minCandidatePts)
        s2.minCandidatePts = 800;
    end
    if ~isfield(s2, 'useStep1OverlapOnly') || isempty(s2.useStep1OverlapOnly)
        s2.useStep1OverlapOnly = true;
    end
    if ~isfield(s2, 'surfaceSmoothSigmaPix') || isempty(s2.surfaceSmoothSigmaPix)
        s2.surfaceSmoothSigmaPix = 0.0;
    end
    if ~isfield(s2, 'maskInterpThresh') || isempty(s2.maskInterpThresh)
        s2.maskInterpThresh = 0.55;
    end
    if ~isfield(s2, 'lambdaMaskPenalty') || isempty(s2.lambdaMaskPenalty)
        s2.lambdaMaskPenalty = 2.0e-2;
    end
    if ~isfield(s2, 'lambdaRxRy') || isempty(s2.lambdaRxRy)
        s2.lambdaRxRy = 2.0e-3;
    end
    if ~isfield(s2, 'lambdaTz') || isempty(s2.lambdaTz)
        s2.lambdaTz = 2.0e-3;
    end
    if ~isfield(s2, 'bound') || isempty(s2.bound)
        s2.bound = struct();
    end
    if ~isfield(s2.bound, 'rxDeg') || isempty(s2.bound.rxDeg)
        s2.bound.rxDeg = 1.0;
    end
    if ~isfield(s2.bound, 'ryDeg') || isempty(s2.bound.ryDeg)
        s2.bound.ryDeg = 1.0;
    end
    if ~isfield(s2.bound, 'rzDeg') || isempty(s2.bound.rzDeg)
        s2.bound.rzDeg = 2.0;
    end
    if ~isfield(s2.bound, 'txMM') || isempty(s2.bound.txMM)
        s2.bound.txMM = 1.5;
    end
    if ~isfield(s2.bound, 'tyMM') || isempty(s2.bound.tyMM)
        s2.bound.tyMM = 1.5;
    end
    if ~isfield(s2.bound, 'tzMM') || isempty(s2.bound.tzMM)
        s2.bound.tzMM = 0.2;
    end
    if ~isfield(s2, 'maxSearchPts') || isempty(s2.maxSearchPts)
        s2.maxSearchPts = inf;
    end
    if ~isfield(s2, 'maxEvalPts') || isempty(s2.maxEvalPts)
        s2.maxEvalPts = inf;
    end
    if ~isfield(s2, 'use_lsqnonlin') || isempty(s2.use_lsqnonlin)
        s2.use_lsqnonlin = (exist('lsqnonlin', 'file') == 2);
    end
    if ~isfield(s2, 'maxIter') || isempty(s2.maxIter)
        s2.maxIter = 80;
    end
    if ~isfield(s2, 'maxFunEval') || isempty(s2.maxFunEval)
        s2.maxFunEval = 5000;
    end
    if ~isfield(s2, 'stepTol') || isempty(s2.stepTol)
        s2.stepTol = 1e-12;
    end
    if ~isfield(s2, 'funTol') || isempty(s2.funTol)
        s2.funTol = 1e-12;
    end
    if ~isfield(s2, 'optTol') || isempty(s2.optTol)
        s2.optTol = 1e-12;
    end

    cfg.step2 = s2;
end

function cfg = apply_eval_defaults(cfg)
    if ~isfield(cfg, 'eval') || isempty(cfg.eval)
        cfg.eval = struct();
    end
    if ~isfield(cfg.eval, 'search') || isempty(cfg.eval.search)
        cfg.eval.search = struct();
    end
    if ~isfield(cfg.eval.search, 'minOverlapFrac') || isempty(cfg.eval.search.minOverlapFrac)
        cfg.eval.search.minOverlapFrac = 0.12;
    end
    if ~isfield(cfg.eval.search, 'minOverlapPts') || isempty(cfg.eval.search.minOverlapPts)
        cfg.eval.search.minOverlapPts = 150;
    end
    if ~isfield(cfg.eval, 'stats') || isempty(cfg.eval.stats)
        cfg.eval.stats = struct();
    end
    if ~isfield(cfg.eval.stats, 'innerErodeRadius') || isempty(cfg.eval.stats.innerErodeRadius)
        cfg.eval.stats.innerErodeRadius = 2;
    end
end

function data = prepare_sag_case_from_generator(subA_raw, subB_raw, truth_raw, cfg)
    [localA_noisy, poseA] = build_local_grid_subap(subA_raw, 'noisy');
    [localB_noisy, poseB] = build_local_grid_subap(subB_raw, 'noisy');
    [localA_clean, ~]     = build_local_grid_subap(subA_raw, 'clean');
    [localB_clean, ~]     = build_local_grid_subap(subB_raw, 'clean');

    truth = struct();
    truth.X    = double(truth_raw.Xm);
    truth.Y    = double(truth_raw.Ym);
    truth.Z    = double(truth_raw.Z_clean);
    truth.mask = logical(truth_raw.mask) & isfinite(truth.Z);

    A_noisy = build_measured_subap_from_local(localA_noisy, poseA, cfg.step2.maxSearchPts, cfg.step2.maxEvalPts);
    B_noisy = build_measured_subap_from_local(localB_noisy, poseB, cfg.step2.maxSearchPts, cfg.step2.maxEvalPts);
    A_clean = build_measured_subap_from_local(localA_clean, poseA, cfg.step2.maxSearchPts, cfg.step2.maxEvalPts);
    B_clean = build_measured_subap_from_local(localB_clean, poseB, cfg.step2.maxSearchPts, cfg.step2.maxEvalPts);

    data = struct();
    data.truth         = truth;
    data.localA_noisy  = localA_noisy;
    data.localB_noisy  = localB_noisy;
    data.localA_clean  = localA_clean;
    data.localB_clean  = localB_clean;
    data.A_noisy       = A_noisy;
    data.B_noisy       = B_noisy;
    data.A_clean       = A_clean;
    data.B_clean       = B_clean;
    data.subapA_pose   = poseA;
    data.subapB_pose   = poseB;
    data.truthRel      = true_relative_pose(poseA, poseB);
end

function [local, pose] = build_local_grid_subap(sub_raw, sourceMode)
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
            assert(isfield(sub_raw, 'z_clean'), 'sub_raw 缺少 z_clean 字段，无法构造 clean 评估。');
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
    local.ds   = ds;
    local.Rsub = Rsub;

    pose = struct();
    pose.R_w = double(sub_raw.pose_R);
    pose.T_w = double(sub_raw.pose_t(:));
    pose.Rsub = Rsub;
    pose.ds = ds;
end

function M = build_measured_subap_from_local(local, pose, maxSearchPts, maxEvalPts)
    X = double(local.X);
    Y = double(local.Y);
    Z = double(local.Z);
    mask = logical(local.mask) & isfinite(Z);

    xL = X(mask);
    yL = Y(mask);
    zL = Z(mask);

    P = pose.R_w * [xL(:).'; yL(:).'; zL(:).'] + pose.T_w(:);
    xW = P(1,:).';
    yW = P(2,:).';
    zW = P(3,:).';

    xVec = X(1,:);
    yVec = Y(:,1);
    Zg = Z;
    Zg(~mask) = NaN;

    Fgrid = griddedInterpolant({yVec, xVec}, Zg, 'linear', 'none');

    M = struct();
    M.pose          = pose;
    M.xLocal        = xL(:);
    M.yLocal        = yL(:);
    M.zLocal        = zL(:);
    M.xWorld        = xW(:);
    M.yWorld        = yW(:);
    M.zWorld        = zW(:);
    M.n             = numel(M.xLocal);
    M.Fgrid         = Fgrid;
    M.idxSearch     = pick_subsample_indices(M.n, maxSearchPts);
    M.idxEval       = pick_subsample_indices(M.n, maxEvalPts);
end

function idx = pick_subsample_indices(n, maxN)
    if isinf(maxN) || maxN >= n || maxN <= 0
        idx = (1:n).';
        return;
    end
    idx = round(linspace(1, n, maxN));
    idx = unique(max(min(idx, n), 1)).';
end

function [Zfeat, maskFeat, aux] = build_sag_feature_map(local, params)
    X = double(local.X);
    Y = double(local.Y);
    Z = double(local.Z);
    M = logical(local.mask) & isfinite(Z);

    if params.marginPix > 0
        maskFeat = erode_mask_disk(M, params.marginPix);
        if nnz(maskFeat) < 50
            maskFeat = M;
        end
    else
        maskFeat = M;
    end

    Zwork = Z;
    planeCoef = [0;0;0];
    if params.removeBestFitPlane
        xv = X(maskFeat);
        yv = Y(maskFeat);
        zv = Zwork(maskFeat);
        Aplane = [xv(:), yv(:), ones(numel(xv),1)];
        planeCoef = Aplane \ zv(:);
        Zplane = planeCoef(1) * X + planeCoef(2) * Y + planeCoef(3);
        Zwork(maskFeat) = Zwork(maskFeat) - Zplane(maskFeat);
    end

    if params.smoothSigmaPix > 0
        Zwork = masked_gaussian_smooth(Zwork, maskFeat, params.smoothSigmaPix);
    end

    zv = Zwork(maskFeat);
    mu = mean(zv, 'omitnan');
    sd = std(zv, 0, 'omitnan');
    if params.useZScore
        if ~(isfinite(sd) && sd > 0)
            sd = 1;
        end
        Zwork(maskFeat) = (Zwork(maskFeat) - mu) / sd;
    else
        Zwork(maskFeat) = Zwork(maskFeat) - mu;
    end

    if isfinite(params.clipSigma) && params.clipSigma > 0
        Zwork(maskFeat) = min(max(Zwork(maskFeat), -params.clipSigma), params.clipSigma);
    end

    Zfeat = nan(size(Z));
    Zfeat(maskFeat) = Zwork(maskFeat);

    aux = struct();
    aux.maskInput = M;
    aux.maskFeat = maskFeat;
    aux.Zdetrended = Zwork;
    aux.planeCoef = planeCoef;
    aux.meanVal = mu;
    aux.stdVal = sd;
end

function Zs = masked_gaussian_smooth(Z, M, sigmaPix)
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

function coarse = coarse_search_sag_se2(A, MA, B, MB, XA, YA, XB, YB, params)
    ds = infer_local_ds(XA, YA);
    thetaList = params.thetaSearchDeg(:).';

    bestScore = -inf;
    best = struct('thetaRad', 0, 'thetaDeg', 0, 'tx', 0, 'ty', 0, ...
                  'overlapN', 0, 'scoreMap', [], 'thetaListDeg', thetaList);

    for thDeg = thetaList
        theta = deg2rad(thDeg);

        [Bw0, Mw0] = warp_B_to_A_grid(B, MB, XB, YB, XA, YA, theta, 0, 0);
        [scoreMap, overlapMap] = masked_ncc_full_integer(A, MA, Bw0, Mw0, params.minOverlapPix);

        if all(~isfinite(scoreMap(:)))
            continue;
        end

        [scoreMax, idxMax] = max(scoreMap(:));
        [iy, ix] = ind2sub(size(scoreMap), idxMax);

        % === 修复 1：减去模板 Bw0 的尺寸，而不是相关结果 scoreMap 的尺寸 ===
        txPix = ix - size(Bw0, 2);
        tyPix = iy - size(Bw0, 1);
        % =====================================================================

        tx = txPix * ds;
        ty = tyPix * ds;

        if scoreMax > bestScore
            bestScore = scoreMax;
            best.thetaRad = theta;
            best.thetaDeg = thDeg;
            best.tx = tx;
            best.ty = ty;
            best.overlapN = overlapMap(iy, ix);
            best.scoreMap = scoreMap;
        end
    end

    coarse = best;
end

function fine = refine_search_sag_se2(A, MA, B, MB, XA, YA, XB, YB, coarse, params)
    ds = infer_local_ds(XA, YA);

    thetaList = linspace(coarse.thetaDeg - params.thetaFineSpanDeg, ...
                         coarse.thetaDeg + params.thetaFineSpanDeg, ...
                         params.thetaFineNum);
    shiftList = linspace(-params.shiftFineSpanPx * ds, ...
                          params.shiftFineSpanPx * ds, ...
                          params.shiftFineNum);

    bestScore = -inf;
    best = coarse;

    for thDeg = thetaList
        theta = deg2rad(thDeg);
        for dtx = shiftList
            for dty = shiftList
                tx = coarse.tx + dtx;
                ty = coarse.ty + dty;

                [score, overlapMask] = masked_ncc_direct( ...
                    A, MA, B, MB, XA, YA, XB, YB, ...
                    theta, tx, ty, params.minOverlapPix, params.maskInterpThresh);

                if score > bestScore
                    bestScore = score;
                    best.thetaRad = theta;
                    best.thetaDeg = thDeg;
                    best.tx = tx;
                    best.ty = ty;
                    best.overlapMask = overlapMask;
                    best.score = score;
                end
            end
        end
    end

    fine = best;
end

function fine = coarse_to_fine_passthrough(coarse)
    fine = coarse;
    if ~isfield(fine, 'score')
        fine.score = NaN;
    end
    if ~isfield(fine, 'overlapMask')
        fine.overlapMask = [];
    end
end

function [Bw, Mw] = warp_B_to_A_grid(B, MB, XB, YB, XA, YA, theta, tx, ty)
    c = cos(theta);
    s = sin(theta);

    qx = XA - tx;
    qy = YA - ty;

    xq =  c * qx + s * qy;
    yq = -s * qx + c * qy;

    Bw = interp2(XB(1,:), YB(:,1), B, xq, yq, 'linear', NaN);
    Mw = interp2(XB(1,:), YB(:,1), double(MB), xq, yq, 'linear', 0) >= 0.5;
end

function [scoreMap, overlapMap] = masked_ncc_full_integer(A, MA, B, MB, minOverlap)
    A0 = A; B0 = B;
    A0(~MA) = 0;
    B0(~MB) = 0;

    num = conv2(rot90(B0,2), A0, 'full');
    denA = conv2(rot90(double(MB),2), A0.^2, 'full');
    denB = conv2(rot90(B0.^2,2), double(MA), 'full');
    overlapMap = conv2(rot90(double(MB),2), double(MA), 'full');

    scoreMap = num ./ sqrt(max(denA, 0) .* max(denB, 0));
    scoreMap(overlapMap < minOverlap) = NaN;
    
    % === 修复 3: 加入重叠面积惩罚项，抑制边缘假峰值 ===
    maxOv = max(overlapMap(:));
    if maxOv > 0
        penalty = (overlapMap ./ maxOv) .^ 0.25; 
        scoreMap = scoreMap .* penalty;
    end
    % ==================================================
end

function [score, overlapMask] = masked_ncc_direct(A, MA, B, MB, XA, YA, XB, YB, theta, tx, ty, minOverlap, maskInterpThresh)
    [Bw, Mw] = warp_B_to_A_grid(B, MB, XB, YB, XA, YA, theta, tx, ty);
    overlapMask = MA & Mw & isfinite(A) & isfinite(Bw);

    if nnz(overlapMask) < minOverlap
        score = -inf;
        overlapMask = false(size(A));
        return;
    end

    qx = XA - tx;
    qy = YA - ty;
    xq =  cos(theta) * qx + sin(theta) * qy;
    yq = -sin(theta) * qx + cos(theta) * qy;

    maskVal = interp2(XB(1,:), YB(:,1), double(MB), xq, yq, 'linear', 0);
    overlapMask = overlapMask & (maskVal >= maskInterpThresh);

    if nnz(overlapMask) < minOverlap
        score = -inf;
        overlapMask = false(size(A));
        return;
    end

    av = A(overlapMask);
    bv = Bw(overlapMask);

    av = av - mean(av, 'omitnan');
    bv = bv - mean(bv, 'omitnan');

    denom = sqrt(sum(av.^2) * sum(bv.^2));
    if denom <= 0 || ~isfinite(denom)
        score = -inf;
    else
        score = sum(av .* bv) / denom;
    end
end

function modelA = build_surface_model_from_sag(localA, params)
    XA = double(localA.X);
    YA = double(localA.Y);
    ZA = double(localA.Z);
    MA = logical(localA.mask) & isfinite(ZA);

    if params.surfaceSmoothSigmaPix > 0
        ZA = masked_gaussian_smooth(ZA, MA, params.surfaceSmoothSigmaPix);
    end

    Zfill = fill_invalid_by_nearest(XA, YA, ZA, MA);

    modelA = struct();
    modelA.X     = XA;
    modelA.Y     = YA;
    modelA.xVec  = XA(1,:);
    modelA.yVec  = YA(:,1);
    modelA.Zfill = Zfill;
    modelA.mask  = MA;
end

function ds = infer_local_ds(X, Y)
    dx = []; dy = [];
    if size(X,2) >= 2
        dx = median(abs(diff(X(1,:))), 'omitnan');
    end
    if size(Y,1) >= 2
        dy = median(abs(diff(Y(:,1))), 'omitnan');
    end
    cand = [dx, dy];
    cand = cand(isfinite(cand) & cand > 0);
    if isempty(cand)
        error('无法自动推断局部网格间隔 ds。');
    end
    ds = cand(1);
end

function p0 = build_initial_pose_from_step1(fine)
    p0 = zeros(6,1);
    p0(1) = 0;
    p0(2) = 0;
    p0(3) = fine.thetaRad;
    p0(4) = fine.tx;
    p0(5) = fine.ty;
    p0(6) = 0;
end

function [PBcand, meta] = build_candidate_points_from_step1(localA, localB, fine2d, p0, params)
    XA = double(localA.X);
    YA = double(localA.Y);
    ZA = double(localA.Z);
    MA = logical(localA.mask) & isfinite(ZA);

    XB = double(localB.X);
    YB = double(localB.Y);
    ZB = double(localB.Z);
    MB = logical(localB.mask) & isfinite(ZB);

    overlapMaskA = MA;
    if params.useStep1OverlapOnly && isfield(fine2d, 'overlapMask') && ~isempty(fine2d.overlapMask)
        overlapMaskA = logical(fine2d.overlapMask) & MA;
        if nnz(overlapMaskA) < params.minCandidatePts
            overlapMaskA = MA;
        end
    end

    xA = XA(overlapMaskA);
    yA = YA(overlapMaskA);

    [R0, t0] = posevec_to_rt_internal(p0);
    Rxy = R0(1:2,1:2);
    txy = t0(1:2);

    qA = [xA.'; yA.'];
    qB = Rxy.' * (qA - txy);

    xB = qB(1,:).';
    yB = qB(2,:).';

    zB = interp2(XB(1,:), YB(:,1), ZB, xB, yB, 'linear', NaN);
    mB = interp2(XB(1,:), YB(:,1), double(MB), xB, yB, 'linear', 0) >= 0.95;

    valid = isfinite(zB) & mB;
    PBcand = [xB(valid), yB(valid), zB(valid)];

    meta = struct();
    meta.numRawAOverlap = nnz(overlapMaskA);
    meta.numCandidateFromB = size(PBcand,1);
end

function [pEst, exitflag, out] = refine_pose_height_residual(PBcand, p0, modelA, cfg)
    [lb, ub] = build_pose_bounds(p0, cfg);

    if cfg.use_lsqnonlin
        fun = @(p) residual_pose_height_residual(p, PBcand, modelA, cfg);
        opts = optimoptions('lsqnonlin', ...
            'Algorithm', 'trust-region-reflective', ...
            'Display', 'off', ...
            'SpecifyObjectiveGradient', false, ...
            'MaxIterations', cfg.maxIter, ...
            'MaxFunctionEvaluations', cfg.maxFunEval, ...
            'FunctionTolerance', cfg.funTol, ...
            'StepTolerance', cfg.stepTol, ...
            'OptimalityTolerance', cfg.optTol);
        [pEst, ~, residual, exitflag, output] = lsqnonlin(fun, p0, lb, ub, opts); %#ok<ASGLU>
        out = struct('algorithm', 'lsqnonlin', 'output', output, 'residual', residual);
    else
        objFun = @(p) sum(residual_pose_height_residual(clip_to_bounds(p, lb, ub), PBcand, modelA, cfg).^2);
        opts = optimset('Display', 'off', ...
            'MaxIter', cfg.maxIter, ...
            'MaxFunEvals', cfg.maxFunEval, ...
            'TolX', cfg.stepTol, ...
            'TolFun', cfg.funTol);
        pEst = fminsearch(objFun, p0, opts);
        pEst = clip_to_bounds(pEst, lb, ub);
        residual = residual_pose_height_residual(pEst, PBcand, modelA, cfg);
        exitflag = 1;
        out = struct('algorithm', 'fminsearch', 'residual', residual);
    end
end

function [lb, ub] = build_pose_bounds(p0, params)
    lb = p0;
    ub = p0;

    lb(1) = -deg2rad(params.bound.rxDeg); ub(1) = deg2rad(params.bound.rxDeg);
    lb(2) = -deg2rad(params.bound.ryDeg); ub(2) = deg2rad(params.bound.ryDeg);
    lb(3) = p0(3) - deg2rad(params.bound.rzDeg); ub(3) = p0(3) + deg2rad(params.bound.rzDeg);
    lb(4) = p0(4) - params.bound.txMM; ub(4) = p0(4) + params.bound.txMM;
    lb(5) = p0(5) - params.bound.tyMM; ub(5) = p0(5) + params.bound.tyMM;
    lb(6) = -params.bound.tzMM; ub(6) = params.bound.tzMM;
end

function p = clip_to_bounds(p, lb, ub)
    p = min(max(p, lb), ub);
end

function r = residual_pose_height_residual(p, PB, modelA, params)
    [R, t] = posevec_to_rt_internal(p);
    PA = apply_transform(PB, R, t);

    x = PA(:,1);
    y = PA(:,2);
    z = PA(:,3);

    zA = interp2(modelA.xVec, modelA.yVec, modelA.Zfill, x, y, 'linear', NaN);
    mv = interp2(modelA.xVec, modelA.yVec, double(modelA.mask), x, y, 'linear', 0);

    valid = isfinite(zA) & (mv > 0);

    N = size(PB,1);
    rHeight = zeros(N,1);
    if any(valid)
        rHeight(valid) = z(valid) - zA(valid);
    end

    mv = min(max(mv, 0), 1);
    rMask = params.lambdaMaskPenalty * (1 - mv);
    rReg = [params.lambdaRxRy * p(1); params.lambdaRxRy * p(2); params.lambdaTz * p(6)];

    r = [rHeight; rMask; rReg];
end

function reg = sag_pose_to_reg(p)
    [R, t] = posevec_to_rt_internal(p);
    reg = struct();
    reg.p = [t(1), t(2), t(3), p(1), p(2), p(3)];
    reg.R = R;
    reg.score = NaN;
    reg.overlapRMSE = NaN;
    reg.overlapRatio = NaN;
    reg.overlapN = NaN;
    reg.residual = [];
    reg.xAov = [];
    reg.yAov = [];
    reg.zAref = [];
    reg.zAfromB = [];
end

function reg = truth_relative_pose_to_reg(truthRel)
    reg = struct();
    reg.p = [truthRel.t(1), truthRel.t(2), truthRel.t(3), truthRel.rx, truthRel.ry, truthRel.rz];
    reg.R = truthRel.R;
    reg.score = NaN;
    reg.overlapRMSE = NaN;
    reg.overlapRatio = NaN;
    reg.overlapN = NaN;
    reg.residual = [];
    reg.xAov = [];
    reg.yAov = [];
    reg.zAref = [];
    reg.zAfromB = [];
end

function reg = attach_overlap_metrics_to_reg(reg, A, B, cfg)
    idx = B.idxEval;

    tx = reg.p(1);
    ty = reg.p(2);
    tz = reg.p(3);
    R  = reg.R;

    xB = B.xLocal(idx);
    yB = B.yLocal(idx);
    zB = B.zLocal(idx);

    PB = [xB(:), yB(:), zB(:)];
    PA = apply_transform(PB, R, [tx, ty, tz]);

    xA = PA(:,1);
    yA = PA(:,2);
    zAfromB = PA(:,3);

    zAref = A.Fgrid(yA, xA);
    valid = isfinite(zAref);

    if nnz(valid) < cfg.search.minOverlapPts
        reg.overlapRMSE = inf;
        reg.overlapRatio = 0;
        reg.overlapN = nnz(valid);
        reg.xAov = [];
        reg.yAov = [];
        reg.zAref = [];
        reg.zAfromB = [];
        return;
    end

    dz = zAfromB(valid) - zAref(valid);

    reg.overlapRMSE = sqrt(mean(dz.^2));
    reg.overlapRatio = nnz(valid) / numel(idx);
    reg.overlapN = nnz(valid);
    reg.xAov = xA(valid);
    reg.yAov = yA(valid);
    reg.zAref = zAref(valid);
    reg.zAfromB = zAfromB(valid);
end

function truthRel = true_relative_pose(SA, SB)
    RA = SA.R_w;
    RB = SB.R_w;
    TA = SA.T_w(:);
    TB = SB.T_w(:);

    R_true = RA.' * RB;
    t_true = RA.' * (TB - TA);
    [rx, ry, rz] = rotm_to_eul_zyx(R_true);

    truthRel = struct();
    truthRel.R  = R_true;
    truthRel.t  = t_true(:).';
    truthRel.rx = rx;
    truthRel.ry = ry;
    truthRel.rz = rz;
end

function poseErr = evaluate_pose_error_sag(reg, truthRel)
    dt = reg.p(1:3) - truthRel.t(:).';
    deul = [ ...
        wrap_to_pi_local(reg.p(4) - truthRel.rx), ...
        wrap_to_pi_local(reg.p(5) - truthRel.ry), ...
        wrap_to_pi_local(reg.p(6) - truthRel.rz)]; %#ok<NASGU>

    poseErr = struct();
    poseErr.dt_mm    = dt(:).';
    poseErr.et_mm    = norm(dt);
    poseErr.eR_deg   = rotm_geodesic_deg(reg.R, truthRel.R);
end

function deg = rotm_geodesic_deg(R1, R2)
    Rerr = R1 * R2.';
    val = (trace(Rerr) - 1) / 2;
    val = min(max(val, -1), 1);
    deg = rad2deg(acos(val));
end

function [fusedMap, errMap, unionMask, stats_alg, ZAmap, ZBmap, overlapMaskMap, innerMask] = ...
    fuse_and_evaluate_surface_pair(data, reg, cfg, sourceMode)

    switch lower(sourceMode)
        case 'noisy'
            A = data.A_noisy;
            B = data.B_noisy;
        case 'clean'
            A = data.A_clean;
            B = data.B_clean;
        otherwise
            error('sourceMode 仅支持 noisy / clean');
    end

    T = data.truth;

    xAw = A.xWorld;
    yAw = A.yWorld;
    zAw = A.zWorld;

    p = reg.p(:).';
    R = reg.R;

    PB = [B.xLocal(:).'; B.yLocal(:).'; B.zLocal(:).'];
    PA = R * PB + [p(1); p(2); p(3)];

    RA = A.pose.R_w;
    TA = A.pose.T_w(:);

    PW = RA * PA + TA;
    xBw = PW(1,:).';
    yBw = PW(2,:).';
    zBw = PW(3,:).';

    FAw = scatteredInterpolant(xAw, yAw, zAw, 'natural', 'none');
    FBw = scatteredInterpolant(xBw, yBw, zBw, 'natural', 'none');

    ZAmap = FAw(T.X, T.Y);
    ZBmap = FBw(T.X, T.Y);
    ZAmap(~T.mask) = NaN;
    ZBmap(~T.mask) = NaN;

    vA = isfinite(ZAmap) & T.mask;
    vB = isfinite(ZBmap) & T.mask;

    unionMask   = vA | vB;
    overlapMask = vA & vB;
    onlyA       = vA & ~vB;
    onlyB       = ~vA & vB;

    fusedMap = nan(size(T.Z));
    fusedMap(overlapMask) = 0.5 * (ZAmap(overlapMask) + ZBmap(overlapMask));
    fusedMap(onlyA)       = ZAmap(onlyA);
    fusedMap(onlyB)       = ZBmap(onlyB);

    errMap = nan(size(T.Z));
    errMap(unionMask) = fusedMap(unionMask) - T.Z(unionMask);

    ev = errMap(unionMask);
    ev = ev(isfinite(ev));

    stats_alg = struct();
    stats_alg.fusedRMSE = safe_rmse(ev);
    stats_alg.fusedMAE  = safe_mae(ev);
    stats_alg.fusedPV   = safe_pv(ev);
    stats_alg.fusedSTD  = safe_std(ev);
    stats_alg.robustPV995 = safe_robust_pv995(ev);

    ovErr = errMap(overlapMask);
    ovErr = ovErr(isfinite(ovErr));
    stats_alg.overlapRMSE = safe_rmse(ovErr);
    stats_alg.overlapMAE  = safe_mae(ovErr);
    stats_alg.overlapPV   = safe_pv(ovErr);

    innerMask = erode_mask_disk(unionMask, cfg.eval.stats.innerErodeRadius);
    innerErr = errMap(innerMask);
    innerErr = innerErr(isfinite(innerErr));
    stats_alg.innerRMSE = safe_rmse(innerErr);

    overlapMaskMap = overlapMask;
end

function tbl = make_summary_table_intensity(xList, successMat, actualNoiseMat_um, ...
    algRmseMat_mm, algInnerRmseMat_mm, ...
    oracleRmseMat_mm, oracleInnerRmseMat_mm, ...
    deltaRmseMat_mm, regEnergyRmseMat_mm, ...
    poseOnlyRmseMat_mm, poseOnlyInnerRmseMat_mm, ...
    algOverlapRmseMat_mm, oracleOverlapRmseMat_mm, poseOnlyOverlapRmseMat_mm, ...
    etMat_mm, eRMat_deg, solveTimeMat_s)

    tbl = table();
    tbl.sigma_n_set_um = xList(:);
    tbl.n_success = sum(successMat, 2);
    tbl.sigma_n_actual_mean_um = mean(actualNoiseMat_um, 2, 'omitnan');
    tbl.sigma_n_actual_std_um  = std(actualNoiseMat_um, 0, 2, 'omitnan');

    tbl.alg_fused_RMSE_mean_um    = 1e3 * mean(algRmseMat_mm, 2, 'omitnan');
    tbl.alg_fused_RMSE_std_um     = 1e3 * std(algRmseMat_mm, 0, 2, 'omitnan');
    tbl.oracle_fused_RMSE_mean_um = 1e3 * mean(oracleRmseMat_mm, 2, 'omitnan');
    tbl.oracle_fused_RMSE_std_um  = 1e3 * std(oracleRmseMat_mm, 0, 2, 'omitnan');

    tbl.delta_fused_RMSE_mean_um      = 1e3 * mean(deltaRmseMat_mm, 2, 'omitnan');
    tbl.delta_fused_RMSE_std_um       = 1e3 * std(deltaRmseMat_mm, 0, 2, 'omitnan');
    tbl.reg_only_energy_RMSE_mean_um  = 1e3 * mean(regEnergyRmseMat_mm, 2, 'omitnan');
    tbl.reg_only_energy_RMSE_std_um   = 1e3 * std(regEnergyRmseMat_mm, 0, 2, 'omitnan');

    tbl.pose_only_clean_fused_RMSE_mean_um = 1e3 * mean(poseOnlyRmseMat_mm, 2, 'omitnan');
    tbl.pose_only_clean_fused_RMSE_std_um  = 1e3 * std(poseOnlyRmseMat_mm, 0, 2, 'omitnan');

    tbl.alg_inner_RMSE_mean_um    = 1e3 * mean(algInnerRmseMat_mm, 2, 'omitnan');
    tbl.alg_inner_RMSE_std_um     = 1e3 * std(algInnerRmseMat_mm, 0, 2, 'omitnan');
    tbl.oracle_inner_RMSE_mean_um = 1e3 * mean(oracleInnerRmseMat_mm, 2, 'omitnan');
    tbl.oracle_inner_RMSE_std_um  = 1e3 * std(oracleInnerRmseMat_mm, 0, 2, 'omitnan');

    tbl.pose_only_clean_inner_RMSE_mean_um = 1e3 * mean(poseOnlyInnerRmseMat_mm, 2, 'omitnan');
    tbl.pose_only_clean_inner_RMSE_std_um  = 1e3 * std(poseOnlyInnerRmseMat_mm, 0, 2, 'omitnan');

    tbl.alg_overlap_RMSE_mean_um    = 1e3 * mean(algOverlapRmseMat_mm, 2, 'omitnan');
    tbl.alg_overlap_RMSE_std_um     = 1e3 * std(algOverlapRmseMat_mm, 0, 2, 'omitnan');
    tbl.oracle_overlap_RMSE_mean_um = 1e3 * mean(oracleOverlapRmseMat_mm, 2, 'omitnan');
    tbl.oracle_overlap_RMSE_std_um  = 1e3 * std(oracleOverlapRmseMat_mm, 0, 2, 'omitnan');

    tbl.pose_only_clean_overlap_RMSE_mean_um = 1e3 * mean(poseOnlyOverlapRmseMat_mm, 2, 'omitnan');
    tbl.pose_only_clean_overlap_RMSE_std_um  = 1e3 * std(poseOnlyOverlapRmseMat_mm, 0, 2, 'omitnan');

    tbl.e_t_mean_um = 1e3 * mean(etMat_mm, 2, 'omitnan');
    tbl.e_R_mean_deg = mean(eRMat_deg, 2, 'omitnan');
    tbl.solve_time_mean_s = mean(solveTimeMat_s, 2, 'omitnan');
end

function tbl = make_summary_table_spectrum(xList, successMat, actualNoiseMat_um, ...
    algRmseMat_mm, algInnerRmseMat_mm, ...
    oracleRmseMat_mm, oracleInnerRmseMat_mm, ...
    deltaRmseMat_mm, regEnergyRmseMat_mm, ...
    poseOnlyRmseMat_mm, poseOnlyInnerRmseMat_mm, ...
    algOverlapRmseMat_mm, oracleOverlapRmseMat_mm, poseOnlyOverlapRmseMat_mm, ...
    etMat_mm, eRMat_deg, solveTimeMat_s)

    tbl = table();
    tbl.Lc_mm = xList(:);
    tbl.n_success = sum(successMat, 2);
    tbl.sigma_n_actual_mean_um = mean(actualNoiseMat_um, 2, 'omitnan');
    tbl.sigma_n_actual_std_um  = std(actualNoiseMat_um, 0, 2, 'omitnan');

    tbl.alg_fused_RMSE_mean_um    = 1e3 * mean(algRmseMat_mm, 2, 'omitnan');
    tbl.alg_fused_RMSE_std_um     = 1e3 * std(algRmseMat_mm, 0, 2, 'omitnan');
    tbl.oracle_fused_RMSE_mean_um = 1e3 * mean(oracleRmseMat_mm, 2, 'omitnan');
    tbl.oracle_fused_RMSE_std_um  = 1e3 * std(oracleRmseMat_mm, 0, 2, 'omitnan');

    tbl.delta_fused_RMSE_mean_um      = 1e3 * mean(deltaRmseMat_mm, 2, 'omitnan');
    tbl.delta_fused_RMSE_std_um       = 1e3 * std(deltaRmseMat_mm, 0, 2, 'omitnan');
    tbl.reg_only_energy_RMSE_mean_um  = 1e3 * mean(regEnergyRmseMat_mm, 2, 'omitnan');
    tbl.reg_only_energy_RMSE_std_um   = 1e3 * std(regEnergyRmseMat_mm, 0, 2, 'omitnan');

    tbl.pose_only_clean_fused_RMSE_mean_um = 1e3 * mean(poseOnlyRmseMat_mm, 2, 'omitnan');
    tbl.pose_only_clean_fused_RMSE_std_um  = 1e3 * std(poseOnlyRmseMat_mm, 0, 2, 'omitnan');

    tbl.alg_inner_RMSE_mean_um    = 1e3 * mean(algInnerRmseMat_mm, 2, 'omitnan');
    tbl.alg_inner_RMSE_std_um     = 1e3 * std(algInnerRmseMat_mm, 0, 2, 'omitnan');
    tbl.oracle_inner_RMSE_mean_um = 1e3 * mean(oracleInnerRmseMat_mm, 2, 'omitnan');
    tbl.oracle_inner_RMSE_std_um  = 1e3 * std(oracleInnerRmseMat_mm, 0, 2, 'omitnan');

    tbl.pose_only_clean_inner_RMSE_mean_um = 1e3 * mean(poseOnlyInnerRmseMat_mm, 2, 'omitnan');
    tbl.pose_only_clean_inner_RMSE_std_um  = 1e3 * std(poseOnlyInnerRmseMat_mm, 0, 2, 'omitnan');

    tbl.alg_overlap_RMSE_mean_um    = 1e3 * mean(algOverlapRmseMat_mm, 2, 'omitnan');
    tbl.alg_overlap_RMSE_std_um     = 1e3 * std(algOverlapRmseMat_mm, 0, 2, 'omitnan');
    tbl.oracle_overlap_RMSE_mean_um = 1e3 * mean(oracleOverlapRmseMat_mm, 2, 'omitnan');
    tbl.oracle_overlap_RMSE_std_um  = 1e3 * std(oracleOverlapRmseMat_mm, 0, 2, 'omitnan');

    tbl.pose_only_clean_overlap_RMSE_mean_um = 1e3 * mean(poseOnlyOverlapRmseMat_mm, 2, 'omitnan');
    tbl.pose_only_clean_overlap_RMSE_std_um  = 1e3 * std(poseOnlyOverlapRmseMat_mm, 0, 2, 'omitnan');

    tbl.e_t_mean_um = 1e3 * mean(etMat_mm, 2, 'omitnan');
    tbl.e_R_mean_deg = mean(eRMat_deg, 2, 'omitnan');
    tbl.solve_time_mean_s = mean(solveTimeMat_s, 2, 'omitnan');
end

function plot_total_vs_oracle(x, yAlg, eAlg, yOracle, eOracle, xlabelStr, titleStr, pdfPath, pngPath, plotCfg)
    fig = figure('Color', 'w', 'Position', [120 120 760 520]);
    ax = axes(fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');

    errorbar(ax, x, yAlg, eAlg, '-o', 'LineWidth', plotCfg.lineWidth, ...
        'MarkerSize', plotCfg.markerSize, 'CapSize', 10);
    errorbar(ax, x, yOracle, eOracle, '-s', 'LineWidth', plotCfg.lineWidth, ...
        'MarkerSize', plotCfg.markerSize, 'CapSize', 10);

    xlabel(ax, xlabelStr, 'Interpreter', 'latex');
    ylabel(ax, 'Fused RMSE ($\mu$m)', 'Interpreter', 'latex');
    legend(ax, {'Algorithm output', 'Oracle noise floor'}, 'Location', 'northwest');
    title(ax, titleStr, 'Interpreter', 'none');
    set(ax, 'FontSize', 12, 'LineWidth', 1.0);
    apply_x_margin(ax, x);

    exportgraphics(fig, pdfPath, 'ContentType', 'vector');
    exportgraphics(fig, pngPath, 'Resolution', 300);
end

function plot_registration_only_metrics(x, yDelta, eDelta, yRegE, eRegE, yPoseOnly, ePoseOnly, xlabelStr, titleStr, pdfPath, pngPath, plotCfg)
    fig = figure('Color', 'w', 'Position', [120 120 760 520]);
    ax = axes(fig); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');

    errorbar(ax, x, yDelta, eDelta, '-o', 'LineWidth', plotCfg.lineWidth, 'MarkerSize', plotCfg.markerSize, 'CapSize', 10);
    errorbar(ax, x, yRegE,  eRegE,  '-s', 'LineWidth', plotCfg.lineWidth, 'MarkerSize', plotCfg.markerSize, 'CapSize', 10);
    errorbar(ax, x, yPoseOnly, ePoseOnly, '-^', 'LineWidth', plotCfg.lineWidth, 'MarkerSize', plotCfg.markerSize, 'CapSize', 10);

    xlabel(ax, xlabelStr, 'Interpreter', 'latex');
    ylabel(ax, 'Registration-related RMSE ($\mu$m)', 'Interpreter', 'latex');
    legend(ax, {'alg - oracle', 'energy-separated', 'clean surface under estimated pose'}, 'Location', 'northwest');
    title(ax, [titleStr ' | registration-only metrics'], 'Interpreter', 'none');
    set(ax, 'FontSize', 12, 'LineWidth', 1.0);
    apply_x_margin(ax, x);

    exportgraphics(fig, pdfPath, 'ContentType', 'vector');
    exportgraphics(fig, pngPath, 'Resolution', 300);
end

function plot_pose_errors(x, et_um, et_std_um, eR_deg, eR_std_deg, xlabelStr, titleStr, pdfPath, pngPath, plotCfg)
    fig = figure('Color', 'w', 'Position', [120 120 820 560]);
    ax1 = axes(fig); hold(ax1, 'on'); box(ax1, 'on'); grid(ax1, 'on');
    yyaxis(ax1, 'left');
    errorbar(ax1, x, et_um, et_std_um, '-o', 'LineWidth', plotCfg.lineWidth, 'MarkerSize', plotCfg.markerSize, 'CapSize', 10);
    ylabel(ax1, '$e_t$ ($\mu$m)', 'Interpreter', 'latex');

    yyaxis(ax1, 'right');
    errorbar(ax1, x, eR_deg, eR_std_deg, '-s', 'LineWidth', plotCfg.lineWidth, 'MarkerSize', plotCfg.markerSize, 'CapSize', 10);
    ylabel(ax1, '$e_R$ (deg)', 'Interpreter', 'latex');

    xlabel(ax1, xlabelStr, 'Interpreter', 'latex');
    legend(ax1, {'translation error', 'rotation error'}, 'Location', 'northwest');
    title(ax1, [titleStr ' | pose errors'], 'Interpreter', 'none');
    set(ax1, 'FontSize', 12, 'LineWidth', 1.0);
    apply_x_margin(ax1, x);

    exportgraphics(fig, pdfPath, 'ContentType', 'vector');
    exportgraphics(fig, pngPath, 'Resolution', 300);
end

function apply_x_margin(ax, x)
    if isempty(x)
        return;
    end
    xmin = min(x(:));
    xmax = max(x(:));
    if xmax <= xmin
        pad = max(1, abs(xmin) * 0.1 + 1e-6);
    else
        pad = 0.05 * (xmax - xmin);
    end
    xlim(ax, [xmin - pad, xmax + pad]);
end

function [R, t] = posevec_to_rt_internal(p)
    rx = p(1);
    ry = p(2);
    rz = p(3);
    R = eul_zyx_to_rotm(rx, ry, rz);
    t = [p(4); p(5); p(6)];
end

function P2 = apply_transform(P1, R, t)
    P2 = (R * P1.' + t(:)).';
end

function R = eul_zyx_to_rotm(rx, ry, rz)
    cx = cos(rx); sx = sin(rx);
    cy = cos(ry); sy = sin(ry);
    cz = cos(rz); sz = sin(rz);

    R = [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx; ...
         sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx; ...
          -sy, cy*sx,            cy*cx];
end

function [rx, ry, rz] = rotm_to_eul_zyx(R)
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

function Zfill = fill_invalid_by_nearest(X, Y, Z, mask)
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

function maskErode = erode_mask_disk(mask, radiusPix)
    if radiusPix <= 0
        maskErode = logical(mask);
        return;
    end
    [xx, yy] = meshgrid(-radiusPix:radiusPix, -radiusPix:radiusPix);
    se = (xx.^2 + yy.^2) <= radiusPix^2;
    maskErode = conv2(double(mask), double(se), 'same') >= sum(se(:)) - 1e-12;
end

function v = safe_rmse(x)
    if isempty(x) || all(~isfinite(x))
        v = NaN;
    else
        x = x(isfinite(x));
        v = sqrt(mean(x.^2));
    end
end

function v = safe_mae(x)
    if isempty(x) || all(~isfinite(x))
        v = NaN;
    else
        x = x(isfinite(x));
        v = mean(abs(x));
    end
end

function v = safe_pv(x)
    if isempty(x) || all(~isfinite(x))
        v = NaN;
    else
        x = x(isfinite(x));
        v = max(x) - min(x);
    end
end

function v = safe_std(x)
    if isempty(x) || all(~isfinite(x))
        v = NaN;
    else
        x = x(isfinite(x));
        v = std(x, 0);
    end
end

function v = safe_robust_pv995(x)
    if isempty(x) || all(~isfinite(x))
        v = NaN;
    else
        x = sort(x(isfinite(x)));
        if isempty(x)
            v = NaN;
            return;
        end
        lo = x(max(1, round(0.0025*numel(x))));
        hi = x(min(numel(x), round(0.9975*numel(x))));
        v = hi - lo;
    end
end

function ang = wrap_to_pi_local(ang)
    ang = mod(ang + pi, 2*pi) - pi;
end