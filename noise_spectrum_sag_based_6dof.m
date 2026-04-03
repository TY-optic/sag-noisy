clear; clc; close all;

cfg = struct();
cfg.mode = 'spectrum';
cfg.generatorFcn = 'generate_noisy_peaks_subaps_6dof';

cfg.noise = struct();
cfg.noise.sigma_n_um = 1.0;
cfg.noise.LcList_mm = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0];

cfg.mc = struct();
cfg.mc.N = 20;
cfg.mc.baseSeed = 20260407;

cfg.pose6dof = struct( ...
    'tx_mm',  [], ...
    'ty_mm',  [], ...
    'tz_mm',  0.010, ...
    'rx_deg', 0.020, ...
    'ry_deg', -0.040, ...
    'rz_deg', 0.200);

cfg.step1 = struct();
cfg.step1.marginPix = 2;
cfg.step1.removeBestFitPlane = true;
cfg.step1.smoothSigmaPix = 0.0;
cfg.step1.useZScore = true;
cfg.step1.clipSigma = 6.0;
cfg.step1.thetaSearchDeg = -4.0 : 0.05 : 4.0;
cfg.step1.minOverlapPix = 1200;
cfg.step1.doFineRefine = true;
cfg.step1.thetaFineSpanDeg = 0.12;
cfg.step1.thetaFineNum = 13;
cfg.step1.shiftFineSpanPx = 1.2;
cfg.step1.shiftFineNum = 13;
cfg.step1.maskInterpThresh = 0.85;

cfg.step2 = struct();
cfg.step2.minCandidatePts = 800;
cfg.step2.useStep1OverlapOnly = true;
cfg.step2.surfaceSmoothSigmaPix = 0.0;
cfg.step2.maskInterpThresh = 0.55;
cfg.step2.lambdaMaskPenalty = 2.0e-2;
cfg.step2.lambdaRxRy = 2.0e-3;
cfg.step2.lambdaTz = 2.0e-3;
cfg.step2.bound = struct('rxDeg',1.0,'ryDeg',1.0,'rzDeg',2.0,'txMM',1.5,'tyMM',1.5,'tzMM',0.2);
cfg.step2.maxSearchPts = inf;
cfg.step2.maxEvalPts = inf;
cfg.step2.use_lsqnonlin = (exist('lsqnonlin', 'file') == 2);
cfg.step2.maxIter = 80;
cfg.step2.maxFunEval = 5000;
cfg.step2.stepTol = 1e-12;
cfg.step2.funTol = 1e-12;
cfg.step2.optTol = 1e-12;

cfg.io = struct();
cfg.io.outDir = fullfile(pwd, 'figures', 'noise_spectrum_sag_based_6dof_oracle_split');

summaryTbl = run_sag_noise_mc(cfg);
