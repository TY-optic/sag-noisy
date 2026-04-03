function [Zfeat, maskFeat, aux] = build_sag_feature_map(local, params)
%BUILD_SAG_FEATURE_MAP
% 直接基于 sag 构造用于二维粗配准/细配准的标量特征图。
% 不使用 PHS；仅可选地做平面去趋势与轻微平滑。
%
% 输入:
%   local.X, local.Y, local.Z, local.mask
%   params.marginPix
%   params.removeBestFitPlane
%   params.smoothSigmaPix
%   params.useZScore
%   params.clipSigma
%
% 输出:
%   Zfeat    : 配准使用的特征图
%   maskFeat : 有效特征区域
%   aux      : 中间信息

X = double(local.X);
Y = double(local.Y);
Z = double(local.Z);
M = logical(local.mask) & isfinite(Z);

if isfield(params, 'marginPix') && params.marginPix > 0
    maskFeat = erode_mask_disk_local(M, params.marginPix);
    if nnz(maskFeat) < 50
        maskFeat = M;
    end
else
    maskFeat = M;
end

Zwork = Z;
planeCoef = [0;0;0];
removePlane = isfield(params, 'removeBestFitPlane') && params.removeBestFitPlane;
if removePlane
    xv = X(maskFeat);
    yv = Y(maskFeat);
    zv = Zwork(maskFeat);
    Aplane = [xv(:), yv(:), ones(numel(xv),1)];
    planeCoef = Aplane \ zv(:);
    Zplane = planeCoef(1) * X + planeCoef(2) * Y + planeCoef(3);
    Zwork(maskFeat) = Zwork(maskFeat) - Zplane(maskFeat);
end

sigmaPix = 0;
if isfield(params, 'smoothSigmaPix') && ~isempty(params.smoothSigmaPix)
    sigmaPix = params.smoothSigmaPix;
end
if sigmaPix > 0
    Zwork = masked_gaussian_smooth_local(Zwork, maskFeat, sigmaPix);
end

zv = Zwork(maskFeat);
mu = mean(zv, 'omitnan');
sd = std(zv, 0, 'omitnan');
useZScore = ~(isfield(params, 'useZScore') && ~params.useZScore);
if useZScore
    if ~(isfinite(sd) && sd > 0)
        sd = 1;
    end
    Zwork(maskFeat) = (Zwork(maskFeat) - mu) / sd;
else
    Zwork(maskFeat) = Zwork(maskFeat) - mu;
end

clipSigma = inf;
if isfield(params, 'clipSigma') && ~isempty(params.clipSigma)
    clipSigma = params.clipSigma;
end
if isfinite(clipSigma) && clipSigma > 0
    Zwork(maskFeat) = min(max(Zwork(maskFeat), -clipSigma), clipSigma);
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

function maskErode = erode_mask_disk_local(mask, radiusPix)
if radiusPix <= 0
    maskErode = mask;
    return;
end
se = strel('disk', radiusPix, 0);
maskErode = imerode(mask, se);
end
