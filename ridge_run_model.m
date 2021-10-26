function [B,R2] = ridge_run_model(epi_file,X,lambda,mask,throw_away)
% Calculate first level ridge regression for given subject

% Inputs:
%
%         epi_file    =     path to subject's preprocessed (and detrended)
%                           epi.nii -file
%
%            X        =     model design matrix without constant term
%                           (will be added automatically)
%          lambda     =     optimal regularisation parameter
%
%           mask      =     path to brainmask.nii -file
%
%        throw_away   =     [start,end], e.g. [2,2] => throw away 2 time
%                           points from the beginning and 2 time points 
%                           from the end ofthe scan.
% Outputs:
%         
%            B        =     4-dim matrix containing 3-dim volumes of first
%                           level coefficients for each predictor in X
%           
%            R2       =     3-dim volume of R²-values in each voxel
%
% Tomi Karjalainen, last modification on October 26th 2021 by Severi
% Santavirta

if(ischar(mask))
    V = spm_vol(mask);
    mask = spm_read_vols(V);
    clear V
end

mask = mask > 0;

n_voxels = sum(mask(:));

if(ischar(epi_file))
    V = spm_vol(epi_file);
    img = spm_read_vols(V);
    clear V;
else
    img = epi_file;
end

siz = size(img);
Y = reshape(img,[prod(siz(1:3)) siz(4)]);
Y = Y(mask,:);

% Scale the BOLD signal at each voxel by the mean
% See Chen et al. (2017) Neuroimage https://www.ncbi.nlm.nih.gov/pubmed/27729277
mu = mean(Y,2);
M = repmat(mu,[1 size(Y,2)]);
Y = Y./M;

% Throw away the given time points from the beginning and the end of scan.
include_idx = throw_away(1)+1:size(X,1)-throw_away(2);
X = X(include_idx,:);
Y = Y(:,include_idx);

% Scale the design matrix columns
X = zscore(X);

n_predictors = size(X,2);

B = nan([prod(siz(1:3)) n_predictors]);
BT = nan([n_voxels n_predictors]);

R2 = nan([prod(siz(1:3)) 1]); % R²
R2_mask = nan([n_voxels 1]);


for i = 1:n_voxels
    y = Y(i,:)';
    b = ridge(y,X,lambda,0); % Constant term will be added into the design matrix
    yhat = [ones(size(X,1),1) X]*b;
    R2_mask(i,:) = (corr(y,yhat))^2;
    BT(i,:) = b(2:end);
end

B(mask,:) = BT;
R2(mask,1) = R2_mask;

B = reshape(B,[siz(1:3) n_predictors]);
R2 = reshape(R2,[siz(1) siz(2) siz(3)]);

end