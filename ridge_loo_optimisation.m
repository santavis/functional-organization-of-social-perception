function u = ridge_loo_optimisation(YY,XX,lambda,throw_away,opt_stats_dir)
% Uses data from multiple subjects and calculates the average
% prediction error across the subjects for the particular lambda. The
% prediction error is calculated using leave-one-subject-out
% cross-validation.
%
% Inputs:
%
%           YY     =     N x 1 cell array, where N is the number of
%                        subjects. Each of the cells contains the BOLD
%                        signals of the subject, arranged in an M x K
%                        matrix, where M is the number of voxels, and K is
%                        the number of data points (TRs).
%
%           XX     =     N x 1 cell array. Each cell contains the design
%                        matrices of the subject, arranged in an K x F
%                        matrix, where K is the number of time-points (TRs)
%                        and F is the number of features. Note that the
%                        design matrix 
%
%         lambda   =     The regularization parameter (referred to by k in
%                        Matlab's documentation), restricted to nonnegative
%                        values.
%
%       throw_away =     [start,end], e.g. [2,2] => Exclude 2 timepoints
%                        from the beginning and 2 timetoints form the end
%                        of the time series.
%
%    opt_stats_dir =     Directory where the cross-validation information
%                        will be saved. Subjectwise prediction errors
%                        should be checked. Even a few subjects with
%                        exceptionally high prediction error (e.g. due to 
%                        problems in preprocessing) may lead to unoptimal
%                        lamda parameter estimation.
%
% Outputs:
%
%            u     =     Average prediction error over the voxels and
%                        subjects.
%
%
% Tomi Karjalainen, January 16th 2020, last modidication on 
% February 8th 2021 by Severi Santavirta

tic;
N = length(YY);

for i = 1:N
    idx = setdiff(1:N,i); % The participants included in model estimation
    for j = 1:(N-1)
        %fprintf('\x03bb = %f : %.0f/%.0f : %.0f/%.0f\n',lambda,i,N,j,N-1);
        jj = idx(j); % The current participant
        
        Y = YY{jj}; % fMRI data from the current participant
        X = XX{jj}; % The design matrix of the current participant
        
        % Scale the BOLD signal at each voxel by the mean
        % See Chen et al. (2017) Neuroimage https://www.ncbi.nlm.nih.gov/pubmed/27729277
        mu = mean(Y,2);
        M = repmat(mu,[1 size(Y,2)]);
        Y = Y./M;
        
        % Throw away the given time points from the beginning and the end of scan.
        include_idx = throw_away(1)+1:size(X,1)-throw_away(2);
        X = X(include_idx,:);
        Y = Y(:,include_idx);
        
        % Standardize the design matrix so that everything is on the same
        % scale
        X = zscore(X);
        
        % Initialize a matrix for the regression coefficients
        if(j == 1)
            B = nan(size(X,2)+1,size(Y,1),N-1);
        end
        
        % Estimate the regression coefficients for all voxels for the
        % participant
        parfor k = 1:size(Y,1)
            y = Y(k,:)';
            B(:,k,j) = ridge(y,X,lambda,0);
        end
    end
    
    % Specify the design matrix and fMRI data of the remaining participant
    X_loo = XX{i};
    Y_loo = YY{i};
    
    % Scale the BOLD signal by its mean
    mu = mean(Y_loo,2);
    M = repmat(mu,[1 size(Y_loo,2)]);
    Y_loo = Y_loo./M;
    
    % Throw away the given time points from the beginning and the end of scan.
    include_idx = throw_away(1)+1:size(X_loo,1)-throw_away(2);
    X_loo = X_loo(include_idx,:);
    Y_loo = Y_loo(:,include_idx);
    
    % Standardize the design matrix
    X_loo = zscore(X_loo);
    
    % For each voxel, average over the participant-wise betas
    B_mean = mean(B,3);
    
    % Initialize the distance matrix
    if(i == 1)
        D = nan(size(Y_loo,1),N);
    end
    
    % For each voxel, calculate the prediction error
    parfor k = 1:size(Y_loo,1)
        y = Y_loo(k,:)';
        yhat = [ones(size(X_loo,1),1) X_loo]*B_mean(:,k);
        D(k,i) = sum((y-yhat).^2).^0.5;
    end
    
    
end

% Average over the prediction errors
u = mean(D(:));

% Save voxelwise prediction errors information

stats = struct;
stats.lambda = lambda;
stats.pe = u;
stats.mean_subjectwise_pe = mean(D,1,'omitnan');
stats.voxel_and_subjectwise_pe = D;
stats.time_elpased = toc;

f = sprintf('%s/opt_voxelwise_pe_lambda_%s.mat',opt_stats_dir,num2str(floor(lambda)));
save(f,'stats');


end