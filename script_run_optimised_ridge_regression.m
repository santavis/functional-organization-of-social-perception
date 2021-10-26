%% Feature cluster first-level ridge regression with nuisance-regressors (lambda is optimized for the whole model, one model for regression)

%   1.    A fraction af voxels common to both population level grey matter mask and epi mask
%         is chosen for ridge parameter optimisation (would be computationally prohibitive to use 
%         e.g. all grey matter voxels for ridge parameter optimisation)
% 
%   2.    Data is processed to matrix form for ridge parameter optimisation
%   
%   3.    Ridge parameter (lambda) is optimised using leave-one-subject-out
%         cross-validation (this part takes to most of the time)
%
%   4.    First level ridge regression is calculated using optimised ridge
%         parameter (lambda)
%
%
% Requirements: SPM software
%
%
% Severi Santavirta, last modification on October 26th, 2021


clear;clc;

%% INPUT
maxNumCompThreads(1); % Limit to 1 CPU.

% Path to ridge regression functions
addpath('PATH/ridge_functions') 

% Directory of detrended epi (.nii) -files. 
preproc_dir = 'PATH/preproc';

% Subjects, this part is spesific for the megafMRI project, names of the
% EPI-files and regressor files are coded by these indices. Needs to be
% modified for other projects
subjects = {'sub-001';'sub-002'}; 

% Directory of regressors. Every subject should have a mat-file of a NxM matrix of convolved regressors where N is the number of frames 
% and M is the number of regressors
regressor_dir = 'PATH/regressors';

% Directory of nuisance regressors. Every subject should have mat-file of a NxM matrix of convolved regressors where N is the number of frames 
% and M is the number of nuisance regressors
nuisance_dir = 'PATH/nuisance_regressors';

% Path to population level grey matter mask
gmMask_path = 'PATH/gm_mask.nii'; 

% Path to population level EPI mask
funcMask_path = 'PATH/epi_mask.nii';

% Path where the mask for ridge parameter (lambda) optimisation will be saved (.nii-file)
% Mask includes a fraction of voxels common to both grey matter mask and EPI mask (would be computationally prohibitive to use 
% e.g. whole grey matter voxel for ridge parameter optimisation)
optMask_path = 'PATH/optMask.nii'; 

% Choose pseudorandomly and uniformly a fraction of X of common voxels from
% grey matter mask and EPI mask
optimVoxels = 0.2; 

% Filename for a mat file. (The processed data for ridge optimisation is saved for later use)
opt_model_path = 'PATH/opt_model.mat';

% How many frames are excluded from start and end
% of the time series in the analysis
throw_away = [2,2];

% Filename for a mat-file. (The optimal lambda value will be saved for later use)
opt_lambda_path = 'PATH/opt_global_lambda_cluster13.mat'; % Where to save optimal lambda values and details about estimation process

% Path to direcory. (The stepwise lambda optimisation information
% will be stored here, see ridge_loo_final.m) 
opt_lambda_cv_stats = 'PATH/opt_lambda_cv_stats'; % Stepwise optimisation information is stored here.

% Path to directory (first level results will be stored here)
first_level_dir = 'PATH/first_level';


% How many cores will be used. Use with caution! With small (e.g. 2mm)  voxelsize, even
% using 2 cores might lead to out-of-memory and computer freezeing.
paral = 2;


%% Choose a random and uniform sample of grey matter voxels where lambda validation will be performed

if(~exist(first_level_dir,'dir'))
    mkdir(first_level_dir);
end

if(~exist(opt_lambda_cv_stats,'dir'))
    mkdir(opt_lambda_cv_stats);
end

if(~exist(optMask_path,'file'))
    fprintf('Creating mask for ridge optimisation\n');
    
    % Choose pseudorandomly and uniformly a fraction of X  of grey matter voxels from areas that are also inside functional mask 
    VgmMask = spm_vol(gmMask_path); VfuncMask = spm_vol(funcMask_path);
    gmMask = spm_read_vols(VgmMask); funcMask = spm_read_vols(VfuncMask);

    [optMask,optMaskVect] = ridge_optimisation_mask(gmMask,funcMask,optimVoxels);

    VgmMask.fname = optMask_path;
    spm_write_vol(VgmMask,optMask);
    clear VgmMask VfuncMask gmMask funckMask
else
    VoptMask = spm_vol(optMask_path);
    optMask = spm_read_vols(VoptMask);
    siz = size(optMask);
    optMaskVect = reshape(optMask,[siz(1)*siz(2)*siz(3),1,1]);
    clear VoptMask siz
end


%% Create subjectwise models for lambda cross-validation (N features of interest + M nuisance regressors)

if(~exist(opt_model_path,'file'))
    fprintf('Creating a model for ridge optimisation\n')
    
    X = cell(size(subjects,1),1); 
    Y = cell(size(subjects,1),1);  
    
    for I = 1:size(subjects,1)
        fprintf('%s/%s\n',num2str(I),num2str(size(subjects,1)));

        reg_subj_path = sprintf('%s/megafmri-localizer-%s_reg-clusters.mat',regressor_dir,subjects{I}); % megafmri spesific names
        nui_subj_path = sprintf('%s/%s_nuisance_reg.mat',nuisance_dir,subjects{I}); % megafmri spesific names
  
        reg_subj = load(reg_subj_path);
        nui_subj = load(nui_subj_path); 
        X{I,1} = cat(2,reg_subj.R(1:467,:),nui_subj.R(1:467,:)); % Hard coded last frame for megafmri, because subjects had different amount of frames after stimulus.
        x_labels = cat(2,reg_subj.cluster_labels,nui_subj.nui_cats);

        img_path = sprintf('%s/r%s_task-localizer_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold-detrended.nii',preproc_dir,subjects{I}); %megafmri spesific names
        Vimg = spm_vol(img_path);
        img = spm_read_vols(Vimg);
        
        siz = size(img);
        optImg = zeros(siz);
        for J = 1:siz(4)
            frame = squeeze(img(:,:,:,J));
            frame(optMask==0) = 0;
            optImg(:,:,:,J) = frame;
        end
        
        optImgVect = reshape(optImg,[siz(1)*siz(2)*siz(3),siz(4)]);
        tmp = sum(optImgVect,2); tmp(tmp>0) = 1;
        Y{I,1} = optImgVect(logical(tmp),1:467);
        clear img Vimg frame tmp optImgVect reg_subj nui_subj
        
    end
    
    save(opt_model_path,'Y','X','x_labels');
    clear reg_subj_path nui_subj_path reg_subj nui_subj img_path img siz frame

else
    load(opt_model_path);
end


%% Optimize lambda over all regressors for ridge regression with leave-one-subject-out cross validation (one model)

 if(exist(opt_lambda_path,'file'))
    load(opt_lambda_path);
 else
    fprintf('Optimizing ridge parameter with cross validation (may take a really long time)\n');
    p = parpool(paral*6); % Does not take a lot of memory. 
    
    % Use matlabs minimizer function to optimize feature cluster
    % spesific lambda values
    fun = @(lambda) ridge_loo_optimisation(Y,X,lambda,throw_away,opt_lambda_cv_stats);
    options = optimset('Display','Iter','TolX',0.5);
    [opt_lambda,min_pe,exitflag,output] = fminbnd(fun,5,500,options); % Depending on dataset the boundaries [5 500] might not be wide enough, needs to be checked after.
  
    save(opt_lambda_path,'opt_lambda','min_pe','exitflag','output');    
    clear options fun
    delete(p);
end

%% Run ridge regression with optimal lambda (one model)

for I = 1:size(subjects,1)
    fprintf('Running ridge regression with optimized ridge parameter, subject %s/%s\n',num2str(I),num2str(size(subjects,1)));
    
    out_dir = sprintf('%s/%s',first_level_dir,subjects{I});
    if(~exist(out_dir,'dir'))
        mkdir(out_dir);
    end
    
    if(~exist(sprintf('%s/R2.nii.gz',out_dir),'file'))
        img_path = sprintf('%s/r%s_task-localizer_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold-detrended.nii',preproc_dir,subjects{I}); %megafmri spesific names
        Vimg = spm_vol(img_path);
        y = spm_read_vols(Vimg(1:467,:)); % Hard coded last frame for megafMRI

        Xs = X{I,1};
        [coef,R2] = ridge_run_model(y,Xs,opt_lambda,funcMask_path,throw_away);

        % Save R²-maps
        R2_file = sprintf('%s/R2.nii',out_dir);
        V = Vimg(1,:);
        V.fname = R2_file;
        spm_write_vol(V,R2);
        gzip(R2_file); delete(R2_file);
    end
    
    % Save feature specific beta maps
    for J = 1:size(coef,4)
        coef_file = sprintf('%s/beta_%s.nii',out_dir,x_labels{J});
        V = Vimg(1,:);
        V.fname = coef_file;
        spm_write_vol(V,squeeze(coef(:,:,:,J)));
        gzip(coef_file); delete(coef_file);
    end
end
    
