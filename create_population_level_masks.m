%% Create population level brain masks for ridge redression optimisation

% For ridge regression optimisation a population level grey matter and EPI
% masks are needed and they are calculated based on each subjects mask from
% preprosessing 
%
% Population level EPI mask
%   Only voxels where all subjects have signal are included.
%
% Population level grey matter mask
%   Only voxels where the population level average probability for grey matter is over 0.5
%   are included.
%
% Severi Santavirta, last modified on October 26th, 2021



% Subjects, this part is spesific for megafMRI project, names of the
% EPI-files and regressor files are coded by these names. Needs to be
% modified for other projects

subjects = {'sub-001';'sub-002'};


%% EPI mask
sum_mask = zeros(65,77,65); % Depends on the voxel size. Please fill accordingly.
for I = 1:size(subjects,1)
    fprintf('%s/%s\n',num2str(I),num2str(size(subjects,1)));
    mask_path = sprintf('PATH/%s_epi_mask.nii.gz',subjects{I}); % Hard coded naming
    V = spm_vol(mask_path);
    mask = spm_read_vols(V);
    sum_mask = sum_mask+mask;
end

avg_mask = sum_mask/size(subjects,1);
avg_mask(avg_mask<1) = 0; 
V.fname = 'PATH/epi_mask.nii';
spm_write_vol(V,avg_mask);


%% Grey matter mask
sum_mask = zeros(193,229,193); % Depends on the voxel size. Please fill accordingly.
for I = 1:size(subjects,1)
    fprintf('%s/%s\n',num2str(I),num2str(size(subjects,1)));
    mask_path = sprintf('PATH/%s_GM_probseg.nii.gz',subjects{I}); % Probability segmentation, hard coded naming
    V = spm_vol(mask_path);
    mask = spm_read_vols(V);
    sum_mask = sum_mask+mask;
end

avg_mask = sum_mask/size(subjects,1);
avg_mask(avg_mask<0.5) = 0;
avg_mask(avg_mask>0) = 1;
V.fname = 'PATH/gm_mask.nii';
spm_write_vol(V,avg_mask);
