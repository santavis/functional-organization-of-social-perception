function [optMask,optMaskVect] = ridge_optimisation_mask(gmMask,funcMask,thr)
% Choose pseudorandomly  a fractions of voxels included in both grey matter and functional
% masks
%
% INPUTS
%           gmMask      = Population level grey matter mask
%           funcMask    = Population level EPI mask
%           thr         = Fraction of grey matter voxels included (0<thr<=1)
%
% Severi Santavirta, last modification on October 26th, 2021

    siz = size(gmMask);   
    optMask = zeros(siz(1),siz(2),siz(3));
    optMask(logical(gmMask) == 1 & logical(funcMask) == 1) = 1;
    optMaskVect = reshape(optMask,[siz(1)*siz(2)*siz(3),1,1]);

    rng(1);
    rndIdx = randperm(sum(optMaskVect),floor(sum(optMaskVect)*thr))'; 

    k = 0;
    optMaskVect2 = zeros(siz(1)*siz(2)*siz(3),1);
    for I = 1:size(optMaskVect,1)
        if(optMaskVect(I))
            k = k+1;
            if(ismember(k,rndIdx))
                optMaskVect2(I) = 1;
            end
        end

    end

    optMaskVect = optMaskVect2;
    optMask = reshape(optMaskVect,[siz(1),siz(2),siz(3)]);
end
