function [data,cats,dur_tw] = video_lowlevelAudio(videofile,time_window,hop)
% This function utilizes MIRtoolbox to extract some predefined low-level
% features mainly important for fMRI/eye-tracking analysis
% INPUT
%       videofile   = myVideo.mp4
%       time_window = temporal length of each time window (in seconds)
%       hop         = how far from the start of last time-window to start the next
%                     time-window (in seconds). If you like to have
%                     interleaved time-windows then hop < time_window
% OUTPUT
%       data        = extracted auditory features
%       cats        = column names in data matrix
%       diff        = Difference between the true duration of the audio
%                     stream and time_window*n_tw
%
% Severi Santavirta, last modified 27th May 2022


audioIn = mirframe(videofile,time_window,'s',hop,'s'); % Calculate everything in 25ms time-windows, windows are not interleaved
n_tw = size(mirgetdata(audioIn),2);
dur_tw = n_tw*time_window;


cats = {'rms','rms_d','zerocrossing','centroid','spread','entropy','rollof85','roughness'};
data = zeros(n_tw,8);

data(:,1) = mirgetdata(mirrms(audioIn)); % rms
data(2:end,2) = diff(data(:,1)); % Derivative
data(:,3) = mirgetdata(mirzerocross(audioIn)); % Zero crossing of audio wawe, "noisiness"
mu = mirgetdata(mircentroid(audioIn)); % Mean of the spectrum 
mu(isnan(mu)) = nanmean(mu); % if silence, has NaN value, substitute with the average value in the sequence;
data(:,4) = mu;  
sd = mirgetdata(mirspread(audioIn)); % SD of the spectrum
sd(isnan(sd)) = nanmean(sd); % if silence, has NaN value, substitute with the average value in the sequence;
data(:,5) = sd;  
ent = mirgetdata(mirentropy(audioIn)); % Entropy of the spectrum
ent(isnan(ent)) = nanmean(ent); % if silence, has NaN value, substitute with the average value in the sequence;
data(:,6) = ent;  
high_nrg = mirgetdata(mirrolloff(audioIn)); % "85% of energy is under this frequency"
high_nrg(isnan(high_nrg)) = nanmean(high_nrg); % if silence, has NaN value, substitute with the average value in the sequence;
data(:,7) = high_nrg;  
data(:,8) = mirgetdata(mirroughness(audioIn)); % Roughness of the sound

end
