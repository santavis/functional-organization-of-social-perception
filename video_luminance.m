function [L,Fr] = video_luminance(videofile)
% Function calculates luminance of a video input.
% Luminance (L) is returned for every video frame.  Fr is video frame
% rate
%
%
% "Luminance" is approximated as rgb2hsv transformation where
% Luminance ~ V = max{R,G,B}
%
% Severi Santavirta

%% Video

try
    obj = VideoReader(videofile);
catch ME
    error('Problems with reading the file %s.',videofile);
end

Fr = obj.FrameRate;
num_frames = ceil(obj.Duration*Fr);
L = nan(num_frames,1);

i = 0;
while(hasFrame(obj))
    i = i + 1;
    frame = readFrame(obj);
    hsv_frame = rgb2hsv(frame);
    v = squeeze(hsv_frame(:,:,3));
    L(i) = mean(v(:));
end

L(isnan(L)) = [];
end
