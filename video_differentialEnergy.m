function dnrg = video_differentialEnergy(videofile)
% Function calculates the difference between voxels between adjacent frames
% Estimate of "movement/change" of image
%
% Severi Santavirta & Juha Lahnakoski

vidIn = VideoReader(videofile);

dnrg = zeros(vidIn.NumFrames,1);
imLast=zeros(vidIn.Height,vidIn.Width,3);

t=0;
while hasFrame(vidIn)
    t=t+1;
    im = double(vidIn.readFrame);
    dnrg(t)=sqrt(nanmean((im(:)-imLast(:)).^2));
    imLast = im;
end

end
