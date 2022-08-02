function [nrg,images,transform,masked_transform,filtered] = video_spatialEnergy(videofile,lowpass,radius)
% Function calculated the Fourier transform for each frame of the video. A
% mask is used to filter unwanted frequecies, circular low-pass or high-pass
% filters can be used. Spatial energy is then calculated as mean of the
% Fourier filtered images.
%
% INPUT
%           videofile           = myFile.mp4
%           lowpass             = 1 (filter out high frequencies), 0 (filter out low frequencies)
%           radius              = radius of the circular frequency mask as
%                                 percentage of the image height (radius/100*image.Height is the radius of the mask)
%
% OUTPUT
%           nrg                 = spatial energy
%           images              = frames of the video
%           transform           = Fourier transform for visualization
%           masked_transform    = Filtered Fourier transform for
%                                 visualization
%           filtered            = Fourier filtered frames for vizualisation
%
% Severi Santavirta & Juha Lahnakoski, 27th of May, 2022


v = VideoReader(videofile);
[Y,X]=ndgrid(1:v.Height,1:v.Width);

rad = radius/100*v.Height;
if(lowpass)
    % High-pass filter (used as low-pass filter)
    fftmask=max((Y.^2+X.^2<rad^2),max((flipud(Y).^2+X.^2<rad^2),max((Y.^2+fliplr(X).^2<rad^2),(flipud(Y).^2+fliplr(X).^2<rad^2))));
else
    % Low-pass filter (used as high-pass filter)
    fftmask = createCirclesMask(size(Y),[size(Y,2)/2 size(Y,1)/2],rad);
end

nrg = zeros(v.NumFrames,1); % Spatial energy
images = zeros(v.Height,v.Width,3,v.NumFrames); % Save original images as double (uint8(images(:,:,:,fr)) for plotting);
transform = zeros(v.Height,v.Width,v.NumFrames); % Save Fourier transforms
masked_transform = zeros(v.Height,v.Width,v.NumFrames); % Masked fourier transform
filtered = zeros(v.Height,v.Width,v.NumFrames); % Save Fourier filtered images

fr=0;
while hasFrame(v)
    fr=fr+1;
    images(:,:,:,fr) = v.readFrame; 
    F = fft2(nanmean(images(:,:,:,fr),3)); % Fourier transform
    
    filtered(:,:,fr) = abs(ifft2(~fftmask.*F)); % Filtered image 
    nrg(fr) = nanmean(nanmean(filtered(:,:,fr))); % Spatial energy
    
    
    F = fftshift(F); % Center FFT
    F = abs(F); % Get the magnitude
    F = log(F+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
    transform(:,:,fr) = mat2gray(F); % Use mat2gray to scale the image between 0 and 1
    masked_transform(:,:,fr) = ~fftmask.*F;
    
    %subplot(1,3,1);
    %imagesc(uint8(images(:,:,:,fr)));
    %title('Image');
    %subplot(1,3,2);
    %imagesc(uint8(filtered(:,:,fr)));
    %title('Filtered');
    %subplot(1,3,3);
    %imagesc(masked_transform(:,:,fr));
    %title('Masked Fourier transform');

    end

end