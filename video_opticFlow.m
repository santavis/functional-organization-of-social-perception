function opticFlow = video_opticFlow(videofile)
% Estimate optic flow based on LK algorithm and basic options. Optic flow is a single
% measure of absolute movement between adjacent frames.
%
% Severi Santavirta

% Methods
o = opticalFlowLK;


vidIn = VideoReader(videofile);

opticFlow = zeros(vidIn.NumFrames,1);

t = 0;

% Plot for checks
%h = figure;
%movegui(h);
%hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
%hPlot = axes(hViewPanel);

while hasFrame(vidIn)
    t=t+1;
    flow = estimateFlow(o,im2gray(readFrame(vidIn)));
    opticFlow(t,1) = sum(sqrt(flow.Vx.^2 + flow.Vy.^2),'all');

    %imshow(vidIn.readFrame)
    %hold on
    %plot(flow,'DecimationFactor',[5 5],'ScaleFactor',60,'Parent',hPlot);
    %hold off 
    
    %pause(0.1)
end
end