%% This program plots the EEG signals
clc
clear all

load('CLASubjectC1512233StLRHand.mat')
% We create a container (dictionary in python)
chann = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3', 'Marks'};
chann_val = cell(1, length(chann));
eeg_reader = containers.Map(chann, chann_val);
% We assign the data, notice that the marker is added 
for i = [1:length(chann)],
    
   if i ~= length(chann),
       eeg_reader(char(chann(i))) = o.data(:,i);
   else
       eeg_reader(char(chann(i))) = o.marker(:,1)*10
   end
   
end


%% PLOT

time = 0:1/o.sampFreq:o.nS/o.sampFreq-1/o.sampFreq;

figure(1)
for i = [1:length(eeg_reader)]
    chann_offset = eeg_reader(char(chann(end - i + 1)))+ (i-1)*30;
    hold on
    plot( time(:) ,chann_offset(:), 'LineWidth', 0.8)
end
for i=1:length(eeg_reader)
    line([0,length(time)],[(i-1)*30,(i-1)*30],'LineStyle','--','Color',[0.4 0.3 0.5])
    %annotation('textbox',[.2 .6-i*0.03 .1 .2],'String',char(chann(i)),'EdgeColor','None', 'Color', [0.2 0.1 0.1] )
end
set(gca,'ytick',[0:30:(length(eeg_reader)-1)*30])
set(gca,'yticklabel',fliplr(chann), 'fontweight','bold')
set(gca,'xlim',[1000 1010],'ylim',[0 800])
xlabel('Time (s)')
ylabel('Channels')




wentropy

%% FLOWING PLOT (REAL-TIME SIMULATION) 
time = 0:1/o.sampFreq:o.nS/o.sampFreq-1/o.sampFreq;


figure(1)
for i = [1:length(eeg_reader)]
    chann_offset = eeg_reader(char(chann(end - i + 1)))+ (i-1)*30;
    hold on
    plot( time((1:30000)) ,chann_offset(1:30000), 'LineWidth', 0.8)
end
for i=1:length(eeg_reader)
    line([0,length(time)],[(i-1)*30,(i-1)*30],'LineStyle','--','Color',[0.4 0.3 0.5])
    %annotation('textbox',[.2 .6-i*0.03 .1 .2],'String',char(chann(i)),'EdgeColor','None', 'Color', [0.2 0.1 0.1] )
end
set(gca,'ytick',[0:30:(length(eeg_reader)-1)*30])
set(gca,'yticklabel',fliplr(chann), 'fontweight','bold')
xlabel('Time (s)')
ylabel('Channels')
annotation(figure(1),'rectangle',...
    [0.833291666666667 0.112179487179487 0.0719166666666666 0.813034188034188],...
    'Color',[1 0 1],...
    'LineWidth',3);
annotation(figure(1),'textbox',...
    [0.8390625 0.873931623931624 0.0614583333333334 0.0405982905982905],...
    'String','Window',...
    'FitBoxToText','off');

figure(1);
Dx=10;

for n=0:1/16:max(time(1:30000))
    
%       axis([time(n) time(n+Dx) y1 y2]),drawnow
   set(gca,'xlim',[n n+Dx],'ylim',[0 800]); drawnow, pause(0.0295)
   
end
set(gca,'xlim',[0 10],'ylim',[0 800])

%%



figure(1);hold all
Dx=1000;y1=0;y2=800;
count = 0
count_2 = 0
for n=1:(max((time)))
    for i = [1:length(eeg_reader)]
        chann_offset = eeg_reader(char(chann(end - i + 1)))+ (i-1)*30;
        hold on
        plot( time(count*Dx+1:(count+1)*Dx) ,chann_offset(count*Dx+1:(count+1)*Dx), 'LineWidth', 0.8); 
    end
    count = count + 1;
    count_2 = count * 200
    
    for i=1:length(eeg_reader)
    line([0,length(x)],[(i-1)*30,(i-1)*30],'LineStyle','--','Color',[0.4 0.3 0.5])
    %annotation('textbox',[.2 .6-i*0.03 .1 .2],'String',char(chann(i)),'EdgeColor','None', 'Color', [0.2 0.1 0.1] )
    end
    set(gca,'ytick',[0:30:(length(eeg_reader)-1)*30])
    set(gca,'yticklabel',fliplr(chann))
    xlabel('Time (s)')
    ylabel('Channels')
    set(gca,'xlim',[0 10],'ylim',[0 800])
    axis([time(n) time(n+Dx) y1 y2]);drawnow
end
%%
% % % x = 0:10:300;
% % % %generate numbers in [-50,50]
% % % y1 = -50 + 100*rand(numel(x),1);
% % % y2 = -50 + 100*rand(numel(x),1);
% % % y3 = -50 + 100*rand(numel(x),1);
% % % y4 = -50 + 100*rand(numel(x),1);
% % % %now add some offset to move the y values up
% % % y2 = y2 + 200; %zero would be at y = 200
% % % y3 = y3 + 400; %zero would be at y = 400
% % % y4 = y4 + 600; %zero would be at y = 600
% % % plot(x,y1,x,y2,x,y3,x,y4)
% % % for i=1:length(eeg_reader)
% % %     line([0,length(x)],[(i-1)*30,(i-1)*30],'LineStyle','--','Color',[0 0 0])
% % % end
% % % set(gca,'ytick',[-50,0,50,150,200,250,350,400,450,550,600,650])
% % % set(gca,'yticklabel',repmat({'-50','0','50'},1,3))
% % %  text(-200,0,'z=50')
% % % text(305,200,'z=150')
% % % text(305,400,'z=250')
% % % text(305,600,'z=350')
% % % xlabel 'Position [mm]'
% % % ylabel('$u_z[10^{-3} m/s]$','interpreter','latex')
