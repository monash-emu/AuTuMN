
xlims=[2010,2035.5];
targetcolour=[.5,.5,.5];
if strcmp(country,'India')==1
    figrow=1;
    ylims=[200,250,40,5];
elseif strcmp(country,'China')==1
    figrow=2;
    ylims=[80,80,4,7];
elseif strcmp(country,'RSA')==1
    figrow=3;
    ylims=[1000,1500,200,15];
    country='South Africa';
end
rows=3;
% rows=2;
% figrow=1;

if scenario<2
    colour=[.8-.8.*scenario,.8-.8.*scenario,1];
elseif scenario==2
    colour='r';
end
if sum(intervention,2)==0
    colour=[0,0,0];
end
% if intervention(5)>0
%     colour='m';
% end

%Incidence subplot
subplot(rows,4,(figrow-1)*4+1)
hold on
plot(time,inc,'Color',colour)
xlim(xlims)
ylim([0,ylims(1,1)])
title(['Incidence, ',country])
plot([2015.5,2025.5,2035.5],[inc(yearpoints(2015,3)),...
    inc(yearpoints(2015,3))/2,inc(yearpoints(2015,3))/10],'x','Color',targetcolour)
line([2015.5,2025.5],[inc(yearpoints(2015,3)),...
    inc(yearpoints(2015,3))/2],'LineStyle',':','Color',targetcolour)
line([2025.5,2035.5],[inc(yearpoints(2015,3))/2,...
    inc(yearpoints(2015,3))/10],'LineStyle',':','Color',targetcolour)
ylabel('per 100,000 per year')
xlabel('Year')
if figrow==1
    text(2015.5+.6,inc(yearpoints(2015,3))+3,'Baseline',...
        'VerticalAlignment','Middle',...
        'HorizontalAlignment','Left',...
        'FontSize',8)
    text(2025.5+.6,inc(yearpoints(2015,3))/2+3,'Interim',...
        'VerticalAlignment','Middle',...
        'HorizontalAlignment','Left',...
        'FontSize',8)
    text(2035.5-.6,inc(yearpoints(2015,3))/10,'Pre-elimination',...
        'VerticalAlignment','Middle',...
        'HorizontalAlignment','Right',...
        'FontSize',8)
end

%Prevalence subplot
subplot(rows,4,(figrow-1)*4+2)
hold on
plot(time,prev,'Color',colour)
xlim(xlims)
ylim([0,ylims(1,2)])
ylabel('per 100,000')
xlabel('Year')
title(['Prevalence, ',country])

%Mortality subplot
subplot(rows,4,(figrow-1)*4+3)
hold on
plot(time,mort,'Color',colour)
xlim(xlims)
ylim([0,ylims(1,3)])
title(['Mortality, ',country])
plot([2015.5,2025.5,2035.5],[mort(yearpoints(2015,3)),...
    mort(yearpoints(2015,3))/4,mort(yearpoints(2015,3))/20],'x','Color',targetcolour)
line([2015.5,2025.5],[mort(yearpoints(2015,3)),...
    mort(yearpoints(2015,3))/4],'LineStyle',':','Color',targetcolour)
line([2025.5,2035.5],[mort(yearpoints(2015,3))/4,...
    mort(yearpoints(2015,3))/20],'LineStyle',':','Color',targetcolour)
ylabel('per 100,000 per year')
xlabel('Year')

%Proportion MDR subplot
subplot(rows,4,(figrow-1)*4+4)
hold on
plot(time,mdrpropinc*100,'Color',colour)
xlim(xlims)
ylim([0,ylims(1,4)])
title(['Proportion MDR in new cases, ',country])
ylabel('Percentage')
xlabel('Year')
if strcmp(country,'South Africa')==1
    country='RSA';
end

%%
%To save later on, after the figures have been populated
% [~,computer]=system('hostname');
% computer=computer(1:10);
% if strcmp(computer,'JTrauerWin')==1
%     cd 'C:\Users\JTrauer\Dropbox\James Trauer\TB MAC\Graphs\' %Burnet computer
% end
% if strcmp(computer,'553D-UOM11')==1
%     cd 'C:\Users\jtrauer\Dropbox\james trauer\TB MAC\Graphs\' %Doherty
% end
% if strcmp(computer,'James-Trau')==1
%     cd '/Users/jamestrauer/Dropbox/James Trauer/TB MAC/Graphs' %Home
% end

intervention7fig=figure(1);
set(gcf,'PaperUnits','centimeters','PaperPosition',[0,0,30,20])
print -dtiff intervention7fig.tif -r200 %Set to -r1000 if we want higher res