%SCRIPT TO WRITE OUTPUTS TO EXCEL SPREADSHEET

global country

%Get a unique computer identifier__________________________________________
[~,computer]=system('hostname');
computer=computer(1:10);

%Change directory__________________________________________________________
if strcmp(computer,'JTrauerWin')==1
    cd 'C:\Users\JTrauer\Dropbox\James Trauer\TB MAC\' %Burnet computer
end
if strcmp(computer,'553D-UOM11')==1
    cd 'C:\Users\jtrauer\Dropbox\james trauer\TB MAC\' %Doherty
end

%Baseline calibrations_____________________________________________________
if strcmp(country,'China')==1
    xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
        baselinecalibs(1,1:15),'Calibration','B9:P9')
elseif strcmp(country,'India')==1
    xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
        baselinecalibs(1,1:9),'Calibration','B10:J10')
elseif strcmp(country,'RSA')==1
    xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
        baselinecalibs(1,1:9),'Calibration','B11:J11')
    xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
        baselinecalibs(1,16:22),'Calibration','Q11:W11')
end

%Epi_Results data__________________________________________________________
%Columns
EpiDALYEcon_Resultsstartcolumn='A';
Epi_Resultsendcolumn='R';
EpiDALY_Resultsstrendcolumn='C';
DALY_1endcolumn='F';
DALY_2endcolumn='F';
DALY_3endcolumn='F';
if strcmp(country,'RSA')==1
    DALY_1endcolumn='N';
    DALY_3endcolumn='N';
end
Econ_Resultsendcolumn='Y';

%Rows
Epi_Resultsstartrow=8;
DALY_12Econstartrow=8;
DALY_3startrow=8;

if strcmp(country,'China')==1
    Epi_Resultsstartrow=Epi_Resultsstartrow+1056;
    DALY_12Econstartrow=DALY_12Econstartrow+1041;
    DALY_3startrow=DALY_3startrow+104;
end
if strcmp(country,'RSA')==1
    Epi_Resultsstartrow=Epi_Resultsstartrow+2032;
    DALY_12Econstartrow=DALY_12Econstartrow+2002;
    DALY_3startrow=DALY_3startrow+200;
end

if sum(intervention,2)<5&&sum(intervention,2)>0
    Epi_Resultsstartrow=Epi_Resultsstartrow+36+...
        (max(intervention)-1)*100+((min(5,scenario*4))-1)*20;
    DALY_12Econstartrow=DALY_12Econstartrow+21+...
        (max(intervention)-1)*100+((min(5,scenario*4))-1)*20;
    DALY_3startrow=DALY_3startrow+2+(max(intervention)-1)*10+...
        ((min(5,scenario*4))-1)*2;
    if intervention(2)>0
        Epi_Resultsstartrow=Epi_Resultsstartrow+300;
        DALY_12Econstartrow=DALY_12Econstartrow+300;
        DALY_3startrow=DALY_3startrow+30;
    end
    if intervention(3)>0
        Epi_Resultsstartrow=Epi_Resultsstartrow+700;
        DALY_12Econstartrow=DALY_12Econstartrow+700;
        DALY_3startrow=DALY_3startrow+70;
    end
    if intervention(4)>0&&intervention(5)==0&&strcmp(country,'India')==1
        Epi_Resultsstartrow=Epi_Resultsstartrow+800;
        DALY_12Econstartrow=DALY_12Econstartrow+800;
        DALY_3startrow=DALY_3startrow+80;
    elseif intervention(4)>0&&intervention(5)==0&&strcmp(country,'China')==1
        Epi_Resultsstartrow=Epi_Resultsstartrow+720;
        DALY_12Econstartrow=DALY_12Econstartrow+720;
        DALY_3startrow=DALY_3startrow+72;
    elseif intervention(4)>0&&intervention(5)==0&&strcmp(country,'RSA')==1
        Epi_Resultsstartrow=Epi_Resultsstartrow+620;
        DALY_12Econstartrow=DALY_12Econstartrow+620;
        DALY_3startrow=DALY_3startrow+62;
    end
    if intervention(5)>0&&strcmp(country,'India')==1
        Epi_Resultsstartrow=Epi_Resultsstartrow+820;
        DALY_12Econstartrow=DALY_12Econstartrow+820;
        DALY_3startrow=DALY_3startrow+82;
    elseif intervention(5)>0&&strcmp(country,'China')==1
        Epi_Resultsstartrow=Epi_Resultsstartrow+740;
        DALY_12Econstartrow=DALY_12Econstartrow+740;
        DALY_3startrow=DALY_3startrow+74;
    elseif intervention(5)>0&&strcmp(country,'RSA')==1
        Epi_Resultsstartrow=Epi_Resultsstartrow+640;
        DALY_12Econstartrow=DALY_12Econstartrow+640;
        DALY_3startrow=DALY_3startrow+64;
    end
    if intervention(6)>0
        Epi_Resultsstartrow=Epi_Resultsstartrow+740;
        DALY_12Econstartrow=DALY_12Econstartrow+740;
        DALY_3startrow=DALY_3startrow+74;
    end
elseif sum(intervention,2)>5
    Epi_Resultsstartrow=Epi_Resultsstartrow+956+((min(5,scenario*4))-1)*20;
    DALY_12Econstartrow=DALY_12Econstartrow+941+((min(5,scenario*4))-1)*20;
    DALY_3startrow=DALY_3startrow+94+((min(5,scenario*4))-1)*2;
    if strcmp(country,'China')==1||strcmp(country,'RSA')==1
        Epi_Resultsstartrow=Epi_Resultsstartrow-80;
        DALY_12Econstartrow=DALY_12Econstartrow-80;
        DALY_3startrow=DALY_3startrow-8;
    end
end

if sum(intervention,2)==0
    Epi_Resultsendrow=Epi_Resultsstartrow+35;
    DALY_12Econendrow=DALY_12Econstartrow+20;
elseif sum(intervention,2)>0
    Epi_Resultsendrow=Epi_Resultsstartrow+19;
    DALY_12Econendrow=DALY_12Econstartrow+19;
end
DALY_3endrow=DALY_3startrow+1;

intervention
scenario
Epi_Resultsxlrange=[num2str(EpiDALYEcon_Resultsstartcolumn),...
    num2str(Epi_Resultsstartrow),':',...
    num2str(Epi_Resultsendcolumn),...
    num2str(Epi_Resultsendrow)];
Epi_Resultsstrxlrange=[num2str(EpiDALYEcon_Resultsstartcolumn),...
    num2str(Epi_Resultsstartrow),':',...
    num2str(EpiDALY_Resultsstrendcolumn),...
    num2str(Epi_Resultsendrow)];
DALY_1xlrange=[num2str(EpiDALYEcon_Resultsstartcolumn),...
    num2str(DALY_12Econstartrow),':',...
    num2str(DALY_1endcolumn),...
    num2str(DALY_12Econendrow)];
DALY_2xlrange=[num2str(EpiDALYEcon_Resultsstartcolumn),...
    num2str(DALY_12Econstartrow),':',...
    num2str(DALY_2endcolumn),...
    num2str(DALY_12Econendrow)];
DALY_12strxlrange=[num2str(EpiDALYEcon_Resultsstartcolumn),...
    num2str(DALY_12Econstartrow),':',...
    num2str(EpiDALY_Resultsstrendcolumn),...
    num2str(DALY_12Econendrow)];
DALY_3xlrange=[num2str(EpiDALYEcon_Resultsstartcolumn),...
    num2str(DALY_3startrow),':',...
    num2str(DALY_3endcolumn),...
    num2str(DALY_3endrow)];
DALY_3strxlrange=[num2str(EpiDALYEcon_Resultsstartcolumn),...
    num2str(DALY_3startrow),':',...
    num2str(EpiDALY_Resultsstrendcolumn),...
    num2str(DALY_3endrow)];
Econ_Resultsxlrange=[num2str(EpiDALYEcon_Resultsstartcolumn),...
    num2str(DALY_12Econstartrow),':',...
    num2str(Econ_Resultsendcolumn),...
    num2str(DALY_12Econendrow)];
Econ_Resultsstrxlrange=[num2str(EpiDALYEcon_Resultsstartcolumn),...
    num2str(DALY_12Econstartrow),':',...
    num2str(EpiDALY_Resultsstrendcolumn),...
    num2str(DALY_12Econendrow)];

xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    Epi_Results,'Epi_Results',Epi_Resultsxlrange)
xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    EpiDALYEconstr,'Epi_Results',Epi_Resultsstrxlrange)
xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    DALY_1,'DALY_1',DALY_1xlrange)
xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    DALY_2,'DALY_2',DALY_2xlrange)
xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    EpiDALYEconstr,'DALY_1',DALY_12strxlrange)
xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    EpiDALYEconstr,'DALY_2',DALY_12strxlrange)
xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    DALY_3,'DALY_3',DALY_3xlrange)
xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    DALY_3str,'DALY_3',DALY_3strxlrange)
xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    Econ_Results,'Econ_Results',Econ_Resultsxlrange)
xlswrite('Targets_output_spreadsheet_post_London_v4_Melbourne_sent_5th_Jan.xlsx',...
    EpiDALYEconstr,'Econ_Results',Econ_Resultsstrxlrange)
