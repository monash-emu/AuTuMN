%FUNCTION TO PRODUCE OUTPUTS
function[yearpoints,inc,inca,prev,mort,morta,latent,mdrpropinc,...
    mdrcommence,cases,...
    dscommenceyear,mdrcommenceyear,pop,popa,accessprops,smearpospropinc,...
    smearnegpropinc,extrapulpropinc,mdrpropretreat,mdrpropprev,...
    onmdrregimen,paramsint,hivprevtb,hivprev,arvcoverage,hivmorttb,...
    baselinecalibs,Epi_Results,EpiDALYEconstr,DALY_1,DALY_2,...
    DALY_3,DALY_3str,Econ_Results,thesistable]...
    =analysisfunction(time,yall,intervention,scenario,...
    paramsint,ppv,thesistable)

global country

%PREPARATION

%Define timepoints
yearpoints=zeros(2035,4);
yearpoints(:,1)=(1:2035)';
paramsyearpoints=zeros(2035,4);
paramsyearpoints(:,1)=(1:2035)';
for a=1989:2035
    yearpoints(a,2)=find(time<a,1,'last')+1;
    yearpoints(a,3)=find(time<a+.5,1,'last');
    yearpoints(a,4)=find(time<a+1,1,'last');
    paramsyearpoints(a,2)=find(paramsint(:,1)<a,1,'last')+1;
    paramsyearpoints(a,3)=find(paramsint(:,1)<a+.5,1,'last');
    paramsyearpoints(a,4)=find(paramsint(:,1)<a+1,1,'last');
end
%Columns of yearpoints matrix are:
%1.     Calendar year
%2.     Time at start of year (1st January)
%3.     Time at mid point of year (1st July)
%4.     Time at end of year (31st December)

%Split the y array into all population, child and adult matrices
y=sum(yall(:,:,:,:),4);
yc=yall(:,:,:,1);
ya=yall(:,:,:,2);

%Produce variables for the incremental change from cell to cell in t and y
tincr=zeros(size(time));
yincr=zeros(size(y));
yincrc=zeros(size(yc));
yincra=zeros(size(ya));
for a=2:size(time,1)
    tincr(a)=time(a)-time(a-1);
    yincr(a,:)=y(a,:)-y(a-1,:);
    yincrc(a,:)=yc(a,:)-yc(a-1,:);
    yincra(a,:)=ya(a,:)-ya(a-1,:);
end

%POPULATION
%Total populations
pop=sum(sum(y(:,1:40,:),2),3); %Total population column vector all
popa=sum(sum(ya(:,1:40,:),2),3); %Adult

%By access proportions
accessprops=zeros(size(time,1),1,3);
if strcmp(country,'RSA')==1
    accessprops(:,1)=sum(sum(y(:,1:40,1:5),2),3)./pop;
    accessprops(:,2)=sum(sum(y(:,1:40,6:10),2),3)./pop;
    accessprops(:,3)=sum(sum(y(:,1:40,11:15),2),3)./pop;
elseif strcmp(country,'India')==1||strcmp(country,'China')==1
    accessprops=sum(y(:,1:40,:),2)./cat(3,pop,pop,pop);
end

%INCIDENCE
%Total incidence
inc=1e5*sum(sum(yincr(:,43:46,:),2),3)./tincr./pop; %All age
inca=1e5*sum(sum(yincra(:,43:46,:),2),3)./tincr./popa; %Adult

%Proportionate incidence by organ involvement
extrapulpropinc=sum(yincr(:,47,:),3)./(sum(sum(yincr(:,47:49,:),3),2));
smearnegpropinc=sum(yincr(:,48,:),3)./(sum(sum(yincr(:,47:49,:),3),2));
smearpospropinc=sum(yincr(:,49,:),3)./(sum(sum(yincr(:,47:49,:),3),2));

%Smear positive incidence
mdrpropinc=sum(sum(yincr(:,[44,46],:),2),3)./...
    sum(sum(yincr(:,43:46,:),2),3); %Proportion incidence MDR

%Proportion incidence from recent infection (adults)
rapidpropa=sum(sum(yincra(:,43:44,:),3),2)./...
    sum(sum(yincra(:,43:46,:),3),2);

%LATENT PROPORTION
latent=sum(sum(y(:,[4,5,21,22],:),2),3)./pop;
latenta=sum(sum(ya(:,[4,5,21,22],:),2),3)./popa;

%ARI CALCULATION
ari=sum(sum(yincr(:,41:42,:),3),2)./...
    (tincr.*sum(sum(y(:,[1:3,5,22],:),3),2));

%TB-SPECIFIC MORTALITY
mort=1e5.*sum(sum(yincr(:,54:55,:),3),2)./(tincr.*pop);
morta=1e5.*sum(sum(yincra(:,54:55,:),3),2)./(tincr.*popa);

%NUMBERS IN COMPARTMENTS
%Prevalent cases
cases=sum(sum(y(:,[6:20,23:40],:),2),3);
casesc=sum(sum(yc(:,[6:20,23:40],:),2),3);
casesa=sum(sum(ya(:,[6:20,23:40],:),2),3);

%Noncases numbers
noncases=sum(sum(y(:,[1:5,21,22],:),2),3);
noncasesc=sum(sum(yc(:,[1:5,21,22],:),2),3);
noncasesa=sum(sum(ya(:,[1:5,21,22],:),2),3);

%Case and non-case numbers for South Africa
if strcmp(country,'RSA')==1
    cases=zeros(size(y,1),5);
    casesc=zeros(size(y,1),5);
    casesa=zeros(size(y,1),5);
    noncases=zeros(size(y,1),5);
    noncasesc=zeros(size(y,1),5);
    noncasesa=zeros(size(y,1),5);
    for a=1:5
        cases(:,a)=sum(sum(y(:,[6:20,23:40],[a,5+a,10+a]),2),3);
        casesc(:,a)=sum(sum(yc(:,[6:20,23:40],[a,5+a,10+a]),2),3);
        casesa(:,a)=sum(sum(ya(:,[6:20,23:40],[a,5+a,10+a]),2),3);
        noncases(:,a)=sum(sum(y(:,[1:5,21,22],[a,5+a,10+a]),2),3);
        noncasesc(:,a)=sum(sum(yc(:,[1:5,21,22],[a,5+a,10+a]),2),3);
        noncasesa(:,a)=sum(sum(ya(:,[1:5,21,22],[a,5+a,10+a]),2),3);
    end
end

%Pre and post diagnosis numbers
prediag=sum(sum(y(:,[6:8,23:25],:),2),3);
postdiag=sum(sum(y(:,[9:20,26:40],:),2),3);

%Under treatment
ondsregimen=sum(sum(y(:,[15:20,32:34],:),2),3);
onmdrregimen=sum(sum(y(:,35:40,:),2),3);
ondsregimenpriv=sum(y(:,[15:20,32:34],2),2);
onmdrregimenpriv=sum(y(:,35:40,2),2);

%CUMULATIVE COUNTS
%TB deaths
tbdeaths=sum(sum(y(:,54:55,:),3),2);

%Deaths
deathsc=sum(sum(yc(:,54:56,:),3),2);
deathsa=sum(sum(ya(:,54:56,:),3),2);

%Detections (including false positives)
detectcount=sum(sum(y(:,50:51,:),2),3);

%GeneXperts
xpertcount=sum(sum(y(:,62,:),2),3);

%Commencements
dscommence=sum(sum(y(:,52,:),2),3);
dscommencea=sum(sum(ya(:,52,:),2),3);
mdrcommence=sum(sum(y(:,53,:),2),3);
mdrcommencea=sum(sum(ya(:,53,:),2),3);

%DSTs
dstcount=sum(y(:,57,:),3);

%PREVALENCE
%Total prevalence
prev=1e5*sum(sum(y(:,[6:20,23:40],:),2),3)./pop;
preva=1e5*sum(sum(ya(:,[6:20,23:40],:),2),3)./popa;

%Proportion prevalence smear positive
smearpospropprev=prev.*sum(sum(y(:,[8:3:20,25:3:40],:),2),3)./...
    sum(sum(y(:,[6:20,23:40],:),2),3);

%Prevalence MDR
mdrpreva=1e5*sum(sum(y(:,23:40,:),2),3)./popa;

%Proportionate prevalence MDR
mdrpropprev=sum(sum(y(:,23:40,:),2),3)./sum(sum(y(:,[6:20,23:40],:),2),3);

%PREVIOUS TREATMENT CALCULATIONS
%Retreatment cases, MDR proportion
mdrpropretreat=sum(yincr(:,59,:),3)./sum(sum(yincr(:,58:59,:),2),3);

%DS patients, treatment experienced proportion
dstreatexp=sum(yincr(:,58,:),3)./sum(sum(yincr(:,[43,45,58],:),2),3);

%SOUTH AFRICA SPECIFIC CALCULATIONS
hivprevtb=0;
hivprev=0;
arvcoverage=0;
hivmorttb=0;
arvnumbers=0;
iptdetections=0;
iptsuspects=0;
if strcmp(country,'RSA')==1
    hivlayers=[2:5,7:10,12:15];
    arvlayers=[2,7,12];
    hivpop=sum(sum(y(:,1:40,hivlayers),2),3); %Total HIV positive pop
    hivprevtb=sum(sum(yincr(:,43:46,hivlayers),2),3)./...
        sum(sum(yincr(:,43:46,:),2),3); %HIV prevalence in new TB cases
    hivprev=hivpop./pop; %HIV prevalence
    arvcoverage=sum(sum(y(:,1:40,arvlayers),2),3)./hivpop; %ARV coverage
    hivmorttb=sum(sum(yincr(:,54:55,hivlayers),3),2)./...
        sum(sum(yincr(:,54:55,:),3),2); %Proportion of TB mortality due to HIV
    arvnumbers=sum(sum(y(:,1:40,arvlayers),2),3); %Numbers on ARVs
    iptdetections=sum(y(:,60,:),3); %IPT detections
    iptsuspects=sum(y(:,61,:),3); %IPT suspects
end

%OUTPUT MATRICES
%Baseline calibration matrix
baselinecalibs=[inca(yearpoints(2012,3)),...
    inc(yearpoints(2012,3)),...
    morta(yearpoints(2012,3)),...
    mort(yearpoints(2012,3)),...
    popa(yearpoints(2012,3))/1e3,...
    pop(yearpoints(2012,3))/1e3,...
    pop(yearpoints(2025,3))/1e3,...
    dscommence(yearpoints(2012,4))-dscommence(yearpoints(2012,2))+...
    mdrcommence(yearpoints(2012,4))-mdrcommence(yearpoints(2012,2)),...
    mdrcommence(yearpoints(2012,4))-mdrcommence(yearpoints(2012,2)),...
    prev(yearpoints(2000,3)),...
    prev(yearpoints(2010,3)),...
    prev(yearpoints(1990,3)),...
    mdrpropinc(yearpoints(2007,3))*1e2,...
    mdrpropretreat(yearpoints(2007,3))*1e2,...
    (mdrpropinc(yearpoints(2012,3))-mdrpropinc(yearpoints(2007,3)))*1e2,...
    (inc(yearpoints(2012,4))-inc(yearpoints(2012,2)))/inc(yearpoints(2012,2))*1e2,...
    mdrpropinc(yearpoints(2002,3))*1e2,...
    mdrpropretreat(yearpoints(2002,3))*1e2,...
    zeros(1,4)];
if strcmp(country,'RSA')==1
    baselinecalibs(1,end-3:end)=[hivprevtb(yearpoints(2012,3))*1e2,...
    hivprev(yearpoints(2012,3))*1e2,...
    arvcoverage(yearpoints(2012,2))*1e2,...
    hivmorttb(yearpoints(2012,3))*1e2];
end

%Epidemiological results matrix
Epi_Results=[max(intervention)*ones(36,1),scenario.*ones(36,1),...
    zeros(36,16)];
for a=1:36
    Epi_Results(a,4:end)=[yearpoints(a+1999,1),...
        dscommencea(yearpoints(a+1999,4))-...
        dscommencea(yearpoints(a+1999,2))+...
        mdrcommencea(yearpoints(a+1999,4))-...
        mdrcommencea(yearpoints(a+1999,2)),... %True positive initiations
        (dscommencea(yearpoints(a+1999,4))-...
        dscommencea(yearpoints(a+1999,2))+...
        mdrcommencea(yearpoints(a+1999,4))-...
        mdrcommencea(yearpoints(a+1999,2)))./...
        paramsint(paramsyearpoints(a+1999,3),6),... %TP+FP initiations (paramsint column 6 is ppv)
        preva(yearpoints(a+1999,3)),...
        morta(yearpoints(a+1999,3)),...
        inca(yearpoints(a+1999,3)),...
        popa(yearpoints(a+1999,3))/1e3,...
        mdrpreva(yearpoints(a+1999,3)),...
        1e2*mdrpropinc(yearpoints(a+1999,3)),... %Percentage MDR in new
        1e2*mdrpropretreat(yearpoints(a+1999,3)),... %Percentage MDR in retreatment
        1e2*latenta(yearpoints(a+1999,3)),...
        1e2*rapidpropa(yearpoints(a+1999,3)),...
        ari(yearpoints(a+1999,3))*1e2,...
        paramsint(paramsyearpoints(a+1999,3),5),... %Time to diagnosis
        (tbdeaths(yearpoints(a+1999,4))-...
        tbdeaths(yearpoints(a+1999,2)))/1e3]; %TB deaths
end

%Number of suspects screened per detection made
suspectperdetect=2;
acfsuspectperdetect=4;

%LTBI numbers on treatment following latent detection rounds
ltbistartpoints=find(paramsint(:,13)>0);
ltbitimestarted=paramsint(ltbistartpoints,1);
ltbinumberstarted=paramsint(ltbistartpoints,13);
ltbiendpoints=zeros(size(ltbistartpoints));
for a=1:size(ltbistartpoints,1)
    ltbiendpoints(a,1)=find(paramsint(:,1)>ltbitimestarted(a,1)+12/52,1,'first');
    paramsint(ltbistartpoints(a,1)+1:ltbiendpoints(a,1),13)=...
        paramsint(ltbistartpoints(a,1)+1:ltbiendpoints(a,1),13)+...
        ltbinumberstarted(a,1)*.82; %82% compliance
end

%DALY_1, DALY_2 and Economic results matrices
    DALY_1=[zeros(21,1),scenario*ones(21,1),zeros(21,1),...
        (2015:2035)',zeros(21,2)];
    if strcmp(country,'RSA')==1
        DALY_1=[zeros(21,1),scenario*ones(21,1),zeros(21,1),...
            (2015:2035)',zeros(21,10)];
    end
    DALY_2=DALY_1;
    Econ_Results=[zeros(21,1),scenario*ones(21,1),zeros(21,23)];
    dscommenceyear=[(2000:2035)',zeros(36,1)];
    mdrcommenceyear=[(2000:2035)',zeros(36,1)];
    for a=1:21
        if strcmp(country,'China')==1||strcmp(country,'India')==1
        DALY_1(a,5:6)=[cases(yearpoints(a+2014,3)),...
            noncases(yearpoints(a+2014,3))];
        elseif strcmp(country,'RSA')==1
            for b=1:5
                DALY_1(a,3+2*b)=cases(yearpoints(a+2014,3),b);
                DALY_1(a,4+2*b)=noncases(yearpoints(a+2014,3),b);
            end            
        end
        DALY_2(a,5:6)=[(deathsc(yearpoints(a+2014,4))-...
            deathsc(yearpoints(a+2014,2)))/1e3,...
            (deathsa(yearpoints(a+2014,4))-...
            deathsa(yearpoints(a+2014,2)))/1e3];
        Econ_Results(a,4:end)=...
            [yearpoints(a+2014,1),... %Years
            mean(prediag(yearpoints(a+2014,2):...
            yearpoints(a+2014,4))),... %Prediag
            mean(postdiag(yearpoints(a+2014,2):...
            yearpoints(a+2014,4))),... %Postdiag
            mean(prediag(yearpoints(a+2014,2):...
            yearpoints(a+2014,4)))+...
            mean(postdiag(yearpoints(a+2014,2):...
            yearpoints(a+2014,4))),... %Pre+postdiag
            (detectcount(yearpoints(a+2014,4))-...
            detectcount(yearpoints(a+2014,2)))*...
            suspectperdetect,... %Suspects screened
            detectcount(yearpoints(a+2014,4))-...
            detectcount(yearpoints(a+2014,2)),... %Detections made
            sum(paramsint(paramsyearpoints(a+2014,2):...
            paramsyearpoints(a+2014,4),11),1)/ppv,... %ACF detections
            sum(paramsint(paramsyearpoints(a+2014,2):...
            paramsyearpoints(a+2014,4),11),1)/ppv*...
            acfsuspectperdetect,... %ACF suspects
            (dstcount(yearpoints(a+2014,4))-...
            dstcount(yearpoints(a+2014,2)))*...
            suspectperdetect,... %DSTs includes pts without MDR and without TB
            dstreatexp(yearpoints(a+2014,3)).*...
            ondsregimen(yearpoints(a+2014,3)),... %Treatment naive
            (1-dstreatexp(yearpoints(a+2014,3))).*...
            ondsregimen(yearpoints(a+2014,3)),... %Treatment experienced
            ondsregimen(yearpoints(a+2014,3)),... %Any
            onmdrregimen(yearpoints(a+2014,3)),...
            1e2*ondsregimenpriv(yearpoints(a+2014,3))/...
            ondsregimen(yearpoints(a+2014,3)),...
            1e2*onmdrregimenpriv(yearpoints(a+2014,3))/...
            onmdrregimen(yearpoints(a+2014,3)),...
            sum(paramsint(paramsyearpoints(a+2014,2):...
            paramsyearpoints(a+2014,4),12),1)/1e3,... %LTBI screens per year
            mean(paramsint(paramsyearpoints(a+2014,2):...
            paramsyearpoints(a+2014,4),13),1)/1e3,... %LTBI patient years on treatment
            zeros(1,5)];
        if strcmp(country,'RSA')==1
            Econ_Results(a,end-4)=...
                iptsuspects(yearpoints(a+2014,4))-...
                iptsuspects(yearpoints(a+2014,2));
            Econ_Results(a,end-3)=...
                iptdetections(yearpoints(a+2014,4))-...
                iptdetections(yearpoints(a+2014,2));
            Econ_Results(a,end-2)=...
                paramsint(paramsyearpoints(a+2014,3),10)*...
                mean(arvnumbers(yearpoints(a+2014,2):...
                yearpoints(a+2014,4))); %IPT coverage x ARV coverage
            Econ_Results(a,end-1)=.05*mean(arvnumbers(yearpoints(a+2014,2):...
                yearpoints(a+2014,4)))+...
                max(0,arvnumbers(yearpoints(a+2014,4))-...
                arvnumbers(yearpoints(a+2014,2))); %ARV initiations
            Econ_Results(a,end)=mean(arvnumbers(yearpoints(a+2014,2):...
                yearpoints(a+2014,4))); %ARV treatment volume
        end
    end
    if strcmp(country,'RSA')
        DALY_1=[DALY_1(:,[1:6,9:14]),DALY_1(:,7:8)];
    end
    for a=1:36
        dscommenceyear(a,2)=dscommence(yearpoints(a+1999,4))-...
            dscommence(yearpoints(a+1999,2));
        mdrcommenceyear(a,2)=mdrcommence(yearpoints(a+1999,4))-...
            mdrcommence(yearpoints(a+1999,2));
    end
    
    %DALY_3 matrix
    DALY_3=[zeros(2,1),scenario*ones(2,1),zeros(2,4)];
DALY_3(:,4:6)=[.14,casesc(yearpoints(2035,4)),...
    noncasesc(yearpoints(2035,4));...
    15,casesa(yearpoints(2035,4)),...
    noncasesa(yearpoints(2035,4))];
if strcmp(country,'RSA')==1
    DALY_3=[zeros(2,1),scenario*ones(2,1),zeros(2,12)];
    for a=1:5
        DALY_3(1,3+2*a)=casesc(yearpoints(2035,4),a);
        DALY_3(1,4+2*a)=noncasesc(yearpoints(2035,4),a);
        DALY_3(2,3+2*a)=casesa(yearpoints(2035,4),a);
        DALY_3(2,4+2*a)=noncasesa(yearpoints(2035,4),a);
    end
    DALY_3=[DALY_3(:,[1:6,9:14]),DALY_3(:,7:8)];
end

%Truncate if necessary
if max(intervention)>0
    Epi_Results=Epi_Results(17:end,:);
    DALY_1=DALY_1(2:end,:);
    DALY_2=DALY_2(2:end,:);
    Econ_Results=Econ_Results(2:end,:);
end

%Creating cell arrays of strings to populate the string sections of the
%spreadsheets:_____________________________________________________________
EpiDALYEconstr=cell(size(Epi_Results,1),3);
for a=1:size(Epi_Results,1)
    if sum(intervention,2)==0
        EpiDALYEconstr{a,1}='0';
    end    
    if intervention(1)==1
        EpiDALYEconstr{a,1}='1a';
    elseif intervention(1)==2
        EpiDALYEconstr{a,1}='1b';
    elseif intervention(1)==3
        EpiDALYEconstr{a,1}='1';
    end
    if intervention(2)==1
        EpiDALYEconstr{a,1}='2a';
    elseif intervention(2)==2
        EpiDALYEconstr{a,1}='2b';
    elseif intervention(2)==3
        EpiDALYEconstr{a,1}='2c';
    elseif intervention(2)==4
        EpiDALYEconstr{a,1}='2';
    end
    if intervention(3)==1
        EpiDALYEconstr{a,1}='3';
    end
    if intervention(4)==1
        EpiDALYEconstr{a,1}='4';
    end
    if intervention(5)==1
        EpiDALYEconstr{a,1}='5';
    end
    if intervention(6)==1
        EpiDALYEconstr{a,1}='6';
    end
    if intervention(1)>0&&intervention(2)>0
        EpiDALYEconstr{a,1}='7';
    end
    EpiDALYEconstr{a,2}=['cty_',num2str(1e2*Epi_Results(a,2))];
    EpiDALYEconstr{a,3}=country;
    if scenario==2
        EpiDALYEconstr{a,2}='adv';
    end
    if scenario==0
        EpiDALYEconstr{a,2}='base_case';
    end
end

DALY_3str=cell(size(DALY_3,1),4);
for a=1:size(DALY_3,1)
    if sum(intervention,2)==0
        DALY_3str{a,1}='0';
    end  
    if intervention(1)==1
        DALY_3str{a,1}='1a';
    elseif intervention(1)==2
        DALY_3str{a,1}='1b';
    elseif intervention(1)==3
        DALY_3str{a,1}='1';
    end
    if intervention(2)==1
        DALY_3str{a,1}='2a';
    elseif intervention(2)==2
        DALY_3str{a,1}='2b';
    elseif intervention(2)==3
        DALY_3str{a,1}='2c';
    elseif intervention(2)==4
        DALY_3str{a,1}='2';
    end
    if intervention(3)==1
        DALY_3str{a,1}='3';
    end
    if intervention(4)==1
        DALY_3str{a,1}='4';
    end
    if intervention(5)==1
        DALY_3str{a,1}='5';
    end
    if intervention(6)==1
        DALY_3str{a,1}='6';
    end
    if intervention(1)>0&&intervention(2)>0
        DALY_3str{a,1}='7';
    end
    DALY_3str{a,2}=['cty_',num2str(1e2*scenario)];
    DALY_3str{a,3}=country;
    if scenario==2
        DALY_3str{a,2}='adv';
    end
    if scenario==0
        DALY_3str{a,2}='base_case';
    end
end
DALY_3str{1,4}='0_14';
DALY_3str{2,4}='15+';

if strcmp(country,'India')==1
    tablecolumn=1;
elseif strcmp(country,'China')==1
    tablecolumn=2;
elseif strcmp(country,'RSA')==1
    tablecolumn=3;
end

if scenario>=1||scenario==0
thesistable(:,(tablecolumn-1)*3+scenario+1)=...
    [round(inc(yearpoints(2025,3))*10)/10;...
    round(inc(yearpoints(2035,3))*10)/10;...
    round(prev(yearpoints(2025,3))*10)/10;...
    round(prev(yearpoints(2035,3))*10)/10;...
    round(mort(yearpoints(2025,3))*10)/10;...
    round(mort(yearpoints(2035,3))*10)/10;...
    round(mdrpropinc(yearpoints(2025,3))*1000)/10;...
    round(mdrpropinc(yearpoints(2035,3))*1000)/10;...
    round(mdrpropretreat(yearpoints(2025,3))*1000)/10;...
    round(mdrpropretreat(yearpoints(2035,3))*1000)/10;...
    round(latent(yearpoints(2025,3))*1000)/10;...
    round(latent(yearpoints(2035,3))*1000)/10;...
    round(ari(yearpoints(2025,3))*1e4)/1e2;...
    round(ari(yearpoints(2035,3))*1e4)/1e2;...
    round((detectcount(yearpoints(2025,4))-...
    detectcount(yearpoints(2016,2)))/1e5)*10;...
    round((detectcount(yearpoints(2035,4))-...
    detectcount(yearpoints(2026,2)))/1e5)*10;...
    round((mean(ondsregimen(yearpoints(2016,2):yearpoints(2025,4)))*...
    10)/1e3);...
    round((mean(ondsregimen(yearpoints(2026,2):yearpoints(2035,4)))*...
    10)/1e3);...
    round((mean(onmdrregimen(yearpoints(2016,2):yearpoints(2025,4)))*...
    10)/1e3);...
    round((mean(onmdrregimen(yearpoints(2026,2):yearpoints(2035,4)))*...
    10)/1e3);...
    0;0;0;0];
    if intervention(3)>0
        thesistable(21,(tablecolumn-1)*3+scenario+1)=...
            (xpertcount(yearpoints(2025,4))-...
            xpertcount(yearpoints(2016,2)))/1e6;
        thesistable(22,(tablecolumn-1)*3+scenario+1)=...
            (xpertcount(yearpoints(2035,4))-...
            xpertcount(yearpoints(2026,2)))/1e6;
    end
    if intervention(4)>0
        thesistable(21,(tablecolumn-1)*3+scenario+1)=...
            sum(paramsint(paramsyearpoints(2016,2):...
            paramsyearpoints(2025,4),11),1)/ppv;
        thesistable(22,(tablecolumn-1)*3+scenario+1)=...
            sum(paramsint(paramsyearpoints(2026,2):...
            paramsyearpoints(2035,4),11),1)/ppv; %ACF detections
        thesistable(23,(tablecolumn-1)*3+scenario+1)=...
            sum(paramsint(paramsyearpoints(2016,2):...
            paramsyearpoints(2025,4),11),1)/ppv*...
            acfsuspectperdetect/1e6;
        thesistable(24,(tablecolumn-1)*3+scenario+1)=...
            sum(paramsint(paramsyearpoints(2026,2):...
            paramsyearpoints(2035,4),11),1)/ppv*...
            acfsuspectperdetect/1e6; %ACF suspects        
    end
    if intervention(5)>0
        thesistable(21,(tablecolumn-1)*3+scenario+1)=...
            sum(paramsint(paramsyearpoints(2016,2):...
            paramsyearpoints(2025,4),12),1)/1e6;
        thesistable(22,(tablecolumn-1)*3+scenario+1)=...
            sum(paramsint(paramsyearpoints(2026,2):...
            paramsyearpoints(2035,4),12),1)/1e6; %LTBI screens per year
        thesistable(23,(tablecolumn-1)*3+scenario+1)=...
            mean(paramsint(paramsyearpoints(2016,2):...
            paramsyearpoints(2025,4),13),1)/1e6*10;
        thesistable(24,(tablecolumn-1)*3+scenario+1)=...
            mean(paramsint(paramsyearpoints(2026,2):...
            paramsyearpoints(2035,4),13),1)/1e6*10; %LTBI patient years on treatment       
    end
    if intervention(6)>0
        thesistable(21,(tablecolumn-1)*3+scenario+1)=...
            mean(paramsint(paramsyearpoints(2016,2):...
            paramsyearpoints(2025,4),10))*...
            mean(arvnumbers(yearpoints(2016,2):...
            yearpoints(2025,4)))/1e6; %IPT coverage x ARV coverage        
        thesistable(22,(tablecolumn-1)*3+scenario+1)=...
            mean(paramsint(paramsyearpoints(2026,2):...
            paramsyearpoints(2035,4),10))*...
            mean(arvnumbers(yearpoints(2026,2):...
            yearpoints(2035,4)))/1e6; %IPT coverage x ARV coverage
    end
end

end