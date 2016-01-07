%SOUTH AFRICA MASTER SCRIPT
%__________________________________________________________________________
format compact

%SET UP POPULATION
global country
country='RSA'; %RSA (required in analysis function)
noaccess=.05; %Provided - proportion without access to care
lowquality=.8.*(1-noaccess); %Provided - proportion with access to
%low quality care (only slower screening)
highquality=1-lowquality-noaccess; %Provided - remainder have access to
%high quality care (faster screening)
%Provided - age stratification
hivpos=.165; %Provided - HIV prevalence in (adult) population
prophigh=.53; %Govender et al. - from Justin
propmid=.47.*.5;
proplow=.47.*.5;
accessprop=cat(3,... %See function for layers description
    highquality.*(1-hivpos),...
    0,...
    highquality.*hivpos.*prophigh,...
    highquality.*hivpos.*propmid,...
    highquality.*hivpos.*proplow,...
    lowquality.*(1-hivpos),...
    0,...
    lowquality.*hivpos.*prophigh,...
    lowquality.*hivpos.*propmid,...
    lowquality.*hivpos.*proplow,...
    noaccess.*(1-hivpos),...
    0,...
    noaccess.*hivpos.*prophigh,...
    noaccess.*hivpos.*propmid,...
    noaccess.*hivpos.*proplow);
%Age stratification (values provided)
adults=3.6824e7;
allage=5.2386e7;
ppv=.99; %Few false positives treated

%Initialise by compartment (around approximate eventual equilibrium values)
y0props=[.01,.26,.08,.1,.52,.01,.01,.01,zeros(1,54)];
y0=zeros(1,size(y0props,2),size(accessprop,3));
for a=1:size(accessprop,3) %Distribute by HIV status and care access
    y0(:,:,a)=y0props.*accessprop(1,1,a);
end
y0=1e7*cat(4,y0.*(allage-adults)/allage,y0.*adults/allage); %1e7 is an
%arbitrary value for population

%CONSOLIDATED RUN IN PERIOD
[trunin,yrunin,paramsrunin,arunin]=rsafunction(1770,y0,245,...
    zeros(1,6),0,zeros(1e5,13),1,accessprop,ppv); %245 year run in from
    %1770AD with interventions off

%Correct total population size
yrunin=allage*yrunin./...
    sum(sum(sum(yrunin(find(trunin>2012.5,1,'first'),1:40,:),2),3),4);
%%

thesistable=zeros(24,9);
ltbisuspectperdetect=4;

%INTERVENTION PHASE________________________________________________________

%Scenarios
scenarios=[.25,... %25% country expert
    .5,... %50% country expert
    .75,... %75% country expert
    1,... %100% country expert
    2]; %Advocate

%Interventions
ints=[0,0,0,0,0,0;... %Baseline
    1,0,0,0,0,0;... %1A - reduce no access only
    2,0,0,0,0,0;... %1B - redistribute high & low quality
    3,0,0,0,0,0;... %1 - reduce no access & redistribute high & low quality
    0,1,0,0,0,0;... %2A - improve initial default
    0,2,0,0,0,0;... %2B - improve DS outcomes
    0,3,0,0,0,0;... %2C - improve MDR outcomes
    0,4,0,0,0,0;... %2 - combination of 2A to 2C
    0,0,0,1,0,0;... %4 - ACF
    0,0,0,1,1,0;... %5 - ACF with LTBI screening
    0,0,0,0,0,1;... %6 - IPT for HIV positives
    3,4,0,1,1,1]; %7 - Combination of 1-5 (3 not applicable to RSA)

for c=1:size(ints,1) %Run 2:size(ints,1) to run all interventions
    a=0;
    while a<5 %a<5 to run all scenarios
        a=a+1; %Update scenario
        intervention=ints(c,:); %Set intervention
        scenario=scenarios(1,a);
        if sum(intervention,2)==0
            scenario=0;
            a=5;
        end
        
        if intervention(4)>0&&sum(intervention(1,1:3),2)==0
            scenario=2;
            a=5;
        end %Only ever run advocate scenario for intervention 4 or 5
        
        %Error if latent screening on without ACF
        if intervention(5)>0&&intervention(4)~=intervention(5)
            error('Latent screening on, but ACF off');
        end
        
        %NON-ACF VERSION
        if intervention(4)==0||scenario<2
            [time,yint,paramsint,aint]=rsafunction(trunin,yrunin,21,...
                intervention,scenario,paramsrunin,arunin,accessprop,ppv);
        end
        
        %ACF VERSION
        if intervention(4)==1&&scenario==2
            sensitivity=.92; %Sensitivity of screening tool
            propscreened=.5; %Proportion screend
            rounds=20; %Number of rounds
            interacfduration=.5; %Time between rounds (biannual)
            postacfduration=11; %Time after ACF ends
            round=1; %Round number
            tacf=trunin;
            yacf=yrunin;
            paramsacf=paramsrunin;
            aacf=arunin;
            %ACF runs:
            while round<=rounds
                [tacf,yacf,paramsacf,aacf]=rsafunction(tacf,yacf,...
                    interacfduration,intervention,scenario,paramsacf,aacf,...
                    accessprop,ppv); %Inter-ACF period
                round=round+1; %Update rounds
                roundtime=find(paramsacf(:,1)==tacf(end),1,'first'); %Find
                %the point in the parameters matrix that represents the
                %current time
                detections=sensitivity*propscreened*...
                    sigmoidscaleup(tacf(end),[2015,0],[2020,1])*...
                    yacf(end,[6:8,23:25],:,:); %Detections
                paramsacf(roundtime,11)=...
                    sum(sum(sum(detections,2),3),4); %Record detections
                yacf(end,[6:8,23:25],:,:)=...
                    yacf(end,[6:8,23:25],:,:)-detections; %Remove detected
                yacf(end,[15:17,35:37],1:5,:)=yacf(end,[15:17,35:37],1:5,:)+...
                    detections(1,:,1:5,:)+...
                    detections(1,:,6:10,:)+...
                    detections(1,:,11:15,:); %Add to treatment compartments
                if intervention(5)==1&&scenario==2
                    latentdetections=.8*.5*...
                        sigmoidscaleup(tacf(end),[2015,0],[2020,1])*...
                        yacf(end,4:5,:,:); %80% sensitivity of test,
                    %50% of population screened under advocate scenario
                    yacf(end,4:5,:,:)=yacf(end,4:5,:,:)-...
                        .82*latentdetections; %82% compliance
                    yacf(end,2,:,:)=yacf(end,2,:,:)+...
                        sum(.82*latentdetections,2); %Put into sb
                    paramsacf(roundtime,12)=...
                        sum(sum(sum(latentdetections,...
                        2),3),4)*ltbisuspectperdetect; %Screens
                    paramsacf(roundtime,13)=...
                        sum(sum(sum(latentdetections,...
                        2),3),4); %Record detections
                end
            end
            [time,yint,paramsint,aint]=rsafunction(tacf,yacf,...
                postacfduration,intervention,scenario,paramsacf,aacf,...
                accessprop,ppv); %Post-ACF period
        end
        
        paramsint=paramsint(2:end,:);
        paramsint=paramsint(1:find(paramsint==2036),:); %Cut off extraneous
        %zeros
        
        %OUTPUTS___________________________________________________________________
        [yearpoints,inc,inca,prev,mort,morta,latent,mdrpropinc,...
            mdrcommence,cases,...
            dscommenceyear,mdrcommenceyear,pop,popa,accessprops,smearpospropinc,...
            smearnegpropinc,extrapulpropinc,mdrpropretreat,mdrpropprev,...
            onmdrregimen,paramsint,hivprevtb,hivprev,arvcoverage,hivmorttb,...
            baselinecalibs,Epi_Results,EpiDALYEconstr,DALY_1,...
            DALY_2,DALY_3,DALY_3str,Econ_Results,thesistable]...
            =analysisfunction(time,yint,intervention,scenario,...
            paramsint,ppv,thesistable);

        %Get rid of NaNs for ACF with big jumps in values:
        getridnan
%         writetosheets
%         figures
%         
%         calibrationcheck
%                 accesscheck
        
    end
end

%Sound to indicate done
beep
pause(.4)
beep