%STUDY MASTER FUNCTION
function[t,y,paramsout,a]=rsafunction(told,yold,duration,...
    intervention,scenario,paramsin,ain,accessprop,ppv)

if sum(intervention,2)==0&&scenario>0
    error('Interventions off, but scenario greater than zero');
end
if sum(intervention,2)>0&&scenario==0
    error('Interventions on, but no scenario specified');
end

paramsout=paramsin;
a=ain;

%Layers are:
%1. High quality TB care, HIV negative
%2. High quality TB care, HIV positive, ARVs
%3. High quality TB care, HIV positive, no ARVs, CD4>350
%4. High quality TB care, HIV positive, no ARVs, CD4 200-350
%5. High quality TB care, HIV positive, no ARVs, CD4<200
%6. Low quality TB care, HIV negative
%7. Low quality TB care, HIV positive, ARVs
%8. Low quality TB care, HIV positive, CD4>350
%9. Low quality TB care, HIV positive, CD4 200-350
%10. Low quality TB care, HIV positive, CD4<200
%11. No TB care, HIV negative
%12. No TB care, HIV positive, ARVs
%13. No TB care, HIV positive, CD4>350
%14. No TB care, HIV positive, CD4 200-350
%15. No TB care, HIV positive, CD4<200
hivlayers=[2:5,7:10,12:15];
lowcd4layers=[5,10,15];
arvlayers=[2,7,12];

%Set initial conditions equal to end value of last phase
y0=yold(end,:,:,:);
comps=size(y0,2); %Set scalar for number of compartments and outputs
access=size(y0,3); %Set scalar for number of types of care access
ages=size(y0,4); %Set scalar for number of age groups

%Convert y0 into Y0 - from a horizontal matrix in layers, to a single
%column vector for the flows to work on.
Y0=reshape(y0,comps*access*ages,1);

%TRANSMISSION PARAMETER
beta=25.4;

%DISEASE DURATION PARAMETERS
deathmodification=1.1;
muez=.2/3*deathmodification*ones(1,1,access,ages);
mup=.7/3*deathmodification*ones(1,1,access,ages);
gammaez=(1-.2*deathmodification)/3*ones(1,1,access,ages);
gammap=(1-.7*deathmodification)/3*ones(1,1,access,ages);
muez(hivlayers)=muez(hivlayers)*1.5;
mup(hivlayers)=mup(hivlayers)*1.5;
muez(lowcd4layers)=muez(lowcd4layers)*2.4;
mup(lowcd4layers)=mup(lowcd4layers)*2.4;

%Decrease effective sojourn time less if in dx compartments
xincrecovery=1.5;
muxez=muez*xincrecovery;
muxp=mup*xincrecovery;
gammaxez=gammaez*xincrecovery;
gammaxp=gammap*xincrecovery;

%DEFINE FIXED PARAMETERS
%Fixed epidemiological parameters
iota=.99; %99% vaccinated, UNICEF data for South Africa
mu=cat(4,ones(1,1,access,1).*.014,ones(1,1,access,1).*.022);
%Higher death rate for adults
chi=.49; %BCG efficacy or partial immunity
rfmdr=.6; %Relative fitness of MDR-TB
omicron=1; %12/12 duration of initial default
ztrans=.24; %Relative transmissibility of smear-negative TB

%ODE FUNCTION______________________________________________________________
[t,y]=ode45(@(t,y)flowfunction(t,y),[told(end),told(end)+duration],Y0);

%__________________________________________________________________________
%FLOWS SUBFUNCTION
    function[flows]=flowfunction(t,y)
        
        %PREPARE A VECTOR FOR FLOWS
        Y=reshape(y,comps,1,access,ages); %Y0 has been fed
        %into the ODE function as a column vector. This now produces a
        %working matrix for the ODE function with rows compartments, layers
        %access groups and fourth dimension age stratification.
        
        %DEFINE THE COMPARTMENTS
        sa=Y(1,1,:,:); sb=Y(2,1,:,:); sc=Y(3,1,:,:);
        las=Y(4,1,:,:); lbs=Y(5,1,:,:);
        ise=Y(6,1,:,:); isz=Y(7,1,:,:); isp=Y(8,1,:,:);
        dwse=Y(9,1,:,:); dwsz=Y(10,1,:,:); dwsp=Y(11,1,:,:);
        ddse=Y(12,1,:,:); ddsz=Y(13,1,:,:); ddsp=Y(14,1,:,:);
        tise=Y(15,1,:,:); tisz=Y(16,1,:,:); tisp=Y(17,1,:,:);
        tnse=Y(18,1,:,:); tnsz=Y(19,1,:,:); tnsp=Y(20,1,:,:);
        lam=Y(21,1,:,:); lbm=Y(22,1,:,:);
        ime=Y(23,1,:,:); imz=Y(24,1,:,:); imp=Y(25,1,:,:);
        dwme=Y(26,1,:,:); dwmz=Y(27,1,:,:); dwmp=Y(28,1,:,:);
        ddme=Y(29,1,:,:); ddmz=Y(30,1,:,:); ddmp=Y(31,1,:,:);
        dxme=Y(32,1,:,:); dxmz=Y(33,1,:,:); dxmp=Y(34,1,:,:);
        time=Y(35,1,:,:); timz=Y(36,1,:,:); timp=Y(37,1,:,:);
        tnme=Y(38,1,:,:); tnmz=Y(39,1,:,:); tnmp=Y(40,1,:,:);
        %Each compartment is now a single column, with layers by care
        %access and fourth dimension by age group.
        
        %DEFINE TRANSMISSION PARAMETERS
        totalpop=sum(sum(sum(Y(1:40,:,:,:),1),3),4); %Total population
        %across all care access types and age groups
        xtrans=.5; %Relative transmissibility MDR when on DS regimen
        lambdas=beta.*(sum(sum(isp+tisp+dwsp+ddsp,3),4)...
            +ztrans.*(sum(sum(isz+tisz+dwsz+ddsz,3),4)))./totalpop;
        %DS-TB, no immunity
        lambdads=chi.*lambdas; %DS-TB, partial immunity
        lambdam=rfmdr.*beta.*(sum(sum(imp+timp+dwmp+ddmp+xtrans*dxmp,3),4)...
            +ztrans.*(sum(sum(imz+timz+dwmz+ddmz+xtrans*dxmz,3),4)))./totalpop;
        %MDR-TB, no immunity
        lambdadm=chi.*lambdam; %MDR-TB, partial immunity        
       
        %INITIAL DEFAULT
        upsilons=.17*ones(1,1,access,ages); %DS TB initial default (provided value)
        upsilonm=.3*ones(1,1,access,ages); %MDR TB initial default (also provided)
        
        %INTERVENTION 2A
        if intervention(2)==1||intervention(2)==4
            if scenario>0&&scenario<=1
                upsilons(1,1,1:5,:)=sigmoidscaleup(t,[2016,.17],...
                    [2022,.17-.12*scenario])*ones(1,1,5,2);
                upsilonm(1,1,1:5,:)=sigmoidscaleup(t,[2016,.3],...
                    [2022,.15-.15*scenario])*ones(1,1,5,2);
            elseif scenario==2
                upsilons(1,1,1:5,:)=sigmoidscaleup(t,[2016,.17],...
                    [2021,0])*ones(1,1,5,2);
                upsilonm(1,1,1:5,:)=sigmoidscaleup(t,[2016,.3],...
                    [2021,0])*ones(1,1,5,2);
            end
        end
        
        %DEFINE DS PROGRAMMATIC PARAMETERS
        
        %Miscellaneous
        sigmas=52; %Time to start treatment after diagnosis
        
        %Treatment outcomes
        sdefault=.2*ones(1,1,access,ages); %Total defaults DS TB
        sdeath=.04*ones(1,1,access,ages);
        tsd=6/12; %Treatment duration for DS-TB
        tisd=2/52; %Infectious duration for DS-TB
        tisprop=.15; %Proportion infectious at end of infectious duration
        
        %INVERVENTION 2B MODIFICATIONS
        if intervention(2)==2||intervention(2)==4
            if scenario>0&&scenario<=1 %Country expert
                sdefault(1,1,1:5,:)=sigmoidscaleup(t,[2016,.2],...
                    [2022,.2-.075*scenario])*ones(1,1,5,2);
                sdeath(1,1,1:5,:)=sigmoidscaleup(t,[2016,.04],...
                    [2022,.04-.015*scenario])*ones(1,1,5,2);
            elseif scenario==2 %Advocate
                sdefault(1,1,1:5,:)=sigmoidscaleup(t,[2016,.2],...
                    [2021,.125])*ones(1,1,5,2);
                sdeath(1,1,1:5,:)=sigmoidscaleup(t,[2016,.04],...
                    [2021,.025])*ones(1,1,5,2);
            end
        end
        
        %CONVERT TREATMENT OUTCOMES TO FLOWS
        
        %Divide up the defaults between ti and tn compartments
        defaulttis=sdefault.*tisd./tsd; %Defaults in tis
        defaulttns=sdefault.*(tsd-tisd)./tsd; %Defaults in tns
        deathtis=sdeath.*tisd./tsd; %Deaths in tis
        deathtns=sdeath.*(tsd-tisd)./tsd; %Deaths in tns
        successtis=1-defaulttis-deathtis; %Cure/completion in tis
        successtns=1-defaulttns-deathtns; %Cure/completion in tns
        
        %tise/tisz/tisp flows
        taus=ones(1,1,access,ages).*(-log(tisprop)/tisd); %Successfully cleared
        %infection flow
        taus(1,1,11:access,:)=zeros(1,1,5,ages); %Set treatment success to zero in the no access layer
        omegatis=taus.*defaulttis./successtis; %Flow defaulting from treatment tis
        omegatis(1,1,11:access,:)=zeros(1,1,5,ages); %No default on treatment either
        mutis=taus.*deathtis./successtis; %Flow dying on treatment tis
        mutis(1,1,11:access,:)=zeros(1,1,5,ages); %Nor any death on treatment
        
        %tnse/tnsz/tnsp flows
        xis=-log(.5).*ones(1,1,access,2)./(tsd-tisd); %Flow of treatment success
        xis(1,1,11:access,:)=zeros(1,1,5,2); %Set treatment succcess to zero in the no access layer
        omegatns=xis.*defaulttns./successtns; %Flow defaulting from treatment tns
        omegatns(1,1,11:access,:)=zeros(1,1,5,2); %No default on treatment either
        mutns=xis.*deathtns./successtns; %Flow dying on treatment tns
        mutns(1,1,11:access,:)=zeros(1,1,5,2); %Nor any death on treatment
        
        %DEFINE MDR PROGRAMMATIC PARAMETERS
        
        %Miscellaneous
        sigmam=sigmas; %Time to start treatment after diagnosis
        
        %Treatment outcomes
        mdefault=.42*ones(1,1,access,ages);
        mdeath=.08*ones(1,1,access,ages);
        tmd=2; %Treatment duration for MDR TB
        timd=3/12; %Infectious duration for MDR TB
        timprop=1-0.85/2; %Kurbatova ref as for MSF Myanmar
        
        %INVERVENTION 2C MODIFICATIONS
        if intervention(2)==3||intervention(2)==4
            if scenario>0&&scenario<=1 %Country expert
                mdefault(1,1,1:5,:)=sigmoidscaleup(t,[2016,.42],...
                    [2026,.42-.14*scenario])*ones(1,1,5,2);
                mdeath(1,1,1:5,:)=sigmoidscaleup(t,[2016,.08],...
                    [2026,.08-.03*scenario])*ones(1,1,5,2);
            elseif scenario==2 %Advocate
                mdefault(1,1,1:5,:)=sigmoidscaleup(t,[2016,.42],...
                    [2021,.21])*ones(1,1,5,2);
                mdeath(1,1,1:5,:)=sigmoidscaleup(t,[2016,.08],...
                    [2021,.04])*ones(1,1,5,2);
            end
        end
        
        %Divide up the defaults between ti and tn compartments
        defaulttim=mdefault.*timd./tmd; %Defaults in tim
        defaulttnm=mdefault.*(tmd-timd)./tmd; %Defaults in tnm
        deathtim=mdeath.*timd./tmd; %Deaths in tim
        deathtnm=mdeath.*(tmd-timd)./tmd; %Deaths in tnm
        successtim=1-defaulttim-deathtim; %Cure/completion in tim
        successtnm=1-defaulttnm-deathtnm; %Cure/completion in tnm
        
        %time/timz/timp flows
        taum=ones(1,1,access,2).*(-log(timprop)/timd); %Successfully cleared infection flow
        omegatim=taum.*defaulttim./successtim; %Flow defaulting from treatment tim
        mutim=taum.*deathtim./successtim; %Flow dying on treatment tim
        
        %tnme/tnmz/tnmp flows
        xim=-log(.5).*ones(1,1,access,ages)./(tmd-timd); %Flow of treatment success
        omegatnm=xim.*defaulttnm./successtnm; %Flow defaulting from treatment tnm
        mutnm=xim.*deathtnm./successtnm; %Flow dying on treatment tnm
        
        %dxme/dxmp flow
        taui=-log(.5)./tsd; %Rate of returning to the im compartment after
        %completing ineffective treatment
        
        %MDR-TB INTRODUCTION
        eta=0;
        if t>1991.5
            eta=.03; %Amplification to MDR - now calibrated to dynamics
        end
        
        %SCALE UP CARE ACCESS DURING RUN IN
        
        %For general care
        deltasp=cat(4,cat(3,ones(1,1,5).*3,ones(1,1,5).*1.5,zeros(1,1,5)),...
            cat(3,ones(1,1,5).*3,ones(1,1,5)*1.5,zeros(1,1,5)));
        
        %INTERVENTION 6 CASE DETECTION
        deltaspinc=zeros(size(deltasp));
        if intervention(6)==1
            if scenario<=2
                if t<=2022
                    deltaspinc(1,1,arvlayers,:)=sigmoidscaleup(t,...
                        [2015,0],[2022,1.1*scenario])*...
                        deltasp(1,1,arvlayers,:);
                elseif t>2022
                    deltaspinc(1,1,arvlayers,:)=sigmoidscaleup(t,...
                        [2022,1.1*scenario],[2025,.1*scenario])*...
                        deltasp(1,1,arvlayers,:);
                end
            end
            if scenario==2
                if t<=2022
                    deltaspinc(1,1,arvlayers,:)=...
                        sigmoidscaleup(t,[2015,0],[2022,1.3])*...
                        deltasp(1,1,arvlayers,:);
                elseif t>2022
                    deltaspinc(1,1,arvlayers,:)=...
                        sigmoidscaleup(t,[2022,1.3],[2025,.1])*...
                        deltasp(1,1,arvlayers,:);
                end
            end
        end %Code laboriously calibrated once - and don't plan to do again
        %- approximately doubled case detection among ARV population during
        %the scale up, then reverts to an extra 10%
        deltasp=deltasp+deltaspinc;
        deltaseinc=deltaspinc*.5;
        deltase=deltasp*.5;
        clinicallymissedsmearneg=sigmoidscaleup(t,[2000,.1],[2011,.09]);
        if t>2011&&t<2012
            clinicallymissedsmearneg=sigmoidscaleup(t,...
                [2011,.09],[2012,.06])*ones(1,1,access,ages);
        elseif t>=2012
            clinicallymissedsmearneg=sigmoidscaleup(t,...
                [2012,.06],[2016,.02])*ones(1,1,access,ages);
        end
        deltasz=deltasp.*(1-clinicallymissedsmearneg);
        deltaszinc=deltaspinc.*(1-clinicallymissedsmearneg);
        
        %For MDR detection
        mdrdetectp=sigmoidscaleup(t,[2000,0],[2011,.11]);
        if t>2011&&t<2012
            mdrdetectp=sigmoidscaleup(t,[2011,.11],[2012,.25])*...
                ones(1,1,access,ages);
        elseif t>=2012
            mdrdetectp=sigmoidscaleup(t,[2012,.25],[2016,.78])*...
                ones(1,1,access,ages);
        end
        %Say 8% empirical diagnosis (similar to India) and 3% DST
        %Reference for imperfect role out of GeneXpert:
        %http://www.nspreview.org/2013/09/04/genexpert-an-imperfect-rollout/

        mdrdetectz=mdrdetectp*.69;
        mdrdetecte=mdrdetectz; %Reference for GeneXpert having similar
        %sensitivity for extrapulmonary as for smear-negative TB:
        %http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3122824/
        labdst=sigmoidscaleup(t,[2006,0],[2011,.03])*...
            ones(1,1,access,ages);
        
        %Ratios
        suspecttocaseratio=2; %Screens per diagnosis (either TP or FP)
        suspecttocaseratioipt=1.5;
        
        %Baseline proportion of MDR cases that are diagnosed as such,
        %rather than being started on DS TB treatment.
        deltame=mdrdetecte.*deltase;
        deltamz=mdrdetectz.*deltasz;
        deltamp=mdrdetectp.*deltasp;
        deltaxme=(1-mdrdetecte).*deltase;
        deltaxmz=(1-mdrdetectz).*deltasz;
        deltaxmp=(1-mdrdetectp).*deltasp;
        
        %SMEAR/PULMONARY STATUS
        smearpos=.416;
        childsmearpos=.04; %Allowing less smear positive in children
        smearneg=.370;
        extrapul=1-smearpos-smearneg;
        eps=.07./2;
        kappaadult=.93./2;
        kappachild=(1-(.07.*(extrapul+smearneg+childsmearpos)))./2;
        epsp=cat(4,childsmearpos.*eps.*ones(1,1,access,1),...
            smearpos.*eps.*ones(1,1,access,1));
        epsz=cat(4,smearneg.*(ones(1,1,access,ages).*eps));
        epse=cat(4,extrapul.*(ones(1,1,access,ages).*eps));
        kappa=cat(4,ones(1,1,access,1).*kappachild,ones(1,1,access,1).*...
            kappaadult);
        nu=.075/20;
        nup=cat(4,childsmearpos.*eps.*ones(1,1,access,1),...
            smearpos.*nu.*ones(1,1,access,1));
        nuz=cat(4,smearneg.*(ones(1,1,access,ages).*nu));
        nue=cat(4,extrapul.*(ones(1,1,access,ages).*nu));
        
        iptcoverage=.05;
        
        %INTERVENTION 6
        if intervention(6)>0
            if scenario<2
                iptcoverage=sigmoidscaleup(t,[2016,.05],...
                    [2022,.8*scenario]);
            elseif scenario==2
                iptcoverage=sigmoidscaleup(t,[2016,.05],...
                    [2021,1]);
            end
        end
        
        arvmod=.9175-.35*iptcoverage;
        
        %HIV positive modifications
        hivprogression=sigmoidscaleup(t,[1970,0],[1990,15]);
        hivprogression=cat(3,1,1+hivprogression*arvmod,1+hivprogression,...
            1+hivprogression*2.5,1+hivprogression*10);
        hivprogression=cat(3,hivprogression,hivprogression,hivprogression);
        hivprogression=cat(4,hivprogression,hivprogression);
        
        epsp=epsp.*hivprogression;
        epsz=epsz.*hivprogression;
        epse=epse.*hivprogression;
        nup=nup.*hivprogression;
        nuz=nuz.*hivprogression;
        nue=nue.*hivprogression;
        
        xpertprop=zeros(1,1,access,ages);
        [flowsmat]=compartmentflows(sa,sb,sc,las,lbs,ise,isz,isp,...
            dwse,dwsz,dwsp,ddse,ddsz,ddsp,tise,tisz,tisp,tnse,tnsz,tnsp,...
            lam,lbm,ime,imz,imp,dwme,dwmz,dwmp,dxme,dxmz,dxmp,time,timz,timp,...
            ddme,ddmz,ddmp,tnme,tnmz,tnmp,lambdas,lambdam,lambdads,lambdadm,...
            xis,xim,kappa,gammaez,gammaxez,gammaxp,gammap,nue,nuz,nup,...
            epse,epsz,epsp,eta,omegatis,omegatns,omegatim,omegatnm,...
            omicron,deltase,deltasz,deltasp,deltame,deltamz,deltamp,...
            deltaxme,deltaxmz,deltaxmp,deltaseinc,deltaszinc,deltaspinc,...
            taui,taus,taum,upsilons,upsilonm,sigmas,sigmam,...
            mu,muez,muxez,mup,muxp,mutis,mutns,mutim,mutnm,...
            ppv,suspecttocaseratio,suspecttocaseratioipt,labdst,xpertprop);
        
        %Health care access proportions
        noaccess=.05;
        highquality=.2*(1-noaccess);
        lowquality=1-noaccess-highquality;
        hivpos=.165; %Provided - HIV prevalence in (adult) population
        prophigh=.53; %Govender et al. - from Justin
        propmid=.47*.5;
        proplow=.47*.5;
        
        %BIRTH TIDYING FOR INTERVENTION 1
        if intervention(1)==1 %Intervention 1A
            if scenario<2
                highquality=sigmoidscaleup(t,...
                    [2016,.2*(1-noaccess)],...
                    [2022,.2*(1-noaccess*(1-scenario))]);
                lowquality=sigmoidscaleup(t,...
                    [2016,.8*(1-noaccess)],...
                    [2022,.8*(1-noaccess*(1-scenario))]);
                noaccess=sigmoidscaleup(t,...
                    [2016,noaccess],...
                    [2022,noaccess*(1-scenario)]);
            elseif scenario==2
                highquality=sigmoidscaleup(t,...
                    [2016,.2*(1-noaccess)],...
                    [2021,.2]);
                lowquality=sigmoidscaleup(t,...
                    [2016,.8*(1-noaccess)],...
                    [2021,.8]);
                noaccess=sigmoidscaleup(t,...
                    [2016,noaccess],...
                    [2021,0]);
            end
        end
        if intervention(1)==2 %Intervention 1B
            if scenario<2
                highquality=sigmoidscaleup(t,...
                    [2016,.2*(1-noaccess)],...
                    [2022,(.2+.8*scenario)*... %Proportion high quality
                    (1-noaccess)]); %Proportion with access
                lowquality=sigmoidscaleup(t,...
                    [2016,.8*(1-noaccess)],...
                    [2022,(.8-.8*scenario)*... %Proportion low quality
                    (1-noaccess)]); %Proportion with access
            elseif scenario==2
                highquality=sigmoidscaleup(t,...
                    [2016,.2*(1-noaccess)],...
                    [2021,(1-noaccess)]);
                lowquality=sigmoidscaleup(t,...
                    [2016,.8*(1-noaccess)],...
                    [2021,0]);
            end
        end
        if intervention(1)==3 %Intervention 1
            if scenario<2
                highquality=sigmoidscaleup(t,...
                    [2016,.2*(1-noaccess)],...
                    [2022,(.2+.8*scenario)*... %Proportion high quality
                    (1-noaccess*(1-scenario))]); %Proportion with access
                lowquality=sigmoidscaleup(t,...
                    [2016,.8*(1-noaccess)],...
                    [2022,(.8-.8*scenario)*... %Proportion low quality
                    (1-noaccess*(1-scenario))]); %Proportion with access
                noaccess=sigmoidscaleup(t,...
                    [2016,noaccess],...
                    [2022,noaccess*(1-scenario)]); %Proportion no access
            elseif scenario==2
                highquality=sigmoidscaleup(t,...
                    [2016,.2*(1-noaccess)],...
                    [2021,1]);
                lowquality=sigmoidscaleup(t,...
                    [2016,.8*(1-noaccess)],...
                    [2021,0]);
                noaccess=sigmoidscaleup(t,...
                    [2016,noaccess],...
                    [2021,0]);
            end
        end
        
        if t<2009
            arvprop=0;
        elseif t>=2009&&t<2017
            arvprop=.09;
        elseif t>=2017&&t<2025
            arvprop=.69;
        elseif t>=2025
            arvprop=.77;
        end
        accessprop=cat(3,...
            highquality*(1-hivpos),...
            highquality*hivpos*arvprop,...
            highquality*hivpos*prophigh*(1-arvprop),...
            highquality*hivpos*propmid*(1-arvprop),...
            highquality*hivpos*proplow*(1-arvprop),...
            lowquality*(1-hivpos),...
            lowquality*hivpos*arvprop,...
            lowquality*hivpos*prophigh*(1-arvprop),...
            lowquality*hivpos*propmid*(1-arvprop),...
            lowquality*hivpos*proplow*(1-arvprop),...
            noaccess*(1-hivpos),...
            0,...
            noaccess*hivpos*prophigh,...
            noaccess*hivpos*propmid,...
            noaccess*hivpos*proplow);        
        
        pi=(totalpop.*(27./1000).*accessprop);
        %Birth rate now calibrated to growth target inferred from 2012 and
        %2025 populations.
        
        %AGEING AND BIRTHS
        flowsmat(1,1,:,1)=flowsmat(1,1,:,1)+pi.*(1-iota); %Adding births to sa
        flowsmat(2,1,:,1)=flowsmat(2,1,:,1)+pi.*iota; %Addding births to sb
        flowsmat(1:40,1,:,1)=flowsmat(1:40,1,:,1)-Y(1:40,1,:,1)./15;
        flowsmat(1:40,1,:,2)=flowsmat(1:40,1,:,2)+Y(1:40,1,:,1)./15;

        %ARV SCALE UP
        arvrate=0;
        if t<=2003
            arvrate=0;
        elseif t>2003&&t<2009
            arvrate=.015;
        elseif t>=2009&&t<2012
            arvrate=.13;
        elseif t>=2012&&t<2016
            arvrate=.22;
        elseif t>=2016&&t<2025
            arvrate=.04;
        elseif t>=2025
            arvrate=0;
        end
        flowsmat(1:40,1,3:5,:)=flowsmat(1:40,1,3:5,:)-...
            arvrate*Y(1:40,1,3:5,:);
        flowsmat(1:40,1,8:10,:)=flowsmat(1:40,1,8:10,:)-...
            arvrate*Y(1:40,1,8:10,:);
        flowsmat(1:40,1,2,:)=flowsmat(1:40,1,2,:)+...
            arvrate*sum(Y(1:40,1,3:5,:),3);
        flowsmat(1:40,1,7,:)=flowsmat(1:40,1,7,:)+...
            arvrate*sum(Y(1:40,1,8:10,:),3);
        
        %INTERVENTION 1A
        if intervention(1)==1&&scenario<2&&t>2016&&t<2022
            intocarerate=[.04,.1,.22,.45]; %Manually calibrated values
            intocare=intocarerate(1,scenario*4);
            flowsmat(1:40,1,1:5,:)=flowsmat(1:40,1,1:5,:)+...
                intocare*.08*Y(1:40,1,11:15,:);
            flowsmat(1:40,1,6:10,:)=flowsmat(1:40,1,6:10,:)+...
                intocare*.92*Y(1:40,1,11:15,:);
            flowsmat(1:40,1,11:15,:)=flowsmat(1:40,1,11:15,:)-...
                intocare*Y(1:40,1,11:15,:);
        end
        if intervention(1)==1&&scenario==2&&t>2016&&t<2021
            intocare=.6;
            flowsmat(1:40,1,1:5,:)=flowsmat(1:40,1,1:5,:)+...
                intocare*.08*Y(1:40,1,11:15,:);
            flowsmat(1:40,1,6:10,:)=flowsmat(1:40,1,6:10,:)+...
                intocare*.92*Y(1:40,1,11:15,:);
            flowsmat(1:40,1,11:15,:)=flowsmat(1:40,1,11:15,:)-...
                intocare*Y(1:40,1,11:15,:);
        end
        
        %INTERVENTION 1B
        if intervention(1)==2&&scenario<2&&t>2016&&t<2022
            lowhighrate=[.047,.118,.23,.9];
            lowhigh=lowhighrate(1,scenario*4);
            flowsmat(1:40,1,1:5,:)=flowsmat(1:40,1,1:5,:)+...
                lowhigh*Y(1:40,1,6:10,:);
            flowsmat(1:40,1,6:10,:)=flowsmat(1:40,1,6:10,:)-...
                lowhigh.*Y(1:40,1,6:10,:);
        end
        if intervention(1)==2&&scenario==2&&t>2016&&t<2021
            lowhigh=.9;
            flowsmat(1:40,1,1:5,:)=flowsmat(1:40,1,1:5,:)+...
                lowhigh*Y(1:40,1,6:10,:);
            flowsmat(1:40,1,6:10,:)=flowsmat(1:40,1,6:10,:)-...
                lowhigh.*Y(1:40,1,6:10,:);
        end
        
        %INTERVENTION 1
        if intervention(1)==3&&scenario<2&&t>2016&&t<2022
            lowhighrate=[.047,.11,.23,.8];
            nohighrate=[.05,.13,.22,.5];
            lowhigh=lowhighrate(1,scenario*4);
            nohigh=nohighrate(1,scenario*4);
            flowsmat(1:40,1,1:5,:)=flowsmat(1:40,1,1:5,:)+...
                lowhigh*Y(1:40,1,6:10,:)+nohigh*Y(1:40,1,11:15,:);
            flowsmat(1:40,1,6:10,:)=flowsmat(1:40,1,6:10,:)-lowhigh.*Y(1:40,1,6:10,:);
            flowsmat(1:40,1,11:15,:)=flowsmat(1:40,1,11:15,:)-nohigh.*Y(1:40,1,11:15,:);
        end
        if intervention(1)==3&&scenario==2&&t>2016&&t<2021
            lowhigh=.9;
            nohigh=.6;
            flowsmat(1:40,1,1:5,:)=flowsmat(1:40,1,1:5,:)+...
                lowhigh*Y(1:40,1,6:10,:)+nohigh*Y(1:40,1,11:15,:);
            flowsmat(1:40,1,6:10,:)=flowsmat(1:40,1,6:10,:)-lowhigh.*Y(1:40,1,6:10,:);
            flowsmat(1:40,1,11:15,:)=flowsmat(1:40,1,11:15,:)-nohigh.*Y(1:40,1,11:15,:);
        end
        
        %PARAMETERS OUTPUT MATRIX
        a=a+1;
        paramsout(a,1:13)=updateparams(t,Y,access,accessprop,...
            ise,isz,isp,ime,imz,imp,...
            deltase,deltasz,deltasp,deltame,deltamz,deltamp,deltaxme,deltaxmz,...
            deltaxmp,gammaez,gammap,muez,mup,ppv,upsilonm,mdefault,mdeath,...
            iptcoverage,arvlayers);
        
        %REARRANGE FLOW VECTOR AGAIN
        flows=reshape(flowsmat,comps*access*ages,1); %Put flow matrix
        %back into single column vector shape to act on the initial condition
        %vector.
        
    end %__________________________________________________________________

y=reshape(y,size(y,1),comps,access,ages); %Reshape into a vertical
%vector with columns compartments and access types layers.
y=[yold;y]; %Concatenate the y vectors together
t=[told;t]; %Just to make sure t and y vectors come out the same length

end