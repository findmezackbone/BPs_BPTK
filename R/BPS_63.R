library(deSolve)
para_BP_individual<- read.csv("D:\\1st\\Project_pharmacy\\R_language\\para_BP_individual.csv")

i<-2

gender=as.numeric(para_BP_individual[2,i+1])   #1 for male, 2 for female
bw= as.numeric(para_BP_individual[1,i+1])    #body weight (kg)
QCC = as.numeric(para_BP_individual[3,i+1])
QliverC = as.numeric(para_BP_individual[4,i+1])
QskinC =as.numeric( para_BP_individual[7,i+1])
QgonadC=as.numeric(para_BP_individual[8,i+1])
QfatC = as.numeric(para_BP_individual[9,i+1])
QbrainC <- as.numeric(para_BP_individual[22,i+1])
QmuscleC <-as.numeric(para_BP_individual[24,i+1])

VplasmaC <-as.numeric(para_BP_individual[16,i+1])
VfatC <- as.numeric(para_BP_individual[15,i+1])
VliverC <- as.numeric(para_BP_individual[10,i+1])
VbrainC <- as.numeric(para_BP_individual[25,i+1])
VgonadC <- as.numeric(para_BP_individual[14,i+1])
VskinC <-  as.numeric(para_BP_individual[13,i+1])
VrichC <- as.numeric(para_BP_individual[26,i+1])
VbodygC <- VplasmaC 
VbodysC <- VplasmaC

QC <- QCC*60 # (L/h) |Cardiac output according to ICRP
Qfat <- QfatC*QC # (L/h) |Blood flow to the fat
Qliver <- QliverC*QC # (L/h) |Blood flow to the liver
Qgonad <- QgonadC*QC # (L/h) |Blood flow to the gonads
Qbrain <- QbrainC*QC # (L/h) |Blood flow to the brain
Qskin=QskinC*QC# (L/h) |Blood flow to the skin
Qslow <- QmuscleC*QC # (L/h)|Blood flow to the slowly perfused tissues
Qrich <- QC-Qliver-Qbrain-Qfat-Qgonad-Qslow-Qskin

Vliver <- VliverC*bw # (L) |Volume of the liver
Vfat <- VfatC*bw # (L) |Volume of the fat
Vgonad <- VgonadC*bw # (L) |Volume of the gonads
Vplasma <- VplasmaC*bw # (L) |Volume of the plasma
Vbrain <- VbrainC*bw # (L) |Volume of the brain
Vskin=VskinC*bw  # (L) |Volume of the skin
Vrich<- VrichC*bw  # (L) |Volume of the richly perfused tissues
Vslow <- bw-Vliver-Vfat-Vgonad-Vplasma-Vbrain-Vrich
Vbodyg<- VbodygC*bw # (L) |Volume of the distribution for BPAG
Vbodys <- VbodysC*bw # (L) |Volume of the distribution for BPAS



MWBPS <- 250.27 # (g/mol) |Molecular weight
pliver <- 0.846 # | (liver/blood)
pfat <- 0.435 # | (fat/blood)
pslow <- 0.881 # | (slowly perfused/blood)
prich <- 0.810 # | (richly perfused/blood)
pgonad <- 0.843 # | (gonads/blood)
pbrain <- 0.810 # | (brain/blood)
pskin <- 1.43# | (skin/blood)


geC <- 3.5 # (1/h/bw^-0.25) |Gastric emptying of BPs
k0C <- 0 # (1/h/bw^-0.25) |Oral uptake of BPs from the stomach into the liver; set to 0
k1C <- 5 # (1/h/bw^-0.25) |Oral uptake of BPs from the small intestine into the liver
k4C <- 0 # (1/h/bw^-0.25) |Fecal elimination of BPs from small intestine after peroral administration; set to 0
kGIingC <- 50 # (1/h/bw^-0.25) |Transport of BPafG from enterocytes into serum
kGIinsC <- 50 # (1/h/bw^-0.25) |Transport of BPafS from enterocytes into serum
kmgutg <-555000 # (nM) |Glucuronidation of BPs in the gut
ksigutg <- 711000 # (nM) |Glucuronidation of BPs in the gut for substrate
vmaxgutgC <- 563 # (nmol/h/kg bw) |Glucuronidation of BPs in the gut
fgutg <- 1 # Correction factor of glucuronidation in the gut
kmguts <- 0.001 # (nM) |Sulfation of BPaf in the gut, not modeled
vmaxgutsC <- 0.001 # (nmol/h/bw^0.75 |Sulfation of BPf in the gut
fguts <- 0 # Correction factor of sulfation in the gut no sulfation in the gut assumed

met1g <- 0.33# |Fraction of BPs-G in the liver taken up directly into serum (set to 1 to deactivate EHR)
met1s <- 1 # |Fraction of BPAS in the liver taken up directly into serum
enterocytes <- 0.1223 # (L)
kmliver <-446000 # (nM) |Glucuronidation of BPs in the liver
vmaxliverC <- 7810 # (nmol/h/g liver) |Glucuronidation of BPs in the liver
fliverg <- 1
kmlivers <- 10100 # (nM) |Sulfation of BPA in the liver, set to the value for SULT1A1 (Takahito 2002)
vmaxliversC <- 149 #

flivers <- 1
EHRtime <- 0.00 # (h) |Time until EHR occurs
EHRrateC <- 2 # (1/h/bw^-0.25) |EHR of BPsg
k4C_IV <- 0 # (1/h/bw^-0.25) |Fecal elimination of BPAG from the EHR compartment
kurinebpaC <- 0.04 # (L/h/bw^0.75) 
kurinebpagC <-1.336 # (L/h/bw^0.75)
kurinebpasC <- 0.03 # (L/h/bw^0.75)
vreabsorptiongC <- 0
vreabsorptionsC<-0
kreabsorptiong<-9200
kreabsorptions<-9200
kenterobpagC <- 0.35 # (1/h/bw^-0.25) |EHR of Bps
kenterobpasC <- 0.0 



vmaxliversCnew <- vmaxliversC*VliverC*1000
vmaxliversCnew <- vmaxliversCnew*bw/(bw^0.75)
vmaxliverCnew <- vmaxliverC*VliverC*1000
vmaxliverCnew <- vmaxliverCnew*bw/(bw^0.75)
vmaxgutgCnew <- vmaxgutgC*bw/(bw^0.75)

vreabsorptiong <- vreabsorptiongC*bw^0.75 # (nmol/h) |vmax of renal resorption of BPAG
vreabsorptions <- vreabsorptionsC*bw^0.75 # (nmol/h) |vmax of renal resorption of BPAS

EHRrate <- EHRrateC/(bw^0.25) # (1/h) |EHR of BPAG
k0 <- k0C/bw^0.25 # (1/h)|Uptake of BPA from the stomach into the liver
ge <- geC/bw^0.25 # (1/h)|Gastric emptying of BPA
k1 <- k1C/bw^0.25 # (1/h)|Uptake of BPA from small intestine into the liver
k4 <- k4C/bw^0.25 # (1/h)|Fecal excretion of BPA after peroral administration from small intestine 
k4_IV <- k4C_IV/bw^0.25 # (1/h) |Fecal excretion of BPAG from the EHR compartment
vmaxliver <- vmaxliverCnew*fliverg*bw^0.75 # (nmol/h) |vmax of BPA glucuronidation in the liver
kGIing <- kGIingC/bw^0.25 # (1/h) |Uptake of BPAG from small intestine into serum
met2g <- 1.0-met1g # () |Fraction of BPAG formed subject to EHR
met2s <- 1.0-met1s # () |Fraction of BPAS formed subject to EHR

kurinebpa <- kurinebpaC*bw^0.75 # (L/h)|Clearance of BPA via urine
kurinebpag <- kurinebpagC*bw^0.75 #(L/h)|Clearance of BPAG via urine
kurinebpas <- kurinebpasC*bw^0.75 #(L/h)|Clearance of BPAS via urine
vmaxlivers <- vmaxliversCnew*flivers*bw^0.75 # (nmol/h) |vmax of BPA sulfation in the liver
kGIins <- kGIinsC/bw^0.25 # (1/h) |Uptake of BPAS from small intestine into serum
vmaxgutg <- vmaxgutgCnew*fgutg*bw^0.75 # (nmol/h) |vmax of BPA glucuronidation in the gut
vmaxguts <- vmaxgutsC*fguts*bw^0.75 # (nmol/h) |vmax of BPA sulfation in the gut
kenterobpag<- kenterobpagC/bw^0.25 # (1/h) |EHR of BPA due to biliary excretion of BPAG
kenterobpas <- kenterobpasC/bw^0.25 # (1/h) |EHR of BPA due to biliary excretion of BPAS
oral_intakerate=0
BSA= as.numeric(para_BP_individual[21,i+1])*100 #m2-dm2 body skin area
if(gender==1) epi_finger=545 else epi_finger=460  #ICRP 89 P196 um
TSC = 185/100000  # Thickness of stratum corneum (um-dm). thick for palm 
TVE = (epi_finger-185)/100000  # Thickness of viable epidermis (um-dm).
SCDX = TSC / 10
if(gender==1) skin_finger=1200+545 else skin_finger=1095+460  #ICRP 89 P196 um
TFO=388/560*skin_finger/100000   #um-dm
DOSE=as.numeric(para_BP_individual[17,i+1])/MWBPS   #(nmol/kg/d)
ABS=as.numeric(para_BP_individual[18,i+1])   #absorption fraction
DOSE_d=DOSE*ABS
AEXP=20/100  # Area of exposed skin (cm^2- dm2).
add_peroid=1/60 #h
exp_peroid=13/6 #h
VWELL=AEXP*0.01 #L
FEXP=0.005 # FRACTION OF Area of exposed skin FOR HIAR
VTSC = AEXP * TSC   # ... (exposed) stratum corneum.L
VTVE = AEXP * TVE    # ... (exposed) viable epidermis.L
VTFO=AEXP*FEXP*TFO 
HSCVE=19.45 
HFOWell= 5.4 
HSCwell=  5.4 
DSC = 17.28E-9 # diffusion (cm^2/h- dm2) 
PFO= 6.39E-5 #cm/h- dm 
u1=5.7E-5 

yini_bps_dermal <- unlist(c(data.frame(
  
  dose=0,AFO=0,AWELL=0,
  CSC01=0,CSC02=0,CSC03=0,CSC04=0,CSC05=0,CSC06=0,CSC07=0,CSC08=0,CSC09=0,
  AVE=0,Input_oral=0,
  
  AST = 0, 
  AGImet = 0, 
  AGImets = 0, 
  AAO = 0,
  ASI = 0, 
  Afeces = 0, 
  Aoral = 0, 
  AGIin = 0, AGIBPAg = 0, AGIins = 0,AGIBPAs = 0, 
  Askin=0,
  Aplasma = 0, 
  AFat = 0, 
  Agonad = 0, 
  AM=0,AMs=0,
  ABPA_delayinbpag = 0, 
  ABPA_delayinbpas = 0,
  ALiver = 0, 
  Abrain = 0, 
  AR = 0, 
  AS = 0, 
  Aurinebpa = 0, 
  
  ABPAg_prod = 0, 
  ABPAg_prod_delay = 0, 
  ABPAg_prod_gut = 0, 
  ABPAg_prod_delay_gut=0, 
  ABPAs_prod = 0, 
  ABPAs_prod_delay = 0,
  ABPAs_prod_gut = 0, 
  ABPAs_prod_delay_gut=0, 
  
  ABPA_delayin= 0, 
  Afecesiv = 0, 
  ABPA_delay = 0, 
  ABPA_delayins = 0, 
  Afecesivs = 0, 
  ABPA_delays = 0,
  
  Areabsorption = 0, 
  Aurineg = 0, 
  Aurinebpag = 0, 
  Areabsorptions = 0, 
  Aurines = 0 ,
  Aurinebpas = 0, 
  
  Abpac = 0,
  Abpas = 0, 
  
  Vurine=0)))

para_bps_dermal <- unlist(c(data.frame(
  oral_intakerate,QC,Qfat,Qliver,Qgonad,Qbrain,
  Qskin,
  Qrich,Qslow,Vliver,Vfat,Vskin,Vgonad,Vplasma,Vbrain,
  Vslow, Vrich,Vbodyg,Vbodys,
  pliver,pfat,pslow, prich,pgonad,pbrain,pskin,kmgutg,ksigutg,kmguts,met1g,met1s,
  enterocytes, kmliver, kmlivers,EHRtime,kreabsorptiong,kreabsorptions,vreabsorptiong,vreabsorptions,
  EHRrate,k0, ge,k1, k4,k4_IV,vmaxliver,kGIing, met2g,met2s,kurinebpa,kurinebpag, kurinebpas,vmaxlivers,
  kGIins,vmaxgutg,vmaxguts,kenterobpag,kenterobpas,
  DOSE_d,VTVE,VTSC, VTFO,TFO,AEXP,SCDX,
  HSCVE,HSCwell,HFOWell,PFO,
  DSC,VWELL,exp_peroid,u1,BSA,FEXP,add_peroid
)))

PBTKmod_bps <- function(t, y, parms)
{
  with (as.list(c(y, parms)),
        {
          if(t<EHRtime){kentero=0}else{kentero=EHRrate} 
          if(t<=exp_peroid){ OO_S_TP1=1} else { OO_S_TP1=0}
          
          CA <- Aplasma/Vplasma
          
          # Dermal Exposure Well
          if(t<=add_peroid) Rdose=DOSE_d/(add_peroid) else Rdose=0 
          CWELL= (AWELL/VWELL)*OO_S_TP1 # ... skin exposure well (nmol/l)
          CSC00 = CWELL * HSCwell
          CVE = AVE/ VTVE
          CvVE = CVE / pskin 
          CSC10 = CVE * HSCVE 
          
          JSC00 = -DSC * (CSC01 - CSC00) / SCDX -u1*CSC00  # ... outer surface of SC.
          JSC10 = -DSC * (CSC10 - CSC09) / SCDX -u1*CSC10   # ... interface of SC with VE.
          
          CFO=AFO/VTFO
          CvFO=CFO/pskin 
          
          JFO=PFO*(CWELL-CFO/HFOWell) 
          
          dAFO=JFO*AEXP*FEXP + Qskin*(AEXP/BSA)*0.25 * (CA - CvFO)
          dAWELL =(Rdose-JSC00*AEXP*(1-FEXP)-AEXP*FEXP*JFO)*OO_S_TP1
          
          dCSC01 = DSC * (CSC00 - 2 * CSC01 + CSC02) / (SCDX * SCDX)+u1*(CSC02-CSC00)/(2*SCDX)
          dCSC02 = DSC * (CSC01 - 2 * CSC02 + CSC03) / (SCDX * SCDX)+u1*(CSC03-CSC01)/(2*SCDX)
          dCSC03 = DSC * (CSC02 - 2 * CSC03 + CSC04) / (SCDX * SCDX)+u1*(CSC04-CSC02)/(2*SCDX)
          dCSC04 = DSC * (CSC03 - 2 * CSC04 + CSC05) / (SCDX * SCDX)+u1*(CSC05-CSC03)/(2*SCDX)
          dCSC05 = DSC * (CSC04 - 2 * CSC05 + CSC06) / (SCDX * SCDX)+u1*(CSC06-CSC04)/(2*SCDX)
          dCSC06 = DSC * (CSC05 - 2 * CSC06 + CSC07) / (SCDX * SCDX)+u1*(CSC07-CSC05)/(2*SCDX)
          dCSC07 = DSC * (CSC06 - 2 * CSC07 + CSC08) / (SCDX * SCDX)+u1*(CSC08-CSC06)/(2*SCDX)
          dCSC08 = DSC * (CSC07 - 2 * CSC08 + CSC09) / (SCDX * SCDX)+u1*(CSC09-CSC07)/(2*SCDX)
          dCSC09 = DSC * (CSC08 - 2 * CSC09 + CSC10) / (SCDX * SCDX)+u1*(CSC10-CSC08)/(2*SCDX)
          
          dAVE = JSC10 * AEXP*(1-FEXP) + Qskin*(AEXP/BSA)*0.75*(CA - CvVE)
          
          if(t<=0.1 && t>=0){onoff_oral =1} else{onoff_oral =0}
          dInput_oral=oral_intakerate*onoff_oral 
          Cgut <- ASI/enterocytes 
          dAST <- dInput_oral-k0*AST-ge*AST 
          RAGImet <- vmaxgutg*Cgut/(kmgutg+Cgut*(1+Cgut/ksigutg)) 
          RAGImets <- vmaxguts*Cgut/(kmguts+Cgut) 
          RAAO <- k1*ASI 
          dASI <- ge*AST-RAGImet-RAAO-RAGImets 
          RAfeces <- k4*ASI 
          RAoral <- k0*AST+RAAO
          
          RAGIin <- kGIing*AGIBPAg 
          RAGIBPAg <- RAGImet - RAGIin
          RAGIins <- kGIins*AGIBPAs 
          RAGIBPAs <- RAGImets - RAGIins
          
          
          Cskin <- Askin/(Vskin-VTSC-VTVE-VTFO) 
          CVskin <- Askin/((Vskin-VTSC-VTVE-VTFO)*pskin) 
          
          CFat <- AFat/Vfat
          CVFat <- AFat/(Vfat*pfat) 
          Cgonad <- Agonad/Vgonad 
          CVgonad <- Agonad/(Vgonad*pgonad) 
          CLiver <- ALiver/Vliver
          CVLiver <- ALiver/(Vliver*pliver) 
          Cbrain <- Abrain/Vbrain
          CVbrain <- Abrain/(Vbrain*pbrain) 
          CR <- AR/Vrich # 
          CVR <- AR/(Vrich*prich) # 
          CVS <- AS/(Vslow*pslow) # (nmol/L) 
          CS <- AS/Vslow # 
          
          CV <-(CVLiver*Qliver+
                  CVskin*(Qskin*(1-AEXP/BSA)) + 
                  CvVE*(Qskin*(AEXP/BSA)*0.75)+
                  CvFO*(Qskin*(AEXP/BSA)*0.25)+
                  +CVFat*Qfat+CVR*Qrich+CVS*Qslow+
                  CVgonad*Qgonad+CVbrain*Qbrain)/QC #
          
          dAskin <- Qskin*(1-AEXP/BSA)*(CA-CVskin) 
          
          dAplasma <- QC*(CV-CA) -kurinebpa*CV 
          dAfat <- Qfat*(CA-CVFat) 
          dAgonad <- Qgonad*(CA-CVgonad)
          
          RAM <- vmaxliver*CVLiver/(kmliver+CVLiver) 
          RAMs <- vmaxlivers*CVLiver/(kmlivers+CVLiver)
          RABPA_delayinbpag <- ABPA_delay*kenterobpag
          RABPA_delayinbpas <- ABPA_delays*kenterobpas 
          
          dALiver <- Qliver*(CA-CVLiver)+RAoral-RAM-RAMs+RABPA_delayinbpag+RABPA_delayinbpas
          dAbrain <- Qbrain*(CA-CVbrain) # (nmol/h)
          dAR <- Qrich*(CA-CVR)
          dAS <- Qslow*(CA-CVS)
          dAurinebpa <- kurinebpa*CV
          
          RABPAg_prod <- met1g*RAM # (nmol/h) |Taken up into systemic circulation
          RABPAg_prod_delay <- met2g*RAM # (nmol/h) |Excreted into bile
          RABPAg_prod_gut<- met1g*RAGIin # (nmol/h)|Taken up into systemic circulation
          RABPAg_prod_delay_gut <- met2g*RAGIin # (nmol/h) |Excreted into bile
          RABPAs_prod <- met1s*RAMs # (nmol/h) |Taken up into systemic circulation 
          RABPAs_prod_delay <- met2s*RAMs # (nmol/h) |Excreted into bile
          RABPAs_prod_gut<- met1s*RAGIins # (nmol/h)|Taken up into systemic circulation
          RABPAs_prod_delay_gut <- met2s*RAGIins # (nmol/h) |Excreted into bile
          
          RABPA_delayin <- ABPA_delay*kentero  
          RAfecesiv <- ABPA_delay*k4_IV 
          RABPA_delay <- RABPAg_prod_delay+RABPAg_prod_delay_gut-RABPA_delayin-
            RAfecesiv-RABPA_delayinbpag 
          
          RABPA_delayins <- ABPA_delays*kentero 
          RAfecesivs <- ABPA_delays*k4_IV
          RABPA_delays <- RABPAs_prod_delay+RABPAs_prod_delay_gut-RABPA_delayins-
            RAfecesivs-RABPA_delayinbpas 
          
          Cbpac <- Abpac/(Vbodyg+1E-34)
          Cbpas <- Abpas/(Vbodys+1E-34)
          
          RAreabsorption <- vreabsorptiong*Cbpac/(kreabsorptiong+Cbpac) 
          dAurineg <- kurinebpag*Cbpac
          RAurinebpag <- kurinebpag*Cbpac-RAreabsorption 
          
          RAreabsorptions <- vreabsorptions*Cbpas/(kreabsorptions+Cbpas) 
          dAurines <- kurinebpas*Cbpas
          RAurinebpas <- kurinebpas*Cbpas-RAreabsorptions 
          
          RAbpac <- RABPAg_prod+RABPAg_prod_gut+RABPA_delayin-RAurinebpag
          RAbpas <- RABPAs_prod+RABPA_delayins+RABPAs_prod_gut-RAurinebpas
          
          dVurine=1.5/24
          
          dydt <-
            c(Rdose,dAFO,
              dAWELL,dCSC01,dCSC02,dCSC03,dCSC04,dCSC05,dCSC06,
              dCSC07,dCSC08,dCSC09,dAVE,dInput_oral,
              
              dAST,RAGImet,RAGImets,RAAO,dASI,
              RAfeces,RAoral, RAGIin,RAGIBPAg,RAGIins,RAGIBPAs,
              dAskin,dAplasma,dAfat,dAgonad,RAM,RAMs,RABPA_delayinbpag,RABPA_delayinbpas,
              dALiver,dAbrain, dAR,dAS,dAurinebpa,
              RABPAg_prod,RABPAg_prod_delay,RABPAg_prod_gut,RABPAg_prod_delay_gut,
              RABPAs_prod,RABPAs_prod_delay,RABPAs_prod_gut,RABPAs_prod_delay_gut,
              RABPA_delayin,RAfecesiv,RABPA_delay,RABPA_delayins,RAfecesivs,RABPA_delays,
              RAreabsorption,dAurineg,RAurinebpag,
              RAreabsorptions,dAurines,RAurinebpas,
              RAbpac,RAbpas,dVurine)
          
          conc <- c(CV=CV,CWELL=CWELL)
          res <- list(dydt, conc)
          return(res)
        })}

zeit <- seq(0,75, 0.005) # (h) time
BPS1<-ode(y=yini_bps_dermal,times=zeit,  func=PBTKmod_bps, 
          parms=para_bps_dermal, method="lsoda")

plot(BPS1[,28])
write.csv(BPS1,"D://1st//Project_pharmacy//R_language//OriginalResult.csv")