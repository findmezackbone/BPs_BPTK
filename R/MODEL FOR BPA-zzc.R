library(deSolve)
library(ggplot2)

start1 <- Sys.time()   #��¼������ʼʱ��
para_BP_individual <- read.csv("D:\\1st\\Project_pharmacy\\R_language\\para_BP_individual.csv")
i = 2
gender=2  #�Ա�1ΪŮ�ԣ�2Ϊ���� 
bw= 60   #���أ�kg
QCC = as.numeric(para_BP_individual[3,i+1])
QliverC = as.numeric(para_BP_individual[4,i+1])
QskinC =as.numeric( para_BP_individual[7,i+1])
QgonadC=as.numeric(para_BP_individual[8,i+1])
QfatC = as.numeric(para_BP_individual[9,i+1])
QbrainC <- as.numeric(para_BP_individual[22,i+1])
QmuscleC <-as.numeric(para_BP_individual[24,i+1])

QC <- QCC*60  #����ѪҺ���٣�L/h��
Qfat <- QfatC*QC  #֬��ѪҺ���٣�L/h��
Qliver <- QliverC*QC  #����ѪҺ���٣�L/h��
Qgonad <- QgonadC*QC  #����ѪҺ���٣�L/h��
Qbrain <- QbrainC*QC  #����ѪҺ���٣�L/h��
Qskin=QskinC*QC  #Ƥ��ѪҺ���٣�L/h��
Qslow <- QmuscleC*QC  #����������֯ѪҺ���٣�L/h��
Qrich <- QC-Qliver-Qbrain-Qfat-Qgonad-Qslow-Qskin  #���ٹ�����֯ѪҺ���٣�L/h��

#����֯���ٵ��������
VplasmaC <-as.numeric(para_BP_individual[16,i+1])
VfatC <- as.numeric(para_BP_individual[15,i+1])
VliverC <- as.numeric(para_BP_individual[10,i+1])
VbrainC <- as.numeric(para_BP_individual[25,i+1])
VgonadC <- as.numeric(para_BP_individual[14,i+1])
VskinC <-  as.numeric(para_BP_individual[13,i+1])
VrichC <- as.numeric(para_BP_individual[26,i+1])
VbodygC <- VplasmaC  #����BPAG�ֲ���֯�������������L/kg��
VbodysC <- VplasmaC  #����BPAS�ֲ���֯�������������L/kg��
enterocytes <- 0.1223  #С�������L��

Vliver <- VliverC*bw  #���������L��
Vfat <- VfatC*bw  #֬�������L��
Vgonad <- VgonadC*bw  #���������L��
Vplasma <- VplasmaC*bw  #Ѫ�������L��
Vbrain <- VbrainC*bw  #���������L��
Vskin=VskinC*bw  #Ƥ�������L��
Vrich<- VrichC*bw  #���ٹ�����֯�����L��
Vslow <- bw-Vliver-Vfat-Vgonad-Vplasma-Vbrain-Vrich  #����������֯�����L��
Vbodyg<- VbodygC*bw  #����BPAG�ֲ���֯�����L��
Vbodys <- VbodysC*bw  #����BPAS�ֲ���֯�����L��

#��������BPA��������##################################################
MWBPA <- 228.28  #BPA��Ħ��������ת����¶��ngΪmol��
#BPA����֯-Ѫ������ϵ�������㻯ѧ���ڲ�ͬ��֯���ټ�ת�������ã�
pliver<- 0.73   #����-Ѫ������ϵ��
pfat <- 5.0  #����-Ѫ������ϵ��
pslow<- 2.7  #����������֯-Ѫ������ϵ��
prich<- 2.8   #���ٹ�����֯-Ѫ������ϵ��
pgonad<- 2.6   #����-Ѫ������ϵ��
pbrain<- 2.8  #����-Ѫ������ϵ��
pskin<- 2.15  #Ƥ��-Ѫ������ϵ��
#BPA��ADME���������ա��ֲ�����л����й������
#���ղ���
geC<- 3.5  #θ�ſ�ϵ�����ڷ���ҩBPA��θת����С����ϵ����1/h/bw^-0.25��
k0C<- 0  #�ڷ���ҩBPA��θ��������ϵ����1/h/bw^-0.25��
k1C<-2.1  #�ڷ���ҩBPA��С����������ϵ����1/h/bw^-0.25��
k4C<- 0  #�ڷ���ҩBPA��С���ķ������ϵ����1/h/bw^-0.25��
kGIingC <- 50  #�ڷ���ҩBPAG�ӳ���Ѫ�е�ϵ����1/h/bw^-0.25��
kGIinsC <- 50  #�ڷ���ҩBPAS�ӳ���Ѫ�е�ϵ����1/h/bw^-0.25��

ge <- geC/bw^0.25  #θ�ſ�ϵ�����ڷ���ҩBPA��θת����С����ϵ����1/h��
k0 <- k0C/bw^0.25  #�ڷ���ҩBPA��θ��������ϵ����1/h��
k1 <- k1C/bw^0.25  #�ڷ���ҩBPA��С����������ϵ����1/h��
k4 <- k4C/bw^0.25  #�ڷ���ҩBPA��С���ķ������ϵ����1/h ��
kGIing <- kGIingC/bw^0.25  #�ڷ���ҩBPAG�ӳ���Ѫ�е�ϵ����1/h��
kGIins <- kGIinsC/bw^0.25  #�ڷ���ҩBPAS�ӳ���Ѫ�е�ϵ����1/h��


kmgutg<- 58400  #������BPA�������ữ�����ϳ���Km��nmol��
vmaxgutgC<- 361  #������BPA�������ữ�����Ӧ�ٶ�Vmax��nmol/h/kg bw��
fgutg <- 1  #������BPA�������ữ��Ӧ��У�����ӣ�1Ϊ���ڣ�0Ϊ�����ڣ�
kmguts <- 0.001    #������BPA�����λ������ϳ���Km��nmol��
vmaxgutsC <- 0.00001  #������BPA�����λ������Ӧ�ٶ�Vmax��nmol/h/bw^0.75��
ksigutg <- 711000  #������BPA��л�ĵ������Ƴ�����nmol��
fguts <- 0  #������BPA�����λ���Ӧ��У�����ӣ�1Ϊ���ڣ�0Ϊ�����ڣ�
kmliver <- 45800  #������BPA�������ữ�����ϳ���Km��nmol��
vmaxliverC <- 9043.2  #������BPA�������ữ�����Ӧ�ٶ�Vmax��nmol/h/g liver��
fliverg <- 1  #������BPA�������ữ��Ӧ��У�����ӣ�1Ϊ���ڣ�0Ϊ�����ڣ�
kmlivers <- 10100  #������BPA�����λ������ϳ���Km��nmol��
vmaxliversC <- 149  #������BPA�����λ������Ӧ�ٶ�Vmax��nmol/h/g liver��
flivers <- 1   #������BPA�����λ���Ӧ��У�����ӣ�1Ϊ���ڣ�0Ϊ�����ڣ�


vmaxgutgCnew <- vmaxgutgC*bw/(bw^0.75) 
vmaxgutg <- vmaxgutgCnew*fgutg*bw^0.75 #������BPA�������ữ�����Ӧ�ٶ�Vmax��nmol/h��
vmaxguts <- vmaxgutsC*fguts*bw^0.75  #������BPA�����λ������Ӧ�ٶ�Vmax��nmol/h��
vmaxliverCnew <- vmaxliverC*VliverC*1000  
vmaxliverCnew <- vmaxliverCnew*bw/(bw^0.75)
vmaxliver <- vmaxliverCnew*fliverg*bw^0.75  #������BPA�������ữ�����Ӧ�ٶ�Vmax��nmol/h��
vmaxliversCnew <- vmaxliversC*VliverC*1000
vmaxliversCnew <- vmaxliversCnew*bw/(bw^0.75)
vmaxlivers <- vmaxliversCnew*flivers*bw^0.75 #������BPA�������ữ�����Ӧ�ٶ�Vmax��nmol/h��




met1g <- 0.9  #������BPAG����Ѫ�еı�����Ϊ1ʱ�򲻴��ڸγ�ѭ����
met1s <- 1  #������BPAS����Ѫ�еı�����Ϊ1ʱ�򲻴��ڸγ�ѭ����
met2g <- 1.0-met1g   #������BPAG����γ�ѭ���ı���
met2s <- 1.0-met1s   #������BPAS����γ�ѭ���ı���
EHRtime <- 0  #�γ�ѭ��������ʼʱ��
EHRrateC <- 0.2  #BPAG�γ�ѭ�������ʻ���������1/h/bw^-0.25��
k4C_IV <- 0  #BPAG�ڸγ�ѭ���еķ������ϵ����1/h/bw^-0.25��
kenterobpagC <- 0.2  #BPAG�γ�ѭ��ʹ��BPA����ѭ�������ʻ���������1/h/bw^-0.25��
kenterobpasC <- 0.0  #BPAS�γ�ѭ��ʹ��BPA����ѭ�������ʻ���������1/h/bw^-0.25��

EHRrate <- EHRrateC/(bw^0.25)  #BPAG�γ�ѭ�������ʣ�1/h��
k4_IV <- k4C_IV/bw^0.25    #BPAG�ڸγ�ѭ���еķ������ϵ����1/h��
kenterobpag<- kenterobpagC/bw^0.25  #BPAG�γ�ѭ��ʹ��BPA����ѭ�������ʣ�1/h��
kenterobpas <- kenterobpasC/bw^0.25  #BPAS�γ�ѭ��ʹ��BPA����ѭ�������ʣ�1/h��


kurinebpaC <- 0.06  #BPA��Һ��й����������L/h/bw^0.75��
kurinebpagC <- 0.35  #BPAG��Һ��й����������L/h/bw^0.75��
kurinebpasC <- 0.03  #BPAS��Һ��й����������L/h/bw^0.75��
vreabsorptiongC <- 0 #��Һ��й��BPAG�����������յ����Ӧ�ٶ�Vmax��nmol/h/bw^0.75��
vreabsorptionsC <- 0 #��Һ��й��BPAS�����������յ����Ӧ�ٶ�Vmax��nmol/h/bw^0.75��
kreabsorptiong <- 9200  #��Һ��й��BPAG�����������յ����ϳ���Km��nmol/L��
kreabsorptions <- 9200  #��Һ��й��BPAS�����������յ����ϳ���Km��nmol/L��


kurinebpa <- kurinebpaC*bw^0.75  #BPA��Һ��й������L/h��
kurinebpag <- kurinebpagC*bw^0.75  #BPAG��Һ��й������L/h��
kurinebpas <- kurinebpasC*bw^0.75  #BPAS��Һ��й������L/h��
vreabsorptiong <- vreabsorptiongC*bw^0.75  #��Һ��й��BPAG�����������յ����Ӧ�ٶ�Vmax��nmol/h��
vreabsorptions <- vreabsorptionsC*bw^0.75  #��Һ��й��BPAS�����������յ����Ӧ�ٶ�Vmax��nmol/h�� 


##Ƥ������

#手指暴露的参数
BSA= 1.6*100 #����Ƥ���������m2-dm2��
epi_finger=460  #��ָ��Ƥ��ȣ�um��
TSC = 185/100000  #��ָ���ʲ��ȣ�um-dm�� 
TVE = (epi_finger-185)/100000  #��ָ���Ա�Ƥ���ȣ�um-dm��
SCDX = TSC / 10  #���ʲ��ǲ��ȣ���Ϊ11�㣬ÿ����Ϊscdx����dm��
skin_finger=1095+460  #��ָƤ����ȣ�um��
TFO=388/560*skin_finger/100000   #��ָë�Һ�ȣ�um-dm��
AEXP=20/100  #��¶Ƥ�������cm2- dm2��
FEXP=0.01  #��¶Ƥ����ë�����������1-FEXPΪ���ʲ����������
add_peroid=1/60  #ͿĨ��ѧ��ʱ����h�����Ӵ����Ũ�Ȼ�ѧ������Ⱪ¶Rdose����ʱ�䣬����add_period��ֹͣ�Ӵ�����ֽ��ֹͣ�Ⱪ¶���룬��dose=0��
exp_peroid=1   #��¶��ѧ��ʱ����h������ѧ�ﱻͿĨ��Ƥ������Ӵ��Ⱪ¶����ʱ�䣬����exp_period����ϴƤ����ֹͣ��¶����SSD����գ�
VWELL=AEXP*0.01  #��¶Ƥ���������������������ȣ�L��
VTSC = AEXP * TSC   # ��¶Ƥ�����ʲ�������������ȣ�L��·
VTVE = AEXP * TVE    # ��¶Ƥ�����Ա�Ƥ��������������ȣ�L��
VTFO=AEXP*FEXP*TFO  #��¶Ƥ��ë��������������ȣ�L��

#胳膊暴露的参数
BSA= 1.6*100  #����Ƥ���������m2-dm2��
epi_forarm=1000  #ǰ�۱�Ƥ��ȣ�um��
TSC = 50/100000  #ǰ�۽��ʲ��ȣ�um-dm�� 
TVE = (epi_forarm-15)/100000  #ǰ�ۻ��Ա�Ƥ���ȣ�um-dm��
SCDX = TSC / 10  #���ʲ��ǲ��ȣ���Ϊ11�㣬ÿ����Ϊscdx����dm��
skin_forearm=1050  #ǰ��Ƥ����ȣ�um��
TFO=388/560*skin_forearm/100000   #ǰ��ë�Һ�ȣ�um-dm��
AEXP=0.9* BSA   #��¶Ƥ�������dm2��
FEXP=0.01  #��¶Ƥ����ë�����������1-FEXPΪ���ʲ����������
add_peroid1=1/60   #ͿĨ��ѧ��ʱ����h��
exp_peroid1=12    #��¶��ѧ��ʱ����h��
VWELL=AEXP*0.01  #��¶Ƥ���������������������ȣ�L��
VTSC = AEXP * TSC   # ��¶Ƥ�����ʲ�������������ȣ�L��
VTVE = AEXP * TVE    # ��¶Ƥ�����Ա�Ƥ��������������ȣ�L��
VTFO=AEXP*FEXP*TFO  #��¶Ƥ��ë��������������ȣ�L��


ABS=0.2828*0.9618  #TP��¶����ϵ����TPת�Ƶ���ָ���ٱ����գ�
ABS1=0.6  #PCP��¶����ϵ��

HSCVE=195.12  #���ʲ�-��Ƥ����ϵ��
HFOWell= 64.6  #Ƥ���������-ë�ҷ���ϵ��
HSCwell=HFOWell  #Ƥ���������-���ʲ����ϵ��

DSC = 23.4E-9   #���ʲ��е���Ч��ɢϵ����cm2/h- dm2��  
PFO= 105.5E-5  #ë�ҵ���͸ϵ����cm/h- dm��
u1=5.7E-5  #����м����Ƥ������ת�Ƶ��ٶȣ�cm/min��

DOSE= 1*bw/MWBPA*1000*1000  #����BPA��¶��
DOSE_d=1*bw/MWBPA*1000*1000*ABS  #TP��¶BPA��
DOSE_d1=DOSE*ABS1  #PCP��¶BPA��

##���ڱ�¶��
D.o <- 0 # (ug/d)
dose.O <- D.o/MWBPA # (nmol/kg/d)
period.O <- 3/60  # ��¶����ʱ�䣨h�� 
koa <- dose.O/period.O  #��¶�ٶȣ�nmol/h�� 

PBTKmod_bps_d1 <- function(t, y, parms)
{
  with (as.list(c(y, parms)),
        {
          if(t<EHRtime){kentero=0}else{kentero=EHRrate} 
          if(t<=exp_peroid){ OO_S_TP1=1} else { OO_S_TP1=0}
          
          CA <- Aplasma/Vplasma
          
          #######################################################
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
          
          #######################################################
          #######################################################  
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

##################################################################################
##############################_____VALIDATION___________##########################
###################### pars ######################################################

########################################################################

###################################init#####################################
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
########################################################################

############################ ####### #####################
oral_intakerate=0
para_bps_dermal1 <- unlist(c(data.frame(
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

############################ ####### #####################

zeit <- seq(0,24, 0.1) # (h) time

start1 <- Sys.time() 
try_bps_d2<-ode(y=yini_bps_dermal, func=PBTKmod_bps_d1, 
                times=zeit, parms=para_bps_dermal1, method="lsoda")

end1 <- Sys.time()   #��¼����ʱ��
difftime(end1, start1, units = "sec")  #��������ʱ��

plot(try_bps_d2[,"Aplasma"])


