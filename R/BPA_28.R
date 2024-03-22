library(deSolve)
gender=1  #性别，1为女性，2为男性 
bw= 60   #体重，kg
#各组织器官的血液流速（计算化学物在不同组织器官间转移量所用）
QCC = 5.9  #心脏血液流速基本参数（L/min）
QliverC = 0.24  #肝脏血液流速基本参数（L/min）
QskinC =0.044  #皮肤血液流速基本参数（L/min）
QgonadC=0.00018  #性腺血液流速基本参数（L/min）
QfatC = 0.074  #脂肪血液流速基本参数（L/min）
QbrainC <- 0.11  #大脑血液流速基本参数（L/min）
QmuscleC <-0.14  #肌肉血液流速基本参数（L/min）

QC <- QCC*60  #心脏血液流速（L/h）
Qfat <- QfatC*QC  #脂肪血液流速（L/h）
Qliver <- QliverC*QC  #肝脏血液流速（L/h）
Qgonad <- QgonadC*QC  #性腺血液流速（L/h）
Qbrain <- QbrainC*QC  #大脑血液流速（L/h）
Qskin=QskinC*QC  #皮肤血液流速（L/h）
Qslow <- QmuscleC*QC  #缓慢灌流组织血液流速（L/h）
Qrich <- QC-Qliver-Qbrain-Qfat-Qgonad-Qslow-Qskin  #快速灌流组织血液流速（L/h）

#各组织器官的体积参数
VplasmaC <-0.04   #血浆体积基本参数（L/kg）
VfatC <- 0.32   #脂肪体积基本参数（L/kg）
VliverC <- 0.023   #肝脏体积基本参数（L/kg）
VbrainC <- 0.022   #大脑体积基本参数（L/kg）
VgonadC <-0.00018   #性腺体积基本参数（L/kg）
VskinC <-  0.038   #皮肤体积基本参数（L/kg）
VrichC <- 0.059   #快速灌流组织体积基本参数（L/kg）
VbodygC <- 0.04  #参与BPAG分布组织体积基本参数（L/kg）
VbodysC <- VplasmaC  #参与BPAS分布组织体积基本参数（L/kg）
enterocytes <- 0.1223  #小肠体积（L）

Vliver <- VliverC*bw  #肝脏体积（L）
Vfat <- VfatC*bw  #脂肪体积（L）
Vgonad <- VgonadC*bw  #性腺体积（L）
Vplasma <- VplasmaC*bw  #血浆体积（L）
Vbrain <- VbrainC*bw  #大脑体积（L）
Vskin=VskinC*bw  #皮肤体积（L）
Vrich<- VrichC*bw  #快速灌流组织体积（L）

Vslow <- bw-Vliver-Vfat-Vgonad-Vplasma-Vbrain-Vrich  #缓慢灌流组织体积（L）
Vbodyg<- VbodygC*bw  #参与BPAG分布组织体积（L）
Vbodys <- VbodysC*bw  #参与BPAS分布组织体积（L）

#二、引入BPA理化参数##################################################
MWBPA <- 228.28  #BPA的摩尔质量，转化暴露量ng为mol用
#BPA的组织-血浆分配系数（计算化学物在不同组织器官间转移量所用）
pliver<- 0.73   #肝脏-血浆分配系数
pfat <- 5.0  #肝脏-血浆分配系数
pslow<- 2.7  #缓慢灌流组织-血浆分配系数
prich<- 2.8   #快速灌流组织-血浆分配系数
pgonad<- 2.6   #性腺-血浆分配系数
pbrain<- 2.8  #大脑-血浆分配系数
pskin<- 2.15  #皮肤-血浆分配系数
#BPA的ADME参数（吸收、分布、代谢、排泄参数）
#吸收参数
geC<- 3.5  #胃排空系数，口服给药BPA由胃转移至小肠的系数（1/h/bw^-0.25）
k0C<- 0  #口服给药BPA从胃进入肝脏的系数（1/h/bw^-0.25）
k1C<-2.1  #口服给药BPA从小肠进入肝脏的系数（1/h/bw^-0.25）
k4C<- 0  #口服给药BPA从小肠的粪便消除系数（1/h/bw^-0.25）
kGIingC <- 50  #口服给药BPAG从肠到血中的系数（1/h/bw^-0.25）
kGIinsC <- 50  #口服给药BPAS从肠到血中的系数（1/h/bw^-0.25）

ge <- geC/bw^0.25  #胃排空系数，口服给药BPA由胃转移至小肠的系数（1/h）
k0 <- k0C/bw^0.25  #口服给药BPA从胃进入肝脏的系数（1/h）
k1 <- k1C/bw^0.25  #口服给药BPA从小肠进入肝脏的系数（1/h）
k4 <- k4C/bw^0.25  #口服给药BPA从小肠的粪便消除系数（1/h ）
kGIing <- kGIingC/bw^0.25  #口服给药BPAG从肠到血中的系数（1/h）
kGIins <- kGIinsC/bw^0.25  #口服给药BPAS从肠到血中的系数（1/h）


kmgutg<- 58400  #肠道中BPA葡萄苷酸化的米氏常数Km（nmol）
vmaxgutgC<- 361  #肠道中BPA葡萄苷酸化的最大反应速度Vmax（nmol/h/kg bw）
fgutg <- 1  #肠道中BPA葡萄苷酸化反应的校正因子（1为存在，0为不存在）
kmguts <- 0.001    #肠道中BPA硫酸盐化的米氏常数Km（nmol）
vmaxgutsC <- 0.00001  #肠道中BPA硫酸盐化的最大反应速度Vmax（nmol/h/bw^0.75）
ksigutg <- 711000  #肠道中BPA代谢的底物抑制常数（nmol）
fguts <- 0  #肠道中BPA硫酸盐化反应的校正因子（1为存在，0为不存在）
kmliver <- 45800  #肝脏中BPA葡萄苷酸化的米氏常数Km（nmol）
vmaxliverC <- 9043.2  #肝脏中BPA葡萄苷酸化的最大反应速度Vmax（nmol/h/g liver）
fliverg <- 1  #肝脏中BPA葡萄苷酸化反应的校正因子（1为存在，0为不存在）
kmlivers <- 10100  #肝脏中BPA硫酸盐化的米氏常数Km（nmol）
vmaxliversC <- 149  #肝脏中BPA硫酸盐化的最大反应速度Vmax（nmol/h/g liver）
flivers <- 1   #肝脏中BPA硫酸盐化反应的校正因子（1为存在，0为不存在）


vmaxgutgCnew <- vmaxgutgC*bw/(bw^0.75) 
vmaxgutg <- vmaxgutgCnew*fgutg*bw^0.75 #肠道中BPA葡萄苷酸化的最大反应速度Vmax（nmol/h）
vmaxguts <- vmaxgutsC*fguts*bw^0.75  #肠道中BPA硫酸盐化的最大反应速度Vmax（nmol/h）
vmaxliverCnew <- vmaxliverC*VliverC*1000  
vmaxliverCnew <- vmaxliverCnew*bw/(bw^0.75)
vmaxliver <- vmaxliverCnew*fliverg*bw^0.75  #肝脏中BPA葡萄苷酸化的最大反应速度Vmax（nmol/h）
vmaxliversCnew <- vmaxliversC*VliverC*1000
vmaxliversCnew <- vmaxliversCnew*bw/(bw^0.75)
vmaxlivers <- vmaxliversCnew*flivers*bw^0.75 #肝脏中BPA葡萄苷酸化的最大反应速度Vmax（nmol/h）




met1g <- 0.9  #肝脏中BPAG进入血中的比例（为1时则不存在肝肠循环）
met1s <- 1  #肝脏中BPAS进入血中的比例（为1时则不存在肝肠循环）
met2g <- 1.0-met1g   #肝脏中BPAG进入肝肠循环的比例
met2s <- 1.0-met1s   #肝脏中BPAS进入肝肠循环的比例
EHRtime <- 0  #肝肠循环发生起始时间
EHRrateC <- 0.2  #BPAG肝肠循环的速率基本参数（1/h/bw^-0.25）
k4C_IV <- 0  #BPAG在肝肠循环中的粪便消除系数（1/h/bw^-0.25）
kenterobpagC <- 0.2  #BPAG肝肠循环使得BPA发生循环的速率基本参数（1/h/bw^-0.25）
kenterobpasC <- 0.0  #BPAS肝肠循环使得BPA发生循环的速率基本参数（1/h/bw^-0.25）

EHRrate <- EHRrateC/(bw^0.25)  #BPAG肝肠循环的速率（1/h）
kentero=EHRrate
k4_IV <- k4C_IV/bw^0.25    #BPAG在肝肠循环中的粪便消除系数（1/h）
kenterobpag<- kenterobpagC/bw^0.25  #BPAG肝肠循环使得BPA发生循环的速率（1/h）
kenterobpas <- kenterobpasC/bw^0.25  #BPAS肝肠循环使得BPA发生循环的速率（1/h）


kurinebpaC <- 0.06  #BPA尿液排泄基本参数（L/h/bw^0.75）
kurinebpagC <- 0.35  #BPAG尿液排泄基本参数（L/h/bw^0.75）
kurinebpasC <- 0.03  #BPAS尿液排泄基本参数（L/h/bw^0.75）
vreabsorptiongC <- 0 #尿液排泄中BPAG的肾脏重吸收的最大反应速度Vmax（nmol/h/bw^0.75）
vreabsorptionsC <- 0 #尿液排泄中BPAS的肾脏重吸收的最大反应速度Vmax（nmol/h/bw^0.75）
kreabsorptiong <- 9200  #尿液排泄中BPAG的肾脏重吸收的米氏常数Km（nmol/L）
kreabsorptions <- 9200  #尿液排泄中BPAS的肾脏重吸收的米氏常数Km（nmol/L）


kurinebpa <- kurinebpaC*bw^0.75  #BPA尿液排泄参数（L/h）
kurinebpag <- kurinebpagC*bw^0.75  #BPAG尿液排泄参数（L/h）
kurinebpas <- kurinebpasC*bw^0.75  #BPAS尿液排泄参数（L/h）
vreabsorptiong <- vreabsorptiongC*bw^0.75  #尿液排泄中BPAG的肾脏重吸收的最大反应速度Vmax（nmol/h）
vreabsorptions <- vreabsorptionsC*bw^0.75  #尿液排泄中BPAS的肾脏重吸收的最大反应速度Vmax（nmol/h） 


##皮肤参数

#皮肤暴露场景1：擦拭纸（TP）手指暴露#
BSA= 1.6*100 #人体皮肤表面积（m2-dm2）
epi_finger=460  #手指表皮厚度（um）
TSC = 185/100000  #手指角质层厚度（um-dm） 
TVE = (epi_finger-185)/100000  #手指活性表皮层厚度（um-dm）
SCDX = TSC / 10  #角质层亚层厚度（分为11层，每层厚度为scdx）（dm）
skin_finger=1095+460  #手指皮肤厚度（um）
TFO=388/560*skin_finger/100000   #手指毛囊厚度（um-dm）
AEXP=20/100  #暴露皮肤面积（cm2- dm2）
FEXP=0.01  #暴露皮肤中毛囊面积分数（1-FEXP为角质层面积分数）
add_peroid=1/60  #涂抹化学物时长（h）（接触充分浓度化学物，引入外暴露Rdose持续时间，到达add_period后停止接触擦拭纸，停止外暴露引入，即dose=0）
exp_peroid=1   #暴露化学物时长（h）（化学物被涂抹后，皮肤表面接触外暴露持续时间，到达exp_period后清洗皮肤，停止暴露，即SSD被清空）
VWELL=AEXP*0.01  #暴露皮肤表面沉积体积（面积×厚度，L）
VTSC = AEXP * TSC   # 暴露皮肤角质层体积（面积×厚度，L）
VTVE = AEXP * TVE    # 暴露皮肤活性表皮层体积（面积×厚度，L）
VTFO=AEXP*FEXP*TFO  #暴露皮肤毛囊体积（面积×厚度，L）

#皮肤暴露场景2：个人护理产品（PCPs）前臂暴露#
BSA= 1.6*100  #人体皮肤表面积（m2-dm2）
epi_forarm=1000  #前臂表皮厚度（um）
TSC = 50/100000  #前臂角质层厚度（um-dm） 
TVE = (epi_forarm-15)/100000  #前臂活性表皮层厚度（um-dm）
SCDX = TSC / 10  #角质层亚层厚度（分为11层，每层厚度为scdx）（dm）
skin_forearm=1050  #前臂皮肤厚度（um）
TFO=388/560*skin_forearm/100000   #前臂毛囊厚度（um-dm）
AEXP=0.9* BSA   #暴露皮肤面积（dm2）
FEXP=0.01  #暴露皮肤中毛囊面积分数（1-FEXP为角质层面积分数）
add_peroid1=1/60   #涂抹化学物时长（h）
exp_peroid1=12    #暴露化学物时长（h）
VWELL=AEXP*0.01  #暴露皮肤表面沉积体积（面积×厚度，L）
VTSC = AEXP * TSC   # 暴露皮肤角质层体积（面积×厚度，L）
VTVE = AEXP * TVE    # 暴露皮肤活性表皮层体积（面积×厚度，L）
VTFO=AEXP*FEXP*TFO  #暴露皮肤毛囊体积（面积×厚度，L）


ABS=0.2828*0.9618  #TP暴露吸收系数（TP转移到手指、再被吸收）
ABS1=0.6  #PCP暴露吸收系数

HSCVE=195.12  #角质层-表皮分配系数
HFOWell= 64.6  #皮肤表面沉积-毛囊分配系数
HSCwell=HFOWell  #皮肤表面沉积-角质层分配系数

DSC = 23.4E-9   #角质层中的有效扩散系数（cm2/h- dm2）  
PFO= 105.5E-5  #毛囊的渗透系数（cm/h- dm）
u1=5.7E-5  #由脱屑而向皮肤表面转移的速度（cm/min）

DOSE= 1*bw/MWBPA*1000*1000  #经口BPA暴露量
DOSE_d=1*bw/MWBPA*1000*1000*ABS  #TP暴露BPA量
DOSE_d1=DOSE*ABS1  #PCP暴露BPA量

##经口暴露量
D.o <- 0 # (ug/d)
dose.O <- D.o/MWBPA # (nmol/kg/d)
period.O <- 3/60  # 暴露持续时间（h） 
koa <- dose.O/period.O  #暴露速度（nmol/h） 

yini_bps_dermal <- unlist(c(data.frame(
  
  x1=0,x2=0,x3=0,
  x4=0,x5=0,x6=0,x7=0,x8=0,x9=0,x10=0,x11=0,
  x12=0,x13=0,
  
  x14= 0, 
  x15 = 0, 
  x16 = 0, 
  x17 = 0,
  x18 = 0, 
  x19 = 0, 
  x20 = 0, 
  x21 = 0, x22 = 0, x23 = 0,x24 = 0, 
  x25=0,
  x26 = 0, 
  x27 = 0, 
  x28 = 0)))

para_bps_dermal <- unlist(c(data.frame(
  QC,Qfat,Qliver,Qgonad,Qbrain,
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
        
          if(t<=exp_peroid){ ON=1} else { ON=0}
          # Dermal Exposure Well
          if(t<=add_peroid) f1=DOSE_d/(add_peroid) else f1=0 

          dx1 = (DSC/(SCDX**2) + u1/(2*SCDX))*x2-(2*DSC)/(SCDX**2)*x1+((DSC*HSCwell)/(VWELL*(SCDX**2))-(u1*HSCwell)/(VWELL*2*SCDX))*(x11*ON)
          dx2=(DSC/(SCDX**2) - u1/(2*SCDX))*x1-(2*DSC)/(SCDX**2)*x2+(DSC/(SCDX**2) +u1/(2*SCDX))*x3
          dx3=(DSC/(SCDX**2) - u1/(2*SCDX))*x2-(2*DSC)/(SCDX**2)*x3+(DSC/(SCDX**2) +u1/(2*SCDX))*x4
          dx4=(DSC/(SCDX**2) - u1/(2*SCDX))*x3-(2*DSC)/(SCDX**2)*x4+(DSC/(SCDX**2) +u1/(2*SCDX))*x5
          dx5=(DSC/(SCDX**2) - u1/(2*SCDX))*x4-(2*DSC)/(SCDX**2)*x5+(DSC/(SCDX**2) +u1/(2*SCDX))*x6
          dx6=(DSC/(SCDX**2) - u1/(2*SCDX))*x5-(2*DSC)/(SCDX**2)*x6+(DSC/(SCDX**2) +u1/(2*SCDX))*x7
          dx7=(DSC/(SCDX**2) - u1/(2*SCDX))*x6-(2*DSC)/(SCDX**2)*x7+(DSC/(SCDX**2) +u1/(2*SCDX))*x8
          dx8=(DSC/(SCDX**2) - u1/(2*SCDX))*x7-(2*DSC)/(SCDX**2)*x8+(DSC/(SCDX**2) +u1/(2*SCDX))*x9
          dx9=(DSC/(SCDX**2) - u1/(2*SCDX))*x8-(2*DSC)/(SCDX**2)*x9+((DSC*HSCVE)/(VTVE*(SCDX**2))+(u1*HSCVE)/(VTVE*2*SCDX))*x12
          #接下来是第10至20个方程
          dx10=-((PFO*AEXP*FEXP)/(VTFO*HFOWell)+(Qskin*AEXP*0.25)/(BSA*VTFO*pskin))*x10+(PFO*AEXP*FEXP)/VWELL*(x11*ON)+(Qskin*AEXP*0.25)/(BSA*Vplasma)*x17
          dx11=((DSC*AEXP*(1-FEXP))/SCDX*x1+(PFO*AEXP*FEXP)/(VTFO*HFOWell)*x10-(((DSC*HSCwell)/(VWELL*SCDX)-(u1*HSCwell)/VWELL)*AEXP*(1-FEXP)-(PFO*AEXP*FEXP)/VWELL)*x11+f1)*ON
          dx12=(DSC*AEXP*(1-FEXP))/SCDX*x9+(((-DSC*HSCVE)/(VTVE*SCDX)-(u1*HSCVE)/VTVE)*AEXP*(1-FEXP)-(Qskin*AEXP*0.75)/(BSA*VTVE*pskin))*x12+(Qskin*AEXP*0.75)/(BSA*Vplasma)*x17
          dx13=-(k0+ge)*x13+f2
          dx14=(-Qskin*(1-AEXP/BSA))/((Vskin-VTSC-VTVE-VTFO)*pskin)*x14+(Qskin*(1-AEXP/BSA))/Vplasma*x17
          dx15=(-Qfat)/(Vfat*pfat)*x15+Qfat/Vplasma*x17
          dx16=(-Qgonad)/(Vgonad*pgonad)*x16+Qgonad/Vplasma*x17
          dx17=((QC-kurinebpa)*Qskin*(AEXP/BSA)*0.25)/(QC*VTFO*pskin)*x10+((QC-kurinebpa)*Qskin*(AEXP/BSA)*0.75)/(QC*VTVE*pskin)*x12+((QC-kurinebpa)*Qskin*(1-AEXP/BSA))/(QC*(Vskin-VTSC-VTVE-VTFO)*pskin)*x14+((QC-kurinebpa)*Qfat)/(QC*Vfat*pfat)*x15+((QC-kurinebpa)*Qgonad)/(QC*Vgonad*pgonad)*x16-QC/Vplasma *x17+((QC-kurinebpa)*Qbrain)/(QC*Vbrain*pbrain) *x18+((QC-kurinebpa)*Qrich)/(QC*Vrich*prich) *x19+((QC-kurinebpa)*Qslow)/(QC*Vslow*pslow) *x20+((QC-kurinebpa)*Qliver)/(QC*Vliver*pliver) *x24
          dx18=(-Qbrain)/(Vbrain*pbrain) *x18+Qbrain/Vplasma *x17
          dx19=(-Qrich)/(Vrich*prich) *x19+Qrich/Vplasma  *x17
          dx20=(-Qslow)/(Vslow*pslow) *x20+Qslow/Vplasma  *x17
          #接下来是第21至28个方程
          dx21=-kGIing*x21+(vmaxgutg*x23)/(enterocytes*kmgutg+x23+(x23**2)/(enterocytes*ksigutg))
          dx22=-kGIins*x22+(vmaxguts*x23)/(enterocytes*kmguts+x23)
          dx23=ge*x13-k1*x23-(vmaxgutg*x23)/(enterocytes*kmgutg+x23+(x23**2)/(enterocytes*ksigutg))-(vmaxguts*x23)/(enterocytes*kmguts+x23)
          dx24=k0*x13-Qliver/Vplasma  *x17+k1*x23-Qliver/(Vliver*pliver) *x24+kenterobpag*x25+kenterobpas*x26-(vmaxliver*x24)/(Vliver*pliver*kmliver+x24)-(vmaxlivers*x24)/(Vliver*pliver*kmlivers+x24)
          dx25=met2g*kGIing*x21-(EHRrate+k4_IV+kenterobpag)*x25+(met2g*vmaxliver*x24)/(Vliver*pliver*kmliver+x24)
          dx26=met2s*kGIins*x22-(EHRrate+k4_IV+kenterobpas)*x26+(met2s*vmaxlivers*x24)/(Vliver*pliver*kmlivers+x24)
          dx27=met1g*kGIing*x21+EHRrate*x25-kurinebpag/(Vbodyg+10**(-34) ) *x27+(met1g*vmaxliver*x24)/(Vliver*pliver*kmliver+x24)
          dx28=met1s*kGIins*x22+EHRrate*x26-kurinebpas/(Vbodys+10**(-34) ) *x28+(met1s*vmaxlivers*x24)/(Vliver*pliver*kmlivers+x24)
          
          
          
          dydt <-
            c(dx1,
              dx2,
              dx3,
              dx4,
              dx5,
              dx6,
              dx7,
              dx8,
              dx9,
              #接下来是第10至20个方程
              dx10,
              dx11,
              dx12,
              dx13,
              dx14,
              dx15,
              dx16,
              dx17,
              dx18,
              dx19,
              dx20,
              #接下来是第21至28个方程
              dx21,
              dx22,
              dx23,
              dx24,
              dx25,
              dx26,
              dx27,
              dx28)
          
          
          res <- list(dydt)
          return(res)
        })}

zeit <- seq(0,96, 0.005) # (h) time
BPA1<-ode(y=yini_bps_dermal,times=zeit,  func=PBTKmod_bps, 
          parms=para_bps_dermal, method="lsoda")
plot(BPA1[,17])
write.csv(BPA1,"D://1st//Project_pharmacy//R_language//R_28_BPA_Result.csv")
