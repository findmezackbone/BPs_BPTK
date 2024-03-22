library(expm)      #加载expm（计算矩阵指数）包

start1 <- Sys.time()   #记录程序起始时间

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


###X1-X9系数矩阵####
a1=c(2,-1,0,0,0,0,0,0,0)
a2=c(-1,2,-1,0,0,0,0,0,0)
a3=c(0,-1,2,-1,0,0,0,0,0)
a4=c(0,0,-1,2,-1,0,0,0,0)
a5=c(0,0,0,-1,2,-1,0,0,0)
a6=c(0,0,0,0,-1,2,-1,0,0)
a7=c(0,0,0,0,0,-1,2,-1,0)
a8=c(0,0,0,0,0,0,-1,2,-1)
a9=c(0,0,0,0,0,0,0,-1,2)
A1_9=rbind(a1,a2,a3,a4,a5,a6,a7,a8,a9)

a11=c(0,1,0,0,0,0,0,0,0)
a22=c(-1,0,1,0,0,0,0,0,0)
a33=c(0,-1,0,1,0,0,0,0,0)
a44=c(0,0,-1,0,1,0,0,0,0)
a55=c(0,0,0,-1,0,1,0,0,0)
a66=c(0,0,0,0,-1,0,1,0,0)
a77=c(0,0,0,0,0,-1,0,1,0)
a88=c(0,0,0,0,0,0,-1,0,1)
a99=c(0,0,0,0,0,0,0,-1,0)
A2_9=rbind(a11,a22,a33,a44,a55,a66,a77,a88,a99)

A_9=-DSC/SCDX/SCDX*A1_9+u1/2/SCDX*A2_9
AA_9 <- solve(A_9)

#c1 <- c(1,1,0,0)
#c2 <- c(1,1,1,0)
#c3 <- c(0,1,1,1)
#c4 <- c(0,0,1,1)
#cc <- rbind(c1,c2,c3,c4)


###初始值赋值####
X1_9_O=c(0,0,0,0,0,0,0,0,0)
X10_O=0
X11_O=0
X12_O=0
X13_O=0
X14_O=0
X15_O=0
X16_O=0
X17_O=0
X18_O=0
X19_O=0
X20_O=0
X21_O=0
X22_O=0
X23_O=0
X24_O=0
X25_O=0
X26_O=0
X27_O=0
X28_O=0

conc <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

###模型微分方程运算####
tau=0.005
expA=expm(A_9*tau)
expA2=(expA-diag(9))%*%AA_9
for (t in seq(0, 4*24, tau)){
  
  if(t<=add_peroid) {f1=DOSE_d/(add_peroid)} else {f1=0} 
  if(t<=0.1 && t>=0){f2=koa} else {f2=0}
  if(t<=exp_peroid){ON=1} else {ON=0}
  
  
  Y1_9=c((DSC/SCDX/SCDX*HSCwell/VWELL-u1/2/SCDX*HSCwell/VWELL)*X11_O,0,0,0,0,0,0,0,(DSC/SCDX/SCDX*HSCVE/VTVE+u1/2/SCDX*HSCVE/VTVE)*X12_O)
  X1_9=expA%*%X1_9_O+expA2%*%Y1_9
  X1_O=X1_9[1]
  X9_O=X1_9[9]
  X10=exp((-PFO*AEXP*FEXP/VTFO/HFOWell-Qskin*AEXP*0.25/BSA/VTFO/pskin)*0.005)*X10_O+((exp((-PFO*AEXP*FEXP/VTFO/HFOWell-Qskin*AEXP*0.25/BSA/VTFO/pskin)*0.005))-1)/(-PFO*AEXP*FEXP/VTFO/HFOWell-Qskin*AEXP*0.25/BSA/VTFO/pskin)*((PFO*AEXP*FEXP/VWELL)*X11_O+(Qskin*AEXP*0.25/BSA/Vplasma)*X17_O)
  X11=(exp((-(DSC*HSCwell/VWELL/SCDX-u1*HSCwell/VWELL)*AEXP*(1-FEXP)-AEXP*FEXP*PFO/VWELL)*0.005)*X11_O+
         (exp((-(DSC*HSCwell/VWELL/SCDX-u1*HSCwell/VWELL)*AEXP*(1-FEXP)-AEXP*FEXP*PFO/VWELL)*0.005)-1)/(-(DSC*HSCwell/VWELL/SCDX-u1*HSCwell/VWELL)*AEXP*(1-FEXP)-AEXP*FEXP*PFO/VWELL)*((DSC*AEXP*(1-FEXP)/SCDX)*X1_O+(AEXP*FEXP*PFO/HFOWell/VTFO)*X10_O+f1))*ON
  X12=exp(((-DSC*HSCVE/VTVE/SCDX-u1*HSCVE/VTVE)*AEXP*(1-FEXP)-Qskin*AEXP*0.75/BSA/VTVE/pskin)*0.005)*X12_O+(exp(((-DSC*HSCVE/VTVE/SCDX-u1*HSCVE/VTVE)*AEXP*(1-FEXP)-Qskin*AEXP*0.75/BSA/VTVE/pskin)*0.005)-1)/((-DSC*HSCVE/VTVE/SCDX-u1*HSCVE/VTVE)*AEXP*(1-FEXP)-Qskin*AEXP*0.75/BSA/VTVE/pskin)*((DSC*AEXP*(1-FEXP)/SCDX)*X9_O+(Qskin*AEXP*0.75/BSA/Vplasma)*X17_O)
  X13=exp(-(k0+ge)*0.005)*X13_O+(exp(-(k0+ge)*0.005)-1)/(-(k0+ge))*f2
  X14=exp((-Qskin*(1-AEXP/BSA)/(Vskin-VTSC-VTVE-VTFO)/pskin)*0.005)*X14_O+(exp((-Qskin*(1-AEXP/BSA)/(Vskin-VTSC-VTVE-VTFO)/pskin)*0.005)-1)/(-Qskin*(1-AEXP/BSA)/(Vskin-VTSC-VTVE-VTFO)/pskin)*((Qskin*(1-AEXP/BSA)/Vplasma)*X17_O)
  X15=exp((-Qfat/Vfat/pfat)*0.005)*X15_O+(exp((-Qfat/Vfat/pfat)*0.005)-1)/(-Qfat/Vfat/pfat)*(Qfat/Vplasma*X17_O)
  X16=exp((-Qgonad/Vgonad/pgonad)*0.005)*X16_O+(exp((-Qgonad/Vgonad/pgonad)*0.005)-1)/(-Qgonad/Vgonad/pgonad)*(Qgonad/Vplasma*X17_O)
  X17=exp((-QC/Vplasma)*0.005)*X17_O+(exp((-QC/Vplasma)*0.005)-1)/(-QC/Vplasma)*(((QC-kurinebpa)*Qskin*AEXP/BSA*0.25/QC/VTFO/pskin)*X10_O+((QC-kurinebpa)*Qskin*AEXP/BSA*0.75/QC/VTVE/pskin)*X12_O+((QC-kurinebpa)*Qskin*(1-AEXP/BSA)/QC/(Vskin-VTFO-VTSC-VTVE)/pskin)*X14_O+
                                                                                   ((QC-kurinebpa)*Qfat/QC/Vfat/pfat)*X15_O+((QC-kurinebpa)*Qgonad/QC/Vgonad/pgonad)*X16_O+(QC-kurinebpa)*Qbrain/QC/Vbrain/pbrain*X18_O+(QC-kurinebpa)*Qrich/QC/Vrich/prich*X19_O+
                                                                                   (QC-kurinebpa)*Qslow/QC/Vslow/pslow*X20_O+(QC-kurinebpa)*Qliver/QC/Vliver/pliver*X24_O)
  X18=exp((-Qbrain/Vbrain/pbrain)*0.005)*X18_O+(exp((-Qbrain/Vbrain/pbrain)*0.005)-1)/(-Qbrain/Vbrain/pbrain)*(Qbrain/Vplasma*X17_O)
  X19=exp((-Qrich/Vrich/prich)*0.005)*X19_O+(exp((-Qrich/Vrich/prich)*0.005)-1)/(-Qrich/Vrich/prich)*(Qrich/Vplasma*X17_O)
  X20=exp((-Qslow/Vslow/pslow)*0.005)*X20_O+(exp((-Qslow/Vslow/pslow)*0.005)-1)/(-Qslow/Vslow/pslow)*(Qslow/Vplasma*X17_O)
  X21=exp(-kGIing*0.005)*X21_O+(exp(-kGIing*0.005)-1)/(-kGIing)*(vmaxgutg*X23_O/(enterocytes*kmgutg+X23_O+X23_O*X23_O/enterocytes/ksigutg))
  X22=exp(-kGIins*0.005)*X22_O+(exp(-kGIins*0.005)-1)/(-kGIins)*(vmaxguts*X23_O/(enterocytes*kmguts+X23_O))
  X23=exp(-k1*0.005)*X23_O+(exp(-k1*0.005)-1)/(-k1)*(ge*X13_O-(vmaxgutg*X23_O/(enterocytes*kmgutg+X23_O+X23_O*X23_O/enterocytes/ksigutg))-(vmaxguts*X23_O/(enterocytes*kmguts+X23_O)))
  X24=exp((-Qliver/Vliver/pliver)*0.005)*X24_O+(exp((-Qliver/Vliver/pliver)*0.005)-1)/(-Qliver/Vliver/pliver)*(k0*X13_O+Qliver/Vplasma*X17_O+k1*X23_O+kenterobpag*X25_O+kenterobpas*X26_O-vmaxliver*X24_O/(Vliver*pliver*kmliver+X24_O)-vmaxlivers*X24_O/(Vliver*pliver*kmlivers+X24_O))
  X25=exp(-(EHRrate+k4_IV+kenterobpag)*0.005)*X25_O+(exp(-(EHRrate+k4_IV+kenterobpag)*0.005)-1)/(-(EHRrate+k4_IV+kenterobpag))*(met2g*kGIing*X21_O+met2g*vmaxliver*X24_O/(Vliver*pliver*kmliver+X24_O))
  X26=exp(-(EHRrate+k4_IV+kenterobpas)*0.005)*X26_O+(exp(-(EHRrate+k4_IV+kenterobpas)*0.005)-1)/(-(EHRrate+k4_IV+kenterobpas))*(met2s*kGIins*X22_O+met2s*vmaxlivers*X24_O/(Vliver*pliver*kmlivers+X24_O))
  X27=exp(-kurinebpag/(Vbodyg+10^(-34))*0.005)*X27_O+(exp(-kurinebpag/(Vbodyg+10^(-34))*0.005)-1)/(-kurinebpag/(Vbodyg+10^(-34)))*(met1g*kGIing*X21_O+EHRrate*X25_O+met1g*vmaxliver*X24_O/(Vliver*pliver*kmliver+X24_O))
  X28=exp(-kurinebpas/(Vbodys+10^(-34))*0.005)*X28_O+(exp(-kurinebpas/(Vbodys+10^(-34))*0.005)-1)/(-kurinebpas/(Vbodys+10^(-34)))*(met1s*kGIins*X22_O+EHRrate*X26_O+met1s*vmaxlivers*X24_O/(Vliver*pliver*kmlivers+X24_O))
  
  X1_9_O=X1_9
  X10_O=X10
  X11_O=X11
  X12_O=X12
  X13_O=X13
  X14_O=X14
  X15_O=X15
  X16_O=X16
  X17_O=X17
  X18_O=X18
  X19_O=X19
  X20_O=X20
  X21_O=X21
  X22_O=X22
  X23_O=X23
  X24_O=X24
  X25_O=X25
  X26_O=X26
  X27_O=X27
  X28_O=X28
  Xt <- c(X1_9_O,X10_O,X11_O,X12_O,X13_O,X14_O,X15_O,X16_O,X17_O,X18_O,X19_O,X20_O,X21_O,X22_O,X23_O,X24_O,X25_O,X26_O,X27_O,X28_O)
  conc <- cbind(conc,Xt)
}

end1 <- Sys.time()   #记录结束时间
difftime(end1, start1, units = "sec")  #计算运行时间

plot(conc[18,])  #作图展示



