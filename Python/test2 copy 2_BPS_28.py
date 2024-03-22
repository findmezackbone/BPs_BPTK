import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


para_BP_individual = np.loadtxt(open("D://1st/Project_pharmacy//R_language//para_BP_individual.csv"),delimiter=",",skiprows=1,usecols=[2,3,4,5])
#读取受试者生理参数(第五位受试者数据有缺失，暂取四位的数据)

i = -1 #用于控制受试者的编号，可取值-1，0,1,2
gender=para_BP_individual[1,i+1]  #1 for male, 2 for female
bw= para_BP_individual[0,i+1]   #体重，kg 


#各组织器官的血液流速（计算化学物在不同组织器官间转移量所用）
QCC = para_BP_individual[2,i+1]	#心脏血液流速基本参数（L/min）
QliverC = para_BP_individual[3,i+1]	#肝脏血液流速基本参数（L/min）
QskinC = para_BP_individual[6,i+1] #皮肤血液流速基本参数（L/min）
QgonadC=para_BP_individual[7,i+1] #性腺血液流速基本参数（L/min）
QfatC = para_BP_individual[8,i+1] #脂肪血液流速基本参数（L/min）
QbrainC =para_BP_individual[21,i+1] #大脑血液流速基本参数（L/min）
QmuscleC =para_BP_individual[23,i+1] #肌肉血液流速基本参数（L/min）



QC = QCC*60  #心脏血液流速（L/h）
Qfat = QfatC*QC  #脂肪血液流速（L/h）
Qliver = QliverC*QC  #肝脏血液流速（L/h）
Qgonad = QgonadC*QC  #性腺血液流速（L/h）
Qbrain = QbrainC*QC  #大脑血液流速（L/h）
Qskin=QskinC*QC  #皮肤血液流速（L/h）
Qslow = QmuscleC*QC  #缓慢灌流组织血液流速（L/h）
Qrich = QC-Qliver-Qbrain-Qfat-Qgonad-Qslow-Qskin  #快速灌流组织血液流速（L/h）

#各组织器官的体积参数

VplasmaC =para_BP_individual[15,i+1] #血浆体积基本参数（L/kg）
VfatC =para_BP_individual[14,i+1] #脂肪体积基本参数（L/kg）
VliverC=para_BP_individual[9,i+1] #肝脏体积基本参数（L/kg）
VbrainC =para_BP_individual[24,i+1] #大脑体积基本参数（L/kg）
VgonadC =para_BP_individual[13,i+1]  #性腺体积基本参数（L/kg）
VskinC =para_BP_individual[12,i+1] #皮肤体积基本参数（L/kg）
VrichC =para_BP_individual[25,i+1] #快速灌流组织体积基本参数（L/kg）

VbodygC = VplasmaC  #参与BPS-G分布组织体积基本参数（L/kg）
VbodysC = VplasmaC  #参与BPS-S分布组织体积基本参数（L/kg）
enterocytes = 0.1223  #小肠体积（L）
#这一段都是测量的试验者实际数据


Vliver = VliverC*bw  #肝脏体积（L）
Vfat = VfatC*bw  #脂肪体积（L）
Vgonad = VgonadC*bw  #性腺体积（L）
Vplasma = VplasmaC*bw  #血浆体积（L）
Vbrain = VbrainC*bw  #大脑体积（L）
Vskin=VskinC*bw  #皮肤体积（L）
Vrich= VrichC*bw  #快速灌流组织体积（L）

Vslow = bw-Vliver-Vfat-Vgonad-Vplasma-Vbrain-Vrich  #缓慢灌流组织体积（L）
Vbodyg= VbodygC*bw  #参与BPS-G分布组织体积（L）
Vbodys = VbodysC*bw  #参与BPS-S分布组织体积（L）

#二、引入BPS理化参数##################################################
MWBPS = 250.27  #BPS的摩尔质量，转化暴露量ng为mol用

#2024/03/13凌晨整理到这里，底下的参数还没有根据张怡宁论文中参数来改动，白天继续

#以下是BPS的组织-血浆分配系数（计算化学物在不同组织器官间转移量所用）
pliver= 0.846   #肝脏-血浆分配系数
pfat = 0.435  #肝脏-血浆分配系数
pslow= 0.881  #缓慢灌流组织-血浆分配系数
prich= 0.810   #快速灌流组织-血浆分配系数
pgonad= 0.843   #性腺-血浆分配系数
pbrain= 0.810  #大脑-血浆分配系数
pskin= 1.43  #皮肤-血浆分配系数
#这一段都是与测试者无关的人体固定常数（但与BPS有关，是BPS情况下特有的数据）




#BPS的ADME参数（吸收、分布、代谢、排泄参数）
#吸收参数
geC= 3.5  #胃排空系数，口服给药BPS由胃转移至小肠的系数（1/h/bw**-0.25）
k0C= 0  #口服给药BPS从胃进入肝脏的系数（1/h/bw**-0.25）
k1C=5  #口服给药BPS从小肠进入肝脏的系数（1/h/bw**-0.25）
k4C= 0  #口服给药BPS从小肠的粪便消除系数（1/h/bw**-0.25）
kGIingC = 50  #口服给药BPSG从肠到血中的系数（1/h/bw**-0.25）
kGIinsC = 50  #口服给药BPSS从肠到血中的系数（1/h/bw**-0.25）

ge = geC/(bw**0.25)  #胃排空系数，口服给药BPS由胃转移至小肠的系数（1/h）
k0 = k0C/(bw**0.25)  #口服给药BPS从胃进入肝脏的系数（1/h）
k1 = k1C/(bw**0.25)  #口服给药BPS从小肠进入肝脏的系数（1/h）
k4 = k4C/(bw**0.25)  #口服给药BPS从小肠的粪便消除系数（1/h ）
kGIing = kGIingC/(bw**0.25)  #口服给药BPSG从肠到血中的系数（1/h）
kGIins = kGIinsC/(bw**0.25)  #口服给药BPSS从肠到血中的系数（1/h）


kmgutg= 555000  #肠道中BPS葡萄苷酸化的米氏常数Km（nmol）
vmaxgutgC= 563  #肠道中BPS葡萄苷酸化的最大反应速度Vmax（nmol/h/kg bw）
fgutg = 1  #肠道中BPS葡萄苷酸化反应的校正因子（1为存在，0为不存在）
kmguts = 0.001    #肠道中BPS硫酸盐化的米氏常数Km（nmol）
vmaxgutsC = 0.001  #肠道中BPS硫酸盐化的最大反应速度Vmax（nmol/h/bw**0.75）
ksigutg = 711000  #肠道中BPS代谢的底物抑制常数（nmol）
fguts = 0  #肠道中BPS硫酸盐化反应的校正因子（1为存在，0为不存在）
kmliver = 446000  #肝脏中BPS葡萄苷酸化的米氏常数Km（nmol）
vmaxliverC = 7810  #肝脏中BPS葡萄苷酸化的最大反应速度Vmax（nmol/h/g liver）
fliverg = 1  #肝脏中BPS葡萄苷酸化反应的校正因子（1为存在，0为不存在）
kmlivers = 10100  #肝脏中BPS硫酸盐化的米氏常数Km（nmol）
vmaxliversC = 149  #肝脏中BPS硫酸盐化的最大反应速度Vmax（nmol/h/g liver）
flivers = 1   #肝脏中BPS硫酸盐化反应的校正因子（1为存在，0为不存在）


vmaxgutgCnew = vmaxgutgC*bw/(bw**0.75) 
vmaxgutg = vmaxgutgCnew*fgutg*(bw**0.75) #肠道中BPS葡萄苷酸化的最大反应速度Vmax（nmol/h）
vmaxguts = vmaxgutsC*fguts*(bw**0.75)  #肠道中BPS硫酸盐化的最大反应速度Vmax（nmol/h）
vmaxliverCnew = vmaxliverC*VliverC*1000  
vmaxliverCnew = vmaxliverCnew*bw/(bw**0.75)
vmaxliver = vmaxliverCnew*fliverg*(bw**0.75)  #肝脏中BPS葡萄苷酸化的最大反应速度Vmax（nmol/h）
vmaxliversCnew = vmaxliversC*VliverC*1000
vmaxliversCnew = vmaxliversCnew*bw/(bw**0.75)
vmaxlivers = vmaxliversCnew*flivers*(bw**0.75) #肝脏中BPS葡萄苷酸化的最大反应速度Vmax（nmol/h）




met1g = 0.33  #肝脏中BPSG进入血中的比例（为1时则不存在肝肠循环）
met1s = 1  #肝脏中BPSS进入血中的比例（为1时则不存在肝肠循环）
met2g = 1.0-met1g   #肝脏中BPSG进入肝肠循环的比例
met2s = 1.0-met1s   #肝脏中BPSS进入肝肠循环的比例
EHRtime = 0  #肝肠循环发生起始时间
EHRrateC = 2  #BPSG肝肠循环的速率基本参数（1/h/bw**-0.25）
k4C_IV = 0  #BPSG在肝肠循环中的粪便消除系数（1/h/bw**-0.25）
kenterobpagC = 0.35  #BPSG肝肠循环使得BPS发生循环的速率基本参数（1/h/bw**-0.25）
kenterobpasC = 0.0  #BPSS肝肠循环使得BPS发生循环的速率基本参数（1/h/bw**-0.25）

EHRrate = EHRrateC/(bw**0.25)  #BPSG肝肠循环的速率（1/h）
k4_IV = k4C_IV/(bw**0.25)    #BPSG在肝肠循环中的粪便消除系数（1/h）
kenterobpag= kenterobpagC/(bw**0.25)  #BPSG肝肠循环使得BPS发生循环的速率（1/h）
kenterobpas = kenterobpasC/(bw**0.25)  #BPSS肝肠循环使得BPS发生循环的速率（1/h）


kurinebpaC = 0.04  #BPS尿液排泄基本参数（L/h/bw**0.75）
kurinebpagC =1.336  #BPSG尿液排泄基本参数（L/h/bw**0.75）
kurinebpasC = 0.03  #BPSS尿液排泄基本参数（L/h/bw**0.75）
vreabsorptiongC = 0 #尿液排泄中BPSG的肾脏重吸收的最大反应速度Vmax（nmol/h/bw**0.75）
vreabsorptionsC = 0 #尿液排泄中BPSS的肾脏重吸收的最大反应速度Vmax（nmol/h/bw**0.75）
kreabsorptiong = 9200  #尿液排泄中BPSG的肾脏重吸收的米氏常数Km（nmol/L）
kreabsorptions = 9200  #尿液排泄中BPSS的肾脏重吸收的米氏常数Km（nmol/L）


kurinebpa = kurinebpaC*(bw**0.75)  #BPS尿液排泄参数（L/h）
kurinebpag = kurinebpagC*(bw**0.75)  #BPSG尿液排泄参数（L/h）
kurinebpas = kurinebpasC*(bw**0.75)  #BPSS尿液排泄参数（L/h）
vreabsorptiong = vreabsorptiongC*(bw**0.75)  #尿液排泄中BPSG的肾脏重吸收的最大反应速度Vmax（nmol/h）
vreabsorptions = vreabsorptionsC*(bw**0.75)  #尿液排泄中BPSS的肾脏重吸收的最大反应速度Vmax（nmol/h） 

oral_intakerate = 0


##皮肤参数

#皮肤暴露场景1：擦拭纸（TP）手指暴露#
BSA= para_BP_individual[20,i+1]*100 #m2-dm2 body skin area
epi_finger=545 if(gender==1)  else 460 #手指表皮厚度（um）
TSC = 185/100000  #手指角质层厚度（um-dm） 
TVE = (epi_finger-185)/100000  #手指活性表皮层厚度（um-dm）
SCDX = TSC / 10  #角质层亚层厚度（分为11层，每层厚度为scdx）（dm）
skin_finger=1200+545 if(gender==1)  else 1095+460 #手指皮肤厚度（um）
TFO=388/560*skin_finger/100000   #手指毛囊厚度（um-dm）
AEXP=20/100  #暴露皮肤面积（cm2- dm2）
FEXP=0.005  #暴露皮肤中毛囊面积分数（1-FEXP为角质层面积分数）
add_peroid=1/60  #涂抹化学物时长（h）（接触充分浓度化学物，引入外暴露Rdose持续时间，到达add_period后停止接触擦拭纸，停止外暴露引入，即dose=0）
exp_peroid=13/6   #暴露化学物时长（h）（化学物被涂抹后，皮肤表面接触外暴露持续时间，到达exp_period后清洗皮肤，停止暴露，即SSD被清空）
VWELL=AEXP*0.01  #暴露皮肤表面沉积体积（面积*厚度，L）
VTSC = AEXP * TSC   # 暴露皮肤角质层体积（面积*厚度，L）
VTVE = AEXP * TVE    # 暴露皮肤活性表皮层体积（面积*厚度，L）
VTFO=AEXP*FEXP*TFO  #暴露皮肤毛囊体积（面积*厚度，L）

ABS=para_BP_individual[17,i+1]  #TP暴露吸收系数（TP转移到手指、再被吸收）
ABS1=0.6  #PCP暴露吸收系数

HSCVE=19.45   #角质层-表皮分配系数
HFOWell=5.4  #皮肤表面沉积-毛囊分配系数
HSCwell=HFOWell  #皮肤表面沉积-角质层分配系数

DSC = 17.28E-9   #角质层中的有效扩散系数（cm2/h- dm2）  
PFO= 6.39E-5  #毛囊的渗透系数（cm/h- dm）
u1=5.7E-5  #由脱屑而向皮肤表面转移的速度（cm/min）
#以上三个是后续需要进一步优化的参数

DOSE=para_BP_individual[16,i+1]/MWBPS  
DOSE_d=DOSE*ABS  #TP暴露BPS量
DOSE_d1=DOSE*ABS1  #PCP暴露BPS量

##经口暴露量(初始情况)
D_O = 0 # (ug/d)
dose_O = D_O/MWBPS # (nmol/kg/d)
period_O = 3/60  # 暴露持续时间（h） 
koa = dose_O/period_O  #暴露速度（nmol/h） 


AGIBPAg = 0
AGIBPAs = 0 
ABPA_delay = 0
ABPA_delays = 0
Abpac = 0
Abpas = 0
kentero=EHRrate

def FUN(XX,t): #定义微分方程组
	ON = 1 if(t<=exp_peroid) else 0
	f1 = DOSE_d/add_peroid if (t<=add_peroid) else 0
	f2 = koa if (t<=0.1) else 0
	

	x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28= XX

	return np.array([
		        (DSC/(SCDX**2) + u1/(2*SCDX))*x2-(2*DSC)/(SCDX**2)*x1+((DSC*HSCwell)/(VWELL*(SCDX**2))-(u1*HSCwell)/(VWELL*2*SCDX))*x11,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x1-(2*DSC)/(SCDX**2)*x2+(DSC/(SCDX**2) +u1/(2*SCDX))*x3,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x2-(2*DSC)/(SCDX**2)*x3+(DSC/(SCDX**2) +u1/(2*SCDX))*x4,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x3-(2*DSC)/(SCDX**2)*x4+(DSC/(SCDX**2) +u1/(2*SCDX))*x5,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x4-(2*DSC)/(SCDX**2)*x5+(DSC/(SCDX**2) +u1/(2*SCDX))*x6,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x5-(2*DSC)/(SCDX**2)*x6+(DSC/(SCDX**2) +u1/(2*SCDX))*x7,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x6-(2*DSC)/(SCDX**2)*x7+(DSC/(SCDX**2) +u1/(2*SCDX))*x8,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x7-(2*DSC)/(SCDX**2)*x8+(DSC/(SCDX**2) +u1/(2*SCDX))*x9,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x8-(2*DSC)/(SCDX**2)*x9+((DSC*HSCVE)/(VTVE*(SCDX**2))+(u1*HSCVE)/(VTVE*2*SCDX))*x12,
				#接下来是第10至20个方程
				-((PFO*AEXP*FEXP)/(VTFO*HFOWell)+(Qskin*AEXP*0.25)/(BSA*VTFO*pskin))*x10+(PFO*AEXP*FEXP)/VWELL*x11+(Qskin*AEXP*0.25)/(BSA*Vplasma)*x17,
				((DSC*AEXP*(1-FEXP))/SCDX*x1+(PFO*AEXP*FEXP)/(VTFO*HFOWell)*x10-(((DSC*HSCwell)/(VWELL*SCDX)-(u1*HSCwell)/VWELL)*AEXP*(1-FEXP)-(PFO*AEXP*FEXP)/VWELL)*x11+f1)*ON,
				(DSC*AEXP*(1-FEXP))/SCDX*x9+(((-DSC*HSCVE)/(VTVE*SCDX)-(u1*HSCVE)/VTVE)*AEXP*(1-FEXP)-(Qskin*AEXP*0.75)/(BSA*VTVE*pskin))*x12+(Qskin*AEXP*0.75)/(BSA*Vplasma)*x17,
				-(k0+ge)*x13+f2,
				(-Qskin*(1-AEXP/BSA))/((Vskin-VTSC-VTVE-VTFO)*pskin)*x14+(Qskin*(1-AEXP/BSA))/Vplasma*x17,
				(-Qfat)/(Vfat*pfat)*x15+Qfat/Vplasma*x17,
				(-Qgonad)/(Vgonad*pgonad)*x16+Qgonad/Vplasma*x17,
				((QC-kurinebpa)*Qskin*(AEXP/BSA)*0.25)/(QC*VTFO*pskin)*x10+((QC-kurinebpa)*Qskin*(AEXP/BSA)*0.75)/(QC*VTVE*pskin)*x12+((QC-kurinebpa)*Qskin*(1-AEXP/BSA))/(QC*(Vskin-VTSC-VTVE-VTFO)*pskin)*x14+((QC-kurinebpa)*Qfat)/(QC*Vfat*pfat)*x15+((QC-kurinebpa)*Qgonad)/(QC*Vgonad*pgonad)*x16-QC/Vplasma *x17+((QC-kurinebpa)*Qbrain)/(QC*Vbrain*pbrain) *x18+((QC-kurinebpa)*Qrich)/(QC*Vrich*prich) *x19+((QC-kurinebpa)*Qslow)/(QC*Vslow*pslow) *x20+((QC-kurinebpa)*Qliver)/(QC*Vliver*pliver) *x24,
				(-Qbrain)/(Vbrain*pbrain) *x18+Qbrain/Vplasma *x17,
				(-Qrich)/(Vrich*prich) *x19+Qrich/Vplasma  *x17,
				(-Qslow)/(Vslow*pslow) *x20+Qslow/Vplasma  *x17,
				#接下来是第21至28个方程
				-kGIing*x21+(vmaxgutg*x23)/(enterocytes*kmgutg+x23+(x23**2)/(enterocytes*ksigutg)),
				-kGIins*x22+(vmaxguts*x23)/(enterocytes*kmguts+x23),
				ge*x13-k1*x23-(vmaxgutg*x23)/(enterocytes*kmgutg+x23+(x23**2)/(enterocytes*ksigutg))-(vmaxguts*x23)/(enterocytes*kmguts+x23),
				k0*x13-Qliver/Vplasma  *x17+k1*x23-Qliver/(Vliver*pliver) *x24+kenterobpag*x25+kenterobpas*x26-(vmaxliver*x24)/(Vliver*pliver*kmliver+x24)-(vmaxlivers*x24)/(Vliver*pliver*kmlivers+x24),
				met2g*kGIing*x21-(EHRrate+k4_IV+kenterobpag)*x25+(met2g*vmaxliver*x24)/(Vliver*pliver*kmliver+x24),
				met2s*kGIins*x22-(EHRrate+k4_IV+kenterobpas)*x26+(met2s*vmaxlivers*x24)/(Vliver*pliver*kmlivers+x24),
				met1g*kGIing*x21+EHRrate*x25-kurinebpag/(Vbodyg+10**(-34) ) *x27+(met1g*vmaxliver*x24)/(Vliver*pliver*kmliver+x24),
				met1s*kGIins*x22+EHRrate*x26-kurinebpas/(Vbodys+10**(-34) ) *x28+(met1s*vmaxlivers*x24)/(Vliver*pliver*kmlivers+x24)

                ])

t = np.arange(0, 1, 0.005)

track = odeint(FUN, (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), t)


def FUN2(XX2,t2): #定义微分方程组

	x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28= XX2

	return np.array([
		        (DSC/(SCDX**2) + u1/(2*SCDX))*x2-(2*DSC)/(SCDX**2)*x1+((DSC*HSCwell)/(VWELL*(SCDX**2))-(u1*HSCwell)/(VWELL*2*SCDX))*x11,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x1-(2*DSC)/(SCDX**2)*x2+(DSC/(SCDX**2) +u1/(2*SCDX))*x3,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x2-(2*DSC)/(SCDX**2)*x3+(DSC/(SCDX**2) +u1/(2*SCDX))*x4,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x3-(2*DSC)/(SCDX**2)*x4+(DSC/(SCDX**2) +u1/(2*SCDX))*x5,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x4-(2*DSC)/(SCDX**2)*x5+(DSC/(SCDX**2) +u1/(2*SCDX))*x6,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x5-(2*DSC)/(SCDX**2)*x6+(DSC/(SCDX**2) +u1/(2*SCDX))*x7,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x6-(2*DSC)/(SCDX**2)*x7+(DSC/(SCDX**2) +u1/(2*SCDX))*x8,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x7-(2*DSC)/(SCDX**2)*x8+(DSC/(SCDX**2) +u1/(2*SCDX))*x9,
                (DSC/(SCDX**2) - u1/(2*SCDX))*x8-(2*DSC)/(SCDX**2)*x9+((DSC*HSCVE)/(VTVE*(SCDX**2))+(u1*HSCVE)/(VTVE*2*SCDX))*x12,
				#接下来是第10至20个方程
				-((PFO*AEXP*FEXP)/(VTFO*HFOWell)+(Qskin*AEXP*0.25)/(BSA*VTFO*pskin))*x10+(PFO*AEXP*FEXP)/VWELL*x11+(Qskin*AEXP*0.25)/(BSA*Vplasma)*x17,
				0,
				(DSC*AEXP*(1-FEXP))/SCDX*x9+(((-DSC*HSCVE)/(VTVE*SCDX)-(u1*HSCVE)/VTVE)*AEXP*(1-FEXP)-(Qskin*AEXP*0.75)/(BSA*VTVE*pskin))*x12+(Qskin*AEXP*0.75)/(BSA*Vplasma)*x17,
				-(k0+ge)*x13,
				(-Qskin*(1-AEXP/BSA))/((Vskin-VTSC-VTVE-VTFO)*pskin)*x14+(Qskin*(1-AEXP/BSA))/Vplasma*x17,
				(-Qfat)/(Vfat*pfat)*x15+Qfat/Vplasma*x17,
				(-Qgonad)/(Vgonad*pgonad)*x16+Qgonad/Vplasma*x17,
				((QC-kurinebpa)*Qskin*(AEXP/BSA)*0.25)/(QC*VTFO*pskin)*x10+((QC-kurinebpa)*Qskin*(AEXP/BSA)*0.75)/(QC*VTVE*pskin)*x12+((QC-kurinebpa)*Qskin*(1-AEXP/BSA))/(QC*(Vskin-VTSC-VTVE-VTFO)*pskin)*x14+((QC-kurinebpa)*Qfat)/(QC*Vfat*pfat)*x15+((QC-kurinebpa)*Qgonad)/(QC*Vgonad*pgonad)*x16-QC/Vplasma *x17+((QC-kurinebpa)*Qbrain)/(QC*Vbrain*pbrain) *x18+((QC-kurinebpa)*Qrich)/(QC*Vrich*prich) *x19+((QC-kurinebpa)*Qslow)/(QC*Vslow*pslow) *x20+((QC-kurinebpa)*Qliver)/(QC*Vliver*pliver) *x24,
				(-Qbrain)/(Vbrain*pbrain) *x18+Qbrain/Vplasma *x17,
				(-Qrich)/(Vrich*prich) *x19+Qrich/Vplasma  *x17,
				(-Qslow)/(Vslow*pslow) *x20+Qslow/Vplasma  *x17,
				#接下来是第21至28个方程
				-kGIing*x21+(vmaxgutg*x23)/(enterocytes*kmgutg+x23+(x23**2)/(enterocytes*ksigutg)),
				-kGIins*x22+(vmaxguts*x23)/(enterocytes*kmguts+x23),
				ge*x13-k1*x23-(vmaxgutg*x23)/(enterocytes*kmgutg+x23+(x23**2)/(enterocytes*ksigutg))-(vmaxguts*x23)/(enterocytes*kmguts+x23),
				k0*x13-Qliver/Vplasma  *x17+k1*x23-Qliver/(Vliver*pliver) *x24+kenterobpag*x25+kenterobpas*x26-(vmaxliver*x24)/(Vliver*pliver*kmliver+x24)-(vmaxlivers*x24)/(Vliver*pliver*kmlivers+x24),
				met2g*kGIing*x21-(EHRrate+k4_IV+kenterobpag)*x25+(met2g*vmaxliver*x24)/(Vliver*pliver*kmliver+x24),
				met2s*kGIins*x22-(EHRrate+k4_IV+kenterobpas)*x26+(met2s*vmaxlivers*x24)/(Vliver*pliver*kmlivers+x24),
				met1g*kGIing*x21+EHRrate*x25-kurinebpag/(Vbodyg+10**(-34) ) *x27+(met1g*vmaxliver*x24)/(Vliver*pliver*kmliver+x24),
				met1s*kGIins*x22+EHRrate*x26-kurinebpas/(Vbodys+10**(-34) ) *x28+(met1s*vmaxlivers*x24)/(Vliver*pliver*kmlivers+x24)

                ])
t2 = np.arange(1,75,0.005)
initial_expose=track[199,:]
initial_expose[10]=0
track2 = odeint(FUN2, initial_expose, t2)

t_total = np.arange(0,75,0.005)
plasma_track_total = np.append(track[:,23],track2[:,23])
print(type(plasma_track_total))

plt.plot(t_total,plasma_track_total,label = 'Python_BPA_result')


plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')

plt.show()
