import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm 

j = 0
time = np.arange(0,75,0.005) #七十五个小时的时间戳
paradata= np.array([[17.28, 6.39, 5.7],[14.91,2.78,4.707],[21.333, 9.666, 7.42],[19.921, 11, 2.4212],[23.032,7.2223,4.99],
                [13.53,10.11,5.444],[18.888,9.999,8.888],[15.5,10.5,7.5],[12.45,6.84,6.75],[20.8,4.61,9.888]])

#plasma:26 , urineBPS:38 , urineBPSg:54
def BPS_BPTK_MultiParas(t = time,volunteer_ID =j, paras = paradata ,mode = '63'):
    
	plasma_data = np.zeros((np.shape(paras)[0],15000))
	urinebps_data = np.zeros((np.shape(paras)[0],15000))
	urinebpsg_data = np.zeros((np.shape(paras)[0],15000))
	para_BP_individual = np.loadtxt(open("R\para_BP_individual.csv"),delimiter=",",skiprows=1,usecols=[2,3,4,5])
	#读取受试者生理参数(第五位受试者数据有缺失，暂取四位的数据)

	i = volunteer_ID-1 #用于控制受试者的编号，可取值-1，0,1,2
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

	HSCVE=19.45   #角质层-表皮分配系数
	HFOWell=5.4  #皮肤表面沉积-毛囊分配系数
	HSCwell=HFOWell  #皮肤表面沉积-角质层分配系数

	
	#以上三个是后续需要进一步优化的参数

	DOSE=para_BP_individual[16,i+1]/MWBPS  
	DOSE_d=DOSE*ABS  #TP暴露BPS量
	

	kentero=EHRrate

	def FUN(XX,t,para1,para2,para3): #定义微分方程组



				
		dose,AFO,AWELL,CSC01,CSC02,CSC03,CSC04,CSC05,CSC06,CSC07,CSC08,CSC09,AVE,AST,AGImet,AGImets,AAO,ASI,Afeces,Aoral, AGIin,AGIBPAg,AGIins,AGIBPAs,Askin,Aplasma,AFat,Agonad,AM,AMs,ABPA_delayinbpag,ABPA_delayinbpas,ALiver,Abrain, AR,AS,Aurinebpa,ABPAg_prod,ABPAg_prod_delay,ABPAg_prod_gut,ABPAg_prod_delay_gut,ABPAs_prod,ABPAs_prod_delay,ABPAs_prod_gut,ABPAs_prod_delay_gut,ABPA_delayin,Afecesiv,ABPA_delay,ABPA_delayins,Afecesivs,ABPA_delays,Areabsorption,Aurineg,Aurinebpag,Areabsorptions,Aurines,Aurinebpas,Abpac,Abpas = XX
				
				# Dermal Exposure Well

		Rdose=DOSE_d/(add_peroid) if(t<=add_peroid) else 0 
		OO_S_TP1=1 if(t<=exp_peroid) else 0

		DSC_0 = para1
		PFO_0 = para2
		u1_0  = para3

		DSC =DSC_0/1000/1000/1000   #角质层中的有效扩散系数（cm2/h- dm2）  
		PFO= PFO_0/1000/100 #毛囊的渗透系数（cm/h- dm）
		u1=  u1_0/1000/100  #由脱屑而向皮肤表面转移的速度（cm/min）


		CA = Aplasma/Vplasma

		# Dermal Exposure Well

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


		
		Cgut = ASI/enterocytes 
		dAST = -k0*AST-ge*AST 
		RAGImet = vmaxgutg*Cgut/(kmgutg+Cgut*(1+Cgut/ksigutg)) 
		RAGImets = vmaxguts*Cgut/(kmguts+Cgut) 
		RAAO = k1*ASI 
		dASI = ge*AST-RAGImet-RAAO-RAGImets 
		RAfeces = k4*ASI 
		RAoral = k0*AST+RAAO

		RAGIin = kGIing*AGIBPAg 
		RAGIBPAg = RAGImet - RAGIin
		RAGIins = kGIins*AGIBPAs 
		RAGIBPAs = RAGImets - RAGIins



		CVskin = Askin/((Vskin-VTSC-VTVE-VTFO)*pskin) 


		CVFat = AFat/(Vfat*pfat) 

		CVgonad = Agonad/(Vgonad*pgonad) 

		CVLiver = ALiver/(Vliver*pliver) 

		CVbrain = Abrain/(Vbrain*pbrain) 

		CVR = AR/(Vrich*prich) # 
		CVS = AS/(Vslow*pslow) # (nmol/L) 


		CV =(CVLiver*Qliver+
				CVskin*(Qskin*(1-AEXP/BSA)) + 
				CvVE*(Qskin*(AEXP/BSA)*0.75)+
				CvFO*(Qskin*(AEXP/BSA)*0.25)+
				+CVFat*Qfat+CVR*Qrich+CVS*Qslow+
				CVgonad*Qgonad+CVbrain*Qbrain)/QC #

		dAskin = Qskin*(1-AEXP/BSA)*(CA-CVskin) 

		dAplasma = QC*(CV-CA) -kurinebpa*CV 
		dAfat = Qfat*(CA-CVFat) 
		dAgonad = Qgonad*(CA-CVgonad)

		RAM = vmaxliver*CVLiver/(kmliver+CVLiver) 
		RAMs = vmaxlivers*CVLiver/(kmlivers+CVLiver)
		RABPA_delayinbpag = ABPA_delay*kenterobpag
		RABPA_delayinbpas = ABPA_delays*kenterobpas 

		dALiver = Qliver*(CA-CVLiver)+RAoral-RAM-RAMs+RABPA_delayinbpag+RABPA_delayinbpas
		dAbrain = Qbrain*(CA-CVbrain) # (nmol/h)
		dAR = Qrich*(CA-CVR)
		dAS = Qslow*(CA-CVS)
		dAurinebpa = kurinebpa*CV

		RABPAg_prod = met1g*RAM # (nmol/h) |Taken up into systemic circulation
		RABPAg_prod_delay = met2g*RAM # (nmol/h) |Excreted into bile
		RABPAg_prod_gut= met1g*RAGIin # (nmol/h)|Taken up into systemic circulation
		RABPAg_prod_delay_gut = met2g*RAGIin # (nmol/h) |Excreted into bile
		RABPAs_prod = met1s*RAMs # (nmol/h) |Taken up into systemic circulation 
		RABPAs_prod_delay = met2s*RAMs # (nmol/h) |Excreted into bile
		RABPAs_prod_gut= met1s*RAGIins # (nmol/h)|Taken up into systemic circulation
		RABPAs_prod_delay_gut = met2s*RAGIins # (nmol/h) |Excreted into bile

		RABPA_delayin = ABPA_delay*kentero  
		RAfecesiv = ABPA_delay*k4_IV 
		RABPA_delay = RABPAg_prod_delay+RABPAg_prod_delay_gut-RABPA_delayin-RAfecesiv-RABPA_delayinbpag 

		RABPA_delayins = ABPA_delays*kentero 
		RAfecesivs = ABPA_delays*k4_IV
		RABPA_delays = RABPAs_prod_delay+RABPAs_prod_delay_gut-RABPA_delayins-RAfecesivs-RABPA_delayinbpas 

		Cbpac = Abpac/(Vbodyg+1E-34)
		Cbpas = Abpas/(Vbodys+1E-34)

		RAreabsorption = vreabsorptiong*Cbpac/(kreabsorptiong+Cbpac) 
		dAurineg = kurinebpag*Cbpac
		RAurinebpag = kurinebpag*Cbpac-RAreabsorption 

		RAreabsorptions = vreabsorptions*Cbpas/(kreabsorptions+Cbpas) 
		dAurines = kurinebpas*Cbpas
		RAurinebpas = kurinebpas*Cbpas-RAreabsorptions 

		RAbpac = RABPAg_prod+RABPAg_prod_gut+RABPA_delayin-RAurinebpag
		RAbpas = RABPAs_prod+RABPA_delayins+RABPAs_prod_gut-RAurinebpas

		



		return np.array([Rdose,dAFO,
			dAWELL,dCSC01,dCSC02,dCSC03,dCSC04,dCSC05,dCSC06,
			dCSC07,dCSC08,dCSC09,dAVE,
			dAST,RAGImet,RAGImets,RAAO,dASI,
			RAfeces,RAoral, RAGIin,RAGIBPAg,RAGIins,RAGIBPAs,
			dAskin,dAplasma,dAfat,dAgonad,RAM,RAMs,RABPA_delayinbpag,RABPA_delayinbpas,
			dALiver,dAbrain, dAR,dAS,dAurinebpa,
			RABPAg_prod,RABPAg_prod_delay,RABPAg_prod_gut,RABPAg_prod_delay_gut,
			RABPAs_prod,RABPAs_prod_delay,RABPAs_prod_gut,RABPAs_prod_delay_gut,
			RABPA_delayin,RAfecesiv,RABPA_delay,RABPA_delayins,RAfecesivs,RABPA_delays,
			RAreabsorption,dAurineg,RAurinebpag,
			RAreabsorptions,dAurines,RAurinebpas,
			RAbpac,RAbpas])

	t = time
	for i in tqdm(range(np.shape(paras)[0])):
		result = odeint(FUN, (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), t , args=(paras[i,0],paras[i,1],paras[i,2]))
		plasma_data[i,:] = result[:,25] #plasma:26 , urineBPS:38 , urineBPSg:54
		urinebps_data[i,:] = result[:,37] 
		urinebpsg_data[i,:] = result[:,53] 

	return result,plasma_data,urinebps_data,urinebpsg_data
