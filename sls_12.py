import pylatt as latt

QM3 = latt.quad('QM3', L = 0.1, K1 = -4.962)
QM2 = latt.quad('QM2', L = 0.25, K1 = 6.266)
QM1 = latt.quad('QM1', L = 0.15, K1 = -6.584737)
DMS = latt.drif('DMS', L = 2.692)
DM2 = latt.drif('DM2', L = 0.1)
DMON = latt.drif('DMON', L = 0.05)
DOC = latt.drif('DOC', L = 0.05)
DM1 = latt.drif('DM1', L = 0.1)
DME = latt.drif('DME', L = 0.3)
DNVB = latt.drif('DNVB', L = 0.01)
DNM1 = latt.drif('DNM1', L = 0.05)
DXM = latt.drif('DXM', L = 0.218)
DNM2 = latt.drif('DNM2', L = 0.15)
OXX_M = latt.drif('OXX_M', L = 0)#258.8591)
OXY_M = latt.drif('OXY_M', L = 0)#340.7881)
OYY_M = latt.drif('OYY_M', L = 0)#1068.1094)
OCXX = latt.drif('OCXX', L = 0)#67.4915)
OCXX2 = latt.drif('OCXX2', L = 0)#141.264)
CH = latt.kick('CH', L = 0)
CV = latt.kick('CV', L = 0)
MON = latt.moni('MON', L = 0)
BNOM = latt.drif('BNOM', L = 0)
SXX_MH = latt.sext('SXX_MH', L = 0.05, K2 = 523.1522179999999)
SXY_MH = latt.sext('SXY_MH', L = 0.05, K2 = -543.428944)
SYY_MH = latt.sext('SYY_MH', L = 0.05, K2 = 309.45156)
SDX = latt.sext('SDX', L = 0.1, K2 = -693.055549)
SFXH = latt.sext('SFXH', L = 0.05, K2 = 826.834506)
SD = latt.sext('SD', L = 0.1, K2 = -563.1248409999999)
SFH = latt.sext('SFH', L = 0.05, K2 = 689.439306)
BN00 = latt.bend('BN00',L=0.021832,angle=0.005454,e1=0,e2=0.005454,K1=0,K2=0)
BN01 = latt.bend('BN01',L=0.022967,angle=0.005455,e1=-0.005454,e2=0.010909,K1=0,K2=0)
BN02 = latt.bend('BN02',L=0.031095,angle=0.005454,e1=-0.010909,e2=0.016363,K1=0,K2=0)
BN03 = latt.bend('BN03',L=0.038596,angle=0.005454,e1=-0.016363,e2=0.021817,K1=0,K2=0)
BN04 = latt.bend('BN04',L=0.045882,angle=0.005454,e1=-0.021817,e2=0.027271,K1=0,K2=0)
BN05 = latt.bend('BN05',L=0.053107,angle=0.005454,e1=-0.027271,e2=0.032725,K1=0,K2=0)
BN06 = latt.bend('BN06',L=0.060418,angle=0.005454,e1=-0.032725,e2=0.038179,K1=0,K2=0)
VBM = latt.bend('VBM',L=0.2061,angle=0.01906770187171,e1=-0.038179,e2=0.0572467912,K1=-2.079498,K2=0)
ANM = latt.bend('ANM',L=0.3,angle=-0.0136135662,e1=0,e2=-0.0136135662,K1=3.508741,K2=0)
AN = latt.bend('AN',L=0.3,angle=-0.0136135662,e1=0,e2=-0.0136135662,K1=3.920132,K2=0)
VB = latt.bend('VB',L=0.2061,angle=0.01906770187171,e1=0.0572467912,e2=-0.038179,K1=-3.875276,K2=0)

BN00r = latt.bend('BN00r',L=0.021832,angle=0.005454,e2=0,e1=0.005454,K1=0,K2=0)
BN01r = latt.bend('BN01r',L=0.022967,angle=0.005455,e2=-0.005454,e1=0.010909,K1=0,K2=0)
BN02r = latt.bend('BN02r',L=0.031095,angle=0.005454,e2=-0.010909,e1=0.016363,K1=0,K2=0)
BN03r = latt.bend('BN03r',L=0.038596,angle=0.005454,e2=-0.016363,e1=0.021817,K1=0,K2=0)
BN04r = latt.bend('BN04r',L=0.045882,angle=0.005454,e2=-0.021817,e1=0.027271,K1=0,K2=0)
BN05r = latt.bend('BN05r',L=0.053107,angle=0.005454,e2=-0.027271,e1=0.032725,K1=0,K2=0)
BN06r = latt.bend('BN06r',L=0.060418,angle=0.005454,e2=-0.032725,e1=0.038179,K1=0,K2=0)
VBMr = latt.bend('VBMr',L=0.2061,angle=0.01906770187171,e2=-0.038179,e1=0.0572467912,K1=-2.079498,K2=0)
ANMr = latt.bend('ANMr',L=0.3,angle=-0.0136135662,e2=0,e1=-0.0136135662,K1=3.508741,K2=0)
ANr = latt.bend('ANr',L=0.3,angle=-0.0136135662,e2=0,e1=-0.0136135662,K1=3.920132,K2=0)
VBr = latt.bend('VBr',L=0.2061,angle=0.01906770187171,e2=0.0572467912,e1=-0.038179,K1=-3.875276,K2=0)

L0001 = [DMS,QM3,DM2,MON,DMON,DOC,SXX_MH,CV,CH,SXX_MH,DOC,OXX_M,DOC,QM2,
         DOC,SXY_MH,CV,CH,SXY_MH,DOC,OXY_M,DOC,MON,DMON,DM1,QM1,DME,MON,
         DMON,DOC,SYY_MH,CV,CH,SYY_MH,DOC,OYY_M,DOC,BNOM,BN00,BN01]
L0002 = [BN02,BN03,BN04,BN05,BN06,DNVB,VBM,DNM1,SDX,DXM,ANM,DOC,OCXX,
         DOC,SFXH,CV,CH,SFXH,DOC,MON,DOC,AN,DNM2,SD,DNM1,VB,DNVB,BN06r,
         BN05r,BN04r,BN03r,BN02r,BN01r,BN00r,BNOM,BNOM,BN00,BN01,BN02,BN03]
L0003 = [BN04,BN05,BN06,DNVB,VBr,DNM1,SD,DNM2,ANr,DOC,OCXX2,DOC,SFH,CV,
         CH,SFH,DOC,MON,DOC,AN,DNM2,SD,DNM1,VB,DNVB,BN06r,BN05r,BN04r,BN03r,
         BN02r,BN01r,BN00r,BNOM,BNOM,BN00,BN01,BN02,BN03,BN04,BN05]
L0004 = [BN06,DNVB,VBr,DNM1,SD,DNM2,ANr,DOC,OCXX2,DOC,SFH,CV,CH,SFH,DOC,
         MON,DOC,AN,DNM2,SD,DNM1,VB,DNVB,BN06r,BN05r,BN04r,BN03r,BN02r,BN01r,
         BN00r,BNOM,BNOM,BN00,BN01,BN02,BN03,BN04,BN05,BN06,DNVB]
L0005 = [VBr,DNM1,SD,DNM2,ANr,DOC,MON,DOC,SFH,CH,CV,SFH,DOC,OCXX2,DOC,
         AN,DNM2,SD,DNM1,VB,DNVB,BN06r,BN05r,BN04r,BN03r,BN02r,BN01r,BN00r,
         BNOM,BNOM,BN00,BN01,BN02,BN03,BN04,BN05,BN06,DNVB,VBr,DNM1]
L0006 = [SD,DNM2,ANr,DOC,MON,DOC,SFH,CH,CV,SFH,DOC,OCXX2,DOC,AN,DNM2,
         SD,DNM1,VB,DNVB,BN06r,BN05r,BN04r,BN03r,BN02r,BN01r,BN00r,BNOM,BNOM,
         BN00,BN01,BN02,BN03,BN04,BN05,BN06,DNVB,VBr,DNM1,SD,DNM2]
L0007 = [ANr,DOC,MON,DOC,SFXH,CH,CV,SFXH,DOC,OCXX,DOC,ANMr,DXM,SDX,DNM1,
         VBMr,DNVB,BN06r,BN05r,BN04r,BN03r,BN02r,BN01r,BN00r,BNOM,DOC,OYY_M,DOC,
         SYY_MH,CH,CV,SYY_MH,DOC,DMON,MON,DME,QM1,DM1,DMON,MON]
L0008 = [DOC,OXY_M,DOC,SXY_MH,CH,CV,SXY_MH,DOC,QM2,DOC,OXX_M,DOC,
         SXX_MH,CH,CV,SXX_MH,DOC,DMON,MON,DM2,QM3,DMS]
RING = [L0001,L0002,L0003,L0004,L0005,L0006,L0007,L0008]
ring = latt.cell(RING)
#ring = latt.beamline(RING,[2.673100e+00,2.442491e-15,0],[2.790055e+00,-1.110223e-14,0])
