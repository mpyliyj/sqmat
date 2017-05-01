import pylatt as latt

BH = latt.bend('BH',L=1.4,angle=0.056099868814,e1=0.028049934407,e2=0.028049934407,K1=-0.220496,K2=0)
DB = latt.drif('DB', L = 0.125)
M1 = latt.drif('M1', L = 0)
Q1 = latt.quad('Q1', L = 0.4, K1 = 1.10271)
Q2 = latt.quad('Q2', L = 0.4, K1 = 1.26667)
Q3 = latt.quad('Q3', L = 0.4, K1 = 0.8611387)
Q4 = latt.quad('Q4', L = 0.4, K1 = 1.537522)
Q5 = latt.quad('Q5', L = 0.4, K1 = -1.031027)
SD = latt.sext('SD', L = 0.2, K2 = -1.0)
SF = latt.sext('SF', L = 0.15, K2 = 1.0)
D = latt.drif('D', L = 0.0)
D1 = latt.drif('D1', L = 0.3)
D2 = latt.drif('D2', L = 0.4)
D3 = latt.drif('D3', L = 0.6)
D4 = latt.drif('D4', L = 1.3)
D5 = latt.drif('D5', L = 1.82)
D6 = latt.drif('D6', L = 3.0)
D7 = latt.drif('D7', L = 0.2851669)
DL = latt.drif('DL', L = 7.95)
DD1 = latt.drif('DD1', L = 0.15)
M2 = latt.drif('M2', L = 0)
M3 = latt.drif('M3', L = 0)
M4 = latt.drif('M4', L = 0)
CH = latt.kick('CH', L = 0)
CV = latt.kick('CV', L = 0)
L000001 = [DL,Q5,D7,Q4,D6,BH,D5,Q3,D4,Q2,D3,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,SF,D1,Q1,D2,SD,DD1,BH,DB,M1,DB,BH,DD1,SD,D2,Q1,D1,SF,M2,M2,SF,D1,Q1,D2]
L000002 = [SD,DD1,BH,DB,M1,DB,BH,D3,Q2,D4,Q3,D5,BH,M3,D6,Q4,D7,Q5,DL,M4]
ring = [L000001,L000002]
#ELEMENTDEFINITIONS:

ring = latt.cell(ring)
