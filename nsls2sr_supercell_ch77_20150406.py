import pylatt as latt

# === Element definition:
hscraper1n = latt.aper("hscraper1n",L=0.0,aper=[-0.05,1.0,-1.0,1.0])
hscraper1p = latt.aper("hscraper1p",L=0.0,aper=[-1.0,0.05,-1.0,1.0])
hscraper2n = latt.aper("hscraper2n",L=0.0,aper=[-0.05,1.0,-1.0,1.0])
hscraper2p = latt.aper("hscraper2p",L=0.0,aper=[-1.0,0.05,-1.0,1.0])
hscraperxn = latt.aper("hscraperxn",L=0.0,aper=[-0.05,1.0,-1.0,1.0])
hscraperxp = latt.aper("hscraperxp",L=0.0,aper=[-1.0,0.05,-1.0,1.0])
vscraper1n = latt.aper("vscraper1n",L=0.0,aper=[-1.0,1.0,-0.0125,1.0])
vscraper1p = latt.aper("vscraper1p",L=0.0,aper=[-1.0,1.0,-1.0,0.0125])
vscraper2n = latt.aper("vscraper2n",L=0.0,aper=[-1.0,1.0,-0.0125,1.0])
vscraper2p = latt.aper("vscraper2p",L=0.0,aper=[-1.0,1.0,-1.0,0.0125])
b1g3c01a = latt.bend("b1g3c01a",L=2.62,angle=0.10471975512,e1=0.05236,e2=0.05236,K1=0.0,K2=0.0)
b1g3c30a = latt.bend("b1g3c30a",L=2.62,angle=0.10471975512,e1=0.05236,e2=0.05236,K1=0.0,K2=0.0)
b1g5c01b = latt.bend("b1g5c01b",L=2.62,angle=0.10471975512,e1=0.05236,e2=0.05236,K1=0.0,K2=0.0)
b1g5c30b = latt.bend("b1g5c30b",L=2.62,angle=0.10471975512,e1=0.05236,e2=0.05236,K1=0.0,K2=0.0)
D0001 = latt.drif("D0001",L=0.0)
D0002 = latt.drif("D0002",L=0.661071)
D0003 = latt.drif("D0003",L=1.23)
D0004 = latt.drif("D0004",L=0.325206)
D0005 = latt.drif("D0005",L=0.33421)
D0006 = latt.drif("D0006",L=0.085)
D0007 = latt.drif("D0007",L=0.081)
D0008 = latt.drif("D0008",L=0.1485)
D0009 = latt.drif("D0009",L=0.1255)
D0010 = latt.drif("D0010",L=0.328)
D0011 = latt.drif("D0011",L=0.184)
D0012 = latt.drif("D0012",L=0.186)
D0013 = latt.drif("D0013",L=0.07602)
D0014 = latt.drif("D0014",L=0.08998)
D0015 = latt.drif("D0015",L=0.1137)
D0016 = latt.drif("D0016",L=0.2848)
D0017 = latt.drif("D0017",L=0.1815)
D0018 = latt.drif("D0018",L=0.601)
D0019 = latt.drif("D0019",L=0.3755)
D0020 = latt.drif("D0020",L=0.2015)
D0021 = latt.drif("D0021",L=0.2552)
D0022 = latt.drif("D0022",L=0.3144)
D0023 = latt.drif("D0023",L=0.0844)
D0024 = latt.drif("D0024",L=0.184)
D0025 = latt.drif("D0025",L=0.184)
D0026 = latt.drif("D0026",L=0.328)
D0027 = latt.drif("D0027",L=0.176)
D0028 = latt.drif("D0028",L=0.26226)
D0029 = latt.drif("D0029",L=0.08924)
D0030 = latt.drif("D0030",L=0.2605)
D0031 = latt.drif("D0031",L=0.716)
D0032 = latt.drif("D0032",L=0.591)
D0033 = latt.drif("D0033",L=0.07822)
D0034 = latt.drif("D0034",L=0.08778)
D0035 = latt.drif("D0035",L=0.2575)
D0036 = latt.drif("D0036",L=0.3021)
D0037 = latt.drif("D0037",L=0.244)
D0038 = latt.drif("D0038",L=0.23125)
D0039 = latt.drif("D0039",L=0.23475)
D0040 = latt.drif("D0040",L=0.0783)
D0041 = latt.drif("D0041",L=0.0877)
D0042 = latt.drif("D0042",L=0.46852)
D00431 = latt.drif("D00431",L=2.96568)
D00432 = latt.drif("D00432",L=2.83148)
D0044 = latt.drif("D0044",L=0.33432)
D0045 = latt.drif("D0045",L=0.09)
D0046 = latt.drif("D0046",L=0.076)
D0047 = latt.drif("D0047",L=0.23475)
D0048 = latt.drif("D0048",L=0.23125)
D0049 = latt.drif("D0049",L=0.244)
D0050 = latt.drif("D0050",L=0.3021)
D0051 = latt.drif("D0051",L=0.2575)
D0052 = latt.drif("D0052",L=0.09008)
D0053 = latt.drif("D0053",L=0.07592)
D0054 = latt.drif("D0054",L=0.591)
D0055 = latt.drif("D0055",L=0.501)
D0056 = latt.drif("D0056",L=0.2755)
D0057 = latt.drif("D0057",L=0.2015)
D0058 = latt.drif("D0058",L=0.2552)
D0059 = latt.drif("D0059",L=0.3144)
D0060 = latt.drif("D0060",L=0.0844)
D0061 = latt.drif("D0061",L=0.184)
D0062 = latt.drif("D0062",L=0.184)
D0063 = latt.drif("D0063",L=0.328)
D0064 = latt.drif("D0064",L=0.176)
D0065 = latt.drif("D0065",L=0.26226)
D0066 = latt.drif("D0066",L=0.08924)
D0067 = latt.drif("D0067",L=0.2605)
D0068 = latt.drif("D0068",L=0.327011)
D0069 = latt.drif("D0069",L=0.388989)
D0070 = latt.drif("D0070",L=0.36634)
D0071 = latt.drif("D0071",L=0.21366)
D0072 = latt.drif("D0072",L=0.0896)
D0073 = latt.drif("D0073",L=0.0764)
D0074 = latt.drif("D0074",L=0.186)
D0075 = latt.drif("D0075",L=0.184)
D0076 = latt.drif("D0076",L=0.5785)
D0077 = latt.drif("D0077",L=0.2235)
D0078 = latt.drif("D0078",L=0.08059)
D0079 = latt.drif("D0079",L=0.08541)
D0080 = latt.drif("D0080",L=0.4629)
D0081 = latt.drif("D0081",L=4.1871)
ch1xg2c30a = latt.kick("ch1xg2c30a",L=0.0,hkick=0,vkick=0)
ch1xg6c01b = latt.kick("ch1xg6c01b",L=0.0,hkick=0,vkick=0)
ch1yg2c30a = latt.kick("ch1yg2c30a",L=0.0,hkick=0,vkick=0)
ch1yg6c01b = latt.kick("ch1yg6c01b",L=0.0,hkick=0,vkick=0)
ch2xg2c30a = latt.kick("ch2xg2c30a",L=0.0,hkick=0,vkick=0)
ch2xg6c01b = latt.kick("ch2xg6c01b",L=0.0,hkick=0,vkick=0)
ch2yg2c30a = latt.kick("ch2yg2c30a",L=0.0,hkick=0,vkick=0)
ch2yg6c01b = latt.kick("ch2yg6c01b",L=0.0,hkick=0,vkick=0)
cl1xg2c01a = latt.kick("cl1xg2c01a",L=0.0,hkick=0,vkick=0)
cl1xg6c30b = latt.kick("cl1xg6c30b",L=0.0,hkick=0,vkick=0)
cl1yg2c01a = latt.kick("cl1yg2c01a",L=0.0,hkick=0,vkick=0)
cl1yg6c30b = latt.kick("cl1yg6c30b",L=0.0,hkick=0,vkick=0)
cl2xg2c01a = latt.kick("cl2xg2c01a",L=0.0,hkick=0,vkick=0)
cl2xg6c30b = latt.kick("cl2xg6c30b",L=0.0,hkick=0,vkick=0)
cl2yg2c01a = latt.kick("cl2yg2c01a",L=0.0,hkick=0,vkick=0)
cl2yg6c30b = latt.kick("cl2yg6c30b",L=0.0,hkick=0,vkick=0)
cm1xg4c01a = latt.kick("cm1xg4c01a",L=0.0,hkick=0,vkick=0)
cm1xg4c01b = latt.kick("cm1xg4c01b",L=0.0,hkick=0,vkick=0)
cm1xg4c30a = latt.kick("cm1xg4c30a",L=0.0,hkick=0,vkick=0)
cm1xg4c30b = latt.kick("cm1xg4c30b",L=0.0,hkick=0,vkick=0)
cm1yg4c01a = latt.kick("cm1yg4c01a",L=0.0,hkick=0,vkick=0)
cm1yg4c01b = latt.kick("cm1yg4c01b",L=0.0,hkick=0,vkick=0)
cm1yg4c30a = latt.kick("cm1yg4c30a",L=0.0,hkick=0,vkick=0)
cm1yg4c30b = latt.kick("cm1yg4c30b",L=0.0,hkick=0,vkick=0)
fh1g1c02a = latt.kick("fh1g1c02a",L=0.0,hkick=0,vkick=0)
fh2g1c30a = latt.kick("fh2g1c30a",L=0.0,hkick=0,vkick=0)
fl1g1c01a = latt.kick("fl1g1c01a",L=0.0,hkick=0,vkick=0)
fl2g1c01a = latt.kick("fl2g1c01a",L=0.0,hkick=0,vkick=0)
fm1g4c01a = latt.kick("fm1g4c01a",L=0.0,hkick=0,vkick=0)
fm1g4c30a = latt.kick("fm1g4c30a",L=0.0,hkick=0,vkick=0)
isbu3 = latt.kick("isbu3",L=0.65,hkick=0,vkick=0)
isbu4 = latt.kick("isbu4",L=0.65,hkick=0,vkick=0)
issp1d = latt.kick("issp1d",L=0.799513,hkick=0,vkick=0)
ph1g2c30a = latt.moni("ph1g2c30a",L=0.0)
ph1g6c01b = latt.moni("ph1g6c01b",L=0.0)
ph2g2c30a = latt.moni("ph2g2c30a",L=0.0)
ph2g6c01b = latt.moni("ph2g6c01b",L=0.0)
pl1g2c01a = latt.moni("pl1g2c01a",L=0.0)
pl1g6c30b = latt.moni("pl1g6c30b",L=0.0)
pl2g2c01a = latt.moni("pl2g2c01a",L=0.0)
pl2g6c30b = latt.moni("pl2g6c30b",L=0.0)
pm1g4c01a = latt.moni("pm1g4c01a",L=0.0)
pm1g4c01b = latt.moni("pm1g4c01b",L=0.0)
pm1g4c30a = latt.moni("pm1g4c30a",L=0.0)
pm1g4c30b = latt.moni("pm1g4c30b",L=0.0)
qh1 = latt.quad("qh1",L=0.268,K1=-0.641957314648)
qh2 = latt.quad("qh2",L=0.46,K1=1.43673057073)
qh3 = latt.quad("qh3",L=0.268,K1=-1.75355042529)
ql1 = latt.quad("ql1",L=0.268,K1=-1.61785473561)
ql2 = latt.quad("ql2",L=0.46,K1=1.76477357129)
ql3 = latt.quad("ql3",L=0.268,K1=-1.51868267756)
qm1 = latt.quad("qm1",L=0.247,K1=-0.812234822773)
qm2 = latt.quad("qm2",L=0.282,K1=1.22615465959)
sh1 = latt.sext("sh1",L=0.2,K2=19.09710400)
sh3 = latt.sext("sh3",L=0.2,K2=-3.10963400)
sh4 = latt.sext("sh4",L=0.2,K2=-19.49461000)
sl1 = latt.sext("sl1",L=0.2,K2=-8.51443400)
sl2 = latt.sext("sl2",L=0.2,K2=32.72035900)
sl3 = latt.sext("sl3",L=0.2,K2=-27.55513600)
sm1a = latt.sext("sm1a",L=0.2,K2=-26.21867845)
sm1b = latt.sext("sm1b",L=0.2,K2=-28.01658206)
sm2 = latt.sext("sm2",L=0.25,K2=30.69679902)
sqhg2c30a = latt.skew("sqhg2c30a",L=0.1,K1=0.0,tilt=0.785398163397)
sqmg4c01a = latt.skew("sqmg4c01a",L=0.1,K1=0.0,tilt=0.785398163397)

# === Beam Line sequence:
BL = [D0001, issp1d, D0002, isbu3, D0003, isbu4, D0004, fh2g1c30a, D0005,
   sh1, D0006, ph1g2c30a, D0007, qh1, D0008, sqhg2c30a, ch1xg2c30a, ch1yg2c30a,
   sqhg2c30a, D0009, hscraperxp, hscraperxn, D0010, qh2, D0011, sh3, D0012,
   qh3, D0013, ph2g2c30a, D0014, sh4, D0015, vscraper1p, vscraper1n, D0016,
   ch2xg2c30a, ch2yg2c30a, D0017, b1g3c30a, D0018, cm1xg4c30a, cm1yg4c30a,
   D0019, qm1, D0020, sm1a, D0021, fm1g4c30a, D0022, pm1g4c30a, D0023, qm2,
   D0024, sm2, D0025, qm2, D0026, hscraper1p, hscraper1n, D0027, sm1b, D0028,
   pm1g4c30b, D0029, qm1, D0030, cm1yg4c30b, cm1xg4c30b, D0031, b1g5c30b,
   D0032, ql3, D0033, pl2g6c30b, D0034, sl3, D0035, cl2yg6c30b, cl2xg6c30b,
   D0036, ql2, D0037, sl2, D0038, cl1yg6c30b, cl1xg6c30b, D0039, ql1, D0040,
   pl1g6c30b, D0041, sl1, D0042, fl1g1c01a, D00431, D00432, fl2g1c01a, D0044,
   sl1, D0045, pl1g2c01a, D0046, ql1, D0047, cl1xg2c01a, cl1yg2c01a, D0048,
   sl2, D0049, ql2, D0050, cl2xg2c01a, cl2yg2c01a, D0051, sl3, D0052, pl2g2c01a,
   D0053, ql3, D0054, b1g3c01a, D0055, sqmg4c01a, cm1xg4c01a, cm1yg4c01a,
   sqmg4c01a, D0056, qm1, D0057, sm1a, D0058, fm1g4c01a, D0059, pm1g4c01a,
   D0060, qm2, D0061, sm2, D0062, qm2, D0063, hscraper2p, hscraper2n, D0064,
   sm1b, D0065, pm1g4c01b, D0066, qm1, D0067, cm1yg4c01b, cm1xg4c01b, D0068,
   vscraper2p, vscraper2n, D0069, b1g5c01b, D0070, ch2yg6c01b, ch2xg6c01b,
   D0071, sh4, D0072, ph2g6c01b, D0073, qh3, D0074, sh3, D0075, qh2, D0076,
   ch1yg6c01b, ch1xg6c01b, D0077, qh1, D0078, ph1g6c01b, D0079, sh1, D0080,
   fh1g1c02a, D0081]
ring = latt.cell(BL)
