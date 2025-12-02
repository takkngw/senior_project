from controller import  Motor, GPS, DistanceSensor,PositionSensor, Compass,Supervisor, InertialUnit
import math
import random
import time
import scipy
import numpy
from decimal import *
import csv
from datetime import datetime
import os
import struct
#-----------------------------------------------------------------------------------------------------------------------
goal = 0
flag = 0
filePolicy = "./source/policy.npy"
fileEpisode = "./source/episodes.npy"   
filebody = "./source/body.npy"
TIME_STEP = 64
stepAngle = 15
MAX_SPEED = 6.28
maxAngle1 = 16
minAngle1 = -16
REWARD0 = 0
REWARD1 = 1
REWARD2 = 0.01
REWARD3 = 3
NREWARD0 = -1
NREWARD1 = -0.05
NREWARD2 = -1
TREWARD = 0
GOALREWARD = 2

NSTEP = 1
NEPISODE = 1
LRATE = 0.1
DRATE = 0.99
BOLTZMANN = 0.1

C_STATE = [0, 0, 0, 0, 0, 0, 0]
O_STATE = [0, 0, 0, 0, 0, 0, 0]

initmot = [-45, 90, -130, 45, 90, -130, -45, -90, 130, 45, -90, 130, 0, -90, 130, 0, 90, -130]
mot = [35, 3, 3, 3, 3, 3, 3]
act = 12
# Bias
KbiasMot = [450] #environment bias
MbiasMot = [45, 45, 90, -90, -130, 130] #motor bias
BiasMot = KbiasMot + MbiasMot

MAXEPISODE = 1000000
MAXSTEP = 100000
EPISODES = []
STEPS = []
REWARDS = []
#-----------------------------------------------------------------------------------------------------------------------
#set Supervisor
robot = Supervisor()
Nrobot = robot.getFromDef("IRSL-XR06-01")
Rtranslation = Nrobot.getField("translation")
Rrotation = Nrobot.getField("rotation")

TS1 = robot.getFromDef("TS1")
TS1translation = TS1.getField("translation")
TS1rotation = TS1.getField("rotation")

TS2 = robot.getFromDef("TS2")
TS2translation = TS2.getField("translation")
TS2rotation = TS2.getField("rotation")

TS3 = robot.getFromDef("TS3")
TS3translation = TS3.getField("translation")
TS3rotation = TS3.getField("rotation")

TS4 = robot.getFromDef("TS4")
TS4translation = TS4.getField("translation")
TS4rotation = TS4.getField("rotation")

TS5 = robot.getFromDef("TS5")
TS5translation = TS5.getField("translation")
TS5rotation = TS5.getField("rotation")

TS6 = robot.getFromDef("TS6")
TS6translation = TS6.getField("translation")
TS6rotation = TS6.getField("rotation")
#-----------------------------------------------------------------------------------------------------------------------
#set distance sensor
ps = []
psNames = [
    'DS1', 'DS2'
]
for i in range(len(psNames)):
    ps.append(robot.getDistanceSensor(psNames[i]))
    ps[i].enable(TIME_STEP)
    
#set motor
Mnames = []
RMnames = [
    'motor1', 'motor2', 'motor3', 'motor4', 'motor5', 'motor6', 'motor7', 'motor8', 'motor9', 'motor10'
    , 'motor11', 'motor12', 'motor13', 'motor14', 'motor15', 'motor16', 'motor17', 'motor18'
]
    
for i in range(len(RMnames)):
    Mnames.append(robot.getMotor(RMnames[i]))
    
#set PositionSensor
RM = []
RMPS = [
    'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'PS7', 'PS8', 'PS9', 'PS10', 'PS11', 'PS12', 'PS13', 'PS14', 'PS15'
    , 'PS16', 'PS17', 'PS18'
]

for n in range(len(RMPS)):
    RM.append(robot.getPositionSensor(RMPS[n]))
    RM[n].enable(TIME_STEP)

#set TouchSensor     
TS = []
TSS = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']

for n in range(len(TSS)):
    TS.append(robot.getTouchSensor(TSS[n]))
    TS[n].enable(TIME_STEP)

#set GPS
GP = []
GPP = ['gps1']

for g in range(len(GPP)):
    GP.append(robot.getGPS(GPP[g]))
    GP[g].enable(TIME_STEP)

#set COMPASS
CO = []
COM = ['compass1']

for c in range(len(COM)):
    CO.append(robot.getCompass(COM[c]))
    CO[c].enable(TIME_STEP)
            
#set InertialUnit
NIU = []
IU = ['IU']

for U in range(len(IU)):
    NIU.append(robot.getInertialUnit(IU[U]))
    NIU[U].enable(TIME_STEP)

#get motor Value
def PSM():
    PPSM = []
    for m in range(len(RM)):
        PPSM.append(RM[m].getValue())
        #print(RM[m])
    return PPSM
#-----------------------------------------------------------------------------------------------------------------------
#get GPS Values
def Gps():
    gpValues0 = []
    gpValues1 = []
    gpValues2 = []
    gpValues3 = []
    gpValues10 = []
    for G in range(len(GPP)):
        for g in range(len(GP[G].getValues())):
            gpValues0.append(GP[G].getValues()[g])
            gpValues1.append(round(GP[G].getValues()[g], 1))
            gpValues2.append(round(GP[G].getValues()[g], 2))
            gpValues3.append(round(GP[G].getValues()[g], 3))
            gpValues10.append(math.floor(gpValues2[g] * 10))
    return gpValues0, gpValues1, gpValues2, gpValues3, gpValues10   

#get distance
def Distance():
    psValues = []
    for i in range(len(psNames)):
         psValues.append(ps[i].getValue())
    return psValues 

#get COMPASS
def COMPASS():
    cpValies1 = []
    cpValies11 = []
    for C in range(len(COM)):
        for p in range(len(CO[C].getValues())):
            cpValies1.append(abs(round(CO[C].getValues()[p],1)))    
            cpValies11.append(math.floor(round(CO[C].getValues()[p],1)))
    return cpValies1, cpValies11

#get TouchSensor    
def touchSensor():
    tS = []
    for t in range(len(TSS)):
        tS.append(TS[t].getValue())
    
    return tS

def IUU():
    NNIU = []
    for I in range(len(IU)):
        for i in range(len(NIU[I].getRollPitchYaw(),)):
            NNIU.append(round(NIU[0].getRollPitchYaw()[i], 1))
    
    return NNIU

#-----------------------------------------------------------------------------------------------------------------------
def TIMESTOP(T):
    for A in range(T):
        if robot.step(TIME_STEP) == -1:
            break
def initialize():
    for m in range(len(RMnames)):
        Mnames[m].setPosition(math.radians(initmot[m]))
    TIMESTOP(30)
    
            
def INITIAL():
    print("----------------------------INITIAL----------------------------")
    for m in range(len(RMnames)):
        Mnames[m].setPosition(math.radians(initmot[m]))
    INITIAL = [0, 0.2, -4.5]
    ROTATION = [0, 1, 0, 3.141592653589793]
    Rtranslation.setSFVec3f(INITIAL)
    Rrotation.setSFRotation(ROTATION)
    Nrobot.resetPhysics()
    
    INITIALF0 = [0.046775, 0, 0]
    INITIALF1 = [-0.046775, 0, 0]
    ROTATIONF0 = [0, 0, -1, 1.5708]
    ROTATIONF1 = [0, 0, 1, 1.5708]
    TS1translation.setSFVec3f(INITIALF0)
    TS2translation.setSFVec3f(INITIALF0)
    TS3translation.setSFVec3f(INITIALF1)
    TS4translation.setSFVec3f(INITIALF1)
    TS5translation.setSFVec3f(INITIALF1)
    TS6translation.setSFVec3f(INITIALF0)
    TS1rotation.setSFRotation(ROTATIONF0)
    TS2rotation.setSFRotation(ROTATIONF0)
    TS3rotation.setSFRotation(ROTATIONF1)
    TS4rotation.setSFRotation(ROTATIONF1)
    TS5rotation.setSFRotation(ROTATIONF1)
    TS6rotation.setSFRotation(ROTATIONF0)
    TS1.resetPhysics()
    TS2.resetPhysics()
    TS3.resetPhysics()
    TS4.resetPhysics()
    TS5.resetPhysics()
    TS6.resetPhysics()

def reward():
    re = []    
    gax,gay,gaz = Gps()[0][0],Gps()[0][1],Gps()[0][2]
    aa= [i for i, x in enumerate(touchSensor()) if x == 1]
    
    #print(aa)
    if Gps()[0][2] > -4.2:
        re.append(REWARD1)
    #if gax < AGP:
        #print(AGP,gax,"mae")
        #re.append(REWARD1)
    #elif gax > AGP:
        #print(AGP,gax,"usiro")
        #re.append(NREWARD0)
    else:
        re.append(NREWARD1)
    
    if Distance()[0] < 0.1:
        re.append(NREWARD1)
    if Distance()[1] < 0.1:
        re.append(NREWARD1)    
    

        
    #print(re)
    rew = sum(re)    
    return rew
#-----------------------------------------------------------------------------------------------------------------------
def getMotAngles(mode):
    mode1 = [Gps()[3][2]*100, math.degrees(PSM()[0]), math.degrees(PSM()[6]), math.degrees(PSM()[7]), math.degrees(PSM()[1]), math.degrees(PSM()[8]), math.degrees(PSM()[5])]
    
    if mode == 0:       
        for m in range(len(C_STATE)):
            C_STATE[m] = math.floor(mode1[m]) + BiasMot[m]
        else:
            for X in range(len(C_STATE) - len(KbiasMot)):
                if 0 < C_STATE[len(KbiasMot) + X]:
                    C_STATE[len(KbiasMot) + X] = 2
                elif C_STATE[len(KbiasMot) + X] < 0:
                    C_STATE[len(KbiasMot) + X] = 1
                elif C_STATE[len(KbiasMot) + X] == 0:
                    C_STATE[len(KbiasMot) + X] = 0
    elif mode == 1:
        for m in range(len(O_STATE)):
            O_STATE[m] = math.floor(mode1[m] + BiasMot[m])   
        else:
            for Y in range(len(O_STATE) - len(KbiasMot)):
                if 0 < O_STATE[len(KbiasMot) + Y] :
                    O_STATE[len(KbiasMot) + Y] = 2
                elif  O_STATE[len(KbiasMot) + Y] < 0:
                    O_STATE[len(KbiasMot) + Y] = 1
                elif O_STATE[len(KbiasMot) + Y] == 0:
                    O_STATE[len(KbiasMot) + Y] = 0
    mode1 = []       
    
def doAct(n, dir):
    if n == 1:
        va = math.degrees(RM[0].getValue())
        vb = math.degrees(RM[3].getValue())
        vc = math.degrees(RM[12].getValue())
        #print(va,vb,vc)
        rada = va + dir * stepAngle
        radb = vb + dir * stepAngle
        radc = vc + -1 * dir * stepAngle
        #print(initmot[0] + maxAngle1,"<=", rada ,"<=",initmot[0] + minAngle1)
        if initmot[0] + maxAngle1 >= rada or rada <= initmot[0] + minAngle1:
            #print("braek")
            rada = va
        #print(initmot[3] + maxAngle1,"<=", radb ,"<=",initmot[3] + minAngle1)
        if initmot[3] + maxAngle1 >= radb or radb <= initmot[3] + minAngle1:
            #print("braek")
            radb = vb
        #print(initmot[12] + maxAngle1,"<=", radc ,"<=",initmot[12] + minAngle1)
        if initmot[12] + maxAngle1 >= radc or radc <= initmot[12] + minAngle1:
            #print("braek")
            radc = vc
        #print(rada,radb,radc)
        Mnames[0].setPosition(math.radians(rada))
        Mnames[3].setPosition(math.radians(radb))
        Mnames[12].setPosition(math.radians(radc))

    elif n == 2:
        va = math.degrees(RM[15].getValue())
        vb = math.degrees(RM[9].getValue())
        vc = math.degrees(RM[6].getValue())
        #print(va,vb,vc)
        rada = va + -1 * dir * stepAngle
        radb = vb + dir * stepAngle
        radc = vc + dir * stepAngle
        #print(initmot[15] + maxAngle1,"<=", rada ,"<=",initmot[15] + minAngle1)
        if initmot[15] + maxAngle1 <= rada or rada <= initmot[15] + minAngle1:
            #print("braek")
            rada = va
        #print(initmot[9] + maxAngle1,"<=", radb ,"<=",initmot[9] + minAngle1)
        if initmot[9] + maxAngle1 <= radb or radb <= initmot[9] + minAngle1:
            #print("braek")
            radb = vb
        #print(initmot[6] + maxAngle1,"<=", radc ,"<=",initmot[6] + minAngle1)
        if initmot[6] + maxAngle1 <= radc or radc <= initmot[6] + minAngle1:
            #print("braek")
            radc = vc 
        #print(rada,radb,radc)
        Mnames[15].setPosition(math.radians(rada))
        Mnames[9].setPosition(math.radians(radb))
        Mnames[6].setPosition(math.radians(radc))

    elif n == 3:
        va = math.degrees(RM[16].getValue())
        vb = math.degrees(RM[10].getValue())
        vc = math.degrees(RM[7].getValue())
        #print(va,vb,vc)
        rada = va + -1 * dir * stepAngle
        radb = vb + dir * stepAngle
        radc = vc + dir * stepAngle
        #print(initmot[16] + maxAngle1,"<=", rada ,"<=",initmot[16] + minAngle1)
        if initmot[16] + maxAngle1 <= rada or rada <= initmot[16] + minAngle1:
            #print("braek")
            rada = va
        #print(initmot[10] + maxAngle1,"<=", radb ,"<=",initmot[10] + minAngle1)
        if initmot[10] + maxAngle1 <= radb or radb <= initmot[10] + minAngle1:
            #print("braek")
            radb = vb
        #print(initmot[7] + maxAngle1,"<=", radc ,"<=",initmot[7] + minAngle1)
        if initmot[7] + maxAngle1 <= radc or radc <= initmot[7] + minAngle1:
            #print("braek")
            radc = vc  
        #print(rada,radb,radc)
        Mnames[16].setPosition(math.radians(rada))
        Mnames[10].setPosition(math.radians(radb))
        Mnames[7].setPosition(math.radians(radc))
                    
    elif n == 4:
        va = math.degrees(RM[13].getValue())
        vb = math.degrees(RM[1].getValue())
        vc = math.degrees(RM[4].getValue())
        #print(va,vb,vc)
        rada = va + -1 * dir * stepAngle
        radb = vb + dir * stepAngle
        radc = vc + dir * stepAngle
        #print(initmot[13] + maxAngle1,"<=", rada ,"<=",initmot[13] + minAngle1)
        if initmot[13] + maxAngle1 <= rada or rada <= initmot[13] + minAngle1:
            #print("braek")
            rada = va
        #print(initmot[1] + maxAngle1,"<=", radb ,"<=",initmot[1] + minAngle1)
        if initmot[1] + maxAngle1 <= radb or radb <= initmot[1] + minAngle1:
            #print("braek")
            radb = vb
        #print(initmot[4] + maxAngle1,"<=", radc ,"<=",initmot[4] + minAngle1)
        if initmot[4] + maxAngle1 <= radc or radc <= initmot[4] + minAngle1:
            #print("braek")
            radc = vc  
        #print(rada,radb,radc)
        Mnames[13].setPosition(math.radians(rada))
        Mnames[1].setPosition(math.radians(radb))
        Mnames[4].setPosition(math.radians(radc))
        
    elif n == 5:
        va = math.degrees(RM[11].getValue())
        vb = math.degrees(RM[17].getValue())
        vc = math.degrees(RM[8].getValue())
        #print(va,vb,vc)
        rada = va + dir * stepAngle
        radb = vb + -1 * dir * stepAngle
        radc = vc + dir * stepAngle
        #print(initmot[11] + maxAngle1,"<=", rada ,"<=",initmot[11] + minAngle1)
        if initmot[11] + maxAngle1 <= rada or rada <= initmot[11] + minAngle1:
            #print("braek")
            rada = va
        #print(initmot[17] + maxAngle1,"<=", radb ,"<=",initmot[17] + minAngle1)
        if initmot[17] + maxAngle1 <= radb or radb <= initmot[17] + minAngle1:
            #print("braek")
            radb = vb
        #print(initmot[8] + maxAngle1,"<=", radc ,"<=",initmot[8] + minAngle1)
        if initmot[8] + maxAngle1 <= radc or radc <= initmot[8] + minAngle1:
            #print("braek")
            radc = vc  
        #print(rada,radb,radc)
        Mnames[11].setPosition(math.radians(rada))
        Mnames[17].setPosition(math.radians(radb))
        Mnames[8].setPosition(math.radians(radc))
        
    elif n == 6:
        va = math.degrees(RM[14].getValue())
        vb = math.degrees(RM[2].getValue())
        vc = math.degrees(RM[5].getValue())
        #print(va,vb,vc)
        rada = va + dir * stepAngle
        radb = vb + -1* dir * stepAngle
        radc = vc + -1 * dir * stepAngle
        #print(initmot[14] + maxAngle1,"<=", rada ,"<=",initmot[14] + minAngle1)
        if initmot[14] + maxAngle1 <= rada or rada <= initmot[14] + minAngle1:
            #print("braek")
            rada = va
        #print(initmot[2] + maxAngle1,"<=", radb ,"<=",initmot[2] + minAngle1)
        if initmot[2] + maxAngle1 <= radb or radb <= initmot[2] + minAngle1:
            #print("braek")
            radb = vb
        #print(initmot[5] + maxAngle1,"<=", radc ,"<=",initmot[5] + minAngle1)
        if initmot[5] + maxAngle1 <= radc or radc <= initmot[5] + minAngle1:
            #print("braek")
            radc = vc 
        #print(rada,radb,radc)
        Mnames[14].setPosition(math.radians(rada))
        Mnames[2].setPosition(math.radians(radb))
        Mnames[5].setPosition(math.radians(radc))
         
def actionSelect():
    value = numpy.zeros(act)
    proba = numpy.zeros(act)
    total = 0
    ret = 0
    for N in range(0,act):
        value[N] = POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][n]
    for M in range(0,act):
        total = total + numpy.exp(value[M]/BOLTZMANN)
    for L in range(0,act):
        proba[L] = (numpy.exp(value[L]/BOLTZMANN))/total  
    
    c = random.random()

    if c >= 0.0 and c < proba[0]:
        doAct(1, 1)
        ret = 0
    elif c >= proba[0] and c < (proba[0] + proba[1]):
        doAct(1, -1)
        ret = 1
    elif c >= (proba[0] + proba[1]) and c < (proba[0] + proba[1] + proba[2]):
        doAct(2, 1)
        ret = 2
    elif c >= (proba[0] + proba[1] + proba[2]) and c < (proba[0] + proba[1] + proba[2] + proba[3]):
        doAct(2, -1)
        ret = 3
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4]):
        doAct(3, 1)
        ret = 4
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5]):
        doAct(3, -1)
        ret = 5
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6]):
        doAct(4, 1)
        ret = 6
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7]):
        doAct(4, -1)
        ret = 7
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8]):
        doAct(5, 1)
        ret = 8
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8] + proba[9]):
        doAct(5, -1)
        ret = 9
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8] + proba[9]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8] + proba[9] + proba[10]):
        doAct(6, 1)
        ret = 10
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8] + proba[9] + proba[10]) and c <= 1.0:
        doAct(6, -1)
        ret = 11   

    return ret  

#-----------------------------------------------------------------------------------------------------------------------
def maxQ():
    tmpValue = numpy.zeros(act)
    for a in range (act):
        tmpValue[a] = POLICY[C_STATE[0]][C_STATE[1]][C_STATE[2]][C_STATE[3]][C_STATE[4]][C_STATE[5]][C_STATE[6]][a]
    ret = numpy.argmax(tmpValue)
    return ret

def updateQ(tmpR, tmpAct):
    TDerror = DRATE * POLICY[C_STATE[0]][C_STATE[1]][C_STATE[2]][C_STATE[3]][C_STATE[4]][C_STATE[5]][C_STATE[6]][maxQ()] - POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][tmpAct]
    POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][tmpAct] = POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][tmpAct] + LRATE * (tmpR + TDerror)

#-----------------------------------------------------------------------------------------------------------------------
if os.path.isfile(filePolicy):
    POLICY = numpy.load(filePolicy)
else:
    POLICY = numpy.zeros((mot[0], mot[1], mot[2], mot[3], mot[4], mot[5], mot[6], act))
  
if os.path.isfile(fileEpisode):    # if episodes file is exit, load latest saved data for variables
    tmpInfo = numpy.load(fileEpisode)
    NEPISODE = int(tmpInfo[0][(len(tmpInfo[0]))-1])
    NSTEP = int(tmpInfo[1][(len(tmpInfo[1]))-1])
    TREWARD = tmpInfo[2][(len(tmpInfo[2]))-1]
    EPISODES = list(tmpInfo[0])
    STEPS = list(tmpInfo[1])
    REWARDS = list(tmpInfo[2])
    flag = 1
    
if flag == 0:  # Normal start mode
    print('---- Reinforcement learning start ----')
    numpy.save(filePolicy, POLICY)  # Save policy as npz file
    EPISODES.append(NEPISODE)
    STEPS.append(NSTEP)
    REWARDS.append(TREWARD)
elif flag == 1:  # Restart mode
    print('---- Reinforcement learning REstart ----')
    flag = 0  # Reset restart flag
    
    
INITIAL()        
print("strat")
TIMESTOP(30)   
while NEPISODE <= MAXEPISODE:
    while goal == 0 and robot.step(TIME_STEP) != -1 and NSTEP <= MAXSTEP:
        print('Old state', O_STATE)
        #for i in range(len(RMPS)):
            #print(math.degrees(RM[i].getValue()))
        AGP = Gps()[0][2] + 100
        getMotAngles(1)
        execdAct = actionSelect()
        TIMESTOP(50)
        getMotAngles(0)
        TIMESTOP(30)
        r = reward()
        updateQ(r, execdAct)
        TREWARD = TREWARD + r
     
        print('Current State', C_STATE)
        print('Episode', NEPISODE, ', Step', NSTEP, ', Reward', r, ', Total reward', TREWARD)
        print('Executed action number is ', execdAct, ' and its value', POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][execdAct])
        STEPS[(NEPISODE - 1)] = NSTEP
        REWARDS[(NEPISODE - 1)] = TREWARD
        
        
        if Gps()[0][2] > -4.2:
            goal = 1
            print("GOAL",'Episode', NEPISODE)
        NSTEP = NSTEP + 1
    initialize()        
    INITIAL()  
    EPISODES.append(NEPISODE)
    STEPS.append(NSTEP)
    REWARDS.append(TREWARD)  
    goal = 0
    NEPISODE = NEPISODE + 1
    NSTEP = 1
    TREWARD = 0

    
    
tmpEpisode = numpy.array([EPISODES, STEPS, REWARDS])
numpy.save(fileEpisode, tmpEpisode)
numpy.save(filePolicy, POLICY)  # Save policy as npz file

print('---- Reinforcement learning finish ----')
logEpisode = tmpEpisode.T
DATE = datetime.now().strftime("%Y%m%d_%H%M%S")
fileLog = open("./episode_log_" + DATE + ".csv", 'w', newline='')
writer = csv.writer(fileLog)
writer.writerows(logEpisode)
fileLog.close()
    
    
    
   
    
    
    
    
    
    