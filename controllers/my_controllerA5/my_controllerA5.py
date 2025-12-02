from controller import  Motor, GPS, DistanceSensor,PositionSensor, Supervisor, InertialUnit, Compass, TouchSensor
import math
import random
import time
import scipy
import numpy
import numpy.linalg as LA
from decimal import *
import csv
from datetime import datetime
import os
import struct


#-----------------------------------------------------------------------------------------------------------------------
TIME_STEP = 64
speed = 0
#-----------------------------------------------------------------------------------------------------------------------

StartingPoint = [0, 0.2, -4.5]
StartDirection = [0, 1, 0, 3.141592653589793]

#右　0, 1, 0, 1.5707963267948966
#左 0, -1, 0, 1.5707963267948966
#上　0, 1, 0, 0
#下　0, 1, 0, 3.141592653589793

INITIALF0 = [0.046775, 0, 0]
INITIALF1 = [-0.046775, 0, 0]
ROTATIONF0 = [0, 0, -1, 1.5708]
ROTATIONF1 = [0, 0, 1, 1.5708]




GoalPoint = [0, 0, -4.2]












stepAngle = 15
maxAngle1 = 16
minAngle1 = -16
#-----------------------------------------------------------------------------------------------------------------------
goal = 0
NAgent = ["IRSL-XR06-01" ,"IRSL-XR06-02"]
Agent = 0
RL = 0
#RL = 0 強化
#RL = 1 転移
T = 1
#転移率

REWARD1 = 1
NREWARD1 = -0.05
NREWARD2 = -0.1
TREWARD = 0


NSTEP = 1
NEPISODE = 1

LRATE = 0.1
DRATE = 0.99
BOLTZMANN = 0.1

MAXEPISODE = 1000

EPISODES = []
STEPS = []
REWARDS = []


EPISODES.append(NEPISODE)
STEPS.append(NSTEP)
REWARDS.append(TREWARD)
#-----------------------------------------------------------------------------------------------------------------------
C_STATE = [0, 0, 0, 0, 0, 0, 0]
O_STATE = [0, 0, 0, 0, 0, 0, 0]

mot = [35, 3, 3, 3, 3, 3, 3]
act = 12


KbiasMot = [450] #environment bias
MbiasMot = [45, 45, 90, -90, -130, 130] #motor bias

#-----------------------------------------------------------------------------------------------------------------------
BiasMot = [100, 1, 1, 1, 1, 1, 1]
biasMot = [450,45, 45, 90, -90, -130, 130]
initmot = [-45, 90, -130, 45, 90, -130, -45, -90, 130, 45, -90, 130, 0, -90, 130, 0, 90, -130]
#-----------------------------------------------------------------------------------------------------------------------
X0 = []
X1 = []
X2 = []
G2 = [] 
G1 = []
G0 = []

NNG = []
NGP = []
  
NUA = []

ANA = []
#-----------------------------------------------------------------------------------------------------------------------
DATE = datetime.now().strftime("%Y%m%d_%H%M%S")
filePolicy = "./source/policy" + DATE +".npy"
filePolicy2 = "./Policy/policy.npy"


#-----------------------------------------------------------------------------------------------------------------------
#set Supervisor
robot = Supervisor()
Nrobot = robot.getFromDef(NAgent[Agent])
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
psNames = ['DS1', 'DS2']

for i in range(len(psNames)):
    ps.append(robot.getDistanceSensor(psNames[i]))
    ps[i].enable(TIME_STEP)
    
#set motor
RAM = []
RMnames = [
    'motor1', 'motor2', 'motor3', 'motor4', 'motor5', 'motor6', 'motor7', 'motor8', 'motor9', 'motor10'
    , 'motor11', 'motor12', 'motor13', 'motor14', 'motor15', 'motor16', 'motor17', 'motor18'
]

for i in range(len(RMnames)):
    RAM.append(robot.getMotor(RMnames[i]))    

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
       
#set InertialUnit
NIU = []
IU = ['IU']

for U in range(len(IU)):
    NIU.append(robot.getInertialUnit(IU[U]))
    NIU[U].enable(TIME_STEP)    

#set COMPASS
CO = []
COM = ['compass1']

for c in range(len(COM)):
    CO.append(robot.getCompass(COM[c]))
    CO[c].enable(TIME_STEP)   

#-----------------------------------------------------------------------------------------------------------------------
#get motor Value
def PSM():
    PPSM = []
    
    for m in range(len(RM)):
        PPSM.append(math.degrees(RM[m].getValue()))

    return PPSM
    
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
            #print(gpValues0,gpValues1,gpValues2)
            
    return gpValues0, gpValues1, gpValues2, gpValues3, gpValues10    

#get TouchSensor    
def touchSensor():
    tS = []
    for t in range(len(TSS)):
        tS.append(TS[t].getValue())
    
    return tS
    
#get distance
def Distance():
    psValues = []
    
    for i in range(len(psNames)):
         psValues.append(ps[i].getValue())
         
    return psValues 

#get InertialUnit  
def IUU():
    NNIU = []
    
    for I in range(len(IU)):
        for i in range(len(NIU[I].getRollPitchYaw(),)):
            NNIU.append(round(NIU[0].getRollPitchYaw()[i], 1))
    
    return NNIU    

#get COMPASS
def COMPASS():
    cpValies1 = []
    cpValies11 = []
    
    for C in range(len(COM)):
        for p in range(len(CO[C].getValues())):
            cpValies1.append(abs(round(CO[C].getValues()[p],2)))    
            cpValies11.append(math.floor(round(CO[C].getValues()[p],0)))
            
    return cpValies1, cpValies11   
    
    
#-----------------------------------------------------------------------------------------------------------------------
def TIMESTOP(T):
    for A in range(T):
        if robot.step(TIME_STEP) == -1:
            break  

def INITIAL():
    print("----------------------------INITIAL----------------------------")
    for m in range(len(RMnames)):
        RAM[m].setPosition(math.radians(initmot[m]))
    TIMESTOP(30)
    Rtranslation.setSFVec3f(StartingPoint)
    Rrotation.setSFRotation(StartDirection)
    Nrobot.resetPhysics()
  
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
    if GoalPoint[2] <= Gps()[0][2]: 
        re.append(REWARD1)

    else:
        re.append(NREWARD1)
    
    if Distance()[0] < 0.1:
        re.append(NREWARD1)
    if Distance()[1] < 0.1:
        re.append(NREWARD1) 
           


    rew = sum(re)    
    
    return rew   
#-------------------------------------------------------------------------------------------------------------------
TIMESTOP(5)
for A1 in range(act):
    G = []
    P = []

    for A2 in range(len(IUU())):
        G.append(0)
    for A1 in range(len(Gps()[0])):
        P.append(0)
    NNG.append(G)  
    NGP.append(P) 
   
#-----------------------------------------------------------------------------------------------------------------------
def getBCUs(MODE):
    if MODE == 0:
        for XX1 in range(len(IUU())):
            U1 = math.degrees(IUU()[XX1]) + 180
            X1.append(U1)
        for GGG1 in range(len(Gps()[0])):
            NG1 = Gps()[0][GGG1]
            G1.append(NG1)
        
    elif MODE == 1:
        for XX0 in range(len(IUU())):
            U0 = math.degrees(IUU()[XX0]) + 180
            X0.append(U0)
        for GGG0 in range(len(Gps()[0])):
            NG0 = Gps()[0][GGG0]
            G0.append(NG0)



        
#-----------------------------------------------------------------------------------------------------------------------
def getMotAngles(mode):
    
    mode1 = [Gps()[3][2], PSM()[0], PSM()[6], PSM()[7], PSM()[1], PSM()[8], PSM()[5]]
    if mode == 0:       
        for m in range(len(C_STATE)):
            C_STATE[m] = math.floor(mode1[m] * BiasMot[m] + biasMot[m])
        else:
            for X in range(len(C_STATE) - len(KbiasMot)):
                if 0 < C_STATE[len(KbiasMot) + X]:
                    C_STATE[len(KbiasMot) + X] = 2
                elif C_STATE[len(KbiasMot) + X] < 0:
                    C_STATE[len(KbiasMot) + X] = 1
                elif  C_STATE[len(KbiasMot) + X] == 0:
                    C_STATE[len(KbiasMot) + X] = 0
    elif mode == 1:
        for m in range(len(O_STATE)):
            O_STATE[m] = math.floor(mode1[m] * BiasMot[m] + biasMot[m])
        else:
            for Y in range(len(O_STATE) - len(KbiasMot)):
                if 0 < O_STATE[len(KbiasMot) + Y] :
                    O_STATE[len(KbiasMot) + Y] = 2
                elif  O_STATE[len(KbiasMot) + Y] < 0:
                    O_STATE[len(KbiasMot) + Y] = 1
                elif O_STATE[len(KbiasMot) + Y] == 0:
                    O_STATE[len(KbiasMot) + Y] = 0
           
    mode1 = []     
    
    
    
    
    
def doAct0(dir , m1 , m2 , m3):
    va = PSM()[m1]
    vb = PSM()[m2]
    vc = PSM()[m3]
    print(va,vb,vc)
    rada = va + dir * stepAngle
    radb = vb + dir * stepAngle
    radc = vc + -1 * dir * stepAngle
    if initmot[m1] + maxAngle1 >= rada or rada <= initmot[m1] + minAngle1:
            #print("braek")
        rada = va
        #print(initmot[3] + maxAngle1,"<=", radb ,"<=",initmot[3] + minAngle1)
    if initmot[m2] + maxAngle1 >= radb or radb <= initmot[m2] + minAngle1:
            #print("braek")
        radb = vb
        #print(initmot[12] + maxAngle1,"<=", radc ,"<=",initmot[12] + minAngle1)
    if initmot[m3] + maxAngle1 >= radc or radc <= initmot[m3] + minAngle1:
            #print("braek")
        radc = vc
    RAM[m1].setPosition(math.radians(rada))
    RAM[m2].setPosition(math.radians(radb))
    RAM[m3].setPosition(math.radians(radc))    
    
    
       
def doAct2(n , num):
    if n == 1:
        speed = 0.05
        for n in range(len(RMnames)):
            RAM[n].setVelocity(speed * num)
        ga = Gps()[3]
        gb1 = Gps()[3]
        while robot.step(TIME_STEP) != -1:
            if abs(ga[0] - gb1[0]) >= 0.248 or abs(ga[2] - gb1[2]) >= 0.248:
                speed = 0
                for n in range(len(RMnames)):
                    RAM[n].setVelocity(speed)
                getBCUs(0)
                break
            gb1 = Gps()[3]

    
    elif n == 2:
        speed = 0.05
        RAM[0].setVelocity(speed * num)

        RAM[1].setVelocity(speed * num * -1)

        ca = COMPASS()[0]
        
        aa= [i for i, x in enumerate(ca) if x == 1]
        aav = aa[0]
        ga = Gps()[3]
        gb1 = Gps()[3]
        while robot.step(TIME_STEP) != -1: 
            if ca[aav] == 0:
                speed = 0
                for n in range(len(RMnames)):
                    RAM[n].setVelocity(speed)
                TIMESTOP(1)   
                speed = 0.05
                for n in range(len(RMnames)):
                    RAM[n].setVelocity(speed)
                while robot.step(TIME_STEP) != -1:
                    if abs(ga[0] - gb1[0]) >= 0.248 or abs(ga[2] - gb1[2]) >= 0.248:
                        speed = 0
                        for n in range(len(RMnames)):
                            RAM[n].setVelocity(speed)
                        break
                    gb1 = Gps()[3] 
                   
                break
            ca = COMPASS()[0]
          
                   

            
def actionSelect():
    psValues = []
    value = numpy.zeros(act)
    proba = numpy.zeros(act)
    total = 0
    ret = 0
    for N in range(0,act):
        value[N] = POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][n]
    for m in range(0,act):
        total = total + numpy.exp(value[m]/BOLTZMANN)
    for l in range(0,act):
        proba[l] = (numpy.exp(value[l]/BOLTZMANN))/total  
    
 
    
    D = Distance()
    
    c = random.random()
    
    if c >= 0.0 and c < proba[0]:
        if Agent == 0:
            doAct0(1, 0, 3, 12)
            ret = 0
        
    elif c >= proba[0] and c < (proba[0] + proba[1]):
        if Agent == 0:
            doAct0(-1, 0, 3, 12)
            ret = 1
        
    elif c >= (proba[0] + proba[1]) and c < (proba[0] + proba[1] + proba[2]):
        if Agent == 0:
            doAct0(1, 6, 9, 15)
            ret = 2
        
    elif c >= (proba[0] + proba[1] + proba[2]) and c < (proba[0] + proba[1] + proba[2] + proba[3]):
        if Agent == 0:
            doAct0(-1, 6, 9, 15)
            ret = 3
        
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4]):
        if Agent == 0:
            doAct0(1, 7, 10, 16)
            ret = 4
        
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5]):
        if Agent == 0:
            doAct0(-1, 7, 10, 16)
            ret = 5
        
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6]):
        if Agent == 0:
            doAct0(1, 1, 4, 13)
            ret = 6
        
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7]):
        if Agent == 0:
            doAct0(-1, 1, 4, 13)
            ret = 7
        
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8]):
        if Agent == 0:
            doAct0(1, 8, 11, 17)
            ret = 8
       
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8] + proba[9]):
        if Agent == 0:
            doAct0(-1, 8, 11, 17)
            ret = 9
        
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8] + proba[9]) and c < (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8] + proba[9] + proba[10]):
        if Agent == 0:
            doAct0(1, 2, 5, 14)
            ret = 10
        
    elif c >= (proba[0] + proba[1] + proba[2] + proba[3] + proba[4] + proba[5] + proba[6] + proba[7] + proba[8] + proba[9] + proba[10]) and c <= 1.0:
        if Agent == 0:
            doAct0(-1, 2, 5, 14)
            ret = 11
        
    
    
    
            
            
            
            
    
    
    return ret               
    
       
#-----------------------------------------------------------------------------------------------------------------------
def maxQ():
    tmpValue = numpy.zeros(act)
    for a in range (act):
        tmpValue[a] = POLICY[C_STATE[0]][C_STATE[1]][C_STATE[2]][C_STATE[3]][C_STATE[4]][C_STATE[5]][C_STATE[6]][a]
    ret = numpy.argmax(tmpValue)
    
    if RL == 1:
        ret = ret * T    
    return ret

def updateQ(tmpR, tmpAct):
    TDerror = DRATE * POLICY[C_STATE[0]][C_STATE[1]][C_STATE[2]][C_STATE[3]][C_STATE[4]][C_STATE[5]][C_STATE[6]][maxQ()] - POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][tmpAct]
    POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][tmpAct] = POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][tmpAct] + LRATE * (tmpR + TDerror)


#-----------------------------------------------------------------------------------------------------------------------

   
 
def updateGYRO(AN):
    #print("------------------------NUPGYRO------------------------")
    
 
    
    for aG in range(len(IUU())):
        X2.append(X0[aG] - X1[aG])
    
    if X2[0] == 0 and X2[1] == 0 and X2[2] == 0:
        pass
    else:
        
        for N in range(len(IUU())):
            H = X2[N]/ 360
            NNG[AN][N] = NNG[AN][N] + H
            
               
               
                
    

def updateCoordinate(AN):
    print("------------------------NUPCoordinate------------------------")     
    for Ga in range(len(Gps()[0])):
        G2.append(G0[Ga] - G1[Ga])
    if G2[0] == 0 and G2[1] == 0 and G2[2] == 0:
        pass
    else:
        for PN in range(len(Gps()[0])):
            NGP[AN][PN] = NGP[AN][PN] + G2[PN]
            
            
                
                
    G0.clear()
    G1.clear()
    G2.clear()
    X0.clear()
    X1.clear()
    X2.clear() 
    
    
    
    
    
    
#-----------------------------------------------------------------------------------------------------------------------

if os.path.isfile(filePolicy2) and RL == 1:
    POLICY = numpy.load(filePolicy2)
elif RL == 0:
    POLICY = numpy.zeros((mot[0], mot[1], mot[2], mot[3], mot[4], mot[5], mot[6], act))

INITIAL()
print("start")
while NEPISODE <= MAXEPISODE:
    while goal == 0 and robot.step(TIME_STEP) != -1:
        getBCUs(1)
        getMotAngles(1)
        TIMESTOP(150)
        execdAct = actionSelect()
        TIMESTOP(150)
        getMotAngles(0)
        getBCUs(0)
        r = reward()
        updateQ(r, execdAct)
        updateGYRO(execdAct)
        updateCoordinate(execdAct)
        TREWARD = TREWARD + r
        print(NNG)
        print(NGP)
        print('Old state', O_STATE)
        print('Current State', C_STATE)
        print('Episode', NEPISODE, ', Step', NSTEP, ', Reward', r, ', Total reward', TREWARD)
        print('Executed action number is ', execdAct, ' and its value', POLICY[O_STATE[0]][O_STATE[1]][O_STATE[2]][O_STATE[3]][O_STATE[4]][O_STATE[5]][O_STATE[6]][execdAct])
        STEPS[(NEPISODE - 1)] = NSTEP
        REWARDS[(NEPISODE - 1)] = TREWARD
        if (GoalPoint[2] - 0.125) < Gps()[3][2] < (GoalPoint[2] + 0.125) and (GoalPoint[0] - 0.125) < Gps()[3][0] < (GoalPoint[0] + 0.125):
            goal = 1
            print("GOAL",'Episode', NEPISODE)
        NSTEP = NSTEP + 1
        print("----------------------------------------------------------------------------------------------------------------------------------------")    
    INITIAL()  
    
      
    goal = 0
    
    NEPISODE = NEPISODE + 1
    EPISODES.append(NEPISODE)
    STEPS.append(NSTEP)
    REWARDS.append(TREWARD) 
    NSTEP = 1
    TREWARD = 0

    


BCUv1 = numpy.array([ANA,NGP,NNG]) 
tmpEpisode = numpy.array([EPISODES, STEPS, REWARDS])
if RL == 0:
    numpy.save(filePolicy, POLICY)  # Save policy as npz file
    fileLog = open("./source/episode_log_" + DATE + ".csv", 'w', newline='')
elif RL == 1:
    numpy.save(filePolicy2, POLICY)  # Save policy as npz file
    fileLog = open("./Policy/episode_log_" + DATE + ".csv", 'w', newline='')
    
print('---- Reinforcement learning finish ----')
logEpisode = tmpEpisode.T
writer = csv.writer(fileLog)
writer.writerows(logEpisode)
fileLog.close()

print('---- BC finish ----')
Log = open("./source/BCU_log.csv", 'w', newline='')
BCU = BCUv1.T
writer = csv.writer(Log)
writer.writerows(BCU)
fileLog.close()


































































