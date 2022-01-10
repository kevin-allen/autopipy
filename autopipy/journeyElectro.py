import os.path
import pandas as pd
import numpy as np
from scipy import stats
import cv2
import sys
from autopipy.navPath import NavPath
from autopipy.lever import Lever
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class JourneyElectro:
    """
    Class containing information about a single journey on the Autopi task. It is specifically built to work with experiments in which the animal position comes from spikeA AnimalPose Class
    
    
    This is used to analyze all journeys of a trial while keeping the trial class relatively simple
     
    It contains a dictionary of NavPath objects (pathD). 
    
    We develop this class because we might want to consider the mouse behavior during all journeys, not just during the journeys when the lever was pressed.
    
    Attributes:
        name: Name of the trial, usually sessionName_trialNo-JourneyNo
        sessionName: Name of the session in which the trial was performed
        trialNo: Trial number within the session
        jouneyNo: Jouney number within the trial
        mousePose: DataFrame with 
        
        ...
    
    Methods:
          
    """
    def __init__(self,sessionName,trialNo,journeyNo,
                 startIndex,endIndex,
                 lever, 
                 zones,
                 arenaRadiusProportionToPeri,
                 leverPresses,
                 mousePose,
                 positionZones):
        
        self.sessionName = sessionName
        self.trialNo = trialNo
        self.journeyNo = journeyNo
        self.name="{}_{}-{}".format(self.sessionName,self.trialNo,self.journeyNo)
        self.startIndex = startIndex
        self.endIndex = endIndex
        
        self.lever = lever
        self.zones = zones
        self.arenaRadiusProportionToPeri = arenaRadiusProportionToPeri
        self.arenaRadius = self.zones["arena"][2]
      
        self.mousePose = mousePose.loc[self.startIndex:self.endIndex]
        self.startTime= self.mousePose.time.iloc[0]
        self.endTime = self.mousePose.time.iloc[-1]
        self.duration = self.endTime-self.startTime
        
        self.leverPresses = leverPresses[leverPresses.time.between(self.startTime,self.endTime)]
        self.nLeverPresses = len(self.leverPresses)
        self.positionZones = positionZones.loc[self.startIndex:self.endIndex]
        
        self.cutAtLastArenaBridgeTransition()
        self.createNavPaths()
        
    def cutAtLastArenaBridgeTransition(self):
        """
        Method to adjust the end of the journey to the time point at which the animal transitioned from the arena to the bridge.
        
        """
        
        lastArena=self.positionZones[self.positionZones.loca=="arena"].iloc[-1:].index.values[0]
        endingDf = self.positionZones.loc[lastArena:,:]
        
        if len(endingDf.loca)!=0:
            if np.sum(endingDf.loca=="bridge") > 0:
                journeyAtBridgeEndIndex = endingDf.loc[endingDf.loca=="bridge"].index.values[0]
                #print("initial end index:", self.endIndex, " new end index:", journeyAtBridgeEndIndex)
                self.endIndex=journeyAtBridgeEndIndex
                self.mousePose = self.mousePose.loc[:self.endIndex,:]
                self.endTime = self.mousePose.time.iloc[-1]
                self.duration = self.endTime-self.startTime
                self.positionZones = self.positionZones.loc[self.startIndex:self.endIndex]
        
        
    def poseForNavPath(self,startTime,endTime):
        """
        Create a numpy array containing the pose (x, y, z, yaw, pitch, roll, time) during a defined time period
        Arguments:
            startTime: start time of the path
            endTime: end time of the path

        Return Pose as a numpy array with 7 columns

        """
        indices = self.mousePose.time.between(startTime,endTime)
        jnk = self.mousePose.x.loc[indices].to_numpy()
        
        
        return np.stack((self.mousePose.x.loc[indices].to_numpy(),
                         self.mousePose.y.loc[indices].to_numpy(),
                         np.zeros_like(jnk),
                         np.zeros_like(jnk),
                         np.zeros_like(jnk),
                         np.zeros_like(jnk),
                         self.mousePose.time.loc[indices].to_numpy(),
                         self.mousePose.resTime.loc[indices].to_numpy()
                        )                        
                    ,axis=1)
    def createNavPath(self,startTime,endTime,target=None,name="navPath"):
        """
        Wrapper to get the NavPath
        """
        mousePose = self.poseForNavPath(startTime,endTime)
        return NavPath(pPose = mousePose[:,0:7], targetPose=None,name = name,resTime=mousePose[:,7])

    def createNavPaths(self):
        """
        Method that fills up the self.navPaths dictionary
        """
        self.navPaths={}
        self.navPaths["all"] = self.createNavPath(self.startTime,self.endTime,target=None,name=self.name+"_"+"all")

        if self.nLeverPresses > 0:

            # leverPressTime
            lpt = self.leverPresses.time.iloc[0]

            # get a boolean indicating whether at lever or not in the self.mousePose
            posi = np.stack([self.mousePose.x,self.mousePose.y],axis=1)
            atLever = self.lever.isAt(posi)
            self.mousePose["atLever"] = atLever

            
             # search before lever press
            self.navPaths["searchPath"] = self.createNavPath(self.startTime,lpt,target=None,name=self.name+"_"+"searchPath")

            
            # arriving at lever before lever press
            beforePress = self.mousePose.loc[self.mousePose.time < lpt]
            arrivingAtLeverTime = beforePress.time.loc[beforePress.atLever==False].max()
            # from start to arriving at the lever before the press
            self.navPaths["searchToLeverPath"] = self.createNavPath(self.startTime,arrivingAtLeverTime,target=None,name=self.name+"_"+"searchToLeverPath")

           
            # after lever press
            self.navPaths["homingPath"] = self.createNavPath(lpt,self.endTime,target=None,name=self.name+"_"+"homingPath")
            
            # leaving lever after lever press
            afterPress = self.mousePose.loc[self.mousePose.time > lpt]
            leavingLeverTime = afterPress.time.loc[afterPress.atLever==False].min()
            # from leaving the lever after the press to end
            self.navPaths["homingFromLeavingLever"] = self.createNavPath(leavingLeverTime,self.endTime,target=None,name=self.name+"_"+"homingFromLeavingLever")


            
            
            
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    def plotPath(self,ax,pltAllPaths=True):
        """
        Plot the path of the mouse for the entire journey
        """
        if pltAllPaths:
            for k in self.navPaths.keys():
                ax.plot(self.navPaths[k].pPose[:,0],self.navPaths[k].pPose[:,1],label=k)
        else:
            ax.plot(self.mousePose.x,self.mousePose.y,color="black", label="path",zorder=1)
       
   
    def plotLeverPresses(self,ax):
        """
        Plot the position of the animal when the mouse pressed the lever
        """
        if self.nLeverPresses != 0:
            ax.scatter(self.leverPresses.mouseX,self.leverPresses.mouseY,c="red",zorder=2,s=5)
