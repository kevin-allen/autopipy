import os.path
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from scipy import stats
import cv2
import sys
from autopipy.journey import Journey
from autopipy.navPath import NavPath
from autopipy.lever import Lever
import matplotlib.pyplot as plt

class TrialElectro:
    """
    Class containing information about a single trial on the Autopi task when done with electrophysiology data.
    
    When calculating angles, the y coordinate is reversed (0-y) so that 90 degree is up.
    This is done because the y-axis is reversed in the videos.
    
    The position data given to this class should be in cm, with 0,0 being the center of the arena.
    
    Contains a list of JourneyElectro (exploratory episode on the arena)
    
    The time inside this object is in ROS time.
    
    The TrialElectro class has a pathD (dictionary of NavPath) which is just copied from the JourneyElectro with the first lever press.
    
    
     
    Attributes:
        name: Name of the trial, usually session_trialNo
        sessionName: Name of the session in which the trial was performed
        trialNo: Trial number within the session
        startTime: start time of the trial
        endTime: end time of the trial
        journeyList: list of Journey object
        ...
    
    Methods:
        checkSessionDirectory()
        extractTrialFeatures()
        poseFromTrialData()
        poseFromBridgeCoordinates()
        videoIndexFromTimeStamp()
        createTrialVideo()
        decorateVideoFrame()
        vectorAngle()
        lightFromCode()
        previousLight()
        getTrialVariables()
        dataToCm()
        dataToPx()
        
    """
    def __init__(self,sessionName,trialNo,startTime,endTime,startTimeWS,endTimeWS):
        self.sessionName = sessionName
        self.trialNo = trialNo
        self.startTime = startTime
        self.endTime = endTime
        self.startTimeWS = startTimeWS
        self.endTimeWS = endTimeWS
        self.name = "{}_{}".format(sessionName,trialNo)
        self.valid = True # valid if the mouse went on the arena and lever was pressed
        self.duration = self.endTime-self.startTime
        self.light = None
        self.arenaRadius = None     
        self.arenaRadiusProportionToPeri = None
        self.mousePose = None
        self.leverPose = None
        
        self.bCoord = None # np array
        self.bCoordMiddle = None # np array x y
        self.test = None
    
    def setMousePosition(self,mousePose):
        """
        Extract the mouse position data for this trial.
        
        Argument:
        mousePose: Pandas DataFrames with x,y,hd,time for the entire session
        Save it in self.mousePose
        
        """
        self.mousePose = mousePose[mousePose["time"].between(self.startTime,self.endTime)]
        if len(self.mousePose.time)==0:
            print("{}, no mouse position data during this trial".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid = False
            
        validMouse = np.sum(~np.isnan(self.mousePose.x)) 
        if validMouse < 20:
            print("{}, the mouse was detected for fewer than 20 frames during the trial".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid = False
        
    def setLeverPosition(self,leverPose):
        """
        Extract the lever position data for this trial
        Argument:
        leverPose: Pandas DataFrames with pose for the lever for the entire session. With these columns:
                    ['time', 'leverPressX', 'leverPressY', 'leverBoxPLX','leverBoxPLY', 'leverBoxPRX', 'leverBoxPRY']
        Save it in self.leverPose
        """
        self.leverPose = leverPose[leverPose["time"].between(self.startTime,self.endTime)]
        if len(self.leverPose.time)==0:
            print("{}, no lever position data during this trial".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid = False 
        
        self.lever = Lever()
        lp = np.array((stats.mode(self.leverPose["leverPressX"].to_numpy().astype(int))[0].item(), stats.mode(self.leverPose["leverPressY"].to_numpy().astype(int))[0].item()))
        pl = np.array((stats.mode(self.leverPose["leverBoxPLX"].to_numpy().astype(int))[0].item(), stats.mode(self.leverPose["leverBoxPLY"].to_numpy().astype(int))[0].item()))
        pr = np.array((stats.mode(self.leverPose["leverBoxPRX"].to_numpy().astype(int))[0].item(), stats.mode(self.leverPose["leverBoxPRY"].to_numpy().astype(int))[0].item()))
        self.lever.calculatePose(lp = lp, pl = pl, pr = pr)
        
        
        
        
        
    def setArenaRadius(self,arenaRadius,arenaRadiusProportionToPeri=0.925):
        self.arenaRadius=arenaRadius
        self.arenaRadiusProportionToPeri = arenaRadiusProportionToPeri
        
    def getLightCondition(self,log):
        """
        Get the last light that was set before the current trial
        
        """
        lightEvents = log[log.event=="light"] 
        if sum(lightEvents.time < self.startTime) == 0 :
            lightCode=np.nan
        else:
            lightCode=lightEvents.param[lightEvents.time< self.startTime].tail(1).to_numpy()[0]
        if lightCode == 1 or np.isnan(lightCode):
            self.light="light"
        else:
            self.light="dark" 
    
    
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    
    def trialPathFigure(self, legend = True, figSize=(10,10), filePath=None):
        """
        Plot the path of the animal on the arena with the lever
        
        Argument
            pathNames List of path names to display
            legend Bool
            figSize Tuple of the figure size
        
        """
        # to plot the arena circle
        arena=np.arange(start=0,stop=2*np.pi,step=0.02)
        
        
        fig, axes = plt.subplots(1,1,figsize=figSize)
        plt.subplots_adjust(wspace=0.3,hspace=0.3)

        # plot the arena and arena periphery
        axes.set_aspect('equal', adjustable='box')
        axes.set_title("{}, {}".format(self.name,self.light))
        axes.plot(np.cos(arena)*self.arenaRadius,np.sin(arena)*self.arenaRadius,label="Arena",color="gray")
        axes.plot(np.cos(arena)*self.arenaRadius*self.arenaRadiusProportionToPeri,
                     np.sin(arena)*self.arenaRadius*self.arenaRadiusProportionToPeri,label="Periphery",color="gray",linestyle='dashed')
        axes.set_xlabel("cm")
        axes.set_ylabel("cm")
        
        
         ## mouse path
        axes.plot(self.mousePose.x,self.mousePose.y,color="black", label="path")
        
        
        ## lever
        axes.plot(self.lever.pointsPlot[:,0],self.lever.pointsPlot[:,1], color = "gray")
        axes.plot(self.lever.enterZonePointsPlot[:,0],self.lever.enterZonePointsPlot[:,1], color = "gray",linestyle="dotted")
        axes.plot(self.lever.exitZonePointsPlot[:,0],self.lever.exitZonePointsPlot[:,1], color = "yellow",linestyle="dotted")
        

        if legend:
            axes.legend(loc="upper right")
        
        if filePath is not None:
            print("Saving to " + filePath)
            plt.savefig(filePath,bbox_inches = "tight")
    