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
import matplotlib.patches as patches
from scipy.interpolate import interp1d

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
        
        self.test = None
    
    
    def setZoneAreas(self, arenaCoordinateFileName, bridgeCoordinateFileName,bridgeLengthCm=12,homeBaseXCm=30,homeBaseYCm=25):
        """
        Set the areas covered by the home base, bridge, arena, arena center
        
        All values are in cm with 0,0 being the center of the arena.
        
        The areas are defined in a list. For rectangular areas, we save the bottom-left coordinate and width height.
        For the arena, the center and radius are saved. 
        
        """
        if self.arenaRadius is None:
            print("{}, call setAreanRadius() before setZoneAreas()".format(self.name))
            return()
        self.zones={}
        
        # we already know the arena coordinate and radius
        self.zones["arena"]=np.array([0,0,self.arenaRadius])
        self.zones["arenaCenter"]=self.zones["arena"]*self.arenaRadiusProportionToPeri
    
        # the bridge and arena were detected from a cropped image (in pixels). 
        # We know the radius in cm of the arena so we can calculate the px_per_cm of this cropped image.
        ac = np.loadtxt(arenaCoordinateFileName)
        bc = np.genfromtxt(bridgeCoordinateFileName,delimiter=",")
        pxPerCm = ac[2]/self.arenaRadius
        # get in cm with 0,0 as arena center
        bc2 = (bc - ac[0:2])/pxPerCm
        # set the bridge length to 10 cm (not done during bridge detection)
        bc2[0,1] = bc2[1,1]-bridgeLengthCm
        bc2[3,1] = bc2[1,1]-bridgeLengthCm
        self.zones["bridge"] = np.array([bc2[0,0], # bottom-left x
                                         bc2[0,1], # bottom-left y
                                         bc2[3,0]-bc2[0,0], #width
                                         bc2[1,1]-bc2[0,1]]) # height
        
        # the arena is everything that is below the bridge (below relative to y axis)
        # we can center it on the bridge, and set its width
        midBX = self.zones["bridge"][0]+self.zones["bridge"][2]/2
        bottomBY=self.zones["bridge"][1]
        self.zones["homeBase"]= np.array([midBX-homeBaseXCm/2,
                                          bottomBY-homeBaseYCm,
                                          homeBaseXCm,
                                          homeBaseYCm])
                    
        
    
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
    
    def interpolateMousePose(self, limit=10):
        """
        Interpolate small gaps in the mouse position data
        
        In order to have as many valid position data points as possible, we will linearly interpolate the data
        when there are just a few np.nan in the data.
        
        We will need to deal with head-direction data using sin and cos decomposition.
        
        We use the pandas interpolation function
        https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.DataFrame.interpolate.html
        
        Arguments
        limit: Maximum number of consecutive NaNs to fill. Must be greater than 0.
        """
        self.mousePose.x = self.mousePose.x.interpolate(limit=limit)
        self.mousePose.y = self.mousePose.y.interpolate(limit=limit)
        
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
            
    def getLeverPresses(self,log):
        """
        Get the time of lever presses within the trial. 
        Calculate the position of the animal at the lever press time
        Cound how many lever presses in total were performed
        """
        lever = log[ (log.event=="lever_press") | (log.event == "leverPress")]
        index = (lever.time>self.startTime) & (lever.time<self.endTime) # boolean array
        leverPressTime = lever.time[index] # ROS time of lever
        
        if len(leverPressTime) != 0:
            # interpolate the x and y position of the animal at the lever press time
            fx = interp1d(self.mousePose.time, self.mousePose.x)
            fy = interp1d(self.mousePose.time, self.mousePose.y)
            mouseX = fx(leverPressTime)
            mouseY = fy(leverPressTime)
        
            self.leverPress = pd.DataFrame({"time": leverPressTime,
                                            "mouseX":mouseX,
                                           "mouseY":mouseY})
        self.nLeverPresses = len(leverPressTime)
        if self.nLeverPresses == 0:
            print("{}, no lever press".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid=False
        
        if np.isnan(self.leverPress.mouseX.head(1).item()):
            print("{}, position of the animal at first lever press unknown".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid=False
    
    
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    
    def trialPathFigure(self, legend = True, figSize=(10,10), zones=False, filePath=None):
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
        axes.set_title("{}, {}, {:.1f} sec".format(self.name,self.light,self.endTime-self.startTime))
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
        axes.plot(self.lever.exitZonePointsPlot[:,0],self.lever.exitZonePointsPlot[:,1], color = "gray",linestyle="dotted")
        
        ## position of the mouse when pressing the lever
        if self.nLeverPresses != 0:
            axes.scatter(self.leverPress.mouseX,self.leverPress.mouseY,c="red")

        if legend:
            axes.legend(loc="upper right")
        
        if zones:
            # draw the zones
            for i in ["bridge","homeBase"]:
                
                rect = patches.Rectangle((self.zones[i][0], self.zones[i][1]), self.zones[i][2], self.zones[i][3], linewidth=1, edgecolor='gray', facecolor='none')
                # Add the patch to the Axes
                axes.add_patch(rect)
                
        
        if filePath is not None:
            print("Saving to " + filePath)
            plt.savefig(filePath,bbox_inches = "tight")
    