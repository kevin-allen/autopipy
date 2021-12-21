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
        
        The zone definitions are stored in a dictionary called self.zones
        
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
        
        The time use is ROS time
        
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
        Extract the lever position data for this trial.
        
        The time is ROS time
        
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
        Count how many lever presses in total were performed
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
    
    
    def setPositionZones(self):
        """
        Create a DataFrame that containes the current position zone of the animal (arena, arenaCenter, bridge, home base).
        
        It creates a data frame called self.positionZones
        """
        
        mousePoints = np.stack([self.mousePose.x.to_numpy(),self.mousePose.y.to_numpy()],axis=1)
        self.distanceFromArenaCenter = np.sqrt(np.sum((mousePoints - self.zones["arenaCenter"][0:2])**2,axis=1))
        
        # bridge
        mouseRelBridge = mousePoints-self.zones["bridge"][0:2]  # position relative to the bottom left of the bridge
        onBridge = np.logical_and(mouseRelBridge[:,1]> 0, mouseRelBridge[:,1] < self.zones["bridge"][3])
        
        # home base
        mouseRelHb = mousePoints-self.zones["homeBase"][0:2]  # position relative to the bottom left of the bridge
        onHb = np.logical_and(mouseRelHb[:,1]> 0, mouseRelHb[:,1] < self.zones["homeBase"][3])
        
        # gap between bridge and arena (y between max of bridge and -40)
        gap = np.logical_and(mouseRelBridge[:,1]>self.zones["bridge"][3],self.mousePose.y < -40)
        
        self.positionZones=pd.DataFrame({"lever": self.lever.isAt(mousePoints), # use the lever object
                                         "arenaCenter": self.distanceFromArenaCenter< self.zones["arenaCenter"][2],
                                         "arena": self.distanceFromArenaCenter< self.zones["arena"][2],
                                         "bridge": onBridge,
                                         "homeBase": onHb,
                                            "gap": gap})
        
        # get the one-hot encoding back into categorical, when several true, the first column is return.
        self.positionZones["loca"] = self.positionZones.iloc[:, :].idxmax(1)
        
        
    def identifyJourneyBorders(self):
        """
        Identify the beginning and end of journeys 
        
        A journey starts when the animal leaves and is going to enter the arena center. 
        The start of one journey is the end of the next journey or the end of the trial
        
        The method generate a DataFrame (slef.journeyStartEndIndices) with a "start" and "end" column.
        
        """
        # get the rows in which the mouse is on the bridge or arenaCenter
        bridgeArenaCenter = self.positionZones[ (self.positionZones.loca=="bridge") | (self.positionZones.loca=="arenaCenter") ]
        # get a df to look for bridge to arenaCenter transition
        df = pd.DataFrame({"start" : bridgeArenaCenter.loca,"end" : bridgeArenaCenter.shift(-1).loca})

        # start of journey is when the animal leaves the bridge to go on the arenaCenter to explore the center
        self.journeyTransitionIndices = df[(df.start=="bridge") & (df.end=="arenaCenter")].index.values - 1 # - 1 because of the shift

        ## if the mouse first appeared as it was on the arena at the beginning of the trial, add one journey starting at the first arena 
        if np.sum(df.end=="arenaCenter")>0:
            if np.sum(df.start=="bridge")==0:
                print("{}, no bridge time in the trial".format(self.name))
                print("{}, add a journey missed because of no brige time before first arenaCenter".format(self.name))
                firstArenaCenterIndex = df[(df.end=="arenaCenter")].index[0]
                self.journeyTransitionIndices = np.insert(self.journeyTransitionIndices,0,firstArenaCenterIndex-1)
            else :
                firstBridgeIndex = df[(df.start=="bridge")].index[0]
                firstArenaCenterIndex = df[(df.end=="arenaCenter")].index[0]
                if firstArenaCenterIndex < firstBridgeIndex:
                    print("{}, add a journey missed because of no brige time before first arenaCenter".format(self.name))
                    self.journeyTransitionIndices = np.insert(self.journeyTransitionIndices,0,firstArenaCenterIndex-1)


        ## get the start and end indices of journeys 
        if len(self.journeyTransitionIndices) > 0:
            jt = np.append(self.journeyTransitionIndices,self.positionZones.index.values[-1]) # get index for end of last journey
            # dataframe with start and end indices for each journey
            self.journeyStartEndIndices = pd.DataFrame({"start" : jt[0:-1], "end" : jt[1:]-1})

            
    def checkTrialValidity(self):
        """
        Check that the trial is valid
        
        - the mouse should go onto the arena
        
        Some other checks are done in self.getLeverPresses, self.getMousePosition, self.getLeverPosition
        
        Additional checks can be added here
        
        Results stored in self.valid (True or False)
        """
        self.valid=True
        
        if np.sum(self.positionZones["arena"])<0:
            print("the mouse was detected on the arena")
            self.valid=False
        
    
    
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    
    
    
    def trialPathFigure(self, legend = True, figSize=(10,10), zones=False, filePath=None, positionZones = False):
        """
        Plot the path of the animal on the arena with the lever
        
        Use to make sure that trial data extraction is working
        
        Argument
                    
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
        if positionZones == False:
            axes.plot(self.mousePose.x,self.mousePose.y,color="black", label="path")
        else :
            # plot the position in different color depending on self.positionZone 
            axes.plot(self.mousePose.x.loc[self.positionZones["homeBase"].to_numpy()],
                      self.mousePose.y.loc[self.positionZones["homeBase"].to_numpy()],
                      color="orange", label="Home base")
            axes.plot(self.mousePose.x.loc[self.positionZones["arena"].to_numpy()],
                      self.mousePose.y.loc[self.positionZones["arena"].to_numpy()],
                      color="green", label="Arena")
            axes.plot(self.mousePose.x.loc[self.positionZones["arenaCenter"].to_numpy()],
                      self.mousePose.y.loc[self.positionZones["arenaCenter"].to_numpy()],
                      color="pink", label="Arena center")
            axes.plot(self.mousePose.x.loc[self.positionZones["bridge"].to_numpy()],
                      self.mousePose.y.loc[self.positionZones["bridge"].to_numpy()],
                      color="gray", label="Bridge")
            axes.plot(self.mousePose.x.loc[self.positionZones["gap"].to_numpy()],
                      self.mousePose.y.loc[self.positionZones["gap"].to_numpy()],
                      color="yellow", label="Gap")
            
        
        
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
                # Add the patch to the axes
                axes.add_patch(rect)
                
            
          
            
            
        if filePath is not None:
            print("Saving to " + filePath)
            plt.savefig(filePath,bbox_inches = "tight")
    
    def trialJourneyFigure(self, zones = True, legend=True,figSize=(10,10), filePath=None,):
        """
        Plot the path of the animal during the different journeys (different colours)
        
        
                    
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
        for j in range(len(self.journeyStartEndIndices.start)):
    
            start = self.journeyStartEndIndices["start"][j]
            end = self.journeyStartEndIndices["end"][j]
            axes.plot(self.mousePose.x.loc[start:end],self.mousePose.y.loc[start:end])
            
         
        
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
                # Add the patch to the axes
                axes.add_patch(rect)
                
            
          
            
            
        if filePath is not None:
            print("Saving to " + filePath)
            plt.savefig(filePath,bbox_inches = "tight")
    