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

class Trial:
    """
    Class containing information about a single trial on the Autopi task.
    
    When calculating angles, the y coordinate is reversed (0-y) so that 90 degree is up.
    This is done because the y-axis is reversed in the videos.
    
    The position data that come into this class are in pixels, as the position data were extracted from a video.
    
    To be able to compare data from different sessions, we convert the position data from pixels to cm and 
    move the 0,0 origin to the center of the arena. This transformation is done when extracting trial features. 
    There are some variables ending by Px that contain data in pixels to be able to draw in video frames.
    The variables ending with Cm are in cm.
    When not explicitely specified, the values are in cm
    Note that the transformation between pixels to cm representation involve a translation to shift the origin to the center of the arena
    
    Contains a list of Journeys (exploratory episode on the arena)
    
    The Trial class has a pathD (dictionary of NavPath) which is just copied from the journey with the first lever press.
    
    
     
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
        self.duration = None
        self.trialVideoLog = None # DataFrame
        self.light = None
        self.startVideoIndex = None
        self.endVideoIndex = None
        
        self.aCoordCm = None # np array (x y r)
        self.aCoordPx = None
        self.bCoordCm = None # np array
        self.bCoordMiddleCm = None # np array x y
        self.bCoordPx = None # np array
        self.bCoordMiddlePx = None # np array x y
        
        self.trialMLPx = None # DataFrame, mouse and lever position data for this trial
        self.trialMLCm = None # DataFrame, mouse and lever position for the trial, in cm
        self.leverPosition = None # Dictionary with the median lever position
        self.leverPx = None # Lever object to draw or to determine whether the animal is at or on the lever, in px world to draw in video
        self.leverCm = None # Lever object to draw or to determine whether the animal is at or on the lever, in cm world for analysis
        
        self.radiusPeripheryCm = None
        self.radiusPeripheryPx = None
        self.radiusLeverProximity = None
        self.traveledDistance = None
        
        self.pathDF = None # DataFrame, instantaneous variable evolving during the trial (e.g., distance run)
        self.leverPress = None # DataFrame with Ros time and video index of lever presses
        self.nLeverPresses = None 
        self.nJourneys = None
        self.journeyTransitionIndices = None
        self.journeyStartEndIndices = None
        self.journeyList = None
        self.journeysAtLever = None
        self.journeysWithPress = None  
        
        self.peripheryAfterFirstLeverPressCoordCm = [None,None]
        self.peripheryAfterFirstLeverPressCoordPx = [None,None]
        self.peripheryAfterFirstLeverPressAngle = None
        self.arenaToBridgeAngle = None
        self.arenaToMousePeriAngle = None
        self.stateDF = None # DataFrame with the categorical position of the mouse (Bridge, Arena, ArenaCenter, Lever, Gap)
        self.stateTime = None # Dictionary containing the time spent at the different locations (Bridge, Arena, ArenaCenter, Lever, Gap)
        ## start and end indices of paths
        self.searchTotalStartIndex = None
        self.searchTotalEndIndex = None
        self.searchArenaStartIndex = None
        self.searchArenaEndIndex = None
        self.searchArenaNoLeverStartIndex = None
        self.searchArenaNoLeverEndIndex = None
        self.homingTotalStartIndex = None
        self.homingTotalEndIndex = None
        self.homingPeriStartIndex = None
        self.homingPeriEndIndex = None
        self.homingPeriNoLeverStartIndex = None
        self.homingPeriNoLeverEndIndex = None
        self.peripheryAfterFirstLeverPressVideoIndex = None
        self.periArenaCenterBridgeAngle=None
        
        self.pathD = {} # inherited from the first journey with lever press
        
        
    def getTrialVariables(self):
        """
        Return a pandas DataFrame with most trial variables
        
        We first save variables that we calculated with this class
        Then we add the path variables
        
        """
        
        ## variables calculated in this Trial class
        self.myDict = {"sessionName": [self.sessionName],
                       "name": [self.name],
                       "valid": [self.valid],
                       "trialNo": [self.trialNo],
                       "startTime": [self.startTime],
                       "endTime": [self.endTime],
                       "startTimeWS": [self.startTimeWS],
                       "endTimeWS": [self.endTimeWS],
                       "duration": [self.duration],
                       "light": [self.light],
                       "arenaRadiusCm": [self.arenaRadiusCm],
                       "leverPositionX" : [self.leverPositionCm["leverX"]],
                       "leverPositionY" : [self.leverPositionCm["leverY"]],
                       "leverPositionOri": [self.leverPositionCm["leverOri"]],
                       "leverPressX": [self.leverPositionCm["leverPressX"]],
                       "leverPressY": [self.leverPositionCm["leverPressY"]],
                       "nLeverPresses" : [self.nLeverPresses],
                       "nJourneys" : [self.nJourneys],
                       "nJourneysAtLever" : [np.sum(self.journeysAtLever)],
                       "nJourneysWithPress" : [np.sum(self.journeysWithPress)],
                       "travelledDistance" : [self.traveledDistance],
                       "peripheryAfterFirstLeverPressX" : [self.peripheryAfterFirstLeverPressCoordCm[0]],
                       "peripheryAfterFirstLeverPressY" : [self.peripheryAfterFirstLeverPressCoordCm[1]],
                       "peripheryAfterFirstLeverPressAngle" : [self.peripheryAfterFirstLeverPressAngle],
                       "angularErrorHomingPeri" : [self.periArenaCenterBridgeAngle],
                      
                      
                       # total time at different locations
                       "timeBridge" : self.stateTime["bridge"],
                       "timeArenaCenter" : self.stateTime["arenaCenter"],
                       "timeArena" : self.stateTime["arena"],
                       "timePeriphery" : self.stateTime["arena"]-self.stateTime["arenaCenter"],
                       "timeHomeBase" : self.stateTime["homeBase"],
                       "timeLever" : self.stateTime["lever"]}
        df = pd.DataFrame(self.myDict)
        
        for k in self.pathD :
            myDf = self.pathD[k].getVariables() # get a df from the NavPath
            myDf = myDf.add_prefix(k+"_") # add a prefix to the column name
            df = pd.concat([df, myDf.reindex(df.index)], axis=1) # join the NavPat df to the trial df
                
        
        return df
    
    
    def getSpeedProfile(self,pathName = "searchTotal"):
        return self.pathD[pathName].speedProfile
    
    def trialMLToCm(self):
        """
        Transform mouse and lever data to cm and 0,0 at the center of arena

        returns a new data frame
        """
        # select column by names ending with X
        df = self.trialMLPx.copy()
        df1 = (df.filter(regex=("X$"))-self.originPxToCm[0])/self.pxPerCm ## to cm
        df2 = (df.filter(regex=("Y$"))-self.originPxToCm[1])/self.pxPerCm ## to cm
        df3 = df.filter(regex=("Heading"))
        df4 = df.filter(regex=("Ori"))
        dfm = pd.concat([df1,df2,df3,df4],axis=1) # remerge
        return dfm
    
    def extractTrialFeatures(self,log=None,mLPosi=None,videoLog=None,aCoord=None,bCoord=None,
                             arenaRadiusProportionToPeri=0.925, printName = False, arenaRadiusCm=40):
        """
        Extract trial features
        
        The input data from mLPosi, aCoord and bCoord are in pixels and are transform in cm.
        
        Arguments
            log: DataFrame with event log of the session
            mLPosi: DataFrame with mouse and lever position for every video frame
            videoLog: DataFrame time for each video frame
            aCoord: np array arena coordinates (x, y, radius)
            bCoord: np array bridge coordinates (four points)
            arenaRadiusProportionToPeri: proportion of arena radius at which the periphery of the arena is
            printName: Boolean indicating whether to print the trial name
            
        """
        if log is None:
            raise TypeError("log is None")
        if mLPosi is None:
            raise TypeError("mLPosi is None")
        if aCoord is None:
            raise TypeError("aCoord is None")
        if bCoord is None:
            raise TypeError("bCoord is None")
           
        
        ##############################################
        ### get data to transform from pixels to cm ##
        ##############################################
        self.arenaRadiusCm = arenaRadiusCm
        self.pxPerCm = aCoord[2]/self.arenaRadiusCm # to go from pixels to cm
        self.originPxToCm = aCoord[:2] # the 0,0 in the cm world     
        #print("A radius cm: {}, A radius px:{}, pxPerCm: {}, originPxToCm: {}".format(self.arenaRadiusCm,aCoord[2],self.pxPerCm, self.originPxToCm))
        self.arenaRadiusProportionToPeri = arenaRadiusProportionToPeri
        #######################################
        ## arena and bridge coordinates in cm##
        #######################################  
        self.aCoordPx = aCoord
        self.aCoordCm = np.array([0,0,self.arenaRadiusCm]) #
        self.bCoordPx = bCoord
        self.bCoordMiddlePx = self.bCoordPx[0] + (self.bCoordPx[2]-self.bCoordPx[0])/2
        self.bCoordCm = (bCoord-self.originPxToCm)/self.pxPerCm
        self.bCoordMiddleCm = self.bCoordCm[0] + (self.bCoordCm[2]-self.bCoordCm[0])/2
        
          
        ## mouse and lever position
        self.trialMLPx = mLPosi[(videoLog.time > self.startTime) & (videoLog.time < self.endTime)] 
        self.trialMLCm = self.trialMLToCm()
    

        ## test the synchronization for the whole session as an extra precaution
        if len(videoLog.time) != len(mLPosi.mouseX):
            print("videoLog {} is not the same length as mLPosi {}".format(len(videoLog.time),len(mLPosi.mouseX)))
            print("{}, self.valid set to False".format(self.name))
            self.valid = False
        
        if printName : 
            print(self.name)
        ###########################
        # get the light condition #
        ###########################
        lightCode = self.previousLight(log)
        self.light = self.lightFromCode(lightCode)
        
       
        
        #####################################################################
        ## there could be trials in which the mouse was never on the arena ##
        ## for example if the experimenter pressed the lever               ##
        ## we need to deal with these trials                               ##
        ## all mouse position will be at np.NAN                            ##
        #####################################################################
        validMouse = np.sum(~np.isnan(self.trialMLPx.mouseX)) 
        if np.sum(~np.isnan(self.trialMLPx.mouseX)) < 20:
            print("{}, the mouse was detected for fewer than 20 frames during the trial".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid = False
        
       
        #################################################################################
        ## make sure that the trial does not start with the mouse already on the arena ##
        ## can happen if the mouse is jumping over the door as it moves down           ##
        ## or if door open event is late                                               ##
        ## we modify self.startTime during this procedure if needed                    ##
        #################################################################################
        isBridge =  ((self.trialMLCm.mouseX.iloc[0] > self.bCoordCm[0,0]) & (self.trialMLCm.mouseX.iloc[0] < self.bCoordCm[2,0]) & (self.trialMLCm.mouseY.iloc[0] > self.bCoordCm[0,1]) &                                        (self.trialMLCm.mouseY.iloc[0] < self.bCoordCm[2,1]))
        isHome = np.isnan(self.trialMLCm.mouseX.iloc[0]) # mouse not in the field of view
        if not(isBridge or isHome):
            maxBackward=-2.0 # we can move the startTime by a maximum of 2 s
            step=0.1 # 100 ms for each backward jump
            #print("{}, mouse is not in the home base or on the bridge at the beginning of the trial".format(self.name))
            back=0
            while(back>maxBackward):
                back=back-step
                self.startTime = self.startTime + back # back is negative
                self.trialMLPx = mLPosi[(videoLog.time > self.startTime) & (videoLog.time < self.endTime)] 
                self.trialMLCm = self.trialMLToCm()
                isBridge =  ((self.trialMLCm.mouseX.iloc[0] > self.bCoordCm[0,0]) & (self.trialMLCm.mouseX.iloc[0] < self.bCoordCm[2,0]) & (self.trialMLCm.mouseY.iloc[0] > self.bCoordCm[0,1]) &                                        (self.trialMLCm.mouseY.iloc[0] < self.bCoordCm[2,1]))
                isHome = np.isnan(self.trialMLCm.mouseX.iloc[0]) # mouse not in the field of view
                if (isBridge or isHome):
                    break
                    
            print("{}, self.startTime was adjusted by {:2.2} s".format(self.name,back))
            ###########################################################
            # update the mouse and lever tracking data for the trial ##
            ###########################################################
            self.trialMLPx = mLPosi[(videoLog.time > self.startTime) & (videoLog.time < self.endTime)]     
            self.trialMLCm = self.trialMLToCm()
            # adjust the startTimeWS
            self.startTimeWS = self.startTime - log.time[log.event=="start"].values[0]
               
        
        #####################################
        ## duration, start and end indices ##
        #####################################
        self.duration = self.endTime-self.startTime
        # get the start and end video indices
        self.trialVideoLog = videoLog[(videoLog.time > self.startTime) & (videoLog.time < self.endTime)]
        self.startVideoIndex = self.trialVideoLog.frame_number.head(1).squeeze()
        self.endVideoIndex = self.trialVideoLog.frame_number.tail(1).squeeze()
        # get the within trial time for each video frame
        self.trialVideoLog["timeWS"]= self.trialVideoLog.time-self.startTime
        
        
        ###################################################
        ## with some camera/raspberry pis, we had gaps  ###
        ## in which there was no video frames           ###
        ## make sure we don't have a gap during the trial##
        ###################################################
        self.maxInterFrameIntervals=self.trialVideoLog.time.diff().max()
        if self.maxInterFrameIntervals > 0.25 :
            print("{}, there are gaps larger than 0.25 in the video for this trial".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid = False
        
        #######################################
        # location of the lever box and lever #
        #######################################
        self.leverPositionPx = {"leverX" : np.nanmedian(self.trialMLPx["leverX"]),
                             "leverY" : np.nanmedian(self.trialMLPx["leverY"]),
                             "leverOri": np.nanmedian(self.trialMLPx["leverOri"]),
                             "leverPressX" : np.nanmedian(self.trialMLPx["leverPressX"]),
                             "leverPressY" : np.nanmedian(self.trialMLPx["leverPressY"])}
        self.leverPositionCm = {"leverX" : np.nanmedian(self.trialMLCm["leverX"]),
                             "leverY" : np.nanmedian(self.trialMLCm["leverY"]),
                             "leverOri": np.nanmedian(self.trialMLCm["leverOri"]),
                             "leverPressX" : np.nanmedian(self.trialMLCm["leverPressX"]),
                             "leverPressY" : np.nanmedian(self.trialMLCm["leverPressY"])}
        
        
        self.leverPx = Lever()
        lp = np.array((stats.mode(self.trialMLPx["leverPressX"].to_numpy().astype(int))[0].item(), stats.mode(self.trialMLPx["leverPressY"].to_numpy().astype(int))[0].item()))
        pl = np.array((stats.mode(self.trialMLPx["leverBoxPLX"].to_numpy().astype(int))[0].item(), stats.mode(self.trialMLPx["leverBoxPLY"].to_numpy().astype(int))[0].item()))
        pr = np.array((stats.mode(self.trialMLPx["leverBoxPRX"].to_numpy().astype(int))[0].item(), stats.mode(self.trialMLPx["leverBoxPRY"].to_numpy().astype(int))[0].item()))
        self.leverPx.calculatePose(lp = lp, pl = pl, pr = pr)
        
        self.leverCm = Lever()
        lp = np.array((stats.mode(self.trialMLCm["leverPressX"].to_numpy().astype(int))[0].item(), stats.mode(self.trialMLCm["leverPressY"].to_numpy().astype(int))[0].item()))
        pl = np.array((stats.mode(self.trialMLCm["leverBoxPLX"].to_numpy().astype(int))[0].item(), stats.mode(self.trialMLCm["leverBoxPLY"].to_numpy().astype(int))[0].item()))
        pr = np.array((stats.mode(self.trialMLCm["leverBoxPRX"].to_numpy().astype(int))[0].item(), stats.mode(self.trialMLCm["leverBoxPRY"].to_numpy().astype(int))[0].item()))
        self.leverCm.calculatePose(lp = lp, pl = pl, pr = pr)
        
        
        ################################################
        # define arena periphery and lever proximity  ##
        ################################################
        # radius from the arena center that defines the periphery of the arena
        self.radiusPeripheryCm = self.aCoordCm[2]*arenaRadiusProportionToPeri
        self.radiusPeripheryPx = self.aCoordPx[2]*arenaRadiusProportionToPeri
        
       
        
        #########################################
        # variables that evolve along the path ##
        #########################################
        
        # np.array, used later on to know if at the lever
        mousePoints = np.stack((self.trialMLCm.mouseX.to_numpy(),self.trialMLCm.mouseY.to_numpy()),axis=1)
        
        
        # we will store all these Series in a DataFrame called pathDF
        distance = np.sqrt(np.diff(self.trialMLCm.mouseX,prepend=np.NAN)**2+
                                np.diff(self.trialMLCm.mouseY,prepend=np.NAN)**2)
        traveledDistance = np.nancumsum(distance) # cumsum
        self.traveledDistance = np.nansum(distance) # sum
        videoFrameTimeDifference = self.trialVideoLog.time.diff().to_numpy()
        speed = distance/videoFrameTimeDifference
        speedNoNAN = np.nan_to_num(speed) # replace NAN with 0.0 to display in video
        distanceFromArenaCenter = np.sqrt((self.trialMLCm.mouseX.to_numpy() - self.aCoordCm[0])**2+ 
                                               (self.trialMLCm.mouseY.to_numpy() - self.aCoordCm[1])**2)
        ## distance from lever
        distanceFromLeverPress = np.sqrt((self.trialMLCm.leverPressX.to_numpy() - self.trialMLCm.mouseX.to_numpy() )**2 + 
                                        (self.trialMLCm.leverPressY.to_numpy() - self.trialMLCm.mouseY.to_numpy())**2)
        distanceFromLever = np.sqrt((self.trialMLCm.leverX.to_numpy() - self.trialMLCm.mouseX.to_numpy() )**2 + 
                                        (self.trialMLCm.leverY.to_numpy() - self.trialMLCm.mouseY.to_numpy())**2)
        
        ## movement heading of the mouse relative to [1,0]
        mv = np.stack((np.diff(self.trialMLCm.mouseX,prepend=np.NAN),
                       np.diff(self.trialMLCm.mouseY,prepend=np.NAN)),axis=1)
        mvHeading = self.vectorAngle(mv,degrees=True,quadrant=True)
        ## vector from mouse to bridge
        mBVXCm = self.bCoordMiddleCm[0] - self.trialMLCm.mouseX.to_numpy() 
        mBVYCm = self.bCoordMiddleCm[1] - self.trialMLCm.mouseY.to_numpy()   
        mouseToBridgeCm = np.stack((mBVXCm,mBVYCm),axis = 1)
        
        mBVXPx = self.bCoordMiddlePx[0] - self.trialMLPx.mouseX.to_numpy() 
        mBVYPx = self.bCoordMiddlePx[1] - self.trialMLPx.mouseY.to_numpy()   
        mouseToBridgePx = np.stack((mBVXPx,mBVYPx),axis = 1)
        
        
        ## angle between movement heading and vector from the mouse to the bridge
        mvHeadingToBridgeAngle = self.vectorAngle(mv,mouseToBridgeCm,degrees=True)
        ## angle between head direction and vector from the mouse to the bridge
        hdv = np.stack((self.trialMLCm.mouseXHeading.to_numpy(),
                       self.trialMLCm.mouseYHeading.to_numpy()),axis = 1)
        hdToBridgeAngle = self.vectorAngle(hdv,mouseToBridgeCm,degrees=True)
        
        # Store these Series into a data frame
        # use the same index as the self.trialMLPx DataFrame
        self.pathDF = pd.DataFrame({"distance" :distance,
                                  "traveledDistance" : traveledDistance,
                                  "mvHeading" : mvHeading,
                                  "mouseToBridgeXCm": mBVXCm,
                                  "mouseToBridgeYCm": mBVYCm,
                                    "mouseToBridgeXPx": mBVXPx,
                                    "mouseToBridgeYPx": mBVYPx,
                                  "mvHeadingToBridgeAngle" : mvHeadingToBridgeAngle,
                                  "hdToBridgeAngle" : hdToBridgeAngle,
                                  "speed" : speed,
                                  "speedNoNAN" : speedNoNAN,
                                  "distanceFromArenaCenter" : distanceFromArenaCenter,
                                  "distanceFromLever" : distanceFromLever,
                                  "distanceFromLeverPress": distanceFromLeverPress},
                                  index = self.trialMLPx.index)         
       
        ######################
        # get lever presses ##
        ######################
        lever = log[ (log.event=="lever_press") | (log.event == "leverPress")]
        index = (lever.time>self.startTime) & (lever.time<self.endTime) # boolean array
        leverPressTime = lever.time[index] # ROS time of lever
        self.leverPressVideoIndex = leverPressTime.apply(self.videoIndexFromTimeStamp) # video index
        self.leverPress = pd.DataFrame({"time": leverPressTime,
                                        "videoIndex":self.leverPressVideoIndex})
        self.nLeverPresses = len(self.leverPress.time)
        if self.nLeverPresses == 0:
            print("{}, no lever press".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid=False
            
        
        
        #################################################
        ## sectioning the trial into different states  ##
        #################################################
        # define each frame as arena, bridge, lever or home base (NAN), one-hot encoding
        self.stateDF=pd.DataFrame({"lever": self.leverCm.isAt(mousePoints) ,
                                   "arenaCenter": self.pathDF.distanceFromArenaCenter<self.radiusPeripheryCm,
                                   "arena": self.pathDF.distanceFromArenaCenter<self.aCoordCm[2],
                                   "bridge": ((self.trialMLCm.mouseX > self.bCoordCm[0,0]) & 
                                              (self.trialMLCm.mouseX < self.bCoordCm[2,0]) & 
                                              (self.trialMLCm.mouseY > self.bCoordCm[0,1]) & 
                                              (self.trialMLCm.mouseY < self.bCoordCm[2,1])),
                                   "homeBase": pd.isna(self.trialMLCm.mouseX)})
        # if there is a na but the mouse is on the lever, treat it as lever time
        self.stateDF.homeBase[(self.stateDF.homeBase==True)&(self.stateDF.lever==True)]=False
        
        
        # if all false, the mouse is not on arena or bridge
        # most likely between the arena and bridge, or poking it over the edge of the arena
        self.stateDF.insert(0, "gap", self.stateDF.sum(1)==0) 
        # get the one-hot encoding back into categorical, when several true, the first column is return.
        self.stateDF["loca"] = self.stateDF.iloc[:, :].idxmax(1)
        self.stateTime = {"gap" : videoFrameTimeDifference[self.stateDF.gap].sum(),
                          "lever" : videoFrameTimeDifference[self.stateDF.lever].sum(),
                         "arenaCenter" : videoFrameTimeDifference[self.stateDF.arenaCenter].sum(),
                         "arena" : videoFrameTimeDifference[self.stateDF.arena].sum(),
                         "bridge" : videoFrameTimeDifference[self.stateDF.bridge].sum(),
                         "homeBase" : videoFrameTimeDifference[self.stateDF.homeBase].sum()}

        ### 3 checks for potential problems
        ####################################################################
        ### If the lever is not detected correctly, it is possible that ####
        ### the animal is never at the lever                            ####
        ### Or if the experimenter presses the lever for some reason    ####
        ### Set to invalid                                              ####
        ####################################################################
        if np.sum(self.stateDF.loca=="lever") == 0 :
            print("{}, no time in the lever zone".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid=False
        ###################################################################
        ### The mouse should be at the lever when the lever was pressed ###
        ### We consider only the first lever press                      ###
        ###################################################################
        if self.nLeverPresses > 0 :
            if np.sum(self.stateDF.loca.loc[self.leverPress.videoIndex.iloc[0]] == "lever") == 0 :
                print("{}, mouse not in the lever zone when the lever was pressed".format(self.name))
                print("{}, self.valid set to False".format(self.name))
                self.valid=False
        ######################################################
        ### The mouse should be on the arena at some point ###
        ######################################################
        if np.sum(self.stateDF.arena) == 0 :
            print("{}, no time on the arena".format(self.name))
            print("{}, self.valid set to False".format(self.name))
            self.valid=False
        
        ###############################################################
        ### There should be some arena center before the lever press ##
        ###############################################################
        if self.nLeverPresses > 0:
            if all(self.stateDF.loca.loc[self.startVideoIndex : self.leverPress.videoIndex.iloc[0]]!="arenaCenter"):
                print("{}, no time on the arenaCenter before lever press".format(self.name))
                print("{}, self.valid set to False".format(self.name))
                self.valid=False

        ############################################################
        ### There should be some arena data after the lever press ##
        ############################################################
        if self.nLeverPresses > 0 :
            
            if (np.sum(self.stateDF.loca.loc[self.leverPress.videoIndex.iloc[0]:self.endVideoIndex]=="arena")+
             np.sum(self.stateDF.loca.loc[self.leverPress.videoIndex.iloc[0]:self.endVideoIndex]=="arenaCenter"))==0:
                print("{}, There is no arena time after the lever press in the trial".format(self.name))
                print("{}, self.valid set to False".format(self.name))
                self.valid=False
        
        
        
           
        ######################################################################
        ## JOURNEYS BOUNDARIES                                              ##
        ## number of journeys on the arena (from the bridge to arena center)##
        ######################################################################
        # only keep bridge and arenaCenter state
        bridgeArenaCenter = self.stateDF[ (self.stateDF.loca=="bridge") | (self.stateDF.loca=="arenaCenter") ]
        # get a df to look for bridge to arenaCenter transition
        df = pd.DataFrame({"start" : bridgeArenaCenter.shift().loca,"end" : bridgeArenaCenter.loca})
        # start of journey is when the animal leaves the bridge to go on the arena to explore the center
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
            jt = np.append(self.journeyTransitionIndices,self.endVideoIndex) # get index for end of last journey
            # dataframe with start and end indices for each journey
            self.journeyStartEndIndices = pd.DataFrame({"start" : jt[0:-1], "end" : jt[1:]-1})
            
        self.journeyList = []
                
        #################################
        ## create a list of journeys   ##
        #################################
        if self.valid : # we don't want to analyze trials without lever press
            
            #######################################################
            ## don't start new journeys after first lever press  ##
            #######################################################
            if any(self.journeyStartEndIndices["start"] > self.leverPress.videoIndex.iloc[0]):
                print("There is a journey starting after the first lever press")
                # get the last end
                print(self.journeyStartEndIndices)
                print(self.leverPress)
                lastEnd=self.journeyStartEndIndices["end"].iloc[-1]
                print("lastEnd: {}".format(lastEnd))
                # remove the last journey
                self.journeyStartEndIndices = self.journeyStartEndIndices.iloc[:-1,:]
                # reset the end of the new last journey
                self.journeyStartEndIndices["end"].iloc[-1] =lastEnd

            ####################################
            ## create the journey list here  ###
            ####################################
            for j in range(len(self.journeyStartEndIndices)):
                self.journeyList.append(Journey(self.sessionName,self.trialNo,j+1,
                                               self.journeyStartEndIndices["start"][j],self.journeyStartEndIndices["end"][j],
                                               self.leverCm, 
                                               self.arenaRadiusCm, 
                                               self.arenaRadiusProportionToPeri, 
                                               self.aCoordCm, self.bCoordCm,
                                               self.trialMLCm,
                                               self.trialVideoLog,
                                               self.pathDF,
                                               self.leverPress,
                                               self.stateDF,
                                               self.leverPositionCm))

           
                      
            #################################################
            ## reaching periphery after first lever press  ##
            #################################################
            for i in range(self.leverPress.videoIndex.iloc[0],
                           self.trialVideoLog.frame_number.iloc[-1]) :
                if (self.pathDF.distanceFromArenaCenter[i] > self.radiusPeripheryCm):
                    self.peripheryAfterFirstLeverPressVideoIndex = i
                    break

            #####################################################################
            ## moue coordinate when reaching periphery after first lever press ##
            #####################################################################
            self.peripheryAfterFirstLeverPressCoordCm = np.array([self.trialMLCm.loc[self.peripheryAfterFirstLeverPressVideoIndex,"mouseX"],
                                                                self.trialMLCm.loc[self.peripheryAfterFirstLeverPressVideoIndex,"mouseY"]])
            self.peripheryAfterFirstLeverPressCoordPx = np.array([self.trialMLPx.loc[self.peripheryAfterFirstLeverPressVideoIndex,"mouseX"],
                                                                self.trialMLPx.loc[self.peripheryAfterFirstLeverPressVideoIndex,"mouseY"]])
            self.peripheryAfterFirstLeverPressAngle = np.asscalar(self.vectorAngle(v = np.expand_dims(self.peripheryAfterFirstLeverPressCoordCm,axis=0),degrees=True,quadrant=True)[0])
            
            #################################
            # Reaching periphery analysis  ##
            #################################
            ## angle of the bridge center relative to arena center 
            arenaToBridgeVector= self.bCoordMiddleCm - self.aCoordCm[:2]
            self.arenaToBridgeAngle = self.vectorAngle(np.expand_dims(arenaToBridgeVector,0),degrees=True)
            ## angle from mouse when reaching periphery relative to arena center
            arenaToMousePeriVector = self.peripheryAfterFirstLeverPressCoordCm-self.aCoordCm[:2]
            self.arenaToMousePeriAngle = self.vectorAngle(np.expand_dims(arenaToMousePeriVector,0),degrees=True)
            ## angular deviation of the mouse when reaching periphery
            self.periArenaCenterBridgeAngle=self.vectorAngle(v = np.expand_dims(arenaToMousePeriVector,0),
                                                             rv = np.expand_dims(arenaToBridgeVector,0),
                                                             degrees=True)
    
        else : 
            # trial is invalid, just create one Journey covering the entire trial length and no lever pressed considered
            # this will give us the required journey.pathD needed to return trial variables even if the trial is invalid
            emptyLeverPress = pd.DataFrame({"time": [],
                                        "videoIndex": []})
            self.journeyList.append(Journey(self.sessionName,self.trialNo,1,
                                            self.startVideoIndex,self.endVideoIndex,
                                            self.leverCm, 
                                            self.arenaRadiusCm, 
                                            self.arenaRadiusProportionToPeri, 
                                            self.aCoordCm, self.bCoordCm,
                                            self.trialMLCm,
                                            self.trialVideoLog,
                                            self.pathDF,
                                            emptyLeverPress,
                                            self.stateDF,
                                            self.leverPositionCm))
        
        self.journeysAtLever=np.sum([j.atLever for j in self.journeyList])
        self.journeysWithPress=np.sum([j.leverPressed for j in self.journeyList])
        self.nJourneys = len(self.journeyList)
            
        ## set trial.pathD to the journey.pathD with the first lever press
        ## makes it easier to access
        jList = [j for j in self.journeyList if j.leverPressed]
        if len(jList) > 0:
            j = jList[0]
        else :
            j = self.journeyList[-1]
        self.pathD = j.pathD
    
    
    def videoIndexFromTimeStamp(self, timeStamp):
        """
        Get the frame or index in the video for a given timestamp (event)
        """
        return self.trialVideoLog.frame_number.iloc[np.argmin(np.abs(self.trialVideoLog.time - timeStamp))]  
        
    def createTrialVideo(self,pathVideoFile,pathVideoFileOut,decorate=True,detailLevel=1):
        """
        Generate a trial video with annotation
        
        """
        
        if not os.path.isfile(pathVideoFile):
            print(pathVideoFile + " does not exist")
            return False
    
        cap = cv2.VideoCapture(pathVideoFile)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            return False
        
        fps = int (cap.get(cv2.CAP_PROP_FPS))
        inWidth = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        inHeight = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  

        if (self.startVideoIndex<0 or self.endVideoIndex>nFrames) :
            print("startVideoIndex {} or endVideoIndex {} is out of possible range for the number of frames in {} {}".format(self.startVideoIndex,self.endVideoIndex,pathVideoFile,nFrames))
            return
        

        ## mask to plot paths
        mask = np.full((inWidth, inHeight), 255, dtype=np.uint8) # to plot the path
        maskSearchTotal = np.full((inWidth, inHeight), 0, dtype=np.uint8)
        masksearchArena = np.full((inWidth, inHeight), 0, dtype=np.uint8) 
        masksearchArenaNoLever = np.full((inWidth, inHeight), 0, dtype=np.uint8)
        maskHomingTotal = np.full((inWidth, inHeight), 0, dtype=np.uint8)
        maskHomingPeri = np.full((inWidth, inHeight), 0, dtype=np.uint8)
        maskHomingPeriNoLever = np.full((inWidth, inHeight), 0, dtype=np.uint8)
        
        # We will combine to their respective mask, then we will add them to get the right color mixture
        searchTotalBG = np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        searchArenaBG =  np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        searchArenaNoLeverBG =  np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        searchTotalBG[:,:,0] = 150 # these values for search paths should not go over 255 on one channel
        searchArenaBG[:,:,0] = 105
        searchArenaNoLeverBG[:,:,1] = 150
        
        homingTotalBG = np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        homingPeriBG = np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        homingPeriNoLeverBG = np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        homingTotalBG[:,:,2] = 150
        homingPeriBG[:,:,2] = 105
        homingPeriNoLeverBG[:,:,1] = 150
        
        maskDict = {"mask" : mask,
                    "maskSearchTotal" : maskSearchTotal,
                    "masksearchArena" : masksearchArena,
                    "masksearchArenaNoLever": masksearchArenaNoLever,
                    "maskHomingTotal": maskHomingTotal,
                    "maskHomingPeri": maskHomingPeri,
                    "maskHomingPeriNoLever" :maskHomingPeriNoLever,
                    "searchTotalBG" : searchTotalBG,
                    "searchArenaBG" : searchArenaBG,
                    "searchArenaNoLeverBG" : searchArenaNoLeverBG,
                    "homingTotalBG": homingTotalBG,
                    "homingPeriBG" : homingPeriBG,
                    "homingPeriNoLeverBG" : homingPeriNoLeverBG}
        
        out = cv2.VideoWriter(pathVideoFileOut, cv2.VideoWriter_fourcc(*'MJPG'), fps, (inWidth,inHeight))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.startVideoIndex)
        
        print("Trial {}, from {} to {}, {} frames".format(self.trialNo,self.startVideoIndex,self.endVideoIndex, self.endVideoIndex-self.startVideoIndex))
        print(pathVideoFileOut)
        
        count = 0
        for i in range(self.startVideoIndex,self.endVideoIndex+1):
            ret, frame = cap.read()
            if decorate:
                frame = self.decorateVideoFrame(frame,i,count,maskDict,detailLevel)
            
            out.write(frame)
            count=count+1
    
        out.release() 
        cap.release() 
    
    
    def decorateVideoFrame(self,frame,index,count,maskDict,detailLevel=2):
        """
        Function to add information to the trial video
        For example, we can plot the path and other varialbes
        
        Arguments:
            frame: the video frame
            index: index of the frame
            count: count of the frame
            maskDict: dictionary containing the mask for the different paths
            detailLevel: define how much information is displayed on the frame; 0= minimal, 1 = some details, 2 = all information
        
        """
        
        
        # cartesian coordinates
        frame = cv2.putText(frame, 
                            "0,0",
                            (0,15), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (150,150,150), 1, cv2.LINE_AA)
        frame = cv2.putText(frame, 
                            "{},{}".format(frame.shape[0],frame.shape[1]),
                            (frame.shape[0]-70,frame.shape[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (150,150,150), 1, cv2.LINE_AA)
        # polar coordinates
        frame = cv2.putText(frame, 
                            "90",
                            (int(frame.shape[0]/2)-10,frame.shape[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (150,150,150), 1, cv2.LINE_AA)
        
        frame = cv2.putText(frame,
                            "0",
                            (frame.shape[0]-15,int(frame.shape[1]/2)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (150,150,150), 1, cv2.LINE_AA)
        frame = cv2.putText(frame,
                            "180",
                            (0,int(frame.shape[1]/2)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (150,150,150), 1, cv2.LINE_AA)
        
        
        ############################
        # first colum of variables #
        ############################
        # trial time
        if detailLevel < 2 :
            frame = cv2.putText(frame, 
                                "Time: {:.1f} sec".format(self.trialVideoLog.timeWS.iloc[count]), 
                                (30,20), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA)
        else :
            frame = cv2.putText(frame, 
                    "Time: {:.1f} sec, {:.1f}".format(self.trialVideoLog.timeWS.iloc[count],self.startTimeWS + self.trialVideoLog.timeWS.iloc[count] ), 
                    (30,20), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100,200,0), 1, cv2.LINE_AA)

            
        # traveled distance
        if detailLevel > 0:
            frame = cv2.putText(frame, 
                                "Distance: {:.1f} cm".format(self.pathDF.traveledDistance[index]), 
                                (30,50), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA)
            # speed
            frame = cv2.putText(frame, 
                                "Speed: {:.0f} cm/sec".format(self.pathDF.speedNoNAN[index]), 
                                (30,80), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA)       

            # Head direction of the mouse
            if ~np.isnan(self.trialMLPx.mouseOri[index]):
                frame = cv2.putText(frame, 
                                    "HD: {:.0f} deg".format(self.trialMLPx.mouseOri[index]), 
                                    (30,110), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA) 
            else:
                frame = cv2.putText(frame, 
                                    "HD: ", 
                                    (30,110), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA) 

            # Angle between head direction and bridge
            if ~np.isnan(self.pathDF.hdToBridgeAngle[index]):
                frame = cv2.putText(frame, 
                                    "hdToBridge : {:.0f} deg".format(self.pathDF.hdToBridgeAngle[index]), 
                                    (30,140), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA) 
            else :
                frame = cv2.putText(frame, 
                                    "hdToBridge : ", 
                                    (30,140), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA) 

            # Angle between mouse movement heading and vector from mouse to the bridge
            if ~np.isnan(self.pathDF.mvHeadingToBridgeAngle[index]):
                frame = cv2.putText(frame, 
                                    "mvHeadToBridge : {:.0f} deg".format(self.pathDF.mvHeadingToBridgeAngle[index]), 
                                    (30,170), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA) 
            else :
                frame = cv2.putText(frame, 
                                    "mvHeadToBridge : ", 
                                    (30,170), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA) 

        if detailLevel > 1 :
        
            # distance to lever
            frame = cv2.putText(frame, 
                                "Distance lever center: {:.1f} cm".format(self.pathDF.distanceFromLever[index]), 
                                (30,200), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA)       
            # mv heading
            frame = cv2.putText(frame, 
                                "MvHead: {:.0f} deg".format(self.pathDF.mvHeading[index]), 
                                (30,230), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA) 

            # mouse head to bridge vector
            frame = cv2.putText(frame, 
                                "toBridge: {:.0f} {:.0f}".format(self.pathDF.mouseToBridgeXCm[index],self.pathDF.mouseToBridgeYCm[index]), 
                                (30,260), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA) 


            # Lever orientation
            frame = cv2.putText(frame, 
                                "lever ori : {:.0f} deg".format(self.trialMLPx.leverOri[index]), 
                                (30,290), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA) 
        
            # Angle between mouse periphery after lever, arena center and bridge
            if self.valid:
                if index > self.peripheryAfterFirstLeverPressVideoIndex :
                    frame = cv2.putText(frame, 
                                        "Peri error: {:.0f} deg".format(self.periArenaCenterBridgeAngle[0]), 
                                        (30,320), 
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (100,200,0), 1, cv2.LINE_AA)

        
        #################
        ## second column
        ##################
        if detailLevel > 0:
            # Light condition
            frame = cv2.putText(frame, 
                                "Light cond.: {}".format(self.light), 
                                (frame.shape[1]-200,20), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA)

            # Location as a categorical variable
            frame = cv2.putText(frame, 
                                "Loca: {}".format(self.stateDF.loca[index]), 
                                (frame.shape[1]-200,50), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA)
            # journey (from bridge to arenaCenter)
            if(np.sum(index>self.journeyTransitionIndices)>0):
                journeyIndex = np.sum(index>self.journeyTransitionIndices)-1
                frame = cv2.putText(frame, 
                                    "Jou:{}/{}, atL:{}, pr:{}".format(journeyIndex+1,self.nJourneys,
                                                                       int(self.journeyList[journeyIndex].atLever),
                                                                       int(self.journeyList[journeyIndex].leverPressed)), 
                                    (frame.shape[1]-200,80), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, 
                                    "{} journeys".format(self.nJourneys), 
                                    (frame.shape[1]-200,80), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA)

            # lever presses
            if self.nLeverPresses > 0 :
                frame = cv2.putText(frame, 
                                    "Lever presse {} of {}".format(np.sum(index>self.leverPress.videoIndex),len(self.leverPress.videoIndex)), 
                                    (frame.shape[1]-200,110), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA)
            else : 
                frame = cv2.putText(frame, 
                                    "Trial without lever press", 
                                    (frame.shape[1]-200,110), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (100,200,0), 1, cv2.LINE_AA)
        if detailLevel > 1:
            frame = cv2.putText(frame, 
                                "Valid trial: {}".format(self.valid), 
                                (frame.shape[1]-200,140), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA)

            
        ###################################
        ### mask operations for the path ##
        ###################################
        # draw the path using a mask
        # because we are working in pixels and not cm, we are using self.trialMLPx instead of self.trialMLCm
        if not np.isnan(self.trialMLPx.loc[index,"mouseX"]) :
            maskDict["mask"] = cv2.circle(maskDict["mask"],
                                  (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                   radius=1, color=(0, 0, 0), thickness=1)
            
            if self.valid :
                # draw the search path into the specific mask
                             
                
                
                if index >= self.pathD.searchTotalStartIndex and index <= self.pathD.searchTotalEndIndex:
                    maskDict["maskSearchTotal"] = cv2.circle(maskDict["maskSearchTotal"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)
                if index >= self.pathD.searchArenaStartIndex and index <= self.pathD.searchArenaEndIndex:
                    maskDict["masksearchArena"] = cv2.circle(maskDict["masksearchArena"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)
                if index >= self.pathD.searchArenaNoLeverStartIndex and index <= self.pathD.searchArenaNoLeverEndIndex:
                    maskDict["masksearchArenaNoLever"] = cv2.circle(maskDict["masksearchArenaNoLever"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)

                if index >= self.pathD.homingTotalStartIndex and index <= self.pathD.homingTotalEndIndex:
                    maskDict["maskHomingTotal"] = cv2.circle(maskDict["maskHomingTotal"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)
                if index >= self.pathD.homingPeriStartIndex and index <= self.pathD.homingPeriEndIndex:
                    maskDict["maskHomingPeri"] = cv2.circle(maskDict["maskHomingPeri"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)
                if index >= self.pathD.homingPeriNoLeverStartIndex and index <= self.pathD.homingPeriNoLeverEndIndex:
                    maskDict["maskHomingPeriNoLever"] = cv2.circle(maskDict["maskHomingPeriNoLever"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)

        
        # these create an image with just the specific path in a specific color        
        searchArenaPath = cv2.bitwise_or(maskDict["searchArenaBG"],maskDict["searchArenaBG"],mask=maskDict["masksearchArena"])
        searchTotalPath = cv2.bitwise_or(maskDict["searchTotalBG"],maskDict["searchTotalBG"],mask=maskDict["maskSearchTotal"])
        searchArenaNoLever = cv2.bitwise_or(maskDict["searchArenaNoLeverBG"],maskDict["searchArenaNoLeverBG"],mask=maskDict["masksearchArenaNoLever"])
        homingTotalPath = cv2.bitwise_or(maskDict["homingTotalBG"],maskDict["homingTotalBG"],mask=maskDict["maskHomingTotal"])
        homingPeriPath = cv2.bitwise_or(maskDict["homingPeriBG"],maskDict["homingPeriBG"],mask=maskDict["maskHomingPeri"])
        homingPeriNoLeverPath = cv2.bitwise_or(maskDict["homingPeriNoLeverBG"],maskDict["homingPeriNoLeverBG"],mask=maskDict["maskHomingPeriNoLever"])

        
        # apply the path mask to the main frame to zero the pixels in the path
        frame = cv2.bitwise_or(frame, frame, mask=maskDict["mask"])
        if self.valid :
            # combine the different colors to get the search paths
            frame = frame + searchTotalPath + searchArenaPath + searchArenaNoLever
            # combine the different colors to get the homing paths
            frame = frame + homingTotalPath + homingPeriPath + homingPeriNoLeverPath

        ####################################### 
        # add mouse position and orientation ##
        #######################################
        # mouse orientaiton (head-direction) line
        if ~np.isnan(self.trialMLPx.loc[index,"mouseX"]):
            # mouse position dot
            frame = cv2.circle(frame,
                               (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                    radius=4, color=(0, 200, 255), thickness=1)
            # vector from mouse head in the HD
            if detailLevel > 0:
                frame = cv2.line(frame,
                                 (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                 (int(self.trialMLPx.loc[index,"mouseX"]+self.trialMLPx.loc[index,"mouseXHeading"]*2),
                                  int(self.trialMLPx.loc[index,"mouseY"]+self.trialMLPx.loc[index,"mouseYHeading"]*2)),
                                (0,200,255),2)

            # head to bridge vector
            if detailLevel > 0:
                frame = cv2.line(frame,
                                 (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                 (int(self.trialMLPx.loc[index,"mouseX"]+self.pathDF.loc[index,"mouseToBridgeXPx"]),
                                  int(self.trialMLPx.loc[index,"mouseY"]+self.pathDF.loc[index,"mouseToBridgeYPx"])),
                                (100,255,255),2)
        
        
        #######################################
        # add lever position and orientation ##
        # detection points                   ##
        #######################################
        # lever orientaiton line
        if ~np.isnan(self.trialMLPx.loc[index,"leverX"]) :
            frame = cv2.line(frame,
                             (int(self.trialMLPx.loc[index,"leverX"]),int(self.trialMLPx.loc[index,"leverY"])),
                             (int(self.trialMLPx.loc[index,"leverX"]+self.trialMLPx.loc[index,"leverXHeading"]*0.75),
                              int(self.trialMLPx.loc[index,"leverY"]+self.trialMLPx.loc[index,"leverYHeading"]*0.75)),
                            (0,0,255),2)
            # lever position dot
            frame = cv2.circle(frame,
                               (int(self.trialMLPx.loc[index,"leverX"]),int(self.trialMLPx.loc[index,"leverY"])),
                                radius=2, color=(0, 0, 255), thickness=2)
            # leverPress position dot
            frame = cv2.circle(frame,
                               (int(self.trialMLPx.loc[index,"leverPressX"]),int(self.trialMLPx.loc[index,"leverPressY"])),
                                radius=2, color=(0, 0, 255), thickness=2)
            # left corner 
            frame = cv2.circle(frame,
                                (int(self.trialMLPx.loc[index,"leverBoxPLX"]),
                                 int(self.trialMLPx.loc[index,"leverBoxPLY"])),
                                radius=4, color=(150, 255, 0), thickness=1)
            # right corner
            frame = cv2.circle(frame,
                                (int(self.trialMLPx.loc[index,"leverBoxPRX"]),
                                 int(self.trialMLPx.loc[index,"leverBoxPRY"])),
                                radius=4, color=(0, 255, 150), thickness=2)
            
            
        # add lever presses as red dots at the center of the lever
        if (self.leverPress.videoIndex==index).sum() == 1 and ~np.isnan(self.trialMLPx.loc[index,"leverX"]) :
             frame = cv2.circle(frame,
                                (int(self.trialMLPx.loc[index,"leverX"]),
                                 int(self.trialMLPx.loc[index,"leverY"])),
                                radius=4, color=(0, 255, 0), thickness=3)
            
            
            
        
        
        ## Draw a point where the animal reaches periphery after first lever press
        if self.valid > 0:
            if index > self.peripheryAfterFirstLeverPressVideoIndex :
                # mouse position dot
                frame = cv2.circle(frame,
                                   (int(self.peripheryAfterFirstLeverPressCoordPx[0]),
                                    int(self.peripheryAfterFirstLeverPressCoordPx[1])),
                                        radius=4, color=(255, 100, 0), thickness=4)
         
        ## Draw the bridge
        for i in range(3):
            frame = cv2.line(frame,
                        (int(self.bCoordPx[i,0]),int(self.bCoordPx[i,1])),
                        (int(self.bCoordPx[i+1,0]),int(self.bCoordPx[i+1,1])),
                            (200,200,200),1)
        frame = cv2.line(frame,
                        (int(self.bCoordPx[3,0]),int(self.bCoordPx[3,1])),
                        (int(self.bCoordPx[0,0]),int(self.bCoordPx[0,1])),
                            (200,200,200),1)
        
        ## Draw the periphery
        frame = cv2.circle(frame,
                                (int(self.aCoordPx[0]),int(self.aCoordPx[1])),
                                radius=int(self.radiusPeripheryPx), color=(50, 50, 50), thickness=1)
        ###########
        ## lever ##
        ###########
        ## Draw the lever, 
        for i in range(len(self.leverPx.pointsPlot[:,0])-1):
            frame = cv2.line(frame,
                            (int(self.leverPx.pointsPlot[i,0]),int(self.leverPx.pointsPlot[i,1])),
                            (int(self.leverPx.pointsPlot[i+1,0]),int(self.leverPx.pointsPlot[i+1,1])),
                            (200,200,200),1)
        
        
        ## Draw the lever enterZone
        if detailLevel > 0:
            for i in range(len(self.leverPx.enterZonePointsPlot[:,0])-1):
                frame = cv2.line(frame,
                                (int(self.leverPx.enterZonePointsPlot[i,0]),int(self.leverPx.enterZonePointsPlot[i,1])),
                                (int(self.leverPx.enterZonePointsPlot[i+1,0]),int(self.leverPx.enterZonePointsPlot[i+1,1])),
                                (200,200,200),2)

        ## Draw the lever exitZone
        if detailLevel > 1:
            for i in range(len(self.leverPx.exitZonePointsPlot[:,0])-1):
                frame = cv2.line(frame,
                                (int(self.leverPx.exitZonePointsPlot[i,0]),int(self.leverPx.exitZonePointsPlot[i,1])),
                                (int(self.leverPx.exitZonePointsPlot[i+1,0]),int(self.leverPx.exitZonePointsPlot[i+1,1])),
                                (200,200,200),2)

        
        self.frame = frame
       
        
        return frame
        
            
    def vectorAngle(self,v,rv=np.array([[1,0]]),degrees=False,quadrant=False) :
        """
    
        Calculate the angles between an array of vectors relative to a reference vector
        Argument:
            v: Array of vectors, one vector per row
            rv: Reference vector
            degrees: Boolean indicating whether to return the value as radians (False) or degrees (True)
            quadrant: Adjust the angle for 3 and 4 quadrants, assume rv is (1,0) and the dimension of v is 2.
        Return:
            Array of angles
        """
        # length of vector
        if v.shape[1]!=rv.shape[1]:
            print("v and rv should have the same number of column")
            return
        vLen = np.sqrt(np.sum(v*v,axis=1))
        vLen[vLen==0] = np.NAN
        rvLen = np.sqrt(np.sum(rv*rv,axis=1))

        # get unitary vector
        uv = v/vLen[:,None]
        urv = rv/rvLen[:,None]

        # get the angle
        theta = np.arccos(np.sum(uv*urv,axis=1))

        if quadrant:
            # deal with the 3 and 4 quadrant
            theta[v[:,-1] < 0] = 2*np.pi - theta[v[:,-1]<0] 

        if degrees :
            theta = theta * 360 / (2*np.pi)
        
        return theta

    def lightFromCode(self,x):
            """
            Get light or dark depending on light_code
            """
            if x == 1 or np.isnan(x):
                return "light"
            else:
                return "dark"      
         
    def previousLight(self,log):
        """
        Get the last light that was set before the current trial
        
        """
        lightEvents = log[log.event=="light"]
                
        if sum(lightEvents.time < self.startTime) == 0 :
            return np.nan
        else:
            return lightEvents.param[lightEvents.time< self.startTime].tail(1).to_numpy()[0]

    def trialPathFigure(self,pathNames = ["searchArenaNoLever","homingPeriNoLever"], legend = True, figSize=(10,10), filePath=None):
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
        axes.plot(np.cos(arena)*self.arenaRadiusCm,np.sin(arena)*self.arenaRadiusCm,label="Arena",color="gray")
        axes.plot(np.cos(arena)*self.arenaRadiusCm*self.arenaRadiusProportionToPeri,
                     np.sin(arena)*self.arenaRadiusCm*self.arenaRadiusProportionToPeri,label="Periphery",color="gray",linestyle='dashed')
        axes.set_xlabel("cm")
        axes.set_ylabel("cm")
        
       
                      
        ## lever
        axes.plot(self.leverCm.pointsPlot[:,0],self.leverCm.pointsPlot[:,1], color = "gray")
        axes.plot(self.leverCm.enterZonePointsPlot[:,0],self.leverCm.enterZonePointsPlot[:,1], color = "gray",linestyle="dotted")
        axes.plot(self.leverCm.exitZonePointsPlot[:,0],self.leverCm.exitZonePointsPlot[:,1], color = "yellow",linestyle="dotted")
        ## mouse position at lever presses
        if self.nLeverPresses > 0:
            axes.scatter(self.trialMLCm.mouseX[self.leverPressVideoIndex],
                         self.trialMLCm.mouseY[self.leverPressVideoIndex],color="red")
        
        
         ## mouse path
        axes.plot(self.trialMLCm.mouseX,self.trialMLCm.mouseY,color="black", label="path")
        
        ## add the requested path, taken from the last journey

        for p in pathNames:
            if self.pathD[p].pPose is not None :
                axes.plot(self.pathD[p].pPose[:,0],
                          self.pathD[p].pPose[:,1],label=p)
        
        ## add a point at which the homingPeriNoLever path start
        if self.pathD["homingPeriNoLever"].pPose is not None :
            axes.scatter(self.pathD["homingPeriNoLever"].pPose[0,0],
                         self.pathD["homingPeriNoLever"].pPose[0,1],label="homingStart")
        
        
        ## reaching periphery point
        if self.valid > 0:
            # mouse position dot
            axes.scatter(self.peripheryAfterFirstLeverPressCoordCm[0],self.peripheryAfterFirstLeverPressCoordCm[1]
                     ,color="blue")

        if legend:
            axes.legend(loc="upper right")
        
        if filePath is not None:
            print("Saving to " + filePath)
            plt.savefig(filePath,bbox_inches = "tight")
    
        #plt.close(fig)
        
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    
