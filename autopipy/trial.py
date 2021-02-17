import os.path
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import cv2
import sys
from autopipy.navPath import NavPath
from autopipy.lever import Lever

class Trial:
    """
    Class containing information about an autopi trial
    
    When calculating angles, the y coordinate is reversed (0-y) so that 90 degree is up.
    This is done because the y-axis is reversed in the videos.
     
    Attributes:
        name: Name of the trial, usually session_trialNo
        sessionName: Name of the session in which the trial was performed
        trialNo: Trial number within the session
        startTime: start time of the trial
        endTime: end time of the trial
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
        
    """
    def __init__(self,sessionName,trialNo,startTime,endTime,startTimeWS,endTimeWS):
        self.sessionName = sessionName
        self.trialNo = trialNo
        self.startTime = startTime
        self.endTime = endTime
        self.startTimeWS = startTimeWS
        self.endTimeWS = endTimeWS
        self.name = "{}_{}".format(sessionName,trialNo)
        self.duration = None
        self.trialVideoLog = None # DataFrame
        self.light = None
        self.startVideoIndex = None
        self.endVideoIndex = None
        self.aCoord = None # np array (x y r)
        self.bCoord = None # np array
        self.bCoordMiddle = None # np array x y
        self.trialML = None # DataFrame, mouse and lever position data for this trial
        self.leverPosition = None # Dictionary with the median lever position
        self.lever = None # Lever object to draw or to determine whether the animal is at or on the lever
        self.radiusPeriphery = None
        self.radiusLeverProximity = None
        self.traveledDistance = None
        self.pathDF = None # DataFrame, instantaneous variable evolving during the trial (e.g., distance run)
        self.leverPress = None # DataFrame with Ros time and video index of lever presses
        self.peripheryAfterFirstLeverPressCoord = None
        self.arenaToBridgeAngle = None
        self.arenaToMousePeriAngle = None
        self.stateDF = None # DataFrame with the categorical position of the mouse (Bridge, Arena, ArenaCenter, Lever, Gap)
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
        self.journeyTransitionIndices = None
        self.nJourneys = None
        ## NavPath objects
        self.searchTotalNavPath = None
        self.searchArenaNavPath = None
        self.searchArenaNoLeverNavPath = None
        self.homingTotalNavPath = None
        self.homingPeriNavPath = None
        self.homingPeriNoLeverNavPath = None
    
    def getTrialVariables(self):
        """
        Return a pandas DataFrame with most trial variables
        """
        self.myDict = {"sessionName": [self.sessionName],
                       "name": [self.name],
                       "trialNo": [self.trialNo],
                       "startTime": [self.startTime],
                       "endTime": [self.endTime],
                       "duration": [self.duration],
                       "light": [self.light],
                       "leverPositionX" : [self.leverPosition["leverX"]],
                       "leverPositionY" : [self.leverPosition["leverY"]],
                       "leverPositionOri": [self.leverPosition["leverOri"]],
                       "leverPressX": [self.leverPosition["leverPressX"]],
                       "leverPressY": [self.leverPosition["leverPressY"]],
                       "nLeverPresses" : [len(self.leverPress.time)],
                       "nJourneys" : [self.nJourneys],
                       "travelledDistance" : [self.traveledDistance],
                       "anglularErrorHomingPeri" : self.periArenaCenterBridgeAngle,
                       # for searchTotal path 
                       "searchTotal-length" : [self.searchTotalNavPath.length],
                       "searchTotal-duration" : [self.searchTotalNavPath.duration],
                        "searchTotal-meanVectorLengthPosi" : [self.searchTotalNavPath.meanVectorLengthPosi],
                        "searchTotal-meanVectorDirectionPosi": [self.searchTotalNavPath.meanVectorDirectionPosi],
                        "searchTotal-meanVectorLengthOri" : [self.searchTotalNavPath.meanVectorLengthOri[0]],
                        "searchTotal-meanVectorDirectionOri" : [self.searchTotalNavPath.meanVectorDirectionOri[0]],
                        "searchTotal-meanSpeed" : [self.searchTotalNavPath.meanSpeed],
                        "searchTotal-medianMVDeviationToTarget" : [self.searchTotalNavPath.medianMVDeviationToTarget],
                        "searchTotal-medianHDDeviationToTarget" : [self.searchTotalNavPath.medianHDDeviationToTarget],
                      # for searchArena path 
                       "searchArena-length" : [self.searchArenaNavPath.length],
                       "searchArena-duration" : [self.searchArenaNavPath.duration],
                        "searchArena-meanVectorLengthPosi" : [self.searchArenaNavPath.meanVectorLengthPosi],
                        "searchArena-meanVectorDirectionPosi": [self.searchArenaNavPath.meanVectorDirectionPosi],
                        "searchArena-meanVectorLengthOri" : [self.searchArenaNavPath.meanVectorLengthOri[0]],
                        "searchArena-meanVectorDirectionOri" : [self.searchArenaNavPath.meanVectorDirectionOri[0]],
                        "searchArena-meanSpeed" : [self.searchArenaNavPath.meanSpeed],
                        "searchArena-medianMVDeviationToTarget" : [self.searchArenaNavPath.medianMVDeviationToTarget],
                        "searchArena-medianHDDeviationToTarget" : [self.searchArenaNavPath.medianHDDeviationToTarget],
                       # for searchArenaNoLever path
                       "searchArenaNoLever-length" : [self.searchArenaNoLeverNavPath.length],
                       "searchArenaNoLever-duration" : [self.searchArenaNoLeverNavPath.duration],
                        "searchArenaNoLever-meanVectorLengthPosi" : [self.searchArenaNoLeverNavPath.meanVectorLengthPosi],
                        "searchArenaNoLever-meanVectorDirectionPosi": [self.searchArenaNoLeverNavPath.meanVectorDirectionPosi],
                        "searchArenaNoLever-meanVectorLengthOri" : [self.searchArenaNoLeverNavPath.meanVectorLengthOri[0]],
                        "searchArenaNoLever-meanVectorDirectionOri" : [self.searchArenaNoLeverNavPath.meanVectorDirectionOri[0]],
                        "searchArenaNoLever-meanSpeed" : [self.searchArenaNoLeverNavPath.meanSpeed],
                        "searchArenaNoLever-medianMVDeviationToTarget" : [self.searchArenaNoLeverNavPath.medianMVDeviationToTarget],
                        "searchArenaNoLever-medianHDDeviationToTarget" : [self.searchArenaNoLeverNavPath.medianHDDeviationToTarget],
                       # for the homingTotal path 
                       "homingTotal-length" : [self.homingTotalNavPath.length],
                       "homingTotal-duration" : [self.homingTotalNavPath.duration],
                        "homingTotal-meanVectorLengthPosi" : [self.homingTotalNavPath.meanVectorLengthPosi],
                        "homingTotal-meanVectorDirectionPosi": [self.homingTotalNavPath.meanVectorDirectionPosi],
                        "homingTotal-meanVectorLengthOri" : [self.homingTotalNavPath.meanVectorLengthOri[0]],
                        "homingTotal-meanVectorDirectionOri" : [self.homingTotalNavPath.meanVectorDirectionOri[0]],
                        "homingTotal-meanSpeed" : [self.homingTotalNavPath.meanSpeed],
                        "homingTotal-medianMVDeviationToTarget" : [self.homingTotalNavPath.medianMVDeviationToTarget],
                        "homingTotal-medianHDDeviationToTarget" : [self.homingTotalNavPath.medianHDDeviationToTarget],
                       # for the homingPeri path
                       "homingPeri-length" : [self.homingPeriNavPath.length],
                       "homingPeri-duration" : [self.homingPeriNavPath.duration],
                        "homingPeri-meanVectorLengthPosi" : [self.homingPeriNavPath.meanVectorLengthPosi],
                        "homingPeri-meanVectorDirectionPosi": [self.homingPeriNavPath.meanVectorDirectionPosi],
                        "homingPeri-meanVectorLengthOri" : [self.homingPeriNavPath.meanVectorLengthOri[0]],
                        "homingPeri-meanVectorDirectionOri" : [self.homingPeriNavPath.meanVectorDirectionOri[0]],
                        "homingPeri-meanSpeed" : [self.homingPeriNavPath.meanSpeed],
                        "homingPeri-medianMVDeviationToTarget" : [self.homingPeriNavPath.medianMVDeviationToTarget],
                        "homingPeri-medianHDDeviationToTarget" : [self.homingPeriNavPath.medianHDDeviationToTarget],
                       # for the homingPeriNoLever
                       "homingPeriNoLever-length" : [self.homingPeriNoLeverNavPath.length],
                       "homingPeriNoLever-duration" : [self.homingPeriNoLeverNavPath.duration],
                        "homingPeriNoLever-meanVectorLengthPosi" : [self.homingPeriNoLeverNavPath.meanVectorLengthPosi],
                        "homingPeriNoLever-meanVectorDirectionPosi": [self.homingPeriNoLeverNavPath.meanVectorDirectionPosi],
                        "homingPeriNoLever-meanVectorLengthOri" : [self.homingPeriNoLeverNavPath.meanVectorLengthOri[0]],
                        "homingPeriNoLever-meanVectorDirectionOri" : [self.homingPeriNoLeverNavPath.meanVectorDirectionOri[0]],
                        "homingPeriNoLever-meanSpeed" : [self.homingPeriNoLeverNavPath.meanSpeed],
                        "homingPeriNoLever-medianMVDeviationToTarget" : [self.homingPeriNoLeverNavPath.medianMVDeviationToTarget],
                        "homingPeriNoLever-medianHDDeviationToTarget" : [self.homingPeriNoLeverNavPath.medianHDDeviationToTarget]} 
        return pd.DataFrame(self.myDict)
        
    
    def extractTrialFeatures(self,log,mLPosi,videoLog,aCoord,bCoord,
                             arenaRadiusProportionToPeri=0.925):
        """
        Extract trial features 
        
        Arguments
            log: DataFrame with event log of the session
            mLPosi: DataFrame with mouse and lever position for every video frame
            videoLog: DataFrame time for each video frame
            aCoord: np array arena coordinates (x, y, radius)
            bCoord: np array bridge coordinates (four points)
            arenaRadiusProportionToPeri: proportion of arena radius at which the periphery of the arena is
            leverProximityRadiusProportion: proportion of the distance between center of lever and leverPress. 
            
        """
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
        
        ###########################
        # get the light condition #
        ###########################
        lightCode = self.previousLight(log)
        self.light = self.lightFromCode(lightCode)
        
        
        ##################################
        ## arena and bridge coordinates ##
        ##################################
        self.aCoord = aCoord
        self.bCoord = bCoord
        self.bCoordMiddle = self.bCoord[0] + (self.bCoord[2]-self.bCoord[0])/2
        
        ########################################################
        # get the mouse and lever tracking data for the trial ##
        ########################################################
        self.trialML = mLPosi[(videoLog.time > self.startTime) & (videoLog.time < self.endTime)]     
        
        
        #######################################
        # location of the lever box and lever #
        #######################################
        self.leverPosition = {"leverX" : np.nanmedian(self.trialML["leverX"]),
                             "leverY" : np.nanmedian(self.trialML["leverY"]),
                             "leverOri": np.nanmedian(self.trialML["leverOri"]),
                             "leverPressX" : np.nanmedian(self.trialML["leverPressX"]),
                             "leverPressY" : np.nanmedian(self.trialML["leverPressY"])}
        self.lever = Lever()
        self.lever.calculatePose(lp = np.array([np.nanmedian(self.trialML["leverPressX"]), np.nanmedian(self.trialML["leverPressY"])]),
                                 pl = np.array([np.nanmedian(self.trialML["leverBoxPLX"]), np.nanmedian(self.trialML["leverBoxPLY"])]),
                                 pr = np.array([np.nanmedian(self.trialML["leverBoxPRX"]), np.nanmedian(self.trialML["leverBoxPRY"])]))
        
        ################################################
        # define arena periphery and lever proximity  ##
        ################################################
        # radius from the arena center that defines the periphery of the arena
        self.radiusPeriphery = aCoord[2]*arenaRadiusProportionToPeri
       
        
        #########################################
        # variables that evolve along the path ##
        #########################################
        
        # np.array, used later on to know if at the lever
        mousePoints = np.stack((self.trialML.mouseX.to_numpy(),self.trialML.mouseY.to_numpy()),axis=1)
        
        
        # we will store all these Series in a DataFrame called pathDF
        distance = np.sqrt(np.diff(self.trialML.mouseX,prepend=np.NAN)**2+
                                np.diff(self.trialML.mouseY,prepend=np.NAN)**2)
        traveledDistance = np.nancumsum(distance) # cumsum
        self.traveledDistance = np.nansum(distance) # sum
        videoFrameTimeDifference = self.trialVideoLog.time.diff().to_numpy()
        speed = distance/videoFrameTimeDifference
        speedNoNAN = np.nan_to_num(speed) # replace NAN with 0.0 to display in video
        distanceFromArenaCenter = np.sqrt((self.trialML.mouseX.to_numpy() - self.aCoord[0])**2+ 
                                               (self.trialML.mouseY.to_numpy() - self.aCoord[1])**2)
        ## distance from lever
        distanceFromLeverPress = np.sqrt((self.trialML.leverPressX.to_numpy() - self.trialML.mouseX.to_numpy() )**2 + 
                                        (self.trialML.leverPressY.to_numpy() - self.trialML.mouseY.to_numpy())**2)
        distanceFromLever = np.sqrt((self.trialML.leverX.to_numpy() - self.trialML.mouseX.to_numpy() )**2 + 
                                        (self.trialML.leverY.to_numpy() - self.trialML.mouseY.to_numpy())**2)
        
        ## movement heading of the mouse relative to [1,0]
        mv = np.stack((np.diff(self.trialML.mouseX,prepend=np.NAN),
                       np.diff(self.trialML.mouseY,prepend=np.NAN)),axis=1)
        mvHeading = self.vectorAngle(mv,degrees=True,quadrant=True)
        ## vector from mouse to bridge
        mBVX = self.bCoordMiddle[0] - self.trialML.mouseX.to_numpy() 
        mBVY = self.bCoordMiddle[1] - self.trialML.mouseY.to_numpy()   
        mouseToBridge = np.stack((mBVX,mBVY),axis = 1)
       
        ## angle between movement heading and vector from the mouse to the bridge
        mvHeadingToBridgeAngle = self.vectorAngle(mv,mouseToBridge,degrees=True)
        ## angle between head direction and vector from the mouse to the bridge
        hdv = np.stack((self.trialML.mouseXHeading.to_numpy(),
                       self.trialML.mouseYHeading.to_numpy()),axis = 1)
        hdToBridgeAngle = self.vectorAngle(hdv,mouseToBridge,degrees=True)
        # Store these Series into a data frame
        # use the same index as the self.trialML DataFrame
        self.pathDF = pd.DataFrame({"distance" :distance,
                                  "traveledDistance" : traveledDistance,
                                  "mvHeading" : mvHeading,
                                  "mouseToBridgeX": mBVX,
                                  "mouseToBridgeY": mBVY,
                                  "mvHeadingToBridgeAngle" : mvHeadingToBridgeAngle,
                                  "hdToBridgeAngle" : hdToBridgeAngle,
                                  "speed" : speed,
                                  "speedNoNAN" : speedNoNAN,
                                  "distanceFromArenaCenter" : distanceFromArenaCenter,
                                  "distanceFromLever" : distanceFromLever,
                                  "distanceFromLeverPress": distanceFromLeverPress},
                                  index = self.trialML.index)         
       
        ######################
        # get lever presses ##
        ######################
        lever = log[log.event=="lever_press"]
        index = (lever.time>self.startTime) & (lever.time<self.endTime) # boolean array
        leverPressTime = lever.time[index] # ROS time of lever
        leverPressVideoIndex = leverPressTime.apply(self.videoIndexFromTimeStamp) # video index
        self.leverPress = pd.DataFrame({"time": leverPressTime,
                                        "videoIndex":leverPressVideoIndex})
        
        #################################################
        ## reaching periphery after first lever press  ##
        #################################################
        for i in range(self.leverPress.videoIndex.iloc[0],
                       self.trialVideoLog.frame_number.iloc[-1]) :
            if (self.pathDF.distanceFromArenaCenter[i] > self.radiusPeriphery):
                self.peripheryAfterFirstLeverPressVideoIndex = i
                break
        
        #####################################################################
        ## moue coordinate when reaching periphery after first lever press ##
        #####################################################################
        self.peripheryAfterFirstLeverPressCoord = np.array([self.trialML.loc[self.peripheryAfterFirstLeverPressVideoIndex,"mouseX"],
                                                            self.trialML.loc[self.peripheryAfterFirstLeverPressVideoIndex,"mouseY"]])
        
        #################################
        # Reaching periphery analysis  ##
        #################################
        ## angle of the bridge center relative to arena center 
        arenaToBridgeVector= self.bCoordMiddle - self.aCoord[:2]
        self.arenaToBridgeAngle = self.vectorAngle(np.expand_dims(arenaToBridgeVector,0),degrees=True)
        ## angle from mouse when reaching periphery relative to arena center
        arenaToMousePeriVector = self.peripheryAfterFirstLeverPressCoord-self.aCoord[:2]
        self.arenaToMousePeriAngle = self.vectorAngle(np.expand_dims(arenaToMousePeriVector,0),degrees=True)
        ## angular deviation of the mouse when reaching periphery
        self.periArenaCenterBridgeAngle=self.vectorAngle(v = np.expand_dims(arenaToMousePeriVector,0),
                                                         rv = np.expand_dims(arenaToBridgeVector,0),
                                                         degrees=True)
        
        #################################################
        ## sectioning the trial into different states  ##
        #################################################
        # define each frame as arena, bridge, lever or home base (NAN), one-hot encoding
        self.stateDF=pd.DataFrame({"lever": self.lever.isAt(mousePoints) ,
                                   "arenaCenter": self.pathDF.distanceFromArenaCenter<self.radiusPeriphery,
                                   "arena": self.pathDF.distanceFromArenaCenter<self.aCoord[2],
                                   "bridge": ((self.trialML.mouseX > self.bCoord[0,0]) & 
                                              (self.trialML.mouseX < self.bCoord[2,0]) & 
                                              (self.trialML.mouseY > self.bCoord[0,1]) & 
                                              (self.trialML.mouseY < self.bCoord[2,1])),
                                   "homeBase": pd.isna(self.trialML.mouseX)})
        # if all false, the mouse is not on arena or bridge
        # most likely between the arena and bridge, or poking it over the edge of the arena
        self.stateDF.insert(0, "gap", self.stateDF.sum(1)==0) 
        # get the one-hot encoding back into categorical, when several true, the first column is return.
        self.stateDF["loca"] = self.stateDF.iloc[:, :].idxmax(1)
        
        
        ######################################################################
        ## number of journeys on the arena (from the bridge to arena center)##
        ######################################################################
        bridgeArenaCenter = self.stateDF[ (self.stateDF.loca=="bridge") | (self.stateDF.loca=="arenaCenter") ]
        df = pd.DataFrame({"start" : bridgeArenaCenter.shift().loca,"end" : bridgeArenaCenter.loca})
        self.journeyTransitionIndices = df[(df.start=="bridge") & (df.end=="arenaCenter")].index.values
        self.nJourneys=len(self.journeyTransitionIndices)
                    
        ####################
        ## identify  paths #
        ####################
        if len(self.leverPress) > 0 : # nothing if this makes sense if there is no lever press   
            ###################
            ## search paths ###
            ###################
            ## searchTotal, from first step on the arena to lever pressing, excluding bridge time
            self.searchTotalStartIndex = self.stateDF.loca.index[self.stateDF.loca=="arena"][0]
            self.searchTotalEndIndex = self.leverPress.videoIndex.iloc[0]        
            ## searchArena, from first step on the arena after the last bridge to lever pressing
            bridgeIndex = self.stateDF[self.stateDF.loca=="bridge"].index
            if len(bridgeIndex[(bridgeIndex.values < self.leverPress.videoIndex.iloc[0])])==0:
                print("no bridge before lever press in trial {}".format(self.trialNo))
                print("This situation could be caused by video synchronization problems")
                lastBridgeIndexBeforePress=self.startVideoIndex
            else :
                lastBridgeIndexBeforePress = (bridgeIndex[(bridgeIndex.values < self.leverPress.videoIndex.iloc[0])])[-1]
            self.searchArenaStartIndex = lastBridgeIndexBeforePress
            self.searchArenaEndIndex = self.leverPress.videoIndex.iloc[0]
            ## searchArenaNoLever, seachLast, excluding time at lever before pressing
            leverIndex = self.stateDF[self.stateDF.loca=="lever"].index
            self.searchArenaNoLeverStartIndex = self.searchArenaStartIndex
            self.searchArenaNoLeverEndIndex = (leverIndex[(leverIndex.values >lastBridgeIndexBeforePress) &
               (leverIndex.values < self.leverPress.videoIndex.iloc[0])])[0]
        
            ##################
            ## homing paths ##
            ##################
            ## homingTotal, from first lever press to first bridge after the press
            self.homingTotalStartIndex = self.leverPress.videoIndex.iloc[0]
            if len(bridgeIndex[(bridgeIndex.values > self.leverPress.videoIndex.iloc[0])])==0:
                print("no bridge after lever press in trial {}".format(self.trialNo))
                print("This situation could be caused by video synchronization problems")
                firstBridgeIndexAfterPress=self.endVideoIndex
            else :
                firstBridgeIndexAfterPress = (bridgeIndex[(bridgeIndex.values > self.leverPress.videoIndex.iloc[0])])[0]
            self.homingTotalEndIndex = firstBridgeIndexAfterPress
            ## homingPeri, from first lever press to periphery
            self.homingPeriStartIndex = self.homingTotalStartIndex
            self.homingPeriEndIndex = self.peripheryAfterFirstLeverPressVideoIndex
            ## homingPeriNoLever, from first lever press to periphery, excluding first lever time period
            notAtLeverIndex = self.stateDF[self.stateDF.lever==0].index
            self.homingPeriNoLeverStartIndex = (notAtLeverIndex[notAtLeverIndex.values > self.leverPress.videoIndex.iloc[0]])[0]
            self.homingPeriNoLeverEndIndex = self.peripheryAfterFirstLeverPressVideoIndex
   
            #######################################################
            ## create the NavPath objects to get paths variables ##
            #######################################################
            searchTotalPose = self.poseFromTrialData(self.searchTotalStartIndex,
                                                     self.searchTotalEndIndex)
            searchArenaPose = self.poseFromTrialData(self.searchArenaStartIndex,
                                                    self.searchArenaEndIndex)
            searchArenaNoLeverPose = self.poseFromTrialData(self.searchArenaNoLeverStartIndex,
                                                           self.searchArenaNoLeverEndIndex)
            homingTotalPose = self.poseFromTrialData(self.homingTotalStartIndex,
                                                     self.homingTotalEndIndex)
            homingPeriPose = self.poseFromTrialData(self.homingPeriStartIndex,
                                                     self.homingPeriEndIndex)
            homingPeriNoLeverPose = self.poseFromTrialData(self.homingPeriNoLeverStartIndex,
                                                     self.homingPeriNoLeverEndIndex)
            
            # targets for the paths
            leverPose = self.poseFromLeverPositionDictionary()
            bridgePose = self.poseFromBridgeCoordinates()
            
            # get the path variables using the NavPath class
            self.searchTotalNavPath = NavPath(pPose = searchTotalPose,
                                              targetPose=leverPose,name = "searchTotal")
            self.searchArenaNavPath =  NavPath(pPose = searchArenaPose,
                                              targetPose=leverPose, name = "searchArena")
            self.searchArenaNoLeverNavPath =  NavPath(pPose = searchArenaNoLeverPose,
                                              targetPose=leverPose, name = "searchArenaNoLever")
            self.homingTotalNavPath = NavPath(pPose = homingTotalPose,
                                              targetPose=bridgePose, name = "homingTotal")
            self.homingPeriNavPath = NavPath(pPose = homingPeriPose,
                                              targetPose=bridgePose, name = "homingPeri")
            self.homingPeriNoLeverNavPath = NavPath(pPose = homingPeriNoLeverPose,
                                              targetPose=bridgePose, name = "homingPeriNoLever")
            
            
            
            
    def poseFromTrialData(self,startIndex,endIndex):
        """
        Create a numpy array containing the pose (x, y, z, yaw, pitch, roll, time) during a defined time period
        Arguments:
            startIndex: start index of the path, index are from the self.trialML or self.trialVideoLog (they are the same)
            endIndex: end index of the path

        Return Pose as a numpy array with 7 columns

        """
        return np.stack((self.trialML.loc[startIndex:endIndex,"mouseX"].to_numpy(),
                    self.trialML.loc[startIndex:endIndex,"mouseY"].to_numpy(),
                    np.zeros_like(self.trialML.loc[startIndex:endIndex,"mouseY"].to_numpy()),
                    self.trialML.loc[startIndex:endIndex,"mouseOri"].to_numpy(),
                    np.zeros_like(self.trialML.loc[startIndex:endIndex,"mouseY"].to_numpy()),
                    np.zeros_like(self.trialML.loc[startIndex:endIndex,"mouseY"].to_numpy()),
                    self.trialVideoLog.timeWS.loc[startIndex:endIndex].to_numpy()),axis=1)
    def poseFromLeverPositionDictionary(self) :
        """
        Create a numpy array containing the pose (x, y, z, yaw, pitch, roll, time)  
        from the leverPosition dictionary.
        """
        return np.array([[self.leverPosition["leverX"],self.leverPosition["leverY"],0,
                          self.leverPosition["leverOri"],0,0,0]])

    def poseFromBridgeCoordinates(self) :
        """
        Create a numpy array containing the pose (x, y, z, yaw, pitch, roll, time)  
        from the middle of the bridge
        """
        return np.array([[self.bCoordMiddle[0],self.bCoordMiddle[1],0,0,0,0,0]])
    
    def videoIndexFromTimeStamp(self, timeStamp):
        """
        Get the frame or index in the video for a given timestamp (event)
        """
        return self.trialVideoLog.frame_number.iloc[np.argmin(np.abs(self.trialVideoLog.time - timeStamp))]  
        
    def createTrialVideo(self,pathVideoFile,pathVideoFileOut):
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
        
            frame = self.decorateVideoFrame(frame,i,count,maskDict)
            
            out.write(frame)
            count=count+1
    
        out.release() 
        cap.release() 
    
    
    def decorateVideoFrame(self,frame,index,count,maskDict):
        
        
        
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
        frame = cv2.putText(frame, 
                            "Time: {:.2f} sec".format(self.trialVideoLog.timeWS.iloc[count]), 
                            (30,20), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)
        # traveled distance
        frame = cv2.putText(frame, 
                            "Distance: {:.1f} pxs".format(self.pathDF.traveledDistance[index]), 
                            (30,50), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)
        # speed
        frame = cv2.putText(frame, 
                            "Speed: {:.0f} pxs/sec".format(self.pathDF.speedNoNAN[index]), 
                            (30,80), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)       
        # distance to lever
        frame = cv2.putText(frame, 
                            "Distance lever: {:.1f} pxs".format(self.pathDF.distanceFromLever[index]), 
                            (30,110), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)       
        # Angle between mouse head and the bridge
        frame = cv2.putText(frame, 
                            "MvHead: {:.0f} deg".format(self.pathDF.mvHeading[index]), 
                            (30,140), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        
        # Head direction of the mouse
        frame = cv2.putText(frame, 
                            "HD: {:.0f} deg".format(self.trialML.mouseOri[index]), 
                            (30,170), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        
        # mouse head to bridge vector
        frame = cv2.putText(frame, 
                            "toBridge: {:.0f} {:.0f}".format(self.pathDF.mouseToBridgeX[index],self.pathDF.mouseToBridgeY[index]), 
                            (30,200), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        
        
        # Angle between mouse movement heading and vector from mouse to the bridge
        frame = cv2.putText(frame, 
                            "mvHeadToBridge : {:.0f} deg".format(self.pathDF.mvHeadingToBridgeAngle[index]), 
                            (30,230), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        # Angle between head direction and bridge
        frame = cv2.putText(frame, 
                            "hdToBridge : {:.0f} deg".format(self.pathDF.hdToBridgeAngle[index]), 
                            (30,260), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        # Lever orientation
        frame = cv2.putText(frame, 
                            "lever ori : {:.0f} deg".format(self.trialML.leverOri[index]), 
                            (30,290), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        
        # Angle between mouse periphery after lever, arena center and bridge
        if index > self.peripheryAfterFirstLeverPressVideoIndex :
            frame = cv2.putText(frame, 
                                "Peri error: {:.0f} deg".format(self.periArenaCenterBridgeAngle[0]), 
                                (30,320), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA)
        
        
        #################
        ## second column
        ##################
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
        frame = cv2.putText(frame, 
                            "Journey {} of {}".format(np.sum(index>self.journeyTransitionIndices),self.nJourneys), 
                            (frame.shape[1]-200,80), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)
        # lever presses
        frame = cv2.putText(frame, 
                            "Lever presse {} of {}".format(np.sum(index>self.leverPress.videoIndex),len(self.leverPress.videoIndex)), 
                            (frame.shape[1]-200,110), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)
        
            
            
        ###################################
        ### mask operations for the path ##
        ###################################
        # draw the path using a mask
        if not np.isnan(self.trialML.loc[index,"mouseX"]) :
            maskDict["mask"] = cv2.circle(maskDict["mask"],
                                  (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                   radius=1, color=(0, 0, 0), thickness=1)
            
            # draw the search path into the specific mask
            if index >= self.searchTotalStartIndex and index <= self.searchTotalEndIndex:
                maskDict["maskSearchTotal"] = cv2.circle(maskDict["maskSearchTotal"],
                                          (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                           radius=1, color=(255, 255, 255), thickness=1)
            if index >= self.searchArenaStartIndex and index <= self.searchArenaEndIndex:
                maskDict["masksearchArena"] = cv2.circle(maskDict["masksearchArena"],
                                          (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                           radius=1, color=(255, 255, 255), thickness=1)
            if index >= self.searchArenaNoLeverStartIndex and index <= self.searchArenaNoLeverEndIndex:
                maskDict["masksearchArenaNoLever"] = cv2.circle(maskDict["masksearchArenaNoLever"],
                                          (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                           radius=1, color=(255, 255, 255), thickness=1)
        
            if index >= self.homingTotalStartIndex and index <= self.homingTotalEndIndex:
                maskDict["maskHomingTotal"] = cv2.circle(maskDict["maskHomingTotal"],
                                          (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                           radius=1, color=(255, 255, 255), thickness=1)
            if index >= self.homingPeriStartIndex and index <= self.homingPeriEndIndex:
                maskDict["maskHomingPeri"] = cv2.circle(maskDict["maskHomingPeri"],
                                          (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                           radius=1, color=(255, 255, 255), thickness=1)
            if index >= self.homingPeriNoLeverStartIndex and index <= self.homingPeriNoLeverEndIndex:
                maskDict["maskHomingPeriNoLever"] = cv2.circle(maskDict["maskHomingPeriNoLever"],
                                          (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
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
        
        # combine the different colors to get the search paths
        frame = frame + searchTotalPath + searchArenaPath + searchArenaNoLever
        # combine the different colors to get the homing paths
        frame = frame + homingTotalPath + homingPeriPath + homingPeriNoLeverPath
        
      
        
        ####################################### 
        # add mouse position and orientation ##
        #######################################
        # mouse orientaiton (head-direction) line
        if ~np.isnan(self.trialML.loc[index,"mouseX"]):
            frame = cv2.line(frame,
                             (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                             (int(self.trialML.loc[index,"mouseX"]+self.trialML.loc[index,"mouseXHeading"]*2),
                              int(self.trialML.loc[index,"mouseY"]+self.trialML.loc[index,"mouseYHeading"]*2)),
                            (0,200,255),2)
            # mouse position dot
            frame = cv2.circle(frame,
                               (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                    radius=4, color=(0, 200, 255), thickness=1)
            # head to bridge vector
            frame = cv2.line(frame,
                             (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                             (int(self.trialML.loc[index,"mouseX"]+self.pathDF.loc[index,"mouseToBridgeX"]),
                              int(self.trialML.loc[index,"mouseY"]+self.pathDF.loc[index,"mouseToBridgeY"])),
                            (100,255,255),2)
        
        
        #######################################
        # add lever position and orientation ##
        #######################################
        # lever orientaiton line
        if ~np.isnan(self.trialML.loc[index,"leverX"]) :
            frame = cv2.line(frame,
                             (int(self.trialML.loc[index,"leverX"]),int(self.trialML.loc[index,"leverY"])),
                             (int(self.trialML.loc[index,"leverX"]+self.trialML.loc[index,"leverXHeading"]*0.75),
                              int(self.trialML.loc[index,"leverY"]+self.trialML.loc[index,"leverYHeading"]*0.75)),
                            (0,0,255),2)
            # lever position dot
            frame = cv2.circle(frame,
                               (int(self.trialML.loc[index,"leverX"]),int(self.trialML.loc[index,"leverY"])),
                                radius=2, color=(0, 0, 255), thickness=2)
            # leverPress position dot
            frame = cv2.circle(frame,
                               (int(self.trialML.loc[index,"leverPressX"]),int(self.trialML.loc[index,"leverPressY"])),
                                radius=2, color=(0, 0, 255), thickness=2)
        
        
        # add lever presses as red dots at the center of the lever
        if (self.leverPress.videoIndex==index).sum() == 1:
             frame = cv2.circle(frame,
                                (int(self.trialML.loc[index,"leverX"]),
                                 int(self.trialML.loc[index,"leverY"])),
                                radius=4, color=(0, 255, 0), thickness=3)
        
        
        ## Draw a point where the animal reaches periphery after first lever press
        if index > self.peripheryAfterFirstLeverPressVideoIndex :
            # mouse position dot
            frame = cv2.circle(frame,
                               (int(self.peripheryAfterFirstLeverPressCoord[0]),
                                int(self.peripheryAfterFirstLeverPressCoord[1])),
                                    radius=4, color=(255, 100, 0), thickness=4)
         
        ## Draw the bridge
        for i in range(3):
            frame = cv2.line(frame,
                        (int(self.bCoord[i,0]),int(self.bCoord[i,1])),
                        (int(self.bCoord[i+1,0]),int(self.bCoord[i+1,1])),
                            (200,200,200),1)
        frame = cv2.line(frame,
                        (int(self.bCoord[3,0]),int(self.bCoord[3,1])),
                        (int(self.bCoord[0,0]),int(self.bCoord[0,1])),
                            (200,200,200),1)
        
        ## Draw the periphery
        frame = cv2.circle(frame,
                                (int(self.aCoord[0]),int(self.aCoord[1])),
                                radius=int(self.radiusPeriphery), color=(50, 50, 50), thickness=1)
                
        ## Draw the lever, 
        for i in range(len(self.lever.points[:,0])-1):
            frame = cv2.line(frame,
                            (int(self.lever.points[i,0]),int(self.lever.points[i,1])),
                            (int(self.lever.points[i+1,0]),int(self.lever.points[i+1,1])),
                            (200,200,200),1)
        # final lever segment
        frame = cv2.line(frame,
                            (int(self.lever.points[len(self.lever.points[:,0])-1,0]),int(self.lever.points[len(self.lever.points[:,0])-1,1])),
                            (int(self.lever.points[0,0]),int(self.lever.points[0,1])),
                            (200,200,200),1)
        
        ## Draw the lever area
        for i in range(len(self.lever.zonePoints[:,0])-1):
            frame = cv2.line(frame,
                            (int(self.lever.zonePoints[i,0]),int(self.lever.zonePoints[i,1])),
                            (int(self.lever.zonePoints[i+1,0]),int(self.lever.zonePoints[i+1,1])),
                            (200,200,200),2)
        # final lever segment
        frame = cv2.line(frame,
                            (int(self.lever.zonePoints[len(self.lever.zonePoints[:,0])-1,0]),int(self.lever.zonePoints[len(self.lever.zonePoints[:,0])-1,1])),
                            (int(self.lever.zonePoints[0,0]),int(self.lever.zonePoints[0,1])),
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

        
    
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    
