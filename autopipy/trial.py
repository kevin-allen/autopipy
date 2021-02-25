import os.path
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from scipy import stats
import cv2
import sys
from autopipy.navPath import NavPath
from autopipy.lever import Lever

class Trial:
    """
    Class containing information about an autopi trial
    
    When calculating angles, the y coordinate is reversed (0-y) so that 90 degree is up.
    This is done because the y-axis is reversed in the videos.
    
    The position data that come into this class are in pixels, as the position data were extracted from a video.
    
    To be able to compare data from different sessions, we convert the position data from pixels to cm and 
    move the 0,0 origin to the center of the arena. This transformation is done when extracting trial features. 
    There are some variables ending by Px that contain data in pixels to be able to draw in video frames.
    The variables ending with Cm are in cm.
    When not explicitely specified, the values are in cm
    Note that the transformation between pixels to cm representation involve a translation to shift the origin to the center of the arena
    
    
     
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
        self.peripheryAfterFirstLeverPressCoordCm = None
        self.peripheryAfterFirstLeverPressCoordPx = None
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
        self.peripheryAfterFirstLeverPressVideoIndex = None
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
                       "travelledDistance" : [self.traveledDistance],
                       "anglularErrorHomingPeri" : self.periArenaCenterBridgeAngle,
                       # for searchTotal path 
                       "searchTotal_length" : [self.searchTotalNavPath.length],
                       "searchTotal_duration" : [self.searchTotalNavPath.duration],
                        "searchTotal_meanVectorLengthPosi" : [self.searchTotalNavPath.meanVectorLengthPosi],
                        "searchTotal_meanVectorDirectionPosi": [self.searchTotalNavPath.meanVectorDirectionPosi],
                        "searchTotal_meanVectorLengthOri" : [self.searchTotalNavPath.meanVectorLengthOri[0]],
                        "searchTotal_meanVectorDirectionOri" : [self.searchTotalNavPath.meanVectorDirectionOri[0]],
                        "searchTotal_meanSpeed" : [self.searchTotalNavPath.meanSpeed],
                        "searchTotal_medianMVDeviationToTarget" : [self.searchTotalNavPath.medianMVDeviationToTarget],
                        "searchTotal_medianHDDeviationToTarget" : [self.searchTotalNavPath.medianHDDeviationToTarget],
                      # for searchArena path 
                       "searchArena_length" : [self.searchArenaNavPath.length],
                       "searchArena_duration" : [self.searchArenaNavPath.duration],
                        "searchArena_meanVectorLengthPosi" : [self.searchArenaNavPath.meanVectorLengthPosi],
                        "searchArena_meanVectorDirectionPosi": [self.searchArenaNavPath.meanVectorDirectionPosi],
                        "searchArena_meanVectorLengthOri" : [self.searchArenaNavPath.meanVectorLengthOri[0]],
                        "searchArena_meanVectorDirectionOri" : [self.searchArenaNavPath.meanVectorDirectionOri[0]],
                        "searchArena_meanSpeed" : [self.searchArenaNavPath.meanSpeed],
                        "searchArena_medianMVDeviationToTarget" : [self.searchArenaNavPath.medianMVDeviationToTarget],
                        "searchArena_medianHDDeviationToTarget" : [self.searchArenaNavPath.medianHDDeviationToTarget],
                       # for searchArenaNoLever path
                       "searchArenaNoLever_length" : [self.searchArenaNoLeverNavPath.length],
                       "searchArenaNoLever_duration" : [self.searchArenaNoLeverNavPath.duration],
                        "searchArenaNoLever_meanVectorLengthPosi" : [self.searchArenaNoLeverNavPath.meanVectorLengthPosi],
                        "searchArenaNoLever_meanVectorDirectionPosi": [self.searchArenaNoLeverNavPath.meanVectorDirectionPosi],
                        "searchArenaNoLever_meanVectorLengthOri" : [self.searchArenaNoLeverNavPath.meanVectorLengthOri[0]],
                        "searchArenaNoLever_meanVectorDirectionOri" : [self.searchArenaNoLeverNavPath.meanVectorDirectionOri[0]],
                        "searchArenaNoLever_meanSpeed" : [self.searchArenaNoLeverNavPath.meanSpeed],
                        "searchArenaNoLever_medianMVDeviationToTarget" : [self.searchArenaNoLeverNavPath.medianMVDeviationToTarget],
                        "searchArenaNoLever_medianHDDeviationToTarget" : [self.searchArenaNoLeverNavPath.medianHDDeviationToTarget],
                       # for the homingTotal path 
                       "homingTotal_length" : [self.homingTotalNavPath.length],
                       "homingTotal_duration" : [self.homingTotalNavPath.duration],
                        "homingTotal_meanVectorLengthPosi" : [self.homingTotalNavPath.meanVectorLengthPosi],
                        "homingTotal_meanVectorDirectionPosi": [self.homingTotalNavPath.meanVectorDirectionPosi],
                        "homingTotal_meanVectorLengthOri" : [self.homingTotalNavPath.meanVectorLengthOri[0]],
                        "homingTotal_meanVectorDirectionOri" : [self.homingTotalNavPath.meanVectorDirectionOri[0]],
                        "homingTotal_meanSpeed" : [self.homingTotalNavPath.meanSpeed],
                        "homingTotal_medianMVDeviationToTarget" : [self.homingTotalNavPath.medianMVDeviationToTarget],
                        "homingTotal_medianHDDeviationToTarget" : [self.homingTotalNavPath.medianHDDeviationToTarget],
                       # for the homingPeri path
                       "homingPeri_length" : [self.homingPeriNavPath.length],
                       "homingPeri_duration" : [self.homingPeriNavPath.duration],
                        "homingPeri_meanVectorLengthPosi" : [self.homingPeriNavPath.meanVectorLengthPosi],
                        "homingPeri_meanVectorDirectionPosi": [self.homingPeriNavPath.meanVectorDirectionPosi],
                        "homingPeri_meanVectorLengthOri" : [self.homingPeriNavPath.meanVectorLengthOri[0]],
                        "homingPeri_meanVectorDirectionOri" : [self.homingPeriNavPath.meanVectorDirectionOri[0]],
                        "homingPeri_meanSpeed" : [self.homingPeriNavPath.meanSpeed],
                        "homingPeri_medianMVDeviationToTarget" : [self.homingPeriNavPath.medianMVDeviationToTarget],
                        "homingPeri_medianHDDeviationToTarget" : [self.homingPeriNavPath.medianHDDeviationToTarget],
                       # for the homingPeriNoLever
                       "homingPeriNoLever_length" : [self.homingPeriNoLeverNavPath.length],
                       "homingPeriNoLever_duration" : [self.homingPeriNoLeverNavPath.duration],
                        "homingPeriNoLever_meanVectorLengthPosi" : [self.homingPeriNoLeverNavPath.meanVectorLengthPosi],
                        "homingPeriNoLever_meanVectorDirectionPosi": [self.homingPeriNoLeverNavPath.meanVectorDirectionPosi],
                        "homingPeriNoLever_meanVectorLengthOri" : [self.homingPeriNoLeverNavPath.meanVectorLengthOri[0]],
                        "homingPeriNoLever_meanVectorDirectionOri" : [self.homingPeriNoLeverNavPath.meanVectorDirectionOri[0]],
                        "homingPeriNoLever_meanSpeed" : [self.homingPeriNoLeverNavPath.meanSpeed],
                        "homingPeriNoLever_medianMVDeviationToTarget" : [self.homingPeriNoLeverNavPath.medianMVDeviationToTarget],
                        "homingPeriNoLever_medianHDDeviationToTarget" : [self.homingPeriNoLeverNavPath.medianHDDeviationToTarget]} 
        return pd.DataFrame(self.myDict)
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

    
    
    def extractTrialFeatures(self,log,mLPosi,videoLog,aCoord,bCoord,
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
        
        
        ##############################################
        ### get data to transform from pixels to cm ##
        ##############################################
        self.arenaRadiusCm = arenaRadiusCm
        self.pxPerCm = aCoord[2]/self.arenaRadiusCm # to go from pixels to cm
        self.originPxToCm = aCoord[:2] # the 0,0 in the cm world     
        #print("A radius cm: {}, A radius px:{}, pxPerCm: {}, originPxToCm: {}".format(self.arenaRadiusCm,aCoord[2],self.pxPerCm, self.originPxToCm))
         
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
    

        ## test the synchronization at for the whole session as an extra precaution
        if len(videoLog.time) != len(mLPosi.mouseX):
            print("videoLog {} is not the same length as mLPosi {}".format(len(videoLog.time),len(mLPosi.mouseX)))
            print("self.valid set to False")
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
            print("self.valid set to False")
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
                    
            print("self.startTime was adjusted by {:2.2} s".format(back))
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
            print("self.valid set to False")
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
        leverPressVideoIndex = leverPressTime.apply(self.videoIndexFromTimeStamp) # video index
        self.leverPress = pd.DataFrame({"time": leverPressTime,
                                        "videoIndex":leverPressVideoIndex})
        self.nLeverPresses = len(self.leverPress.time)
        if self.nLeverPresses == 0:
            print("{}, no lever press".format(self.name))
            print("self.valid set to False")
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
        
        ####################################################################
        ### If the lever is not detected correctly, it is possible that ####
        ### the animal is never at the lever                            ####
        ### Or if the experimenter presses the lever for some reason    ####
        ### Set to invalid                                              ####
        ####################################################################
        if np.sum(self.stateDF.loca=="lever") == 0 :
            print("{}, no time in the lever zone".format(self.name))
            print("self.valid set to False")
            self.valid=False
        
        ###################################################################
        ### The mouse should be at the lever when the lever was pressed ###
        ### We consider only the first lever press                      ###
        ###################################################################
        if self.nLeverPresses > 0 :
            if np.sum(self.stateDF.loca.loc[self.leverPress.videoIndex.iloc[0]] == "lever") == 0 :
                print("{}, mouse not in the lever zone when the lever was pressed".format(self.name))
                print("self.valid set to False")
                self.valid=False
        
        ######################################################
        ### The mouse should be on the arena at some point ###
        ######################################################
        if np.sum(self.stateDF.arena) == 0 :
            print("{}, no time on the arena".format(self.name))
            print("self.valid set to False")
            self.valid=False
        
        
        ####################
        ## identify  paths #
        ####################
        if self.valid : # no lever or no valid mouse position,
            
            
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
            
            ###################
            ## search paths ###
            ###################
            ## searchTotal, from first step on the arena to lever pressing, excluding bridge time
            self.searchTotalStartIndex = self.stateDF.loca.index[self.stateDF.arena==True][0]
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
            ## in very rare cases, there is no lever zone before the lever press
            if np.sum( (leverIndex.values > lastBridgeIndexBeforePress) &  (leverIndex.values < self.leverPress.videoIndex.iloc[0])) == 0 :
                print("no lever time between leaving the bridge and pressing the lever")
                print("setting the end of the search path at the lever press")
                self.searchArenaNoLeverEndIndex = leverIndex[0]
            else :
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
            
        else : ## there was no lever press or valid mouse position
            self.periArenaCenterBridgeAngle = np.NAN
            # get the path variables using the NavPath class, these will be empty and return np.NAN
            self.searchTotalNavPath = NavPath(pPose = np.empty((1,7)),name = "searchTotal")
            self.searchArenaNavPath =  NavPath(pPose =  np.empty((1,7)), name = "searchArena")
            self.searchArenaNoLeverNavPath =  NavPath(pPose =  np.empty((1,7)),name = "searchArenaNoLever")
            self.homingTotalNavPath = NavPath(pPose =  np.empty((1,7)), name = "homingTotal")
            self.homingPeriNavPath = NavPath(pPose =  np.empty((1,7)), name = "homingPeri")
            self.homingPeriNoLeverNavPath = NavPath(pPose =  np.empty((1,7)), name = "homingPeriNoLever")
        
            
            
    def poseFromTrialData(self,startIndex,endIndex):
        """
        Create a numpy array containing the pose (x, y, z, yaw, pitch, roll, time) during a defined time period
        Arguments:
            startIndex: start index of the path, index are from the self.trialMLCm or self.trialVideoLog (they are the same)
            endIndex: end index of the path

        Return Pose as a numpy array with 7 columns

        """
        return np.stack((self.trialMLCm.loc[startIndex:endIndex,"mouseX"].to_numpy(),
                    self.trialMLCm.loc[startIndex:endIndex,"mouseY"].to_numpy(),
                    np.zeros_like(self.trialMLCm.loc[startIndex:endIndex,"mouseY"].to_numpy()),
                    self.trialMLCm.loc[startIndex:endIndex,"mouseOri"].to_numpy(),
                    np.zeros_like(self.trialMLCm.loc[startIndex:endIndex,"mouseY"].to_numpy()),
                    np.zeros_like(self.trialMLCm.loc[startIndex:endIndex,"mouseY"].to_numpy()),
                    self.trialVideoLog.timeWS.loc[startIndex:endIndex].to_numpy()),axis=1)
    def poseFromLeverPositionDictionary(self) :
        """
        Create a numpy array containing the pose (x, y, z, yaw, pitch, roll, time)  
        from the leverPosition dictionary.
        """
        return np.array([[self.leverPositionCm["leverX"],self.leverPositionCm["leverY"],0,
                          self.leverPositionCm["leverOri"],0,0,0]])

    def poseFromBridgeCoordinates(self) :
        """
        Create a numpy array containing the pose (x, y, z, yaw, pitch, roll, time)  
        from the middle of the bridge
        """
        return np.array([[self.bCoordMiddleCm[0],self.bCoordMiddleCm[1],0,0,0,0,0]])
    
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
            frame = self.decorateVideoFrame(frame,i,count,maskDict)
            
            out.write(frame)
            count=count+1
    
        out.release() 
        cap.release() 
    
    
    def decorateVideoFrame(self,frame,index,count,maskDict):
        """
        Function to add information to the trial video
        For example, we can plot the path and other varialbes
        
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
        frame = cv2.putText(frame, 
                            "Time: {:.1f} sec, {:.1f}".format(self.trialVideoLog.timeWS.iloc[count],self.startTimeWS + self.trialVideoLog.timeWS.iloc[count] ), 
                            (30,20), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)
        # traveled distance
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
        # distance to lever
        frame = cv2.putText(frame, 
                            "Distance lever center: {:.1f} cm".format(self.pathDF.distanceFromLever[index]), 
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
                            "HD: {:.0f} deg".format(self.trialMLPx.mouseOri[index]), 
                            (30,170), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        
        # mouse head to bridge vector
        frame = cv2.putText(frame, 
                            "toBridge: {:.0f} {:.0f}".format(self.pathDF.mouseToBridgeXCm[index],self.pathDF.mouseToBridgeYCm[index]), 
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
                if index >= self.searchTotalStartIndex and index <= self.searchTotalEndIndex:
                    maskDict["maskSearchTotal"] = cv2.circle(maskDict["maskSearchTotal"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)
                if index >= self.searchArenaStartIndex and index <= self.searchArenaEndIndex:
                    maskDict["masksearchArena"] = cv2.circle(maskDict["masksearchArena"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)
                if index >= self.searchArenaNoLeverStartIndex and index <= self.searchArenaNoLeverEndIndex:
                    maskDict["masksearchArenaNoLever"] = cv2.circle(maskDict["masksearchArenaNoLever"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)

                if index >= self.homingTotalStartIndex and index <= self.homingTotalEndIndex:
                    maskDict["maskHomingTotal"] = cv2.circle(maskDict["maskHomingTotal"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)
                if index >= self.homingPeriStartIndex and index <= self.homingPeriEndIndex:
                    maskDict["maskHomingPeri"] = cv2.circle(maskDict["maskHomingPeri"],
                                              (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                               radius=1, color=(255, 255, 255), thickness=1)
                if index >= self.homingPeriNoLeverStartIndex and index <= self.homingPeriNoLeverEndIndex:
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
            frame = cv2.line(frame,
                             (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                             (int(self.trialMLPx.loc[index,"mouseX"]+self.trialMLPx.loc[index,"mouseXHeading"]*2),
                              int(self.trialMLPx.loc[index,"mouseY"]+self.trialMLPx.loc[index,"mouseYHeading"]*2)),
                            (0,200,255),2)
            # mouse position dot
            frame = cv2.circle(frame,
                               (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                                    radius=4, color=(0, 200, 255), thickness=1)
            # head to bridge vector
            frame = cv2.line(frame,
                             (int(self.trialMLPx.loc[index,"mouseX"]),int(self.trialMLPx.loc[index,"mouseY"])),
                             (int(self.trialMLPx.loc[index,"mouseX"]+self.pathDF.loc[index,"mouseToBridgeXPx"]),
                              int(self.trialMLPx.loc[index,"mouseY"]+self.pathDF.loc[index,"mouseToBridgeYPx"])),
                            (100,255,255),2)
        
        
        #######################################
        # add lever position and orientation ##
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
                
        ## Draw the lever, 
        for i in range(len(self.leverPx.points[:,0])-1):
            frame = cv2.line(frame,
                            (int(self.leverPx.points[i,0]),int(self.leverPx.points[i,1])),
                            (int(self.leverPx.points[i+1,0]),int(self.leverPx.points[i+1,1])),
                            (200,200,200),1)
        # final lever segment
        frame = cv2.line(frame,
                            (int(self.leverPx.points[len(self.leverPx.points[:,0])-1,0]),int(self.leverPx.points[len(self.leverPx.points[:,0])-1,1])),
                            (int(self.leverPx.points[0,0]),int(self.leverPx.points[0,1])),
                            (200,200,200),1)
        
        ## Draw the lever area
        for i in range(len(self.leverPx.zonePoints[:,0])-1):
            frame = cv2.line(frame,
                            (int(self.leverPx.zonePoints[i,0]),int(self.leverPx.zonePoints[i,1])),
                            (int(self.leverPx.zonePoints[i+1,0]),int(self.leverPx.zonePoints[i+1,1])),
                            (200,200,200),2)
        # final lever segment
        frame = cv2.line(frame,
                            (int(self.leverPx.zonePoints[len(self.leverPx.zonePoints[:,0])-1,0]),int(self.leverPx.zonePoints[len(self.leverPx.zonePoints[:,0])-1,1])),
                            (int(self.leverPx.zonePoints[0,0]),int(self.leverPx.zonePoints[0,1])),
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
    
    
