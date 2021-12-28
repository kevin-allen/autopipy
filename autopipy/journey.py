import os.path
import pandas as pd
import numpy as np
from scipy import stats
import cv2
import sys
from autopipy.navPath import NavPath
from autopipy.lever import Lever
import matplotlib.pyplot as plt

class Journey:
    """
    Class containing information about a single journey on the Autopi task.
    
    This is used to analyze all journeys of a trial while keeping the trial class relatively simple
    
    Work with data that are in cm, if you want the output in cm!
     
    It contains a dictionary of NavPath objects (pathD). 
    
    We develop this class because we might want to consider the mouse behavior during all journeys, not just during the journeys when the lever was pressed.
    
    Attributes:There
        name: Name of the trial, usually sessionName_trialNo-JourneyNo
        sessionName: Name of the session in which the trial was performed
        trialNo: Trial number within the session
        jouneyNo: Jouney number within the trial
        
        ...
    
    Methods:
          
    """
    def __init__(self,sessionName,trialNo,journeyNo,
                 startIndex,endIndex,
                 lever, 
                 arenaRadius, arenaRadiusProportionToPeri, aCoord,
                 bCoord,
                 trialML,
                 trialVideoLog,
                 pathDF,
                 leverPress,
                 stateDF,
                 leverPosition):
        
        self.sessionName = sessionName
        self.trialNo = trialNo
        self.journeyNo = journeyNo
        self.name="{}_{}-{}".format(self.sessionName,self.trialNo,self.journeyNo)
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.lever = lever
        self.arenaRadiusCm = arenaRadius
        self.arenaRadiusProportionToPeri = arenaRadiusProportionToPeri
        self.aCoord = aCoord
        self.bCoord = bCoord
        self.trialML = trialML.loc[self.startIndex : self.endIndex,:] # only keep what we need
        self.trialVideoLog = trialVideoLog.loc[self.startIndex : self.endIndex,:]
        self.pathDF = pathDF.loc[self.startIndex : self.endIndex,:] # only keep what we need
        self.leverPress = leverPress[(leverPress.videoIndex>self.startIndex) & (leverPress.videoIndex <self.endIndex)] # only keep what we need
        self.stateDF = stateDF.loc[self.startIndex : self.endIndex,:] # only keep what we need
        self.leverPosition = leverPosition
        self.pathD = {}
        #print("lengths, trialML: {}, trialVideoLog: {}, pathDF: {}, leverPress: {}, stateDF: {}".format(len(self.trialML),len(self.trialVideoLog),len(self.pathDF),len(self.leverPress),len(self.stateDF)))
        #print("Creating jouney {}".format(self.name))
        self.bCoordMiddle = self.bCoord[0] + (self.bCoord[2]-self.bCoord[0])/2
        self.radiusPeriphery = self.aCoord[2]*self.arenaRadiusProportionToPeri
        
        
        
        #check if the mouse found the lever and press it for each journey    
        self.atLever = any(self.stateDF["loca"]=="lever")
        self.nLeverPresses = len(self.leverPress)
        self.leverPressed = self.nLeverPresses>0
        
        #print("name {}, atLever: {}, nLeverPressed: {}, leverPressed: {}".format(self.name, self.atLever, self.nLeverPresses, self.leverPressed))
        
       
        ### we now get the start and end indices to create the NavPath objects
        
        if self.leverPressed:
            # we need several NavPath objects to analyze homing

            #################################################
            ## reaching periphery after first lever press  ##
            #################################################
            self.peripheryAfterFirstLeverPressVideoIndex = startIndex
            for i in range(self.leverPress.videoIndex.iloc[0],
                           self.endIndex) :
                if (self.pathDF.distanceFromArenaCenter[i] > self.radiusPeriphery):
                    self.peripheryAfterFirstLeverPressVideoIndex = i
                    break
   
                
            ###################
            ## search paths ###
            ###################
            
            ## searchTotal, from first step on the arena to lever pressing, excluding bridge time
            self.searchTotalStartIndex = self.stateDF.loca.index[self.stateDF.arena==True][0]
            self.searchTotalEndIndex = self.leverPress.videoIndex.iloc[0]        
            
            ## searchArena, from first step on the arena after the last bridge to lever pressing
            bridgeIndex = self.stateDF[self.stateDF.loca=="bridge"].index
            if len(bridgeIndex[(bridgeIndex.values < self.leverPress.videoIndex.iloc[0])])==0:
                #print("{}, no bridge before lever press".format(self.name))
                #print("This situation could be caused by video synchronization problems or mouse not being visible when on the bridge")
                lastBridgeIndexBeforePress=self.startIndex
            else :
                lastBridgeIndexBeforePress = (bridgeIndex[(bridgeIndex.values < self.leverPress.videoIndex.iloc[0])])[-1]
            self.searchArenaStartIndex = lastBridgeIndexBeforePress
            self.searchArenaEndIndex = self.leverPress.videoIndex.iloc[0]
            
            ## searchArenaNoLever, seachLast, excluding time at lever before pressing
            leverIndex = self.stateDF[self.stateDF.loca=="lever"].index
            self.searchArenaNoLeverStartIndex = self.searchArenaStartIndex
            ## in very rare cases, there is no lever zone before the lever press
            if np.sum( (leverIndex.values > lastBridgeIndexBeforePress) &  (leverIndex.values < self.leverPress.videoIndex.iloc[0])) == 0 :
                print("{}, no lever time between leaving the bridge and pressing the lever".format(self.name))
                print("{}, setting the end of the search path at the lever press".format(self.name))
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
                print("{}, no bridge after lever press".format(self.name))
                print("This situation could be caused by video synchronization problems or mouse not being visible when on the bridge")
                firstBridgeIndexAfterPress=self.endIndex
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
            allPose =  self.poseFromTrialData(self.startIndex,
                                                     self.endIndex)
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
            
            ## check the poses and report if the length is less than 2
            poseList=[allPose,
                      searchTotalPose, 
                      searchArenaPose, 
                       searchArenaNoLeverPose, 
                       homingTotalPose, 
                       homingPeriPose, 
                       homingPeriNoLeverPose]
            poseNameList=["allPose",
                          "searchTotalPose", 
                      "searchArenaPose", 
                       "searchArenaNoLeverPose", 
                       "homingTotalPose", 
                       "homingPeriPose", 
                       "homingPeriNoLeverPose"] 
            for a,b in zip(poseList,poseNameList):
                if (len(a) < 2) :
                    print("{}, {} path has a length < 2".format(self.name,b))
            
            # targets for the paths
            leverPose = self.poseFromLeverPositionDictionary()
            bridgePose = self.poseFromBridgeCoordinates()
            
            # get the path variables using the NavPath class
            self.pathD["all"] = NavPath(pPose = allPose,name = "all")
            self.pathD["searchTotal"] = NavPath(pPose = searchTotalPose,
                                              targetPose=leverPose,name = "searchTotal")
            self.pathD["searchArena"] =  NavPath(pPose = searchArenaPose,
                                              targetPose=leverPose, name = "searchArena")
            self.pathD["searchArenaNoLever"] =  NavPath(pPose = searchArenaNoLeverPose,
                                            targetPose=leverPose, name = "searchArenaNoLever")
            self.pathD["homingTotal"] = NavPath(pPose = homingTotalPose,
                                              targetPose=bridgePose, name = "homingTotal")
            self.pathD["homingPeri"] = NavPath(pPose = homingPeriPose,
                                              targetPose=bridgePose, name = "homingPeri")
            self.pathD["homingPeriNoLever"] = NavPath(pPose = homingPeriNoLeverPose,
                                              targetPose=bridgePose, name = "homingPeriNoLever")
            
        else :
            # we need a single NavPath object to know how long the trajectory was
            allPose =  self.poseFromTrialData(self.startIndex,
                                                     self.endIndex)
            self.pathD["all"] = NavPath(pPose = allPose,name = "all")
            ## create empty NavPath so that all journeys have the same content
            self.pathD["searchTotal"] = NavPath(pPose = np.empty((1,7)),name = "searchTotal")
            self.pathD["searchArena"] =  NavPath(pPose =  np.empty((1,7)), name = "searchArena")
            self.pathD["searchArenaNoLever"] =  NavPath(pPose =  np.empty((1,7)),name = "searchArenaNoLever")
            self.pathD["homingTotal"] = NavPath(pPose =  np.empty((1,7)), name = "homingTotal")
            self.pathD["homingPeri"] = NavPath(pPose =  np.empty((1,7)), name = "homingPeri")
            self.pathD["homingPeriNoLever"] = NavPath(pPose =  np.empty((1,7)), name = "homingPeriNoLever")
        
    
    def poseFromTrialData(self,startIndex,endIndex):
        """
        Create a numpy array containing the pose (x, y, z, yaw, pitch, roll, time) during a defined time period
        Arguments:
            startIndex: start index of the path, index are from the self.trialMLCm or self.trialVideoLog (they are the same)
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
    