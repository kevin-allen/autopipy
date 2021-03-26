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
    
    This is used to analyze all the different journeys of a trial while keeping the trial class relatively simple
    
    Work with data that are in cm, if you want the output in cm!
     
     
    Attributes:
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
                 pathDF,
                  leverPress,
                 stateDF):
        
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
        self.trialML = trialML
        self.pathDF = pathDF
        self.leverPress = leverPress
        self.stateDF = stateDF
       
        print("Creating jouney {}".format(self.name))
       
        self.bCoordMiddle = self.bCoord[0] + (self.bCoord[2]-self.bCoord[0])/2
        self.radiusPeriphery = self.aCoord[2]*self.arenaRadiusProportionToPeri
        
        
        
        #check if the mouse found the lever and press it for each journey    
        self.atLever = any(self.stateDF.loc[self.startIndex : self.endIndex,"loca"]=="lever")
        self.nLeverPresses = np.sum((self.leverPress.videoIndex > self.startIndex) & (self.leverPress.videoIndex < self.endIndex))
        self.leverPressed = self.nLeverPresses>0
        
        #print("atLever: {}, nLeverPressed: {}, leverPressed: {}".format(self.atLever, self.nLeverPresses, self.leverPressed))
        
        # we create a dictionary of navPath
        # what is in the dictionary will depend on whether or not the lever was pressed
        