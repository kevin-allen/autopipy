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
        
    def cutAtLastArenaBridgeTransition(self):
        """
        Method to adjust the end of the journey to the time point at which the animal transitioned from the arena to the bridge.
        
        """
        
        lastArena=self.positionZones[self.positionZones.loca=="arena"].iloc[-1:].index.values[0]
        endingDf = self.positionZones.loc[lastArena:,:]
        journeyAtBridgeEndIndex = endingDf.loc[endingDf.loca=="bridge"].index.values[0]
       
        print("initial end index:", self.endIndex, " new end index:", journeyAtBridgeEndIndex)
        self.endIndex=journeyAtBridgeEndIndex
        self.mousePose = self.mousePose.loc[:self.endIndex,:]
        self.endTime = self.mousePose.time.iloc[-1]
        self.duration = self.endTime-self.startTime
        self.positionZones = self.positionZones.loc[self.startIndex:self.endIndex]
        
        
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    
    def pathFigure(self, legend = True, figSize=(10,10), zones=True, filePath=None, positionZones = False):
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
        axes.set_title("{}, {:.1f} sec".format(self.name,self.endTime-self.startTime))
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


        ## points on the start and end of the jouney
        
        axes.scatter(self.mousePose.x.iloc[0],self.mousePose.y.iloc[0],c="green",label="Start")
        axes.scatter(self.mousePose.x.iloc[-1],self.mousePose.y.iloc[-1],c="blue",label="End")
        

        ## lever
        axes.plot(self.lever.pointsPlot[:,0],self.lever.pointsPlot[:,1], color = "gray")
        axes.plot(self.lever.enterZonePointsPlot[:,0],self.lever.enterZonePointsPlot[:,1], color = "gray",linestyle="dotted")
        axes.plot(self.lever.exitZonePointsPlot[:,0],self.lever.exitZonePointsPlot[:,1], color = "gray",linestyle="dotted")

        ## position of the mouse when pressing the lever
        if self.nLeverPresses != 0:
            axes.scatter(self.leverPresses.mouseX,self.leverPresses.mouseY,c="red",label="Lever press")

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
