import os.path
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import cv2
import sys

class trial:
    """
    Class containing information about an autopi trial
    
    Attributes:
        name: Name of the trial, usually session_trialNo
        sessionName: Name of the session in which the trial was performed
        trialNo: Trial number within the session
        startTime: start time of the trial
        endTime: end time of the trial
    
    Methods:
        checkSessionDirectory():
    """
    def __init__(self,sessionName,trialNo,startTime,endTime,startTimeWS,endTimeWS):
        self.sessionName = sessionName
        self.trialNo = trialNo
        self.startTime = startTime
        self.endTime = endTime
        self.startTimeWS = startTimeWS
        self.endTimeWS = endTimeWS
        self.name = "{}_{}".format(sessionName,trialNo)
        
    
    def extractTrialFeatures(self,log,mLPosi,videoLog,aCoord,bCoord,
                             arenaRadiusProportionToPeri=0.925,
                             leverProximityRadiusProportion=1.4):
        """
        Extract trial features 
        
        Arguments
            log: DataFrame with event log of the session
            mLPosi: DataFrame with mouse and lever position for every video frame
            videoLog: time for each video frame
            aCoord: arena coordinates (x, y, radius)
            bCoord: bridge coordinates (four points)
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
        
        ################################################
        # define arena periphery and lever proximity  ##
        ################################################
        # radius from the arena center that defines the periphery of the arena
        self.radiusPeriphery = aCoord[2]*arenaRadiusProportionToPeri
        # radius from the lever center that is defined as being at the lever
        self.radiusLeverProximity = np.nanmedian(np.sqrt( (self.trialML.leverX-self.trialML.leverPressX)**2 +
        (self.trialML.leverY-self.trialML.leverPressY)**2))*leverProximityRadiusProportion
        
        #########################################
        # variables that evolve along the path ##
        #########################################
        # we will store all these Series in a DataFrame called pathDF
        distance = np.sqrt(np.diff(self.trialML.mouseX,prepend=np.NAN)**2+
                                np.diff(self.trialML.mouseY,prepend=np.NAN)**2)
        traveledDistance = np.nancumsum(distance)
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
                       0-np.diff(self.trialML.mouseY,prepend=np.NAN)),axis=1) #0- so that north is 90 degrees
        mvHeading = self.vectorAngle(mv,degrees=True,quadrant=True)
        ## vector from mouse to bridge
        mBVX = self.bCoordMiddle[0] - self.trialML.mouseX.to_numpy()
        mBVY = self.bCoordMiddle[1] - self.trialML.mouseY.to_numpy()
        mouseToBridge = np.stack((self.bCoordMiddle[0] - self.trialML.mouseX.to_numpy() ,
                                  0-(self.bCoordMiddle[1] - self.trialML.mouseY.to_numpy())),axis = 1) #0- so that north is 90 degrees
        ## mouse to bridge angle relative to 1,0
        mouseToBridgeAngle = self.vectorAngle(mouseToBridge,degrees=True)
        ## angle between movement heading and vector from the mouse to the bridge
        mvHeadingToBridgeAngle = self.vectorAngle(mv,mouseToBridge,degrees=True)
        ## angle between head direction and vector from the mouse to the bridge
        hdv = np.stack((self.trialML.mouseXHeading.to_numpy(),
                       0-self.trialML.mouseYHeading.to_numpy()),axis = 1) # 0- so that north is 90
        hdToBridgeAngle = self.vectorAngle(hdv,mouseToBridge,degrees=True)
        # Store these Series into a data frame
        # use the same index as the self.trialML DataFrame
        self.pathDF = pd.DataFrame({"distance" :distance,
                                  "traveledDistance" : traveledDistance,
                                  "mvHeading" : mvHeading,
                                  "mouseToBridgeX": mBVX,
                                  "mouseToBridgeY": mBVY,
                                  "mouseToBridgeAngle" : mouseToBridgeAngle,
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
        arenaToBridgeVector[1]=0-arenaToBridgeVector[1] # 90 is north
        self.arenaToBridgeAngle = self.vectorAngle(np.expand_dims(arenaToBridgeVector,0),degrees=True)
        ## angle from mouse when reaching periphery relative to arena center
        arenaToMousePeriVector = self.peripheryAfterFirstLeverPressCoord-self.aCoord[:2]
        arenaToMousePeriVector[1]=0-arenaToMousePeriVector[1] # 90 is north
        self.arenaToMousePeriAngle = self.vectorAngle(np.expand_dims(arenaToMousePeriVector,0),degrees=True)
        ## angular deviation of the mouse when reaching periphery
        self.periArenaCenterBridgeAngle=self.vectorAngle(v = np.expand_dims(arenaToMousePeriVector,0),
                                                         rv = np.expand_dims(arenaToBridgeVector,0),
                                                         degrees=True)
        
        #################################################
        ## sectioning the trial into different states  ##
        #################################################
        # define each frame as arena, bridge, lever or home base (NAN), one-hot encoding
        self.stateDF=pd.DataFrame({"lever": self.pathDF.distanceFromLever<self.radiusLeverProximity,
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
            ## searchLast, from first step on the arena after the last bridge to lever pressing
            bridgeIndex = self.stateDF[self.stateDF.loca=="bridge"].index
            if len(bridgeIndex[(bridgeIndex.values < self.leverPress.videoIndex.iloc[0])])==0:
                print("no bridge before lever press in trial {}".format(self.trialNo))
                print("This situation could be caused by video synchronization problems")
                lastBridgeIndexBeforePress=self.startVideoIndex
            else :
                lastBridgeIndexBeforePress = (bridgeIndex[(bridgeIndex.values < self.leverPress.videoIndex.iloc[0])])[-1]
            self.searchLastStartIndex = lastBridgeIndexBeforePress
            self.searchLastEndIndex = self.leverPress.videoIndex.iloc[0]
            ## searchLastNoLever, seachLast, excluding time at lever before pressing
            leverIndex = self.stateDF[self.stateDF.loca=="lever"].index
            self.searchLastNoLeverStartIndex = self.searchLastStartIndex
            self.searchLastNoLeverEndIndex = (leverIndex[(leverIndex.values >lastBridgeIndexBeforePress) &
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
        maskSearchLast = np.full((inWidth, inHeight), 0, dtype=np.uint8) 
        maskSearchLastNoLever = np.full((inWidth, inHeight), 0, dtype=np.uint8)
        maskHomingTotal = np.full((inWidth, inHeight), 0, dtype=np.uint8)
        maskHomingPeri = np.full((inWidth, inHeight), 0, dtype=np.uint8)
        maskHomingPeriNoLever = np.full((inWidth, inHeight), 0, dtype=np.uint8)
        
        # We will combine to their respective mask, then we will add them to get the right color mixture
        searchTotalBG = np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        searchLastBG =  np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        searchLastNoLeverBG =  np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        searchTotalBG[:,:,0] = 150 # these values for search paths should not go over 255 on one channel
        searchLastBG[:,:,0] = 105
        searchLastNoLeverBG[:,:,1] = 150
        
        homingTotalBG = np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        homingPeriBG = np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        homingPeriNoLeverBG = np.full((inWidth,inHeight,3),0,dtype=np.uint8)
        homingTotalBG[:,:,2] = 150
        homingPeriBG[:,:,2] = 105
        homingPeriNoLeverBG[:,:,1] = 150
        
        maskDict = {"mask" : mask,
                    "maskSearchTotal" : maskSearchTotal,
                    "maskSearchLast" : maskSearchLast,
                    "maskSearchLastNoLever": maskSearchLastNoLever,
                    "maskHomingTotal": maskHomingTotal,
                    "maskHomingPeri": maskHomingPeri,
                    "maskHomingPeriNoLever" :maskHomingPeriNoLever,
                    "searchTotalBG" : searchTotalBG,
                    "searchLastBG" : searchLastBG,
                    "searchLastNoLeverBG" : searchLastNoLeverBG,
                    "homingTotalBG": homingTotalBG,
                    "homingPeriBG" : homingPeriBG,
                    "homingPeriNoLeverBG" : homingPeriNoLeverBG}
        
        out = cv2.VideoWriter(pathVideoFileOut, cv2.VideoWriter_fourcc(*'MJPG'), fps, (inWidth,inHeight))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.startVideoIndex)
        
        print("Trial {}, from {} to {}, {} frames".format(self.trialNo,self.startVideoIndex,self.endVideoIndex, self.endVideoIndex-self.startVideoIndex))
        count = 0
        for i in range(self.startVideoIndex,self.endVideoIndex+1):
            ret, frame = cap.read()
        
            frame = self.decorateVideoFrame(frame,i,count,maskDict)
            
            out.write(frame)
            count=count+1
    
        out.release() 
        cap.release() 
    
    
    def decorateVideoFrame(self,frame,index,count,maskDict):
        
        # trial time
        frame = cv2.putText(frame, 
                            "Time: {:.2f} sec".format(self.trialVideoLog.timeWS.iloc[count]), 
                            (20,20), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)
        # traveled distance
        frame = cv2.putText(frame, 
                            "Distance: {:.1f} pxs".format(self.pathDF.traveledDistance[index]), 
                            (20,50), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)
        # speed
        frame = cv2.putText(frame, 
                            "Speed: {:.0f} pxs/sec".format(self.pathDF.speedNoNAN[index]), 
                            (20,80), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)       
        # distance to lever
        frame = cv2.putText(frame, 
                            "Distance lever: {:.1f} pxs".format(self.pathDF.distanceFromLever[index]), 
                            (20,110), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)       
        # Angle between mouse head and the bridge
        frame = cv2.putText(frame, 
                            "Mv heading: {:.0f} deg".format(self.pathDF.mvHeading[index]), 
                            (20,140), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        
        
        # Angle between mouse head and the bridge
        frame = cv2.putText(frame, 
                            "MouseToBridge angle: {:.0f} deg".format(self.pathDF.mouseToBridgeAngle[index]), 
                            (20,170), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        # Angle between mouse movement heading and vector from mouse to the bridge
        frame = cv2.putText(frame, 
                            "MouseHeadingToBridge angle: {:.0f} deg".format(self.pathDF.mvHeadingToBridgeAngle[index]), 
                            (20,200), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        # Angle between mouse movement heading and vector from mouse to the bridge
        frame = cv2.putText(frame, 
                            "hdToBridge angle: {:.0f} deg".format(self.pathDF.hdToBridgeAngle[index]), 
                            (20,230), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA) 
        # Angle between mouse periphery after lever, arena center and bridge
        if index > self.peripheryAfterFirstLeverPressVideoIndex :
            frame = cv2.putText(frame, 
                                "peri error: {:.0f} deg".format(self.periArenaCenterBridgeAngle[0]), 
                                (20,260), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100,200,0), 1, cv2.LINE_AA) 
        
        # Location as a categorical variable
        frame = cv2.putText(frame, 
                            "Loca: {}".format(self.stateDF.loca[index]), 
                            (frame.shape[1]-200,20), 
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
            if index >= self.searchLastStartIndex and index <= self.searchLastEndIndex:
                maskDict["maskSearchLast"] = cv2.circle(maskDict["maskSearchLast"],
                                          (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                           radius=1, color=(255, 255, 255), thickness=1)
            if index >= self.searchLastNoLeverStartIndex and index <= self.searchLastNoLeverEndIndex:
                maskDict["maskSearchLastNoLever"] = cv2.circle(maskDict["maskSearchLastNoLever"],
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
        
        searchLastPath = cv2.bitwise_or(maskDict["searchLastBG"],maskDict["searchLastBG"],mask=maskDict["maskSearchLast"])
            
        searchTotalPath = cv2.bitwise_or(maskDict["searchTotalBG"],maskDict["searchTotalBG"],mask=maskDict["maskSearchTotal"])
      
        searchLastNoLever = cv2.bitwise_or(maskDict["searchLastNoLeverBG"],maskDict["searchLastNoLeverBG"],mask=maskDict["maskSearchLastNoLever"])
        homingTotalPath = cv2.bitwise_or(maskDict["homingTotalBG"],maskDict["homingTotalBG"],mask=maskDict["maskHomingTotal"])
        homingPeriPath = cv2.bitwise_or(maskDict["homingPeriBG"],maskDict["homingPeriBG"],mask=maskDict["maskHomingPeri"])
        homingPeriNoLeverPath = cv2.bitwise_or(maskDict["homingPeriNoLeverBG"],maskDict["homingPeriNoLeverBG"],mask=maskDict["maskHomingPeriNoLever"])
        
        # apply the path mask to the main frame to zero the pixels in the path
        frame = cv2.bitwise_or(frame, frame, mask=maskDict["mask"])
        
        # combine the different colors to get the search paths
        frame = frame + searchTotalPath + searchLastPath + searchLastNoLever
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
        
        ## Draw the lever zone, if there is a lever
        if ~ (np.isnan(self.trialML.loc[index,"leverX"]) or np.isnan(self.trialML.loc[index,"leverX"])) :
            frame = cv2.circle(frame,
                               (int(self.trialML.loc[index,"leverX"]),int(self.trialML.loc[index,"leverY"])),
                               radius=int(self.radiusLeverProximity), color=(50, 50, 50), thickness=1)
        
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

          
        
        
    
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    
    
    
          
#         def previousLight(x,lightEvents):
#             """
#             Get the last light that was set before the current trial
#             """
#             if sum(lightEvents.time < x) == 0 :
#                 return np.nan
#             else:
#                 return lightEvents.param[lightEvents.time< x].tail(1).to_numpy()[0]
#         def lightFromCode(x):
#             """
#             Get light or dark depending on light_code
#             """
#             if x == 1 or np.isnan(x):
#                 return "light"
#             else:
#                 return "dark"
    
#         lightEvents = self.log[self.log.event=="light"]
#         lightCode = trials.start_time.apply(previousLight,args=(lightEvents,))
#         trials["light"] = lightCode.apply(lightFromCode)
                  
                     
                  
#         def leverMissing(tr,leverEvents):
#             """
#             Check if there is a lever press during the trials

#             Return a boolean, True if missing
#             """  
#             if np.logical_and(leverEvents.time > tr["start_time"],
#                               leverEvents.time < tr["end_time"]).any():
#                 return False
#             else :
#                 return True

#         leverEvents = self.log[np.logical_or(self.log.event=="lever_press", self.log.event=="leverPress")]
        
#         missTrials = trials.apply(func = leverMissing,axis=1,args = (leverEvents,))
#         if missTrials.any():
#             print("There was(were) {} trial(s) without lever press".format(missTrials.sum()))
#         trials["lever_pressed"] = ~missTrials # reverse True to False
    