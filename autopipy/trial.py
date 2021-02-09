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
        
    
    def extractTrialFeatures(self,log,mLPosi,videoLog,aCoord,bCoord):
        """
        Extract trial features 
        
        Arguments
            log: DataFrame with event log of the session
            mLPosi: DataFrame with mouse and lever position for every video frame
            videoLog: time for each video frame
            aCoord: arena coordinates (x, y, radius)
            bCoord: bridge coordinates (four points)
            
        """
        self.duration = self.endTime-self.startTime
        
        # get the start and end video indices
        self.trialVideoLog = videoLog[(videoLog.time > self.startTime) & (videoLog.time < self.endTime)]
        self.startVideoIndex = self.trialVideoLog.frame_number.head(1).squeeze()
        self.endVideoIndex = self.trialVideoLog.frame_number.tail(1).squeeze()
        
        # get the within trial time for each video frame
        self.trialVideoLog["timeWS"]= self.trialVideoLog.time-self.startTime
        
        # get the mouse and lever tracking data for the trial
        self.trialML = mLPosi[(videoLog.time > self.startTime) & (videoLog.time < self.endTime)]       
        
        # get lever presses
        lever = log[log.event=="lever_press"]
        index = (lever.time>self.startTime) & (lever.time<self.endTime)
        leverPress = lever.time[index]
        leverPressVideoIndex = leverPress.apply(self.videoIndexFromTimeStamp)
        self.leverPress = pd.DataFrame({"time":leverPress,
                                         "videoIndex":leverPressVideoIndex})

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
  
        out = cv2.VideoWriter(pathVideoFileOut, cv2.VideoWriter_fourcc(*'MJPG'), fps, (inWidth,inHeight))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.startVideoIndex)
        
        print("video:{}".format(pathVideoFileOut))
        print("from {} to {}".format(self.startVideoIndex,self.endVideoIndex))
        count = 0
        for i in range(self.startVideoIndex,self.endVideoIndex+1):
            ret, frame = cap.read()
        
            self.decorateVideoFrame(frame,i,count)
            
            out.write(frame)
            count=count+1
    
        out.release() 
        cap.release() 
    
    
    def decorateVideoFrame(self,frame,index,count):
        
        # add the trial time
        frame = cv2.putText(frame, 
                            "Time: {:.2f} sec".format(self.trialVideoLog.timeWS.iloc[count]), 
                            (20,20), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100,200,0), 1, cv2.LINE_AA)
        
        # add mouse position and orientation
        # mouse orientaiton line
        if ~np.isnan(self.trialML.loc[index,"mouseX"]):
            frame = cv2.line(frame,
                             (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                             (int(self.trialML.loc[index,"mouseX"]+self.trialML.loc[index,"mouseXHeading"]),
                              int(self.trialML.loc[index,"mouseY"]+self.trialML.loc[index,"mouseYHeading"])),
                            (0,200,255),2)
            # mouse position dot
            frame = cv2.circle(frame,
                               (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                    radius=4, color=(0, 200, 255), thickness=1)
        
        
        
        # add mouse position and orientation
        # lever orientaiton line
        if ~np.isnan(self.trialML.loc[index,"leverX"]) :
            frame = cv2.line(frame,
                             (int(self.trialML.loc[index,"leverX"]),int(self.trialML.loc[index,"leverY"])),
                             (int(self.trialML.loc[index,"leverX"]+self.trialML.loc[index,"leverXHeading"]*2),
                              int(self.trialML.loc[index,"leverY"]+self.trialML.loc[index,"leverYHeading"]*2)),
                            (0,0,255),2)
            # lever position dot
            frame = cv2.circle(frame,
                               (int(self.trialML.loc[index,"leverX"]),int(self.trialML.loc[index,"leverY"])),
                                radius=4, color=(0, 0, 255), thickness=1)
        
        
        # add lever presses as red dots
        if (self.leverPress.videoIndex==index).sum() == 1:
             frame = cv2.circle(frame,
                                (int(self.trialML.loc[index,"mouseX"]),int(self.trialML.loc[index,"mouseY"])),
                                radius=4, color=(0, 255, 0), thickness=3)
        
        
        ## Draw the path up to now 
        
        ## Draw search and homing paths with different color
        
        
        for i in range(count):
            if ~np.isnan(self.trialML.leverX.iloc[i]) :
                frame = cv2.circle(frame,
                                  (int(self.trialML.leverX.iloc[i]),int(self.trialML.leverY.iloc[i])),
                                   radius=2, color=(50, 50, 50), thickness=1)
        
            
        ## Draw the bridge
        
        
        ## Draw the periphery
        
        
        
        
        
        
    
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
    