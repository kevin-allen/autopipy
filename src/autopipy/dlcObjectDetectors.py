"""
This file contains the object detectors that are based on dlc models

These classes inherits from the dlc class, so they can easily do inference, label video, etc.
"""

from dlc import dlc
import os.path
import numpy as np
import pandas as pd
import cv2


class leverDetector(dlc):
    """
    Class to implement the detection of the lever using a deeplabcut model
    
    Assumes that the 3 parts of the lever are detected
        lever
        lever_right
        lever_left
    
    Methods
        positionOrientationFromFile() calculate the position and orientation for all the data in the .h5 file
        positionOrientationOneFrame() calculate the position and orientation for one frame
        
    """
    def __init__(self, pathConfigFile):
        super().__init__(pathConfigFile)
        
        self.requiredBodyParts = ['lever', 'lever_right', 'lever_left']
        if not self.bodyParts == self.requiredBodyParts :
            print("The body parts are not equal to"+self.requiredBodyParts)
            return False
        
    def positionOientationFromFile(self,pathVideoFile):
        """
        Calculate the position and orientation of the lever
        
        Output is stored in self.posiOri
        """
        
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
        
        self.pathVideoFile=pathVideoFile
        
        # get the tracking data from file
        self.loadPositionData(self.pathVideoFile) # load from file to self.out
        
        self.posiOri = np.apply_along_axis(self.positionOrientationOneFrame, 1, self.out)
        
    def labelVideoLever(self,pathVideoFile,pathOutputFile):
        """
        Function to create a label video specific to lever detection
        
        There will be an arrow specifying the direction of the lever and a dot for its center
        
        Arguments
            pathVideoFile: video to get the frames from
            pathOutputFile: labeled video to create
        """
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
        self.pathVideoFile = pathVideoFile
        
        # load data from file and get lever center and orientation
        self.positionOientationFromFile(self.pathVideoFile) # see self.out and self.posiOri
        
        # pathOutputFile
        
        cap = cv2.VideoCapture(self.pathVideoFile)
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
    
    
        # read one frame to get the size
        ret, frame = cap.read()
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        frameRate = cap.get(cv2.CAP_PROP_FPS) # float
        

        # move back so that the user can see this frame
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)-1
        cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame)
    
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(pathOutputFile, fourcc , frameRate, (int(width),int(height)))
      
        ## loop through the video
        index = 0
        numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        numOut = self.out.shape[0]
        if numFrames != numOut:
            print("Number of tracking points not equal to number of frames in video")
            return False
        
        while(cap.isOpened() and index < numFrames ):
            ret, frame = cap.read() 
            self.detectionImage(frame,self.out[index,],self.posiOri[index,]) # will modify frame
            out.write(frame)   
            index=index+1
        out.release()
        cap.release()
      
    
    
    def detectionImage(self,frame,oneOut,onePosiOri):
        """
        Takes an image and position data and draw the detection on the image
        
        Arguments
            frame An opencCV image
            oneOut numpy array with the tracking data of one frame
            onePosiOri numpy array with the position of the lever center, the lever orientation, and vector from center to lever press
        """
        
        ## open cv drawings
        if np.isnan(onePosiOri[0]):
            frame = cv2.putText(frame, "No lever detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                                             1, (100,200,0), 2, cv2.LINE_AA) 
        else :

            frame = cv2.line(frame,(int(onePosiOri[0]),int(onePosiOri[1])),(int(onePosiOri[0]+onePosiOri[3]*2),int(onePosiOri[1]+onePosiOri[4]*2)),(255,200,0),5)
            frame = cv2.circle(frame, (int(onePosiOri[0]),int(onePosiOri[1])), radius=4, color=(0, 255, 0), thickness=-1)
            
            frame = cv2.circle(frame, (int(oneOut[0]),int(oneOut[1])), radius=4, color=(0, 255, 255), thickness=-1)
            frame = cv2.circle(frame, (int(oneOut[3]),int(oneOut[4])), radius=4, color=(0, 255, 255), thickness=-1)
            frame = cv2.circle(frame, (int(oneOut[6]),int(oneOut[7])), radius=4, color=(0, 255, 255), thickness=-1)
            
                   
            frame = cv2.putText(frame, "{:.1f}".format(onePosiOri[2]), (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                                                 1, (100,200,0), 2, cv2.LINE_AA) 
            frame = cv2.putText(frame, "{:.1f} {:.1f}".format(onePosiOri[0],onePosiOri[1]), (50,100), cv2.FONT_HERSHEY_SIMPLEX ,  
                                                 1, (100,200,0), 2, cv2.LINE_AA)
           
        
    def positionOrientationOneFrame(self,frameTrackingData,probThreshold = 0.5):
        """
        The data are lever (x,y,prob), lever_right (x,y,prob), lever_left (x,y,prob)
        
        The 0,0 coordinates are top-left
        If the probability is below the threshold, return None
    
        Arguments
            frameTrackingData 1D numpy array containing the tracking values for one frame
        Return
            List containing x, y , theta_deg, X, Y
                x and y are the middle point of the lever box
                theta_deg is the angle of the lever relative to East. North is 90 degrees
                X and Y are the vector from the x and y to the lever (part that is pressed by the animal)
        """
    
        if frameTrackingData[2] < probThreshold or frameTrackingData[5] < probThreshold or frameTrackingData[8] < probThreshold:
            return [np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]
    
        ## middle point at the back of the lever (it does not matter if the two points are swapped, which is good)
        P4x = (frameTrackingData[3]+frameTrackingData[6])/2
        P4y = (frameTrackingData[4]+frameTrackingData[7])/2

        ## lever position (middle point)
        x= (frameTrackingData[0]+P4x)/2
        y= (frameTrackingData[1]+P4y)/2
    
    
        ## vector from middle point to tip of lever
        X=frameTrackingData[0]-x
        Y=frameTrackingData[1]-y
        v=np.array([X,Y])
        ## make it a unitary vector, get the length
        l = np.sqrt(np.dot(v,v))
        u = v/l
        U = np.array([1,0]) # ref vector
        ## get the angle between ref vector and our unitory vector
        theta = np.arccos(np.dot(u,U))
        if u[1]>0:
            theta = 2*np.pi-theta
        theta_deg = theta/(2*np.pi)*360

        return [x,y,theta_deg,X,Y]
