"""
This file contains the object detectors that are based on dlc models

These classes inherits from the dlc class, so they can easily do inference, label video, etc.
"""

from autopipy.dlc import dlc
import os.path
import numpy as np
from scipy.stats import mode  
import pandas as pd
import cv2
import glob
import sys

class leverDetector(dlc):
    """
    Class to implement the detection of the lever using a deeplabcut model
    This class inherits from dlc, but has additional functions that are specific to the lever.
    For example, it can calculate the position and orientation of the lever and create label video with this.
    
    It assumes that the 3 parts of the lever are detected by deeplabcut
        lever
        lever_right
        lever_left
    
    Methods
        positionOrientationFromFile() calculate the position and orientation for all the data in the .h5 file
        positionOrientationOneFrame() calculate the position and orientation for one frame
        labelVideoLever() label a video
        detectionImage() add points and line to an image to show the lever detection.
        
    """
    def __init__(self, pathConfigFile):
        super().__init__(pathConfigFile)
        
        self.requiredBodyParts = ['lever', 'lever_right', 'lever_left']
        if not self.bodyParts == self.requiredBodyParts :
            print("The body parts are not equal to"+self.requiredBodyParts)
            return False
        
    def positionOientationFromFile(self,pathVideoFile):
        """
        Calculate the position and orientation of the lever for a video file
        
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
        
        This is similar to the labelVideo() function of deeplabcut. 
        But in addition it shows the calculated position and orientation of the lever
        
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
        The function is used when labelling videos
        
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
        Function to get the position and orientation of the lever based on deeplabcut detection.
        
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

    def __str__(self):
        """
        Print function
        """
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    


class mouseLeverDetector(dlc):
    """
    Class to implement the detection of the mouse and lever using a deeplabcut model
    This class inherits from dlc, but has additional functions that are specific to the lever.
    For example, it can calculate the position and orientation of the lever and the location of the mouse
    
    It assumes that the 3 parts of the lever are detected by deeplabcut
        lever
        lever_right
        lever_left
    
    Methods
        
    """
    def __init__(self, pathConfigFile):
        super().__init__(pathConfigFile)
        
        self.requiredBodyParts = ['lever', 'boxPL', 'boxPR', 'nose', 'earL', 'earR', 'tail']
        if not self.bodyParts == self.requiredBodyParts :
            print("The body parts are not equal to"+self.requiredBodyParts)
            return False
    def __str__(self):
        """
        Print function
        """
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    def positionOientationFromFile(self,pathVideoFile):
        """
        Calculate the position and orientation of the mouse and lever for a video file
        
        Output is stored in self.posiOri
        """
        
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
        
        self.pathVideoFile=pathVideoFile
        
        # get the tracking data from file
        self.loadPositionData(self.pathVideoFile) # load from file to self.out
        
        self.posiOri = np.apply_along_axis(self.positionOrientationOneFrame, 1, self.out)
        
        
    def positionOrientationOneFrame(self,frameTrackingData,probThreshold = 0.5):
        """
        Function to get the position and orientation of the mouse and lever based on deeplabcut detection.
        
        Data in self.out are in the order ['lever', 'boxPL', 'boxPR', 'nose', 'earL', 'earR', 'tail']
        There are 3 numbers per object: x, y and probability
                
        The 0,0 coordinates are top-left
        If the probability is below the threshold, return None
    
        There are 2 objects (lever and mouse), we can use the first 3 parts of each (ignoring the tail).
        For both object, we find a middle point P4 between parts 2 and 3, center is P4 and part 1 is position.
        Orientation is from P4 to part 1.
        
        Arguments
            frameTrackingData 1D numpy array containing the tracking values for one frame
        Return
            List containing 
                lever.x, lever.y , lever.theta_deg, lever.X, lever.Y
                mouse.x, mouse.y , mouse.theta_deg, mouse.X, mouse.Y
                
                x and y are the middle point of the lever box for lever
                theta_deg is the angle of the lever or animal relative to East. North is 90 degrees
                
                X and Y are the vector from the x and y heading in the calculated orientation
        """
        ret = np.empty(10) # to return the results
        ret[:] = np.nan # by default set to NaN
               
        # LEVER
        # if we know where the lever is
        inds=0 #index at which the parts for the lever starts in the frameTrackingData
        if frameTrackingData[inds+2] > probThreshold and frameTrackingData[inds+5] > probThreshold and frameTrackingData[inds+8] > probThreshold:
    
            ## middle point at the back of the lever (it does not matter if the two points are swapped, which is good)
            P4x = (frameTrackingData[inds+3]+frameTrackingData[inds+6])/2
            P4y = (frameTrackingData[inds+4]+frameTrackingData[inds+7])/2

            ## lever position (middle point)
            x= (frameTrackingData[inds+0]+P4x)/2
            y= (frameTrackingData[inds+1]+P4y)/2
    
            ## vector from middle point to tip of lever
            X=frameTrackingData[inds+0]-x
            Y=frameTrackingData[inds+1]-y
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

            ret[0:5] = [x,y,theta_deg,X,Y]
        
        # MOUSE
        # if we know where the mouse is, values for mouse starts at 
        inds=9 #index at which the parts for the lever starts in the frameTrackingData
        if frameTrackingData[inds+2] > probThreshold and frameTrackingData[inds+5] > probThreshold and frameTrackingData[inds+8] > probThreshold:
    
            ## middle point at the back of the lever (it does not matter if the two points are swapped, which is good)
            P4x = (frameTrackingData[inds+3]+frameTrackingData[inds+6])/2
            P4y = (frameTrackingData[inds+4]+frameTrackingData[inds+7])/2

            ## lever position (middle point)
            x= (frameTrackingData[inds+0]+P4x)/2
            y= (frameTrackingData[inds+1]+P4y)/2
    
            ## vector from middle point to tip of lever
            X=frameTrackingData[inds+0]-x
            Y=frameTrackingData[inds+1]-y
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

            ret[5:10] = [x,y,theta_deg,X,Y]
        
        return ret
        
    def labelVideoMouseLever(self,pathVideoFile,pathOutputFile):
        """
        Function to create a label video specific to lever detection
        
        This is similar to the labelVideo() function of deeplabcut. 
        But in addition it shows the calculated position and orientation of the lever
        
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
    
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        frameRate = cap.get(cv2.CAP_PROP_FPS) # float
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(pathOutputFile, fourcc , frameRate, (int(width),int(height)))
      
        ## loop through the video
        index = 0
        
        numOut = self.out.shape[0]
        if nFrames != numOut:
            print("Number of tracking points not equal to number of frames in video")
            return False
        
        print("Saving labeled video in " + pathOutputFile)
        while(cap.isOpened() and index < nFrames ):
            ret, frame = cap.read() 
            self.detectionImage(frame,self.out[index,],self.posiOri[index,]) # will modify frame
            out.write(frame)
            if index % 10 == 0:
                sys.stdout.write('\r')
                sys.stdout.write("{} of {} frames".format(index,nFrames))
                sys.stdout.flush()
            index+=1
                
            
        out.release()
        cap.release()
        
        
    def detectionImage(self,frame,oneOut,onePosiOri):
        """
        Takes an image and position data and draw the detection on the image
        The function is used when labelling videos
        
        Arguments
            frame An opencCV image
            oneOut numpy array with the tracking data of one frame
            onePosiOri numpy array with the position of the lever center, the lever orientation, and vector from center to lever press
        """
        
        ## open cv drawings
        if np.isnan(onePosiOri[0]):
            frame = cv2.putText(frame, "No lever detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                                             0.75, (100,200,0), 1, cv2.LINE_AA) 
        else :

            # lever direction
            inds=0
            frame = cv2.line(frame,(int(onePosiOri[inds+0]),int(onePosiOri[inds+1])),(int(onePosiOri[inds+0]+onePosiOri[inds+3]),int(onePosiOri[inds+1]+onePosiOri[inds+4])),(255,200,0),2)
            # lever position
            frame = cv2.circle(frame, (int(onePosiOri[inds+0]),int(onePosiOri[inds+1])), radius=4, color=(0, 255, 0), thickness=-1)
            # lever orientation
            frame = cv2.putText(frame, "{:.1f}".format(onePosiOri[inds+2]), (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                                                 0.75, (100,200,0), 1, cv2.LINE_AA) 
            # lever position
            frame = cv2.putText(frame, "{:.1f} {:.1f}".format(onePosiOri[inds+0],onePosiOri[inds+1]), (50,100), cv2.FONT_HERSHEY_SIMPLEX ,  
                                                 0.75, (100,200,0), 1, cv2.LINE_AA)
           
        if np.isnan(onePosiOri[5]):  
            frame = cv2.putText(frame, "No mouse detected", (50,150), cv2.FONT_HERSHEY_SIMPLEX ,  
                                             0.75, (100,200,0), 1, cv2.LINE_AA) 
        else:
            # mouse direction
            inds=5
            frame = cv2.line(frame,(int(onePosiOri[inds+0]),int(onePosiOri[inds+1])),(int(onePosiOri[inds+0]+onePosiOri[inds+3]*2),int(onePosiOri[inds+1]+onePosiOri[inds+4]*2)),(0,200,255),2)
            # mouse position
            frame = cv2.circle(frame, (int(onePosiOri[inds+0]),int(onePosiOri[inds+1])), radius=2, color=(255, 0, 0), thickness=-1)
            # mouse orientation
            frame = cv2.putText(frame, "{:.1f}".format(onePosiOri[inds+2]), (50,150), cv2.FONT_HERSHEY_SIMPLEX ,  
                                                 0.75, (0,200,100), 1, cv2.LINE_AA) 
            # mouse position
            frame = cv2.putText(frame, "{:.1f} {:.1f}".format(onePosiOri[inds+0],onePosiOri[inds+1]), (50,200), cv2.FONT_HERSHEY_SIMPLEX ,  
                                                 0.75, (0,200,1), 1, cv2.LINE_AA)
            


class bridgeDetector(dlc):
    """
    Class to implement the detection of the bridge using a deeplabcut model
    This class inherits from dlc, but has additional functions that are specific to the bridge.
    For example, it can calculate the position of the center of the bridge
    
    It assumes that the 2 parts of the bridge are detected by deeplabcut
        leftAnt
        rightAnt
         
    Methods
        positionFromFile() calculate the position and orientation for all the data in the .h5 file
        positionOneFrame() calculate the position and orientation for one frame
        labelVideoBridge() label a video
        detectionImage() add points and line to an image to show the lever detection.
        
    """
    def __init__(self, pathConfigFile):
        super().__init__(pathConfigFile)
        
        self.requiredBodyParts = ['leftAnt', 'rightAnt']
        if not self.bodyParts == self.requiredBodyParts :
            print("The body parts are not equal to"+self.requiredBodyParts)
            return False
        
    def positionFromFile(self,pathVideoFile):
        """
        Calculate the position of the end of the bridge
        
        Output is stored in self.posiOri
        
        Format, leftAnt.x leftAnt.y, rightAnt.x, rightAnt.x, center.x, center.y
        
        """
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
        
        self.pathVideoFile=pathVideoFile
        
        # get the tracking data from file
        self.loadPositionData(self.pathVideoFile) # load from file to self.out
        
        self.posi = np.apply_along_axis(self.positionOneFrame, 1, self.out)

    def positionOneFrame(self,frameTrackingData,probThreshold = 0.5):
        """
        Function to get the position of the bridge based on deeplabcut detection.
        
        The 0,0 coordinates are top-left
        If the probability is below the threshold, return None
    
        Arguments
            frameTrackingData 1D numpy array containing the tracking values for one frame
        Return
            Array with [leftAnt.x, leftAnt.y, rightAnt.x, rightAnt.x, center.x, center.y]
                
        """
    
        if frameTrackingData[2] < probThreshold or frameTrackingData[5] < probThreshold:
            return [np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]
    
        ## middle point at the back of the lever (it does not matter if the two points are swapped, which is good)
        x = (frameTrackingData[0]+frameTrackingData[3])/2
        y = (frameTrackingData[1]+frameTrackingData[4])/2

        return [frameTrackingData[0],frameTrackingData[1], frameTrackingData[3], frameTrackingData[4],x,y]

        
    def labelVideoBridge(self,pathVideoFile,pathOutputFile):
        """
        Function to create a label video specific to lever detection
        
        This is similar to the labelVideo() function of deeplabcut. 
        But in addition it shows the center of the edge of the bridge
        
        Arguments
            pathVideoFile: video to get the frames from
            pathOutputFile: labeled video to create
        """
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
        self.pathVideoFile = pathVideoFile
        
        # load data from file and get lever center and orientation
        self.positionFromFile(self.pathVideoFile) # see self.out and self.posi
        
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
        print("Creating " + pathOutputFile)
        while(cap.isOpened() and index < numFrames ):
            ret, frame = cap.read() 
            self.detectionImage(frame,self.out[index,],self.posi[index,]) # will modify frame
            out.write(frame)   
            index=index+1
        out.release()
        cap.release()
       
    def detectionImage(self,frame,oneOut,onePosi):
        """
        Takes an image and position data and draw the detection on the image
        The function is used when labelling videos
        
        Arguments:
            frame An opencCV image
            oneOut numpy array with the tracking data of one frame
            onePosi numpy array with the position of the bridge
        """
        
        ## open cv drawings
        if np.isnan(onePosi[0]):
            frame = cv2.putText(frame, "No bridge detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                                             1, (100,200,0), 2, cv2.LINE_AA) 
        else :    
            frame = cv2.circle(frame, (int(onePosi[0]),int(onePosi[1])), radius=4, color=(0, 255, 0), thickness=-1)
            frame = cv2.circle(frame, (int(onePosi[2]),int(onePosi[3])), radius=4, color=(0, 255, 0), thickness=-1)
            frame = cv2.circle(frame, (int(onePosi[4]),int(onePosi[5])), radius=4, color=(255, 255, 0), thickness=-1)
        
    def __str__(self):
        """
        Print function
        """
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    def detectBridgeCoordinates(self,pathVideoFile,numFrames=1000, skip=300, tmpDir="/tmp/"):
        """
        Perform bridge detection on a range of frames from a video
        
        It will create a new video in the /tmp/ directory to perform the analysis using dlc.
        
        Assumes that the bridge is at the top of the image
        
        Arguments:
            pathVideoFile: File from which to get the frame from
            numFrames: Number of frames to analyze
            skip: Number of frames to skip at the beginning of the video file
            tmpDir: Directory where to do the detection
        
        Returns
            4x2 np.array containing the coordinates of the bridge
            top-left,bottom-left,bottom-right,top-right
        """
        
        if not os.path.isdir(tmpDir):
            print(tmpDir+" is not a directory")
            return 0
        
        pathTmpVideo = tmpDir+'tmpVid.avi'
        base = pathTmpVideo.split(".")[0]
        
        # remove any old files that might be there
        fileList = glob.glob(base+"*")
        for file in fileList:
            os.remove(file)
    
        if not os.path.isfile(pathVideoFile):
            print(pathVideoFile + " does not exist")
            return False

        count = 0 
        cap = cv2.VideoCapture(pathVideoFile)

        fps = int (cap.get(cv2.CAP_PROP_FPS))
        width = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(pathTmpVideo, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width,height)) 

        while cap.isOpened() and count < numFrames + skip: 
            ret, frame = cap.read() 
            count = count+1 
            if not ret: 
                print("Can't receive frame (stream end?). Exiting ...") 
                break 
            if count > skip: 
                out.write(frame) 
             
        cap.release() 
        out.release() 
           
        print(pathTmpVideo+" created for bridge detection") 
        
        
        self.inferenceVideo(pathTmpVideo)
        
        # get the position of the bridge, now in self.posi
        self.positionFromFile(pathTmpVideo) 
        # self.posi now has this information per frame leftAnt.x, leftAnt.y, rightAnt.x, rightAnt.x, center.x, center.y 
            
        # remove old files
        fileList = glob.glob(base+"*")
        for file in fileList:
            os.remove(file)
        
        # get the mode of leftAnt.x, leftAnt.y, rightAnt.x, rightAnt.y
        x1 = mode(self.posi[:,0].astype(int))[0]
        y1 = mode(self.posi[:,1].astype(int))[0]
        x2 = mode(self.posi[:,2].astype(int))[0]
        y2 = mode(self.posi[:,3].astype(int))[0]
       
        # we extend the bridge to the top edge of the image, this is heading into the home base.
        # top-left, bottom-left, bottom-right, top-right
        self.bridgeCoordinates = np.array([[x1[0],0],
                         [x1[0],y1[0]],
                         [x2[0],y2[0]],
                         [x2[0],0]])
       
        
        return self.bridgeCoordinates


    def labelImage(self,pathVideoFile,outputImageFile):
        """
        Save an image in a file with the detected bridge
        
        Arguments:
            pathVideoFile
            outputImageFile
        """
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
           
        self.pathVideoFile = pathVideoFile
        cap = cv2.VideoCapture(self.pathVideoFile)
        ret, frame = cap.read()
        
        for i in range(self.bridgeCoordinates.shape[0]):
            cv2.circle(frame,(self.bridgeCoordinates[i,0],self.bridgeCoordinates[i,1]),3,(0,255,0),2)
           
        # save the last frame with detected circle
        print("labelImage: " + outputImageFile)
        cv2.imwrite(outputImageFile,frame)

        