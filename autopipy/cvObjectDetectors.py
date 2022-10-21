import os.path
import numpy as np
import cv2

class ArenaDetector:
    """
    Class to detect the arena from a video
    
    Attributes:
    
    Methods:
        
    """
    def __init__(self):
        self.coordinates = np.empty(3)
        return
        
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    def detectArenaCoordinates (self, pathVideoFile, minRadius=180, maxRadius=220, numFrames=100, blur=11, circleMethod='min',randomFrameSelection=True):
        """
        Detect the arena from images of a videos
        
        minRadius: minimum radius to be considered
        maxRadius: maximum radius to be considered
        numFrames: how many frames to analyze
        blur: frames will be blurred before circle detection
        randomFrameSelection: pick random frames from the video instead of picking the first frames
        circleMethod: how to pick the final circle coordinate from all circles that were detected ("min" or "median" radius)
        """
    
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
           
        self.pathVideoFile = pathVideoFile
        cap = cv2.VideoCapture(self.pathVideoFile)
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        l = []

        ## loop throught the frames and detect the arnea
        ## save the x,y of center and radius in the list l
        if randomFrameSelection:
            selectIndices = np.sort(np.random.randint(0,nFrames-1,numFrames))
        
       
        count=0
        while cap.isOpened() and count < numFrames :
            if randomFrameSelection:
                cap.set(cv2.CAP_PROP_POS_FRAMES,selectIndices[count])
                
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.medianBlur(img,blur)
                cimg = cv2.cvtColor(blurred,cv2.COLOR_GRAY2BGR)
                circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,50,
                                          param1=75,param2=45,minRadius=minRadius,maxRadius=maxRadius)#p1=70,p2=50 
                ## next try was p1=75, p2=45
                circles = np.uint16(np.around(circles))

                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                    # draw the center of the circle
                    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
                    l.append(i)
            except:
                pass
            
             
            count=count+1
        
        ## clean up
        cap.release()
        
        # Select circle
        l = np.asarray(l)
        if l.shape[0]==0 :
            raise ValueError("problem with circle detection, array size 0")
    
        if circleMethod == 'min':
            ## selected the smallest detected circle
            self.coordinates=l[np.argmin(l[:,2]),:]   # coordinates = [x, y, r]
        elif circleMethod == 'median':
            self.coordinates=l[np.argmin(np.abs(l[:,2] - np.median(l[:,2]))),:]   # coordinates = [x, y, r]

        return self.coordinates
    
    def labelImage(self,pathVideoFile,outputImageFile,skip=30):
        """
        Save an image in a file with the detected arena
        
        Arguments:
            pathVideoFile
            outputImageFile
        """
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
           
        self.pathVideoFile = pathVideoFile
        cap = cv2.VideoCapture(self.pathVideoFile)
        index=0
        while index < skip:
            ret, frame = cap.read()
            index+=1
        
        cv2.circle(frame,(self.coordinates[0],self.coordinates[1]),self.coordinates[2],(0,255,0),2)
        cv2.circle(frame,(self.coordinates[0],self.coordinates[1]),2,(0,0,255),3)
        
        # save the last frame with detected circle
        print("labelImage: " + outputImageFile)
        cv2.imwrite(outputImageFile,frame)

        