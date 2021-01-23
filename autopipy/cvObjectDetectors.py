import os.path
import numpy as np
import cv2

class arenaDetector:
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
    
    def detectArenaCoordinates (self, pathVideoFile, minRadius=180, maxRadius=220, numFrames=100, blur=11, circle='min'):
    
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
           
        self.pathVideoFile = pathVideoFile
        cap = cv2.VideoCapture(self.pathVideoFile)

        l = []

        ## loop throught the frames and detect the arnea
        ## save the x,y of center and radius in the list l
        count=0
        while cap.isOpened() and count < numFrames :
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
                                          param1=70,param2=50,minRadius=minRadius,maxRadius=maxRadius)

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
            print("problem with circle detection, array size 0")
    
        if circle == 'min':
            ## selected the smallest detected circle
            self.coordinates=l[np.argmin(l[:,2]),:]   # coordinates = [x, y, r]
        elif circle == 'median':
            self.coordinates=l[np.argmin(np.abs(l[:,2] - np.median(l[:,2]))),:]   # coordinates = [x, y, r]

        return self.coordinates
    
    def labelImage(self,pathVideoFile,outputImageFile):
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
        ret, frame = cap.read()
        
        cv2.circle(frame,(self.coordinates[0],self.coordinates[1]),self.coordinates[2],(0,255,0),2)
        cv2.circle(frame,(self.coordinates[0],self.coordinates[1]),2,(0,0,255),3)
        
        # save the last frame with detected circle
        print("labelImage: " + outputImageFile)
        cv2.imwrite(outputImageFile,frame)

        