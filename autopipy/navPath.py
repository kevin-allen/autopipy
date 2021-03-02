import numpy as np

class NavPath:
    """
    Class representing a path of an object (animal, airplane, etc.) in the context of navigation.
    
    To make this class as flexible as possible, we will define position as x, y, z and orientation as yaw, pitch and roll.
    
    The class calculates some statistics about the path.
    
    If a target is defined, statistics regardign the relationship between the path and the target will be calculated.
    
    When calculating angles, the y coordinate is reversed (0-y) so that 90 degree is up.
    This is done because the y-axis is reversed in the videos.
    
    
    
    Attributes:
        pPose: numpy array with columns x y z yaw pitch roll time, the rows are different time points
        target: numpy array with Pose (x, y, z, yaw, pitch, roll) of a theoretical target for the path
        length
        duration
        meanVectorLengthPosi
        meanVectorDirectionPosi
        meanVectorLengthOri
        meanVectorDirectionOri
        meanSpeed
        medianMVDeviationToTarget
        medianHDDeviationToTarget
    Methods:
    """
    def __init__(self, pPose, targetPose=None,name=None):
        self.pPose = pPose
        self.targetPose = targetPose
        self.name = name
        if self.pPose is None or self.pPose.shape[0] < 2 :  
            print("{} :empty NavPath created".format(self.name))
            self.length = np.NAN
            self.duration = np.NAN
            self.meanVectorLengthPosi = np.NAN
            self.meanVectorDirectionPosi = np.NAN
            self.meanVectorLengthOri = np.array([np.NAN,np.NAN,np.NAN])
            self.meanVectorDirectionOri = np.array([np.NAN,np.NAN,np.NAN])
            self.meanSpeed = np.NAN
            self.medianMVDeviationToTarget=np.NAN
            self.medianHDDeviationToTarget=np.NAN
            return
        
        
        posi = self.pPose[:,0:3] # get position data
        self.mv = np.diff(posi,axis = 0) # change in position for 3 columns
        mv3 = np.sqrt(np.sum(self.mv*self.mv,axis = 1)) # distance between consecutive position
        
        # length of the path in 3D
        self.length = np.sum(mv3)
        
        # difference between largest and smallest time point
        self.duration = np.max(pPose[:,6])-np.min(pPose[:,6]) # duration from the start to the end
        
        # (first to last Pose distance) / length of the path, 1 if straght line, 0 if came back to same point
        mvEnds=posi[-1,:]-posi[0,:]
        self.distanceEnds =  np.sqrt(np.sum(mvEnds*mvEnds))
        self.meanVectorLengthPosi =  self.distanceEnds/ self.length # mean vector length of position vectors
        self.meanVectorDirectionPosi = self.direction(mvEnds[0],mvEnds[1]) # mean direction of the movement
        
        # how concentrated were the angles for yaw, pitch and roll, 1 if always same orientation, 0 if homogeneously distributed
        ori = self.pPose[:,3:6]
        
        # mean vector length of each Euler angle (yaw, pitch, roll)
        self.meanVectorLengthOri = np.apply_along_axis(self.meanVectorLength, 0, ori)
        
        # mean direction of each Euler angle (yaw, pitch, roll)
        self.meanVectorDirectionOri = np.apply_along_axis(self.meanVectorDirection, 0, ori)
        
        # mean linear speed
        self.meanSpeed = self.length/self.duration 
    
        
        # if we have a target, calculate whether the path lead to the target       
        
        self.medianMVDeviationToTarget=np.NAN
        self.medianHDDeviationToTarget=np.NAN
        if targetPose is not None:
            ## mv heading relative to vector to target
            posi = self.pPose[:,0:3] # get position in the path
            posiT = self.targetPose[:,0:3] # the position of the target
            mv = np.diff(posi,axis = 0,prepend=np.NAN) # movement vector
            tv = posiT - posi # toTargetVector
            angles=self.vectorAngle(mv,tv,degrees=True,quadrant=False)
            self.medianMVDeviationToTarget = np.nanmedian(angles)
            
            ## orientation (yaw only!!! should be more generic!!!)        
            # get 2D vectors from yaw angle
            hdv = self.unityVectorsFromAngles(self.pPose[:,3]) # only yaw
            angles=self.vectorAngle(hdv,tv[:,0:2],degrees=True,quadrant=False)
            self.medianHDDeviationToTarget = np.nanmedian(angles)
            
    def unityVectorsFromAngles(self,theta,degree=True):
        if degree:
            theta = theta*np.pi/180
        return np.stack((np.cos(theta),np.sin(theta)),axis =1)

           
            
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

    
    
    def direction(self, x, y, degree = True,negativeAngle=False):
        """
        Calculate the direction of a vector in 2D
        Arguments
            x: x component
            y: y component
            degree: whether to return the data as degree
        """
        direction = np.arctan2(y,x)
        if negativeAngle==False:
            if direction < 0:
                direction = 2*np.pi+direction
        if degree:
            direction = direction/np.pi*180
        return direction
        
    
    def meanVectorLength(self, theta, degree=True):
        """
        Calculate the mean direction and mean vector length
        Arguments:
            theta: angles for which you what the mean vector
            degree: whether we are working in degrees or radians
            negativeAngle: whether you want angles from -180 to 180 or from 0 to 360
        """
        if degree:
            theta = theta*np.pi/180
        # get the unity vectors for these angles
        v = np.stack((np.cos(theta),np.sin(theta)),axis =1)
        mv = np.sum(v,axis=0)/len(v)
        length = np.sqrt(np.sum((mv*mv)))
        
        return length
    

    
    def meanVectorDirection(self, theta, degree=True, negativeAngle=False):
        if degree:
            theta = theta*np.pi/180
        # get the unity vectors for these angles
        v = np.stack((np.cos(theta),np.sin(theta)),axis =1)
        mv = np.sum(v,axis=0)/len(v)
        
        direction = np.arctan2(mv[1],mv[0])
        if negativeAngle==False:
            if direction < 0:
                direction = 2*np.pi+direction
        if degree:
            direction = direction/np.pi*180
        return direction
    
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    