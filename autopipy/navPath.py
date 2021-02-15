import numpy as np

class NavPath:
    """
    Class representing a path of an object (animal, airplane, etc.) in the context of navigation.
    
    To make this class as flexible as possible, we will define position as x, y, z and orientation as yaw, pitch and roll.
    
    The class calculates some statistics about the path.
    
    If a target is defined, statistics regardign the relationship between the path and the target will be calculated.
        
    Attributes:
        pPose: numpy array with columns x y z yaw pitch roll time, the rows are different time points
        target: numpy array with Pose (x, y, z, yaw, pitch, roll) of a theoretical target for the path
        
    Methods:
    """
    def __init__(self, pPose, targetPose=None):
        self.pPose = pPose
        
        posi = self.pPose[:,0:3] # get position data
        mv = np.diff(posi,axis = 0) # change in position for 3 columns
        mv3 = np.sqrt(np.sum(mv*mv,axis = 1)) # distance between consecutive position
        
        # length of the path in 3D
        self.length = np.sum(mv3)
        
        # difference between largest and smallest time point
        self.duration = np.max(pPose[:,6])-np.min(pPose[:,6]) # duration from the start to the end
        
        # (first to last Pose distance) / length of the path, 1 if straght line, 0 if came back to same point
        mvEnds=posi[-1,:]-posi[0,:]

        self.distanceEnds =  np.sqrt(np.sum(mvEnds*mvEnds))
        self.meanVectorLengthPosi =  self.distanceEnds/ self.length # mean vector length of position vectors
        
        # how concentrated were the angles for yaw, pitch and roll, 1 if always same orientation, 0 if homogeneously distributed
        ori = self.pPose[:,3:6]
        
        # mean vector length of each Euler angle (yaw, pitch, roll)
        self.meanVectorLengthOri = np.apply_along_axis(self.meanVectorLength, 0, ori)
        
        # mean direction of each Euler angle (yaw, pitch, roll)
        self.meanDirection = np.apply_along_axis(self.meanVectorDirection, 0, ori)
        
        # mean linear speed
        self.meanSpeed = self.length/self.duration 
         
    def meanVectorLength(self, theta, degree=True, negativeAngle=False):
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
    