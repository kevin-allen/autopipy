import numpy as np

class NavPath:
    """
    Class representing a path of an object (animal, airplane, etc.) in the context of navigation.
    
    To make this class as flexible as possible, we will define position as x, y, z and orientation as yaw, pitch and roll.
    
    The class calculates some statistics about the path.
    
    If a target is defined, statistics regardign the relationship between the path and the target will be calculated.
        
    Attributes:
        pathData: numpy array with columns x y z yaw pitch roll time
        target: Pose (x, y, z, yaw, pitch, roll) of a theoretical target of the path
        
    Methods:
    """
    def __init__(self, pathData, target):
        self.pathData = pathData
        self.pathLength = np.NAN # sum of position vectors
        self.pathDuration = np.NAN # duration from the start to the end
        self.meanVectorLengthPosi = np.NAN # mean vector length of position vectors
        self.meanVectorLengthOri = np.NAN # mean vector length of each Euler angle (yaw, pitch, roll)
        self.meanDirection = np.NAN # mean direction of each Euler angle (yaw, pitch, roll)
        self.meanSpeed = np.NAN # 