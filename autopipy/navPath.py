import numpy as np
import pandas as pd
from scipy import stats

class NavPath:
    """
    Class representing a path of an object (animal, airplane, etc.) in the context of navigation.
    
    To make this class as flexible as possible, we will define position as x, y, z and orientation as yaw, pitch and roll.
    
    The class calculates some statistics about the path.
    
    If a target is defined, statistics regardign the relationship between the path and the target will be calculated.
      
    
    
    Attributes:
        pPose: numpy array with columns [x, y, z, yaw, pitch, roll, time], the rows are different time points
        target: numpy array with Pose (x, y, z, yaw, pitch, roll) of a theoretical target for the path
        length
        duration
        meanVectorLengthPosi
        meanVectorDirectionPosi
        meanVectorLengthOri
        meanVectorDirectionOri
        meanSpeed
        speedProfile: speed over the path, path split in 10 equal bins
        oriAngularDistance
        oriAngularSpeed
        mvAngularDistance
        mvAngularSpeed
        medianMVDeviationToTarget
        medianHDDeviationToTarget
    Methods:
    """
    def __init__(self, pPose, targetPose=None,name=None,resTime=None,trialNo=None):
        """
        Attributes:
        pPose: 2D numpy array with 7 or 8 columns [x, y, z, yaw, pitch, roll, time] or [x, y, z, yaw, pitch, roll, timeRos, timeRes], angles are in degrees
        targetPose: 2D numpy array with Pose (x, y, z, yaw, pitch, roll), angles are in degrees
        resTime: 1D numpy array with alternative time (e.g. electrophysiology time)
        """
        

        self.pPose = pPose
        self.targetPose = targetPose
        self.name = name
        self.resTime=resTime
        self.trialNo=trialNo
        
        if self.pPose.shape[1] != 7 and self.pPose.shape[1] != 8:
            print("{} :pPose should have 7 or 8 columns [x, y, z, yaw, pitch, roll, RosTime, (resTime)]".format(self.name))
            self.pPose = None
        if self.pPose.shape[0] < 2 :  
            #print("{} :pPose has a length of {}".format(self.name,self.pPose.shape[0]))
            self.pPose = None
        
        if self.pPose is None :
            #print("{} :empty NavPath created".format(self.name))
            self.length = np.NAN
            self.duration = np.NAN
            
            # vector analysis
            # the vectors used in the calculation of these should probaby weighted by running speed
            self.meanVectorLengthPosi = np.NAN
            self.meanVectorDirectionPosi = np.NAN
            self.meanVectorLengthOri = np.array([np.NAN,np.NAN,np.NAN])
            self.meanVectorDirectionOri = np.array([np.NAN,np.NAN,np.NAN])
            
            self.meanSpeed = np.NAN
            self.speedProfile = np.empty((10))
            self.speedProfile[:] = np.NAN
            
            # should we have a speed cutoff or weight given to speed???
            self.oriAngularDistance =  np.array([np.NAN,np.NAN,np.NAN])
            self.oriAngularSpeed = np.array([np.NAN,np.NAN,np.NAN])
            self.mvAngularDistance =  np.NAN
            self.mvAngularSpeed = np.NAN
            
            # vector analysis
            # the vectors used in the calculation of these should probaby weighted by running speed
            self.medianMVDeviationToTarget=np.NAN
            self.medianHDDeviationToTarget=np.NAN
            
            # variables quantifying turns around the target
            self.entryAngleAroundTarget = np.NAN # angle of first data point relative to target
            self.exitAngleAroundTarget = np.NAN # angle of the last point relative to target
            self.cumSumDiffAngleAroundTarget = None # vector with the cum sum of difference in angle around the target
            self.endCumSumDiffAngleAroundTarget = np.NAN  # last data point in the cum sum of difference in angle around the target
            self.rangeCumSumDiffAngleAroundTarget = np.NAN
            
            
            return
        

        self.startTime = self.pPose[:,6].min()
        self.endTime=self.pPose[:,6].max()
        
        posi = self.pPose[:,0:3] # get position data
        self.mv = np.diff(posi,axis = 0,append=np.nan) # movement vector in x y z dimensions
        mv3 = np.sqrt(np.nansum(self.mv*self.mv,axis = 1)) # length of vectors (pythagoras) 
        
        # run distance from beginning of path
        self.distanceRun = np.nancumsum(mv3)
        self.distanceRunProp = self.distanceRun/np.nanmax(self.distanceRun)
        
        # time from beginning
        self.internalTime = self.pPose[:,6]- self.pPose[:,6].min()
        self.internalTimeProp = self.internalTime/np.nanmax(self.internalTime)
        
        # length of the path in 3D
        self.length = np.nansum(mv3)

        # difference between largest and smallest time point
        self.duration = np.nanmax(pPose[:,6])-np.nanmin(pPose[:,6]) # duration from the start to the end
        
        # (first to last Pose distance) / length of the path, 1 if straght line, 0 if came back to same point
        mvEnds=posi[-1,:]-posi[0,:]
        self.distanceEnds =  np.sqrt(np.nansum(mvEnds*mvEnds))
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
        timeDiff=np.diff(self.pPose[:,6],axis = 0,append=np.nan)
        
        # speed profile in the path divided into 10 equal bins 
        self.speed=mv3/timeDiff
        ##WARNING##
        # we need to remove na for the stats.binned_statistic
        # we might want to change this behavior in the future
        # I do not expect to have many nan in the paths, based on labeled videos
        self.speedShort = self.speed[~np.isnan(self.speed)] 
        if len(self.speedShort) > 9 :
            x=np.arange(len(self.speedShort))
            ind = np.isfinite(self.speedShort)
            binRes = stats.binned_statistic(x[ind],self.speedShort[ind],"mean",bins=10)
            self.speedProfile = binRes.statistic     
        else :
            self.speedProfile = np.empty((10))
            self.speedProfile[:] = np.NAN
            
        # calculate the sum of difference in head orientation for the 3 axes
        oriDiff= np.abs(np.diff(ori,axis=0)) # change in orientation for yaw, pitch, roll
        # we need to remove np.nan values if there are some
        
        oriDiff=oriDiff[~np.isnan(oriDiff).any(axis=1), :]
        oriDiff = np.where(oriDiff>180,360-oriDiff,oriDiff)
        self.oriAngularDistance = np.nansum(oriDiff, axis= 0)
        self.oriAngularSpeed = self.oriAngularDistance/self.duration
        
        # calculate the sum of difference in movement direction for x, y, z dimensions
        mvAngle = np.empty(self.mv.shape[0]-1)
        for i in np.arange(self.mv.shape[0]-1):
            # reshape to make suitable for vectorAngle function
            v1 = np.reshape(self.mv[i,:],(1,-1))
            v2 = np.reshape(self.mv[i+1,:],(1,-1))
            mvAngle[i] = self.vectorAngle(v=v1,rv=v2,degrees=True) 
        self.mvAngularDistance = np.nansum(mvAngle)
        self.mvAngularSpeed=self.mvAngularDistance/self.duration
          
        
        # if we have a target, calculate whether the path lead to the target       
        self.medianMVDeviationToTarget=np.NAN
        self.medianHDDeviationToTarget=np.NAN
        
        
        if targetPose is not None:
            ## mv heading relative to vector to target
            posi = self.pPose[:,0:3] # get position in the path
            posiT = self.targetPose[:,0:3] # the position of the target
            
            self.targetDistance = np.sqrt(np.nansum((posi-posiT)**2,axis=1))
            
            mv = np.diff(posi,axis = 0,prepend=np.NAN) # movement vector
            tv = posiT - posi # toTargetVector, where is the target relative to the animal
            
            
            
            angles=self.vectorAngle(mv,tv,degrees=True,quadrant=False)
            self.medianMVDeviationToTarget = np.nanmedian(angles)
            
            ## orientation (yaw only!!! should be more generic!!!)        
            # get 2D vectors from yaw angle
            hdv = self.unityVectorsFromAngles(self.pPose[:,3]) # only yaw
            angles=self.vectorAngle(hdv,tv[:,0:2],degrees=True,quadrant=False)
            self.medianHDDeviationToTarget = np.nanmedian(angles)
            
            # get a vector from target to animal 
            self.vTargetToAnimal = posi-posiT # contains 3 columns x,y,z
            
            # replace the head direction data with the angle between the vector of the animal position (origin 0,0) and the vector 1,0
            self.targetToAnimalAngle = np.arctan2(self.vTargetToAnimal[:,1], self.vTargetToAnimal[:,0]) # relative to 1,0 vector
            
            ########################
            ## angle around target #
            ########################
            
            targetPoint = np.array([self.targetPose[0,0],self.targetPose[0,1]])
            animalPoints = np.vstack([self.pPose[:,0],self.pPose[:,1]]).T
            animalPoints = animalPoints - targetPoint # put the target at zero, as the origin of our vectors
            #animalPoints = animalPoints[~np.isnan(animalPoints).any(axis=1),:] # remove NAN, can't do this because of instantaneous data
            animalAngleAroundTarget = np.arctan2(animalPoints[:,1],animalPoints[:,0])
            
            if np.sum(~np.isnan(animalAngleAroundTarget)) > 2:
                self.entryAngleAroundTarget = animalAngleAroundTarget[~np.isnan(animalAngleAroundTarget)][0]
                self.exitAngleAroundTarget = animalAngleAroundTarget[~np.isnan(animalAngleAroundTarget)][-1]
            else:
                self.entryAngleAroundTarget = np.nan
                self.exitAngleAroundTarget = np.nan
            
            # get the rotation matrix of each vector
            M=np.apply_along_axis(self.rotMatrix,1,animalPoints)
            # apply the rotation matrix of previous vector to current vector
            rotAnimalPoints = np.empty(shape = (animalPoints.shape[0]-1,2))
            for i in range(1,animalPoints.shape[0]):
                rotAnimalPoints[i-1,:] = animalPoints[i,:]@M[i-1,:]
            diffAnimalAngleAroundTarget = np.arctan2(rotAnimalPoints[:,1],rotAnimalPoints[:,0])
            # vector with the cumsum of angles around lever
            self.cumSumDiffAngleAroundTarget = np.cumsum(diffAnimalAngleAroundTarget)
            
            # last data point in the cum sum of difference in angle around the target, and range (peak-to-peak)
            if np.sum(~np.isnan(self.cumSumDiffAngleAroundTarget)) > 2:
                self.endCumSumDiffAngleAroundTarget =  self.cumSumDiffAngleAroundTarget[~np.isnan(self.cumSumDiffAngleAroundTarget)][-1] 
                self.rangeCumSumDiffAngleAroundTarget = np.ptp(self.cumSumDiffAngleAroundTarget[~np.isnan(self.cumSumDiffAngleAroundTarget)])
            else:
                self.endCumSumDiffAngleAroundTarget = np.nan
                self.rangeCumSumDiffAngleAroundTarget = np.nan
        
            # add one np.nan to make it the correct size, now was -1 normal size
            self.cumSumDiffAngleAroundTarget = np.hstack([np.array([np.nan]),self.cumSumDiffAngleAroundTarget])
            
        else:
            self.targetDistance = np.zeros_like(self.pPose[:,0])
            self.targetDistance[:]=np.nan
            self.vTargetToAnimal = np.zeros_like(self.pPose[:,0:2])
            self.vTargetToAnimal[:] = np.nan
            
            # default values
            self.entryAngleAroundTarget = np.nan # angle of first data point relative to target
            self.exitAngleAroundTarget = np.nan # angle of the last point relative to target
            self.cumSumDiffAngleAroundTarget = np.zeros_like(self.pPose[:,0]) # vector with the cum sum of difference in angle around the target
            self.cumSumDiffAngleAroundTarget[:] = np.nan
            self.endCumSumDiffAngleAroundTarget = np.nan  # last data point in the cum sum of difference in angle around the target
            self.rangeCumSumDiffAngleAroundTarget = np.nan
            
            
            
            
    def instantaneousBehavioralVariables(self):
        """
        Method returning instantaneous behavioral variables from the path.
        We get as many rows as there are in the self.pPose array
        
        Arguments
        
        Return:
        DataFrame time, internal time, distance run, speed, x, y as columns
        """
        if self.pPose is not None :
            if self.resTime is None:
                return pd.DataFrame({"name" : self.name,
                                     "trialNo": self.trialNo,
                                     "timeRos": self.pPose[:,6],
                                     "iTime": self.internalTime,
                                     "iTimeProp": self.internalTimeProp,
                                     "distance": self.distanceRun,
                                     "distanceProp": self.distanceRunProp,
                                     "speed": self.speed,
                                     "x": self.pPose[:,0],
                                     "y": self.pPose[:,1],
                                     "targetDistance":self.targetDistance,
                                     "targetToAnimalX":self.vTargetToAnimal[:,0],
                                     "targetToAnimalY":self.vTargetToAnimal[:,1],
                                     "targetToAnimalAngle":self.targetToAnimalAngle,
                                     "cumSumDiffAngleAroundTarget":self.cumSumDiffAngleAroundTarget})
            else:
                return pd.DataFrame({"name" : self.name,
                                     "trialNo": self.trialNo,
                                     "timeRos": self.pPose[:,6],
                                     "timeRes": self.resTime,
                                     "iTime": self.internalTime,
                                     "iTimeProp": self.internalTimeProp,
                                     "distance": self.distanceRun,
                                     "distanceProp":self.distanceRunProp,
                                     "speed": self.speed,
                                     "x": self.pPose[:,0],
                                     "y": self.pPose[:,1],
                                     "targetDistance":self.targetDistance,
                                     "targetToAnimalX":self.vTargetToAnimal[:,0],
                                     "targetToAnimalY":self.vTargetToAnimal[:,1],
                                     "targetToAnimalAngle":self.targetToAnimalAngle,
                                     "cumSumDiffAngleAroundTarget":self.cumSumDiffAngleAroundTarget})
    
    def getVariables(self):
        self.myDict = {
            "length" : [self.length],
            "duration" : [self.duration],
            "meanVectorLengthPosi" : [self.meanVectorLengthPosi],
            "meanVectorDirectionPosi": [self.meanVectorDirectionPosi],
            "meanVectorLengthOri" : [self.meanVectorLengthOri[0]],
            "meanVectorDirectionOri" : [self.meanVectorDirectionOri[0]],
            "meanSpeed" : [self.meanSpeed],
            "mvAngularDistance" : [self.mvAngularDistance],
            "mvAngularSpeed" : [self.mvAngularSpeed],
            "oriAngularDistance" : [self.oriAngularDistance[0]],
            "oriAngularSpeed" : [self.oriAngularSpeed[0]],
            "medianMVDeviationToTarget" : [self.medianMVDeviationToTarget],
            "medianHDDeviationToTarget" : [self.medianHDDeviationToTarget],
            "entryAngleAroundTarget" : [self.entryAngleAroundTarget],
            "exitAngleAroundTarget" : [self.exitAngleAroundTarget],
            "endCumSumDiffAngleAroundTarget" : [self.endCumSumDiffAngleAroundTarget],
            "rangeCumSumDiffAngleAroundTarget" : [self.rangeCumSumDiffAngleAroundTarget]
                        
            
        }
        return pd.DataFrame(self.myDict)
                
    
    def unityVectorsFromAngles(self,theta,degree=True):
        if degree:
            theta = theta*np.pi/180
        return np.stack((np.cos(theta),np.sin(theta)),axis =1)

    def rotMatrix(self,point):
        """
        Return a 2D rotation matrix

        Argument 1D np array with x and y coordinate
        """
        a = np.arctan2(point[1],point[0])
        return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
 
            
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
        vLen = np.sqrt(np.nansum(v*v,axis=1))
        vLen[vLen==0] = np.NAN
        rvLen = np.sqrt(np.nansum(rv*rv,axis=1))

        # get unitary vectors
        uv = v/vLen[:,None]
        urv = rv/rvLen[:,None]

        # get the angle, dot product, then acos
        theta = np.arccos(np.clip(np.nansum(uv*urv,axis=1),  -1.0, 1.0))

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
        Calculate the mean vector length
        Arguments:
            theta: angles for which you what the mean vector
            degree: whether we are working in degrees or radians
        Return:
        Mean vector length of theta
        """
        if degree:
            theta = theta*np.pi/180
        # get the unity vectors for these angles
        v = np.stack((np.cos(theta),np.sin(theta)),axis =1)
        mv = np.nansum(v,axis=0)/len(v)
        length = np.sqrt(np.nansum((mv*mv)))
        
        return length
    

    
    def meanVectorDirection(self, theta, degree=True, negativeAngle=False):
        """
        Calculate the mean direction
        Arguments:
            theta: angles for which you what the mean vector
            degree: whether we are working in degrees or radians
            negativeAngle: whether you want angles from -180 to 180 or from 0 to 360
        Return:
        Mean direction of theta
        """
        if degree:
            theta = theta*np.pi/180
        # get the unity vectors for these angles
        v = np.stack((np.cos(theta),np.sin(theta)),axis =1)
        mv = np.nansum(v,axis=0)/len(v)
        
        direction = np.arctan2(mv[1],mv[0])
        if negativeAngle==False:
            if direction < 0:
                direction = 2*np.pi+direction
        if degree:
            direction = direction/np.pi*180
        return direction
    
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    