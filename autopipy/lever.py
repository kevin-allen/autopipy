import numpy as np
import matplotlib.path as mpltPath

class Lever:
    """
    Class representing a lever in space. It is used the points detected by dlc models to estimate where the lever is
    The class is use to estimate whether a mouse is near the lever or on the lever.
    The lever will be modeled as a rectangle and a triangle.
    
    Attributes:
        points: np.array of the points to trace the countour of the lever
        zonePoints: np.array of points, scalled up version of the lever countour to establish if the mouse is at the lever
        pose: np.array with the x y z yaw pitch roll of the lever (middle point)
        leverPath: matplotlib path of the lever
        leverZonePath = matplotlib path of the lever zone
        
    Methods:
        calculatePose()
        isAt()
        
    """
    def __init__(self):
        self.scallingFactorLarge=1.65 # scaling factor to define the lever zone
        self.scallingFactorSideWalls = 0.8 # length of the side walls relative to the long axis (posterior middle point to lever press)
        
    def isAt(self,points):
        """
        Function to establish if a point is within the lever zone defined by the zonePoints polygone
        Arguments
            points, np.array [n,2] containing the points to test
        """
        return self.leverZonePath.contains_points(points)
    
    def calculatePose(self,lp=None,pl=None,pr=None):
        """
        The inputs are 3  1D np.arrays of length 2. They correspond to the following points
            LPX LPY  x and y coordinate of the lever press
            PLX PLY x and y coordinates of the posterior left corner of the lever box, anterior is the press
            PRX PRY x and y coordinates of the posterior right corner of the lever box
            
        Two polygones are created
        slef.points : points following the lever
        self.zonePoints: points surrounding the lever, scalled up shape of the lever to establish if the mouse is around the lever
        """
        self.lp=lp
        self.pl=pl
        self.pr=pr
        
        ###################################
        # calculate the pose of the lever #
        ###################################
        
        ## middle point at the back of the lever (it does not matter if the two points are swapped, which is good)
        self.p4 = (pl+pr)/2
        
        ## lever position (middle point)
        self.leverCenter = (lp+self.p4)/2
        
        ## vector from middle point to tip of lever
        vCenterToPress= lp-self.leverCenter
        
        ## get the angle of this vector relative to 1,0
        ## make it a unitary vector, get the length
        l = np.sqrt(np.dot(vCenterToPress,vCenterToPress))
        u = vCenterToPress/l
        ## get the angle between 1,0 and our unitory vector
        theta = np.arctan2(u[1],u[0])
        if theta<0:
            theta = 2*np.pi+theta
            
        self.theta_deg = theta/(2*np.pi)*360
        self.pose = np.array([[self.leverCenter[0],self.leverCenter[1],0,self.theta_deg,0,0]]) # x y z yaw pitch roll
        
        ###########################################
        # determine the area covered by the lever #
        ###########################################
        # we create a polygone that should match the lever shape
        # we start at 0,0 to draw
         
        # The best anchor for determine the lever position is the long vector from the middle point of the vPosterior to the lever press
        
        # vector from pr to pl
        vPosterior=pl-pr # vector
        vPosteriorLength = np.sqrt(np.dot(vPosterior,vPosterior)) # length of vector
                
        # mid point of vPosterior
        midPosterior = pr + vPosterior/2
        
        # the longest vector to draw relative to
        vLong = lp-midPosterior # long vector, from midPosterior to leverpress
        vLongLength = np.sqrt(np.dot(vLong,vLong)) # length
        uvLong = vLong/vLongLength # unit vector
        
        # we will align everything to the vLong vector,
        # we first need the 2 perpendicular vectors with a length of vPosteriorLength/2
        uBase = self.rotateVector(uvLong,90)
        vBase = uBase*vPosteriorLength # give it the length of vPosterior
             
        # our 5 points forming the lever
        p0 = np.array([0,0])
        p1 = p0 + vBase/2
        p2 = p1 + vLong * self.scallingFactorSideWalls
        p3 = p0 + vLong
        p5 = p0 - vBase/2
        p4 = p5 + vLong * self.scallingFactorSideWalls
                
        self.points = np.stack((p0,p1,p2,p3,p4,p5))

        # We can scale this up to have an area surrounding the actual lever
        
        self.zonePoints = self.points * self.scallingFactorLarge
        
        
        # Point 0,0 is the middle of the base (posterior side)
        # We need to translate the points so that they are on the original lever
        # We can align with the lever press 
        self.points = self.points  + (lp-p3)
        
        self.pointsPlot = np.append(self.points,self.points[0,:]).reshape((-1,2))
        
        # Align with the lever press, then move 
        self.zonePoints = self.zonePoints + (lp-p3) - (vLong*(self.scallingFactorLarge-1))/2
        
        self.zonePointsPlot = np.append(self.zonePoints,self.zonePoints[0,:]).reshape((-1,2))
        
        self.leverPath = mpltPath.Path(self.points)
        self.leverZonePath = mpltPath.Path(self.zonePoints)
        
    def rotateVector(self,v,angle,degree=True):
        # for other angles
        # A  cos(theta) -sin(theta)
        #    sin(theta)  cos(theta)
        if degree:
            theta = angle/180*np.pi
        A = np.array( [[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) # matrix to rotate vector
        return np.dot(A,v)
        