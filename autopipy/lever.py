import numpy as np
import matplotlib.path as mpltPath

class Lever:
    """
    Class representing a lever in space. It is used the points detected by dlc models to estimate where the lever is
    The class is use to estimate whether a mouse is near the lever or on the lever.
    
    There are 2 ways to estimate if the mouse is at the lever.
    1. We use two zones to estimate if the mouse is at the lever. The entry zone is smaller than the exit zone.
       This was created because there were many false "leaving the lever" events when using one zone for both entering and exiting the lever zone.
       This is somehow arbitrary
      
    2. We use the occupancy histogram of the mouse with the distance from lever on the x-axis. This creates a peak near the lever and occupancy is then flat after 10 cm.
       We use the distribution to set a limit of the Lever zone based on the empirical data.
    
    The lever will be modeled as a rectangle and a triangle.
    
    Attributes:
        scalingFactorEnterZone
        scalingFactorExitZone
        isAtLeaver
        pose
        points
        pointsPlot
        enterZonePoints
        enterZonePointsPlot
        leverPath
        enterZoneLeverPath
        exitZoneLeverPath
        
        
    Methods:
        calculatePose()
        isAt()
        leverDistance()
        plotLever()
        
    """
    def __init__(self):
        self.scalingFactorEnterZone=1.65 # scaling factor to define whether the mouse is at the lever (arriving in the zone)
        self.scalingFactorExitZone=2.00 # scaling factor to define whether the mouse is at the lever (exiting the zone)
        self.scalingFactorSideWalls = 0.8 # length of the side walls relative to the long axis (posterior middle point to lever press)
        self.isAtLever = False
        self.leverZoneMaxDistance = np.nan # maximal distance to be considered at the lever
        self.pose = np.array([[np.nan]*6])
        
    def isAt(self,points, method = "zones"):
        """
        Function to determine if the animal was at the lever. 
        
        Two methods can be used by this function.
        1. Different lever zones are used depending if the animal was or not in the lever zone previously
           The zone to test whether the mouse has left the lever zone is slightly larger than that to test whether the mouse has entered the lever zone
        2. A limit distance is used (usually obtained from the distribution of distance from lever for the whole session)
        
        
        Arguments
            points, np.array [n,2] containing the points to test
            method, string can be set to "zones" or "maxDistance"
        """
        res = np.empty(points.shape[0],dtype=bool) # to return the results
        
        # we have to loop through
        if method == "zones":
            for i in range(points.shape[0]):

                if points[0,0] is None : # keep previous value, assumptionis that we have lost mouse tracking next or on the lever, why are indices [0,0], makes little sense
                    res[i] = self.isAtLever
                else:
                    if self.isAtLever : # check if animal has left
                        v = self.exitZoneLeverPath.contains_points(np.expand_dims(points[i,:],axis=0))
                        if v == False:
                            self.isAtLever=False # mouse is not at the lever
                        res[i] = v
                    else : # check if animal has entered
                        v = self.enterZoneLeverPath.contains_points(np.expand_dims(points[i,:],axis=0))
                        if v == True:
                            self.isAtLever=True # mouse is at the lever
                        res[i] = v
        elif method == "maxDistance":
            if np.isnan(self.leverZoneMaxDistance):
                raise ValueError("set lever.leverZoneMaxDistance before using lever.isAt() with method = 'maxDistance'")
            
            for i in range(points.shape[0]):
                if np.isnan(points[i,0]) or np.isnan(points[i,1]) : # keep previous value, assumptionis that we have lost mouse tracking next or on the lever
                    res[i] = self.isAtLever
                else:
                    self.isAtLever =  self.leverDistance(points[i,:]) < self.leverZoneMaxDistance # is at lever if distance is smaller than self.maxLeverDistance
                    res[i] = self.isAtLever
        
        else:
            raise ValueError("Not a valid method, use 'zones' or 'maxDistance'")
        
        return res
            

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
        #print("leverCenter:",self.leverCenter)
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
        p2 = p1 + vLong * self.scalingFactorSideWalls
        p3 = p0 + vLong
        p5 = p0 - vBase/2
        p4 = p5 + vLong * self.scalingFactorSideWalls
                
        self.points = np.stack((p0,p1,p2,p3,p4,p5))

        # We can scale this up to have an area surrounding the actual lever
        
        self.enterZonePoints = self.points * self.scalingFactorEnterZone
        self.exitZonePoints = self.points * self.scalingFactorExitZone

        # Point 0,0 is the middle of the base (posterior side)
        # We need to translate the points so that they are on the original lever
        # We can align with the lever press 
        self.points = self.points  + (lp-p3)
        self.pointsPlot = np.append(self.points,self.points[0,:]).reshape((-1,2))
        
        # Align with the lever press, then move 
        self.enterZonePoints = self.enterZonePoints + (lp-p3) - (vLong*(self.scalingFactorEnterZone-1))/2
        self.enterZonePointsPlot = np.append(self.enterZonePoints,self.enterZonePoints[0,:]).reshape((-1,2))
        self.exitZonePoints = self.exitZonePoints + (lp-p3) - (vLong*(self.scalingFactorExitZone-1))/2
        self.exitZonePointsPlot = np.append(self.exitZonePoints,self.exitZonePoints[0,:]).reshape((-1,2))
        
        # matplotlib path
        self.leverPath = mpltPath.Path(self.points)
        self.enterZoneLeverPath = mpltPath.Path(self.enterZonePoints)
        self.exitZoneLeverPath = mpltPath.Path(self.exitZonePoints)
        
    def leverDistance(self,P):
        """
        Calculate the distance of point P from the lever

        We get the distance between point and all the segments forming the lever
        The minimal distance found is the lever distance from the point

        """
        
        if np.isnan(P[0]) or np.isnan(P[1]) :
            return np.nan
        
        return np.min([pointDistanceFromSegment(self.points[i],self.points[i+1],P) for i in range(self.points.shape[0]-1)])
    
        
    def rotateVector(self,v,angle,degree=True):
        # for other angles
        # A  cos(theta) -sin(theta)
        #    sin(theta)  cos(theta)
        if degree:
            theta = angle/180*np.pi
        A = np.array( [[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) # matrix to rotate vector
        return np.dot(A,v)
    def plotLever(self,ax=None, zones=True):
        """
        Function to plot the lever on an matplotlib axis
        """
        if ax is None:
            ax = plt.gca()
        
        if hasattr(self, 'pointsPlot'):
            ax.plot(self.pointsPlot[:,0],self.pointsPlot[:,1], color = "gray")
            if zones:
                ax.plot(self.enterZonePointsPlot[:,0],self.enterZonePointsPlot[:,1], color = "gray",linestyle="dotted")
                ax.plot(self.exitZonePointsPlot[:,0],self.exitZonePointsPlot[:,1], color = "gray",linestyle="dotted")

    
            
            
def pointDistanceFromSegment(S0,S1,P):
    """
    Get the distance of point P from a segment going from point S0 to point S1

    S0, S1 and P are 1D numpy vector of length 2

    Not member of the lever class as it is more general and could be used by other classes

    Returns Distance of point from segment
    """

    # get a vector from 0,0 in direction of S0->S1
    S = S1-S0
    # bring the point in the 0,0 reference frame 
    P = P -S0
    # we need the projection of vector P on vector S
    # C is a constant that will scale S to the projection of P on S
    C = S.dot(P)/S.dot(S)

    # we scale vector S by C to get our projection, which is the closest point on vector S from P
    projP = S*C 


    # if C is not between 0 and 1, then the projection is not within our segment,
    # we then have to get the distance from (0,0) or S (end of segment)
    if C < 0:
        ref = np.array([0,0])
    elif C > 1:
        ref = S
    else:
        ref = projP

    return np.sqrt(np.sum((P-ref)**2)) # get the distance

