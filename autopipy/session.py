import os.path
import os
import pandas as pd
import numpy as np
from autopipy.trial import trial
class session:
    """
    Class containing information about an autopi session
    
    Attributes:
        name: Name of the session. Usually used as the beginning of the file names
        path: Directory path of the data for this session
        fileBase: path + name
        arenaTopVideo: Boolean indicating whether we should expect a video for the arnea
        homeBaseVideo:  Boolean indicating whether we should have a video of the home base
        requiredFileExts: List containing the extensions of the file we should have in the directory
        arenaTopCropped: Boolean indicating whether we have an arena top cropped video
        dataFileCheck: Boolean indicating whether to test for the presence of data file in the session directory
        fileNames: Dictionary to get important file names
        log Data frame with the content of the log file
        
    Methods:
        checkSessionDirectory()
        loadLogFile()
        loadPositionTrackingData()
        segmentTrialsFromLog()
        extractTrialFeatures()
        createTrialList()
        createTrialVideos()
    """
    def __init__(self,name,path,arenaTopVideo=True,homeBaseVideo=True, dataFileCheck=True):
        self.name = name
        self.path = path
        self.fileBase = path+"/"+name
        self.arenaTopVideo = arenaTopVideo
        self.homeBaseVideo = homeBaseVideo
        
        self.requiredFileExts = ["log","protocol"]
        if self.arenaTopVideo:
            self.requiredFileExts.append("arena_top.avi")
            self.requiredFileExts.append("arena_top.log")
            
        if self.homeBaseVideo:
            self.requiredFileExts.append("home_base.avi")
            self.requiredFileExts.append("home_base.log")
            
        # check that we have valid data
        if dataFileCheck:
            if self.checkSessionDirectory():
                self.dirOk=True
            else:
                print("problem with the directory " + self.path)
                self.dirOk=False
    
        # check if we have an arena_top.cropped.avi file
        self.arenaTopCropped=False
        if self.arenaTopVideo:
            if os.path.isfile(self.fileBase + "." + "arena_top.cropped.avi"):
                self.arenaTopCropped=True  
        
        #####################################################
        # create a dictonary to quickly get the file names ##
        #########################################################
        # all file names with important data should be set here #
        #########################################################
        self.fileNames = {"log": self.fileBase+".log",
                         "protocol": self.fileBase+".protocol",
                         "arena_top.avi": self.fileBase+".arena_top.avi",
                         "arena_top.log": self.fileBase+".arena_top.log",
                         "arena_top.cropped.avi": self.fileBase+".arena_top.cropped.avi",
                         "mouseLeverPosition.csv": self.fileBase+".mouseLeverPosition.csv",
                         "arenaCoordinates": self.path+"/"+"arenaCoordinates",
                         "bridgeCoordinates": self.path+"/"+"bridgeCoordinates"} 
        return
        
    def checkSessionDirectory(self):
        # check that the directory is there
        if os.path.isdir(self.path) == False :
            raise IOError(self.path + " does not exist") # raise an exception
            
        # check that the files needed are there
        for ext in self.requiredFileExts:
            fileName = self.fileBase + "." + ext
            if os.path.isfile(fileName)== False:
                print(fileName + " does not exist")
                raise IOError(fileName + " does not exist") # raise an exception
        return True
    
    def loadLogFile(self):
        self.log = pd.read_csv(self.fileNames["log"],sep=" ")
    
    def loadPositionTrackingData(self):
        self.mouseLeverPosi = pd.read_csv(self.fileNames["mouseLeverPosition.csv"])
        self.videoLog = pd.read_csv(self.fileNames["arena_top.log"], delimiter=" ")
        self.arenaCoordinates = np.loadtxt(self.fileNames["arenaCoordinates"])
        self.bridgeCoordinates = np.genfromtxt(self.fileNames["bridgeCoordinates"],delimiter=",")
    
    
    def segmentTrialsFromLog(self):
        """
        Identify the begining and end of each trial from the log file
        
        A trial is define by the opening and closing of the home base door. 
        
        The trial starts when the door opens and ends when the door closes.
        
        We don't do any analysis of the trial here. Such analysis is done with the trial class.
        
        A pandas data frame called `trials` will be stored as a session attribute
        """
        
        print("{} trial segmentation".format(self.name))
        self.log = pd.read_csv(self.fileNames["log"],sep=" ")
        
        
        # check that there is one start and end events
        if len(self.log[self.log.event=="start"]) != 1:
            raise RuntimeError("There should be one and only one start event in the log file") # raise an exception
        if len(self.log[self.log.event=="end"]) != 1:
            raise RuntimeError("There should be one and only one end event in the log file") # raise an exception
            
        # check that there are door events in the log
        if ~(self.log.event.unique()=="door").any() :
            raise RuntimeError("There should be door events in the log file") # raise an exception
        
        
        # get the door events after the start of the session
        doorEvents = self.log[np.logical_and(self.log.event=="door" ,
                                             (self.log.time - self.log.time[self.log.event=="start"].to_numpy()) > 0)]

        # make sure it starts with opening the door (value of 0)
        while doorEvents.param.iloc[0] == 1:
            print("Remove door closing event at the beginning of the session, index {}".format(doorEvents.index[0]))
            doorEvents = doorEvents.drop(doorEvents.index[0]) # remove first row

        # make sure it ends with closing the door (value of 1)
        if doorEvents.param.iloc[-1] == 0:
            print("Remove last opening of the door")
            doorEvents = doorEvents[:-1] # drop last row

        # make sure that the door alternates between up and down states.
        diffDoor = doorEvents.param.diff() # if == 0 we have repeat of the same door command
        if any(diffDoor == 0): # this is a problem
            print("problem with the door alternation")
            for probIndex in doorEvents.index[diffDoor==0]: # loop over problematic indices
                print("Problem with index {}".format(probIndex))
                if(doorEvents.param.loc[probIndex]==0):
                # remove the index before 
                    indexBefore = doorEvents[doorEvents.time<doorEvents.time.loc[probIndex]].tail(1).index
                    print("Removing the first of two door openings (index :{})".format(indexBefore))
                    doorEvents = doorEvents.drop(indexBefore)
                else :
                    print("Removing the second of two door closings (index :{})".format(probIndex))
                    doorEvents = doorEvents.drop(probIndex)

        #######################################
        # create a data frame for the trials  #
        #######################################
        myDict = {"sessionName" : [self.name for i in range(sum(doorEvents.param==0))],
                  "trialNo" : range(1,len(doorEvents.time[doorEvents.param==0])+1),
                  "startTime" : doorEvents.time[doorEvents.param==0].array,
                  "endTime" : doorEvents.time[doorEvents.param==1].array,
                  "startTimeWS" : doorEvents.time[doorEvents.param==0].array-self.log.time[self.log.event=="start"].array,
                  "endTimeWS" : doorEvents.time[doorEvents.param==1].array-self.log.time[self.log.event=="start"].array}
        
        trials = pd.DataFrame(data = myDict)

        print("Number of trials : {}".format(len(trials)))
  
        self.trials = trials

    def createTrialList(self):
        """
        Create a list of trial object
        
        Use the trials attribute that was generated by the function segmentTrialsFromLog
        
        """
        def getTrialFromSeries(x):
            return trial(x.sessionName,x.trialNo,x.startTime,x.endTime,x.startTimeWS,x.endTimeWS)
        self.trialList = self.trials.apply(getTrialFromSeries,axis=1).tolist()
    
    def extractTrialFeatures(self):
        """
        Does most of the job of extracting information of the trials in our trialList
        """
        self.segmentTrialsFromLog() # get the beginning and end of trials
        self.createTrialList() # create a List of trial object
        self.loadLogFile()
        self.loadPositionTrackingData()
        for trial in self.trialList:
            trial.extractTrialFeatures(log = self.log,
                                       mLPosi = self.mouseLeverPosi,
                                       videoLog = self.videoLog,
                                       aCoord = self.arenaCoordinates,
                                       bCoord = self.bridgeCoordinates)   
    def createTrialVideos(self):
        """
        Create trial videos in a trialVideos directory
        """
        trialVideosDir = self.path+"/trialVideos"
        print("Saving videos in "+trialVideosDir)
        if not os.path.exists(trialVideosDir) :
            try:
                os.mkdir(trialVideosDir)
            except OSError:
                print ("Creation of the directory %s failed" % trialVideosDir)
                return
            else:
                print ("Successfully created the directory %s " % trialVideosDir)
        
        for trial in self.trialList:
            trial.createTrialVideo(pathVideoFile = self.fileNames["arena_top.cropped.avi"],
                                   pathVideoFileOut = "{}/{}.trial_{}.avi".format(trialVideosDir,self.name,trial.trialNo))
        
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    