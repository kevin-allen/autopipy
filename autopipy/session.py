import os.path
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from autopipy.trial import Trial
from autopipy.trialElectro import TrialElectro
from tqdm import tqdm

class Session:
    """
    Class containing information about an autopi session
    
    Attributes:
        name: Name of the session. Usually used as the beginning of the file names
        path: Directory path of the data for this session
        subject: Name of the subect. This assumes that the session name is in the format subject-date-time
        sessionDateTime: Datetime of the session. This assumes that the session name is in the format subject-date-time
        fileBase: path + name
        arenaTopVideo: Boolean indicating whether we should expect a video for the arnea
        homeBaseVideo:  Boolean indicating whether we should have a video of the home base
        requiredFileExts: List containing the extensions of the file we should have in the directory
        arenaTopCropped: Boolean indicating whether we have an arena top cropped video
        dataFileCheck: Boolean indicating whether to test for the presence of data file in the session directory
        fileNames: Dictionary to get important file names
        log Data frame with the content of the log file
        mousePose: DataFrame with time, (resTime), x, y, hd
        
    Methods:
        checkSessionDirectory()
        loadLogFile()
        loadPositionTrackingData()
        segmentTrialsFromLog()
        extractTrialFeatures(): Use this function to extract the trial features
        createTrialList()
        createTrialVideos()
    """
    def __init__(self,name,path,arenaTopVideo=True,homeBaseVideo=True, dataFileCheck=True):
        self.name = name
        self.path = path
        self.subject = self.name.split("-")[0]
        self.sessionDateTime = datetime.strptime(self.name.split("-")[1]+self.name.split("-")[2], '%d%m%Y%H%M')
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
        self.dirOk=False
        if dataFileCheck:
            if self.checkSessionDirectory():
                self.dirOk=True
            else:
                print("problem with the directory " + self.path)

    
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
        
        ################################################################
        ### other attributes that can be set using different methods  ##
        ################################################################
        self.log = None # pandas dataframe, content of the log file
        self.mouseLeverPosi = None # pandas dataframe, position of mouse and lever
        self.videoLog = None # pandas dataframe, time point for every video frame that were used to get the mouse and lever position
        self.arenaCoordinates = None # numpy array containing the arenaCoordinates (x,y,r)
        self.bridgeCoordinates = None # numpy array containing the 4 corners of the bridge, one corner per row 
        self.trials = None # pandas dataframe containing trial start and end time 
        self.trialList = None # list of Trial objects
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
    
    def loadProtocolFile(self):
        """
        Load the protocol file into a Pandas DataFrame and add session name, subject and datatime
        
        """
        self.protocol = pd.read_csv(self.fileNames["protocol"],header = None,sep=" ", names = ["script","duration","master","ip"])
        self.protocol["session"] = self.name
        self.protocol["subject"] = self.subject
        self.protocol["sessionDateTime"] = self.sessionDateTime
        
    
    def loadLogFile(self):
        """
        Load the log file into a Pandas DataFrame
        """
        self.log = pd.read_csv(self.fileNames["log"],sep=" ")
        
        # check that there is one start and end events
        if len(self.log[self.log.event=="start"]) != 1:
            raise RuntimeError("There should be one and only one start event in {}".format(self.fileNames["log"])) # raise an exception
        if len(self.log[self.log.event=="end"]) != 1:
            raise RuntimeError("There should be one and only one end event in {}".format(self.fileNames["log"])) # raise an exception
    
    def logProcessing(self):
        """
        Do 3 simple steps with the log data
        1) remove anything before start
        2) remove anything after end
        3) add a time column relative to the start event
        """
        
        startIndex = self.log[self.log["event"]=="start"].index.values[0]
        endIndex = self.log[self.log["event"]=="end"].index.values[0]
        self.log = self.log.loc[startIndex:endIndex]
        self.log["timeWS"] = self.log["time"]-self.log.time.iloc[0]
        self.log["session"] = self.name
        self.log["subject"] = self.subject
        self.log["sessionDateTime"] = self.sessionDateTime
        
    def loadPositionTrackingData(self):
        self.mouseLeverPosi = pd.read_csv(self.fileNames["mouseLeverPosition.csv"])
        self.videoLog = pd.read_csv(self.fileNames["arena_top.log"], delimiter=" ")
        self.arenaCoordinates = np.loadtxt(self.fileNames["arenaCoordinates"])
        self.bridgeCoordinates = np.genfromtxt(self.fileNames["bridgeCoordinates"],delimiter=",")
        
        print("Lenght of mouseLeverPosi: {}".format(len(self.mouseLeverPosi)))
        print("Lenght of videoLog: {}".format(len(self.videoLog)))
    
        if len(self.mouseLeverPosi) != len(self.videoLog):
            print("Length of mouseLeverPosi ({}) is not equal to videoLog ({})".format(len(self.mouseLeverPosi),len(self.videoLog)))
            print("Problem with the length of the arena_top.avi and arena_top.log")
            newDf = self.fixVideoLog(self.videoLog)
            self.videoLog = newDf
            
    def testVideoLogSynchronization(self):
        """
        
        Method to assess whether the video and the video log are synchronized and whether there are problems associated with these files
        
        Test if we have the same number of video frames as entries in the video log
        Test if we have all the frames in the log
        Test whether there are long gap between frames
        Test what is the sampling rate of the video
        """
        
        cap = cv2.VideoCapture(self.fileNames["arena_top.avi"])
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release() 
        
        self.videoLog = pd.read_csv(self.fileNames["arena_top.log"], delimiter=" ") # One row per frame, presumably
        
        self.syncrhoDifference = nFrames-len(self.videoLog.time)
        self.synchroProblemIndices = self.videoLog.index[self.videoLog.frame_number.diff()!=1]
        self.synchroFirstFrame = self.videoLog.frame_number[0]
        self.synchroGapLengths = self.videoLog.frame_number.diff()[self.videoLog.frame_number.diff()>1]
        self.synchroMeanFrameTimeDiff=self.videoLog.time.diff().mean()
        self.synchroMaxFrameTimeDiff=self.videoLog.time.diff().max()
        self.synchroProblemTimeDiff = np.sum(self.videoLog.time.diff()>0.25)
        self.synchroMeanFrameRate= nFrames/(self.videoLog.time.max()-self.videoLog.time.min())
        print("{}, video len: {}, video-log len:{}, first frame: {}, max log gap: {}, mean time diff: {:.3}, max time diff: {:.3}, num problem diff: {}, frame rate: {:.3}".format(self.name, nFrames, len(self.videoLog), self.synchroFirstFrame, self.synchroGapLengths.max(), self.synchroMeanFrameTimeDiff, self.synchroMaxFrameTimeDiff,
                                                                                              self.synchroProblemTimeDiff,
                                                                                              self.synchroMeanFrameRate))
    def fixVideoLog(self,vl):
        removeCount=0
        addCount=0
        while np.sum(vl.frame_number.diff() != 1) > 1:

            problemIndices = vl.index[vl.frame_number != vl.index]
            firstProblemIndex = problemIndices[0]
            print("first problem index: {}".format(firstProblemIndex))
            print(vl.loc[(firstProblemIndex-1):(firstProblemIndex+1),:])
            if vl.loc[firstProblemIndex,"frame_number"] < firstProblemIndex :
                print("Duplicate frames in the Videolog at index {}".format(firstProblemIndex))
            else :
                print("Missing frame in the Videolog at index {}".format(firstProblemIndex))
                print("Inserting the missing entry")
                # get data frame before
                df1 = vl.loc[:firstProblemIndex-1,:]
                # get one line dataframe
                df2 = pd.DataFrame({"frame_number" : [vl.frame_number[firstProblemIndex]-1],
                               "time" : [vl.time[firstProblemIndex-1]+ vl.time[firstProblemIndex] - vl.time[firstProblemIndex-1] ]  })
                # get data frame after
                df3 = vl.loc[firstProblemIndex:,:]
                # concatenate data frames
                dfNew = pd.concat([df1,df2,df3])
                print("fixed DataFrame")
                dfNew = dfNew.set_index(np.arange(len(dfNew.time)))
                print(dfNew.loc[(firstProblemIndex-1):(firstProblemIndex+1),:])
                addCount=addCount+1
                vl = dfNew

                problemIndices = vl.index[vl.frame_number != vl.index]
                print("Problem indices after fix: {}".format(len(problemIndices)))
        print("Removed {} rows and added {} rows".format(removeCount,addCount))
        return vl
    
    def segmentTrialsFromLog(self, verbose=True):
        """
        Identify the begining and end of each trial from the log file
        
        A trial in this function is defined by the opening and closing of the home base door. 
        
        The trial starts when the door opens and ends when the door closes.
        
        We don't do any analysis of the trial here. Such analysis is done with the trial class.
        
        A pandas data frame called `trials` will be stored as a session attribute
        
        Arguments:
        
        verbose: Boolean indicating wheter to print information
        
        """
        
        if verbose:
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
            if verbose:
                print("Remove door closing event at the beginning of the session, index {}".format(doorEvents.index[0]))
            doorEvents = doorEvents.drop(doorEvents.index[0]) # remove first row

        # make sure it ends with closing the door (value of 1)
        if doorEvents.param.iloc[-1] == 0:
            if verbose:
                print("Remove last opening of the door")
            doorEvents = doorEvents[:-1] # drop last row

        # make sure that the door alternates between up and down states.
        diffDoor = doorEvents.param.diff() # if == 0 we have repeat of the same door command
        if any(diffDoor == 0): # this is a problem
            if verbose:
                print("problem with the door alternation")
            for probIndex in doorEvents.index[diffDoor==0]: # loop over problematic indices
                if verbose:
                    print("Problem with index {}".format(probIndex))
                if(doorEvents.param.loc[probIndex]==0):
                # remove the index before 
                    indexBefore = doorEvents[doorEvents.time<doorEvents.time.loc[probIndex]].tail(1).index
                    if verbose:
                        print("Removing the first of two door openings (index :{})".format(indexBefore))
                    doorEvents = doorEvents.drop(indexBefore)
                else :
                    if verbose:
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

        if verbose:
            print("Number of trials : {}".format(len(trials)))
  
        self.trials = trials

    def createTrialList(self):
        """
        Create a list of Trial objects
        
        Use the trials attribute that was generated by the function segmentTrialsFromLog
        
        """
        def getTrialFromSeries(x):
            return Trial(x.sessionName,x.trialNo,x.startTime,x.endTime,x.startTimeWS,x.endTimeWS)
        self.trialList = self.trials.apply(getTrialFromSeries,axis=1).tolist()
        
    def createTrialElectroList(self,arenaRadiusCm=40):
        """
        Create a list of TrialElectro objects
        
        """
        
        def getTrialFromSeries(x):
            return TrialElectro(x.sessionName,x.trialNo,x.startTime,x.endTime,x.startTimeWS,x.endTimeWS)
        self.trialList = self.trials.apply(getTrialFromSeries,axis=1).tolist()
        
        #for trial in self.trialList:
        #    trial.
        
    def getTrial(self,trialNo):
        """
        Get a single trial from the trial list using the trial number
        """
        if self.trialList is None:
            print("Create the trialList before calling session.getTrial()")
            return None
        
        return [t for t in self.trialList if t.trialNo==trialNo][0]
    
    def extractTrialElectroFeatures(self,mousePose,verbose=False):
        """
        Extract trial features for session with electrophysiology
        
        Arguments: 
        mousePose: numpy array, the spikeA.AnimalPose.Pose
        
        The leverPose is generated from the lever_pose.ipynb` notebook
        
        """
        if verbose:
            print("Mouse pose format:", mousePose.shape)
        self.mousePose = pd.DataFrame({"time":mousePose[:,7], # ROS time
                                       "resTime":mousePose[:,0], # rec time
                                       "x":mousePose[:,1],
                                       "y":mousePose[:,2],
                                       "hd":mousePose[:,4]})
        self.leverPose = pd.read_csv(self.path+"/leverPose",index_col = False) 
        
        for trial in self.trialList:
            trial.extractTrialFeatures(arenaCoordinatesFile=self.fileNames["arenaCoordinates"],
                                       bridgeCoordinatesFile = self.fileNames["bridgeCoordinates"],
                                       log = self.log,
                                       mousePose = self.mousePose,
                                       leverPose = self.leverPose,verbose=verbose)
    
        self.nJourneys = np.sum([ t.nJourneys for t in self.trialList])
        self.nTrials = len(self.trialList)
    
    def navPathIntervals(self):
        """
        Function returning a DataFrame with the start and end time of each NavPaths 

        All NavPaths are included in the DataFrame.

        Return
        DataFrame with the name, trial, journey, type, light, nLeverPresses, startTimeRos, endTimeRos of each NavPath
        """
        myList=[]
        for t in self.trialList:
            for j in t.journeyList:
                for k in j.navPaths.keys():
                   
                    df = pd.DataFrame({"name": [j.navPaths[k].name],
                                       "trial": [t.name],
                                       "trialNo": [t.trialNo],
                                       "journey": [j.journeyNo],
                                       "type": [k],
                                       "light": [t.light],
                                       "nLeverPresses": [j.nLeverPresses],
                                 "startTimeRos": [j.navPaths[k].startTime],
                                 "endTimeRos": [j.navPaths[k].endTime]})
                    myList.append(df)

        
        return pd.concat(myList)

    def navPathInstantaneousBehavioralData(self):
        """
        Function returning the instantaneous behavioral data of all NavPaths 

        Arguments:
        
        Return
        DataFrame with a bunch of behavioral variables varying in time
        """
        myList=[]
        for t in self.trialList:
            for j in t.journeyList:
                for k in j.navPaths.keys():
                    myList.append(j.navPaths[k].instantaneousBehavioralVariables())
                            
        return pd.concat(myList)
    
    
    def extractTrialFeatures(self):
        """
        Does most of the job of extracting information of the trials in our trialList
        """
        self.segmentTrialsFromLog() # get the beginning and end of trials
        self.createTrialList() # create a List of trial object
        self.testVideoLogSynchronization()
        self.loadLogFile()
        self.loadPositionTrackingData()
        for trial in self.trialList:
            trial.extractTrialFeatures(log = self.log,
                                       mLPosi = self.mouseLeverPosi,
                                       videoLog = self.videoLog,
                                       aCoord = self.arenaCoordinates,
                                       bCoord = self.bridgeCoordinates)   
    def getTrialVariablesDataFrame(self):
        """
        Get a DatatFrame with the variables of each trial, one row per trial
        
        We will add the subject name the the data frame as it is useful when analysing data at the project level.
        
        """
        dfList = [trial.getTrialVariables() for trial in self.trialList]
        self.trialVariables = pd.concat(dfList)
        self.trialVariables["subject"]=self.subject # add the subject name to the dataframe
        self.trialVariables["date"]= self.sessionDateTime # add the data to the data frame
    
    def getTrialPathSpeedProfile(self,pathName="searchTotal"):
        """
        Get a numpy array containing the speed profile of a given path
        """
        aList = [ trial.getSpeedProfile(pathName) for trial in self.trialList]
        self.speedProfile = np.stack(aList, axis=0)
    
    def createTrialPlots(self):
        """
        Create trial plots in a trialPlots directory
        """
        trialPlotsDir = self.path+"/trialPlots"
        print("Saving plots in "+trialPlotsDir)
        if not os.path.exists(trialPlotsDir) :
            try:
                os.mkdir(trialPlotsDir)
            except OSError:
                print ("Creation of the directory %s failed" % trialPlotsDir)
                return
            else:
                print ("Successfully created the directory %s " % trialPlotsDir)
        
        for trial in self.trialList:
            trial.trialPathFigure(filePath = "{}/{}.trial_{}.png".format(trialPlotsDir,self.name,trial.trialNo))
                                  
                                   
    
    def createTrialVideos(self,decorate=True):
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
                                   pathVideoFileOut = "{}/{}.trial_{}.avi".format(trialVideosDir,self.name,trial.trialNo),
                                   decorate=decorate)
        
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    def trialPathFigure(self,fileName=None,arenaRadius = 40,arenaRadiusProportionToPeri=0.925):
        """
        Create a single figure with all trial paths. All of the trials are put on top of each other
        
        It is a great way to double-check that the trial analysis when as expected.
        
        Arguments:
            fileName: file in which to save the figure, by default it goes in self.fileBase+"_trialPaths.pdf"
            arenaRadius: radius of the arena in cm
            arenaRadiusProportionToPeri: proportion of the arena radius where the periphery of the arena is defined
        """
        # to plot the arena circle
        arena=np.arange(start=0,stop=2*np.pi,step=0.02)
        
        
        fig, axes = plt.subplots(2,4,figsize=(20,10))
        plt.subplots_adjust(wspace=0.3,hspace=0.3)

        # what needs to be applied to all graphs
        for ax in axes.flatten():
            ax.set_aspect('equal', adjustable='box')
            ax.plot(np.cos(arena)*arenaRadius,np.sin(arena)*arenaRadius,label="Arena",color="gray")
            ax.plot(np.cos(arena)*arenaRadius*arenaRadiusProportionToPeri,np.sin(arena)*arenaRadius*arenaRadiusProportionToPeri,label="Periphery",color="gray",linestyle='dashed')
            ax.set_xlabel("cm")
            ax.set_ylabel("cm")
        axes[0,0].set_title("Search-light paths")
        axes[0,1].set_title("Search-light no lever paths")
        axes[0,2].set_title("Search-dark paths")
        axes[0,3].set_title("Search-dark no lever paths")

        axes[1,0].set_title("Homing-light paths")
        axes[1,1].set_title("Homing-light to peri paths")
        axes[1,2].set_title("Homing-dark paths")
        axes[1,3].set_title("Homing-dark to peri paths")
        
        
        lightTrials = [t for t in self.trialList if t.light=="light" and t.valid]
        darkTrials =  [t for t in self.trialList if t.light=="dark" and t.valid]

        for t in lightTrials:
            axes[0,0].plot(t.searchTotalNavPath.pPose[:,0],t.searchTotalNavPath.pPose[:,1])

        for t in lightTrials:
            axes[0,1].plot(t.searchArenaNoLeverNavPath.pPose[:,0],t.searchArenaNoLeverNavPath.pPose[:,1])

        for t in darkTrials:
            axes[0,2].plot(t.searchTotalNavPath.pPose[:,0],t.searchTotalNavPath.pPose[:,1])

        for t in darkTrials:
            axes[0,3].plot(t.searchArenaNoLeverNavPath.pPose[:,0],t.searchArenaNoLeverNavPath.pPose[:,1])

        for t in lightTrials:
            axes[1,0].plot(t.homingTotalNavPath.pPose[:,0],t.homingTotalNavPath.pPose[:,1])

        for t in lightTrials:
            axes[1,1].plot(t.homingPeriNoLeverNavPath.pPose[:,0],t.homingPeriNoLeverNavPath.pPose[:,1])

        for t in darkTrials:
            axes[1,2].plot(t.homingTotalNavPath.pPose[:,0],t.homingTotalNavPath.pPose[:,1])

        for t in darkTrials:
            axes[1,3].plot(t.homingPeriNoLeverNavPath.pPose[:,0],t.homingPeriNoLeverNavPath.pPose[:,1])


        if fileName is None:
            fileName = self.fileBase+"_trialPaths.pdf"
        print("Saving trialPaths in " + fileName)
        plt.savefig(fileName)
    def plotTrialsWithJourneys(self,fileName):
        """
        Method to plot all trials of a session with the extracted journeys
        
        The main purpose is to visualize trials/journeys for potential problems with trial/journey extraction.
        
        There is one row per trial.
        The first column is the complete trial.
        The following rows (up to 5) are the individual journeys performed by the mouse.
        
        The figure is saved into a pdf file.
        
        Argument:
        fileName: name of the file to save the pdf
        """
        
        print("Generating",fileName)
        nRow=8
        nCol=5

        with PdfPages(fileName) as pdf:
            for i in tqdm(range(len(self.trialList))):
                trial = self.trialList[i]
                currentRow = i%nRow
                if currentRow == 0:
                    fig, axes = plt.subplots(nRow,nCol,figsize=(12,4*nRow))

                trial.plotTrialAndJourneys(axes[currentRow,:])

                if currentRow == nRow-1:
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()
