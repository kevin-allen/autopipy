import os.path
import pandas as pd
import numpy as np

class session:
    """
    Class containing information about an autopi session
    
    Attributes:
        name    Name of the session. Usually used as the beginning of the file names
        path    Directory path of the data for this session
        fileBase path plus name
        arenaTopVideo Boolean indicating whether we should expect a video for the arnea
        homeBaseVide  Boolean indicating whether we should have a video of the home base
        requiredFileExts List containing the extensions of the file we should have in the directory
        arenaTopCropped Boolean indicating whether we have an arena top cropped video
        dataFileCheck Boolean indicating whether to test for the presence of data file in the session directory
        log Data frame with the content of the log file
    
    Methods:
        checkSessionDirectory():
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
        
        # create a dictonary to quickly get the file names.
        self.fileNames = {"log": self.fileBase+".log",
                         "protocol": self.fileBase+".protocol",
                         "arena_top.avi": self.fileBase+".arena_top.avi",
                         "arena_top.cropped.avi": self.fileBase+".arena_top.cropped.avi"}
        
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
    
    def segmentTrialsFromLog(self):
        """
        Identify the begining and end of each trial from the log file
        
        A trial is define by the opening and closing of the home base door. 
        
        The trial starts when the door opens and ends when the door closes.
        
        
        A pandas data frame called trials will be stored as a session attribute
        """
        
        print("{} trial segmentation".format(self.name))
        self.log = pd.read_csv(self.fileNames["log"],sep=" ")
        
        
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

        # create a data frame for the trials
        myDict = {"start_time" : doorEvents.time[doorEvents.param==0].array,
         "end_time" : doorEvents.time[doorEvents.param==1].array,
         "trial_no" : range(1,len(doorEvents.time[doorEvents.param==0])+1),
         "start_time_ws" : doorEvents.time[doorEvents.param==0].array-self.log.time[self.log.event=="start"].array,
         "end_time_ws" : doorEvents.time[doorEvents.param==1].array-self.log.time[self.log.event=="start"].array,
         "duration" : doorEvents.time[doorEvents.param==1].array - doorEvents.time[doorEvents.param==0].array}
        trials = pd.DataFrame(data = myDict)

        print("Number of trials : {}".format(len(trials)))
        
        def previousLight(x,lightEvents):
            """
            Get the last light that was set before the current trial
            """
            if sum(lightEvents.time < x) == 0 :
                return np.nan
            else:
                return lightEvents.param[lightEvents.time< x].tail(1).to_numpy()[0]
        def lightFromCode(x):
            """
            Get light or dark depending on light_code
            """
            if x == 1 or np.isnan(x):
                return "light"
            else:
                return "dark"
    
        lightEvents = self.log[self.log.event=="light"]
        trials["light_code"] = trials.start_time.apply(previousLight,args=(lightEvents,))
        trials["light"] = trials.light_code.apply(lightFromCode)
    
        def leverMissing(tr,leverEvents):
            """
            Check if there is a lever press during the trials

            Return a boolean, True if missing
            """  
            if np.logical_and(leverEvents.time > tr["start_time"],
                              leverEvents.time < tr["end_time"]).any():
                return False
            else :
                return True

        leverEvents = self.log[np.logical_or(self.log.event=="lever_press", self.log.event=="leverPress")]
        
        missTrials = trials.apply(func = leverMissing,axis=1,args = (leverEvents,))
        if missTrials.any():
            print("There was(were) {} trial(s) without lever press".format(missTrials.sum()))
        trials["lever_pressed"] = ~missTrials # reverse True to False
    
    
    
        self.trials = trials

    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    