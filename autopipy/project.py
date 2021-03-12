import os.path
import pandas as pd
import numpy as np
from autopipy.session import Session

class Project:
    """
    Class containing information about an autopi project.
    A project normally contains several sessions and has a tree directory and a model directory
    
    Attributes:
        name    Project name
        dataPath    Path of the directory containing the data for all the project
        dlcModelPath   Directory with the deeplabcut models
        
    Methods:
        createSessionList()
        mouseNameFromSessionName()
        sessionPathFromSessionName()
    """
    def __init__(self,name, dataPath, dlcModelPath = None):
        self.name = name
        self.dataPath = dataPath
        self.dlcModelPath = dlcModelPath
        self.sessionList = None
        
        print("Project name: "+ self.name)
        print("dataPath: "+ self.dataPath)
        print("dlcModelPath: "+ self.dlcModelPath)
        return

    def mouseNameFromSessionName(self,sessionName):
        """
        Get the mouse name from the session name
        
        Function used to generate the list of session objects
        """
        return sessionName.split("-")[0]
    
    def sessionPathFromSessionName(self, sessionName):
        """
        Get the full session path from the session name and the dataPath variable
        
        Used to generate the list of session objects
        """
        return self.dataPath + "/" + self.mouseNameFromSessionName(sessionName)+ "/" + sessionName
    
    def createSessionList(self, sessionNameList, needVideos=True):
        """
        Generate a sessionList for the project
        
        Arguments:
            sessionNameList: A list of session names
            needVideos: Boolean determining whether the session object should have video files
            
        The list is stored in the project attribute sessionList 
        """
        if not isinstance(sessionNameList,list):
            raise TypeError("sessionNameList is not a list")
        
        if needVideos:
            self.sessionList =  [ Session(name = sessionName, path = self.sessionPathFromSessionName(sessionName),dataFileCheck=False) for sessionName in sessionNameList]
        else :
            self.sessionList =  [ Session(name = sessionName, path = self.sessionPathFromSessionName(sessionName),dataFileCheck=False, arenaTopVideo=False,homeBaseVideo=False) for sessionName in sessionNameList]
    
    def getSession(self, sessionName):
        """
        Return a session from the session list based on sessionName
        """
        if self.sessionList is None:
            print("Create the sessionList before calling project.getSession()")
            return None
        
        return [ses for ses in self.sessionList if ses.name==sessionName ][0]
        
    def getTrialVariables(self):
        """
        Concatenate the trial variables of all sessions in the project
        
        You should call call extractTrialFeatures() getTrialVariablesDataFrame() on each session before calling this
        
        return a pandas dataframe
        """
    
        dfList = [ses.trialVariables for ses in self.sessionList]
        return pd.concat(dfList)
    
    
    def getTrialPathSpeedProfile(self):
        """
        Get a dictionary with the speed profile of all trials
        """
        ses = self.sessionList[0]
        if ses is None:
            print("Create the sessionList before calling project.getTrialPathSpeedProfile()")
            return None
        
        tr = ses.trialList[0]
        if tr is None:
            print("Call ses.extractTrialFeatures() on the session of sessionList before calling project.getTrialPathSpeedProfile()")
            return None
        
        self.speedProfile = {} # will be a dictionary, one key per path type
        for k in ses.trialList[0].pathD:
            [ ses.getTrialPathSpeedProfile(pathName=k) for ses in self.sessionList ]
            self.aList = [ ses.speedProfile for ses in self.sessionList ]
            self.a = np.concatenate(self.aList, axis=0)
            self.speedProfile[k]=self.a
        
        return self.speedProfile

    def getLogData(self):
        """
        Returns a Pandas DataFrame containing the log file of each session
        """
        for ses in self.sessionList :
            ses.loadLogFile()
            ses.logProcessing()
        logList = [ses.log for ses in self.sessionList]
        
        return pd.concat(logList)
    def getProtocolData(self):
        """
        Returns a Pandas DataFrame containing the protocol file of each session
        """
        for ses in self.sessionList :
            ses.loadProtocolFile()
        myList = [ses.protocol for ses in self.sessionList]
        
        return pd.concat(myList)