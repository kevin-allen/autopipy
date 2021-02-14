import os.path
import pandas as pd
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
    def __init__(self,name, dataPath, dlcModelPath):
        self.name = name
        self.dataPath = dataPath
        self.dlcModelPath = dlcModelPath
        
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