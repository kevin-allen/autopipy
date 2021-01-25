import os.path
from autopipy.session import session

class project:
    """
    Class containing information about an autopi project.
    A project normally contains several sessions and has a tree directory and a model directory
    
    Attributes:
        name    Project name
        dataPath    Directory path of the data for this session
        dlcModelPath   Directory with the deeplabcut models
        
        
    Methods:
        createSessionList()
        runOnSession():
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
        return sessionName.split("-")[0]
    
    def sessionPathFromSessionName(self, sessionName):
        return self.dataPath + "/" + self.mouseNameFromSessionName(sessionName)+ "/" + sessionName
    
    def createSessionList(self, sessionNameList):
        self.sessionList =  [ session(name = sessionName, path = self.sessionPathFromSessionName(sessionName),dataFileCheck=False) for sessionName in sessionNameList]