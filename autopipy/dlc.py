import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
import os.path
import h5py
import numpy as np
import pandas as pd
import os 
from datetime import datetime


class Dlc():
    """
    Class to simplify inference with a deeplabcut model. This is just a simple wrapper.
    The aims were to
        Simplify inference from video or single images
        Get the data in a numpy array or a pandas data frame
    
    This class will be the parent class of the object detectors that use dlc (e.g. lever detector, mouse detector). 
    This avoids duplicating this code.
    
    Attributes:
        pathConfigFile: path to the configuration file of the deeplabcut model
        pathVideoFile: path to the video file analyzed
        pathVideoOutputH5: path to the output data
        out: output data, read from the h5 file, one row per frame
    Methods:
        inferenceVideo()
        loadPositionData()
        getVideoOutputH5()
        getBodyParts()
        
    """
    
    def __init__(self,pathConfigFile):
        self.pathConfigFile = pathConfigFile
        self.videoType = "avi"
        
        if not os.path.isfile(self.pathConfigFile):
            print(self.pathConfigFile + " does not exist")
            return False
        
        self.getBodyParts()
    
    def inferenceVideo(self, pathVideoFile, saveCsv = True, overwrite=False):
        """
        Perform inference on a video with our model.
        Simply runs deeplabcut.analyze_videos()
        
        Arguments:
            pathVideoFile
            saveCsv Boolean indicating whether to save the data into a csv file
        """
        
        startTime = datetime.now()
        dt_string = startTime.strftime("%d/%m/%Y %H:%M:%S")
        print("Start date and time =", dt_string)
        
        
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
        
        self.pathVideoFile = pathVideoFile
        
        # get the name of the file with the output data
        self.pathVideoOutputH5=self.getVideoOutputH5(pathVideoFile)
        
        if overwrite:
            print("Removing previous inference data: " + self.pathVideoOutputH5)
            if os.path.isfile(self.pathVideoOutputH5):
                os.remove(self.pathVideoOutputH5)
        
        
        print("Running dlc.analyze_video on "+ pathVideoFile)
        deeplabcut.analyze_videos(self.pathConfigFile,[pathVideoFile])
        
        
        # load the position data by default
        self.loadPositionData(pathVideoFile)
            
        if saveCsv:
            df = self.getDataFrameOut(pathVideoFile)
            fileName = os.path.splitext(self.pathVideoOutputH5)[0]+".csv"
            print("Saving position data to "+fileName)
            df.to_csv(fileName,index=False)
            
        endTime = datetime.now()
        dt_string = endTime.strftime("%d/%m/%Y %H:%M:%S")
        print("End date and time =", dt_string)
        
            
            
    def labelVideo(self, pathVideoFile):
        """
        Create a labelled video with deeplabcut function create_labeled_video()
        
        Arguments:
            pathVideoFile
        """
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False     
        self.pathVideoFile = pathVideoFile
        print("Running dlc.create_label_video on "+ pathVideoFile)
        deeplabcut.create_labeled_video(self.pathConfigFile,self.pathVideoFile)
        
        self.pathVideoOutputH5=self.getVideoOutputH5(pathVideoFile)
        fileName = os.path.splitext(self.pathVideoOutputH5)[0]+"_labeled.mp4"
        print("Created "+fileName)
    
    
    def loadPositionData(self, pathVideoFile):
        """
        Load the position data from the h5 file that has been generated during inference.
        Will store the data as a np array in self.out
        
        Arguments:
            pathVideoFile
        """
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
        
        self.pathVideoFile = pathVideoFile
         
        # get the name of the file with the output data
        self.pathVideoOutputH5=self.getVideoOutputH5(pathVideoFile)
        
        if not os.path.isfile(self.pathVideoOutputH5):
            print(self.pathVideoOutputH5 + " does not exist")
            print("Run inferenceVideo first")
            return False
        
        # read the data
        print("dlc.loadPositionData() read data from "+self.pathVideoOutputH5)
        self.out = h5py.File(self.pathVideoOutputH5,mode="r")['df_with_missing']['table'][:]
        
        # turn this into a np.array with one row per frame
        self.out = np.concatenate([x[1] for x in self.out]).reshape(-1,len(self.bodyParts)*3)
        
    def getDataFrameOut(self, pathVideoFile):
        """
        Transform self.out into a pandas data frame, with the column names.
        """
        
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
        self.pathVideoFile = pathVideoFile
        
        self.loadPositionData(self.pathVideoFile)
        
        cn = [[part+".x", part+".y", part+".prob"]for part in self.bodyParts]
        colnames = [item for sublist in cn for item in sublist]
        return pd.DataFrame(data=self.out,columns=colnames)
          
    def getVideoOutputH5(self, pathVideoFile,shuffle=1,trainingsetindex=0, modelprefix=""):
        """
        Get the name of the file with the position data that is generated during inference
        
        Arguments
            pathVideoFile
            shuffle
            trainingsetindex
            modelprefix
        """
        if not os.path.isfile(pathVideoFile): 
            print(pathVideoFile + " does not exist")
            return False
        
        self.pathVideoFile=pathVideoFile
        directory = os.path.dirname(pathVideoFile)
        video = os.path.basename(pathVideoFile)
        
        cfg = auxiliaryfunctions.read_config(self.pathConfigFile)
        
        trainFraction = cfg["TrainingFraction"][trainingsetindex]
        DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
            cfg, shuffle, trainFraction, modelprefix=modelprefix) 
               
        return os.path.splitext(pathVideoFile)[0]+DLCscorer+".h5"
        
    def getBodyParts(self):
        """
        Get the list of body parts from the config file
        """
        cfg = auxiliaryfunctions.read_config(self.pathConfigFile)
        self.bodyParts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, "all")
        
    def __str__(self):
        """
        Print function
        """
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
        
