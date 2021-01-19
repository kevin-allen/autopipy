import os.path

class session:
    """
    Class containing information about an autopi session
    
    Attributes:
        path    Directory path of the data for this session
        name    Name of the session. Usually used as the beginning of the file names
        fileBase path plus name
        arenaTopVideo Boolean indicating whether we should expect a video for the arnea
        homeBaseVide  Boolean indicating whether we should have a video of the home base
        requiredFileExts List containing the extensions of the file we should have in the directory
        arenaTopCropped Boolean indicating whether we have an arena top cropped video
    
    Methods:
        checkSessionDirectory():
    """
    def __init__(self,path,name,arenaTopVideo=True,homeBaseVideo=True):
        self.path = path
        self.name = name
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
        if self.checkSessionDirectory():
            self.dirOk=True
        else:
            print("problem with the directory " + self.path)
    
        # check if we have an arena_top.cropped.avi file
        self.arenaTopCropped=False
        if self.arenaTopVideo:
            if os.path.isfile(self.fileBase + "." + "arena_top.cropped.avi"):
                self.arenaTopCropped=True  
        
        return
        
    def checkSessionDirectory(self):
        # check that the directory is there
        if os.path.isdir(self.path) == False :
            print(self.path + "does not exist")
            return False
        # check that the files needed are there
        for ext in self.requiredFileExts:
            fileName = self.fileBase + "." + ext
            if os.path.isfile(fileName)== False:
                print(fileName + " does not exist")
                return False
        return True

    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    