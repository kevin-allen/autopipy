#!/usr/bin/env python
# coding: utf-8

# # Video processing
# 
# We will implement the first steps of video processing for a single recording session.
# 

# With the `arena_top.avi` file
# 
# * Detect the bridge and arena in arena_top.avi
# * Crop the arena_top.avi to 480x480
# 
# With the cropped video
# 
# * Detect arena in cropped video
# * Detect bridge in cropped video
# * Detect the mouse and lever in cropped video
# 
import os.path
import autopipy
from autopipy.session import session
from autopipy.cvObjectDetectors import arenaDetector
from autopipy.dlcObjectDetectors import bridgeDetector
from autopipy.video_utilities import maskCropVideoToBridgeArena
from autopipy.dlcObjectDetectors import mouseLeverDetector
from autopipy.dlcObjectDetectors import mouseLeverDetector
from autopipy.video_utilities import arenaBridgeDetectionImage
import sys, getopt


def main(argv):
    sessionName = ''
    projectDir = '/adata/electro'
    modelDir= "/adata/models"
    
    try:
        opts, args = getopt.getopt(argv,"hp:m:",["projectDir=","modelDir="])
    except getopt.GetoptError:
        print ('video_processing.py sessionName -p <projectDir> -m <modelDir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('video_processing.py -p <projectDir> -m <modelDir>')
            sys.exit()

        elif opt in ("-p", "--projectDir"):
            projectDir = arg
        
        elif opt in ("-m", "--modelDir"):
             modelDir = arg

    if len(args) != 1:
        print("You need to supply the sessionName")

    sessionName=args[0]
    mouseName=sessionName.split("-")[0]
    sessionPath=projectDir+"/" + mouseName + "/" + sessionName
    
    
    print ('sessionName: '+sessionName)
    print ("mouseName: " + mouseName)
    print ('projectDir: '+ projectDir)
    print ('modelDir: '+modelDir)
    print ("sessionPath: "+ sessionPath)
    video_processing(sessionPath,sessionName,modelDir)

    
def video_processing(sessionPath,sessionName,modelDir):
    
    s = session(path=sessionPath,name=sessionName)

    videoFile = s.fileNames["arena_top.avi"] #Nice trick to get file names using a dict
    arenaImageFile=s.path+"/arenaDetection.png"
    arenaD = arenaDetector()
    aCoord = arenaD.detectArenaCoordinates(pathVideoFile=videoFile, minRadius=180, 
                                  maxRadius=220, numFrames=100, blur=11, circle='min')
    arenaD.labelImage(pathVideoFile=videoFile,outputImageFile=arenaImageFile)


    configFile = modelDir+"/detectBridgeDLC/arena_top-Allen-2020-08-20/config.yaml"
    bridgeImageFile = s.path+"/bridgeDetection.png"
    bridgeD = bridgeDetector(pathConfigFile=configFile)
    bCoord = bridgeD.detectBridgeCoordinates(pathVideoFile=videoFile,numFrames=100, skip=30)
    bridgeD.labelImage(pathVideoFile=videoFile,outputImageFile=bridgeImageFile)


    videoFile=s.fileNames["arena_top.avi"]
    croppedVideoFile = os.path.splitext(videoFile)[0]+".cropped.avi"
    maskCropVideoToBridgeArena(pathVideoFile=videoFile, pathOutputFile=croppedVideoFile, 
                               arenaCoordinates=aCoord, bridgeCoordinates = bCoord)


    configFile=modelDir+"/arena_top-Allen-2019-10-30/config.yaml"
    croppedVideoFile = os.path.splitext(videoFile)[0]+".cropped.avi"
    mouseLeverD = mouseLeverDetector(pathConfigFile=configFile)
    mouseLeverD.inferenceVideo(pathVideoFile=croppedVideoFile,overwrite=True)

    labeledVideoFile = os.path.splitext(croppedVideoFile)[0]+".labeled.avi"
    mouseLeverD.labelVideoMouseLever(pathVideoFile=croppedVideoFile,pathOutputFile=labeledVideoFile)


    arenaImageFile=s.path+"/arenaDetectionCropped.png"
    arenaD = arenaDetector()
    aCoord = arenaD.detectArenaCoordinates(pathVideoFile=croppedVideoFile, minRadius=180, 
                                  maxRadius=220, numFrames=100, blur=11, circle='min')
    arenaD.labelImage(pathVideoFile=croppedVideoFile,outputImageFile=arenaImageFile)


    configFile = modelDir+"/bridgeDetection_480_480-Allen-2021-01-23/config.yaml"
    bridgeImageFile = s.path+"/bridgeDetectionCropped.png"
    bridgeD = bridgeDetector(pathConfigFile=configFile)
    bCoord = bridgeD.detectBridgeCoordinates(pathVideoFile=croppedVideoFile,numFrames=100, skip=30)
    bridgeD.labelImage(pathVideoFile=videoFile,outputImageFile=bridgeImageFile)


    outputImageFile=s.path+"/arenaBridgeDetectionCropped.png"
    arenaBridgeDetectionImage(pathVideoFile=croppedVideoFile,
                              outputImageFile=outputImageFile,
                              arenaCoordinates = aCoord,
                              bridgeCoordinates = bCoord)

        
if __name__ == "__main__":
   main(sys.argv[1:])
