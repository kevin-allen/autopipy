import os.path
import numpy as np
import cv2
import sys


def maskCropVideoToBridgeArena(pathVideoFile, pathOutputFile, arenaCoordinates, bridgeCoordinates, outWidth=480, outHeight=480, arenaRadiusFactor=1.125, bridgeWidthFactor=1.5):
    """
    Function that applies a mask and crop operation to a video 
    The mask will keep the arena and the bridge, and some extra margin.
    
    There is one function performing 2 operations as this will be faster to perform them at the same time rather than one after the other.
    
    Arguments:
        pathVideoFile: video to mask and crop
        pathOutputFile: where to save the masked and cropped video
        arenaCoordinates: [x, y , radius]
        bridgeCoordinates: np.array of shape 4x2
        outWidth: width of the cropped video
        outHeight: height of the cropped video
        
    """
    if not os.path.isfile(pathVideoFile):
        print(pathVideoFile + " does not exist")
        return False
    
    cap = cv2.VideoCapture(pathVideoFile)

    fps = int (cap.get(cv2.CAP_PROP_FPS))
    inWidth = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    inHeight = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
    # modify the arena radius to show the edges
    aCoord = arenaCoordinates.copy() # to preserver original array
    aCoord[2] = aCoord[2]*arenaRadiusFactor
    
    # modify the bridge coordinates
    bCoord = bridgeCoordinates.copy() # to preserve original array
    bridgeWidth = bridgeCoordinates[3,0]-bridgeCoordinates[0,0]
    extraMargin= bridgeWidth*(bridgeWidthFactor-1)/2
    bCoord[0,0]-=extraMargin
    bCoord[1,0]-=extraMargin
    bCoord[2,0]+=extraMargin
    bCoord[3,0]+=extraMargin
    
    
    # get the mask
    outsideMask = createMaskArenaBridge(inWidth, inHeight,aCoord, bCoord)
    
    # check that the video is large enough for the crop
    if inWidth<outWidth:
        print("width needed is larger than original image width")
        return False
    if inHeight<outHeight:
        print("height needed is larger than original image height")
        return False
    
    
    # try to center the x on the arena center
    xmin = aCoord[0] - outWidth // 2
    xmax = aCoord[0] + outWidth // 2
    if xmin < 0:
        diff = np.abs(xmin)
        xmin = 0
        xmax = xmax + diff
    if xmax > inWidth:
        diff = np.abs(xmax - width)
        xmax = width
        xmin = xmin - diff

    # try to center the y on the arena center
    ymin = aCoord[0] - outHeight // 2
    ymax = aCoord[0] + outHeight // 2
    if ymin < 0:
        diff = np.abs(ymin)
        ymin = 0
        ymax = ymax + diff
    if ymax > inHeight:
        diff = np.abs(ymax - inHeight)
        ymax = inHeight
        ymin = ymin - diff
    
    
#     print("crop from {},{} to {},{}".format(xmin,ymin,xmax,ymax))
#     print("{} by {} pixels".format(xmax-xmin,ymax-ymin))
#     print("needed: {} by {}".format(outWidth,outHeight))
    
    out = cv2.VideoWriter(pathOutputFile, cv2.VideoWriter_fourcc(*'MJPG'), fps, (outWidth,outHeight)) 
    
    print("Cropping and masking {} frames in {}".format(nFrames,pathVideoFile))
    print("Output file {}".format(pathOutputFile))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("")
            print("Exiting. Video saved as %s" % pathOutputFile)
            break
        # apply mask
        frame[outsideMask, :] = 0
        # crop image
        frame = frame [ymin:ymax, xmin:xmax, :]
        # save image
        out.write(frame)
        if count % 10 == 0:
            sys.stdout.write('\r')
            sys.stdout.write("{} of {} frames".format(count,nFrames))
            sys.stdout.flush()
        count+=1
    
    cap.release() 
    out.release() 

def createMaskArenaBridge(imageWidth, imageHeight, arenaCoordinates, bridgeCoordinates):
    """
    Create a mask for a video that contains the arena and the bridge
    
    Arguments:
        imageWidth: width of the image in pixels
        imageHeight: heigh of the image in pixels
        arenaCoordinates: [x,y,radius]
        bridgeCoordinates: np.array of shape 4x2
    Return:
        outsideMask, np.array with 0s and 1s. 1s are pixels that we want to zero
    """
    
    x_center = arenaCoordinates[0]
    y_center = arenaCoordinates[1]
    r = arenaCoordinates[2]
    y, x = np.ogrid[:imageHeight, :imageWidth]
    outsideMask = (x - x_center)**2 + (y-y_center)**2 > r**2    # True if outside of arena
    # bridge
    outsideMask[bridgeCoordinates[0,1]:bridgeCoordinates[2,1], bridgeCoordinates[0,0]:bridgeCoordinates[2,0]] = False # False if on the bridge
    
    return outsideMask
    

def arenaBridgeDetectionImage(pathVideoFile, outputImageFile, arenaCoordinates, bridgeCoordinates,skip=30):
    if not os.path.isfile(pathVideoFile):
        print(pathVideoFile + " does not exist")
        return False
    
    cap = cv2.VideoCapture(pathVideoFile)
    index=0
    while index < skip:
        ret, frame = cap.read()
        index+=1
    drawArena(frame, arenaCoordinates)
    drawBridge(frame, bridgeCoordinates)
     # save the last frame with detected circle
    print("labelImage: " + outputImageFile)
    cv2.imwrite(outputImageFile,frame)


def drawBridge(frame, bCoord):
    for i in range(bCoord.shape[0]):
            cv2.circle(frame,(bCoord[i,0],bCoord[i,1]),3,(0,255,0),2)

def drawArena(frame, aCoord):
    cv2.circle(frame,(aCoord[0],aCoord[1]),aCoord[2],(0,255,0),2)
    cv2.circle(frame,(aCoord[0],aCoord[1]),2,(0,0,255),3)
