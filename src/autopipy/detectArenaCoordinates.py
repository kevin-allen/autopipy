import argparse
import os
import sys

import numpy as np
import cv2

#
# detect a large circle in a video
# or
# detect the arena in autopi arena_top video
#
# save image of detected arena and arena coordinates
# in the same directory as the video
#
# example of usage
# python detectArenaCoordinates.py /adata/projects/autopi_2019/mn5145/mn5145-06082019-0249/mn5145-06082019-0249.arena.avi 150 250 100 11 -v
# #
# if running on cropped videos
# python ~/repo/autopi_analysis/python/detectArenaCoordinates.py mn4672-20112019-1632.arena_top.cropped.avi 150 220 100 11 -v
# you might need to activate a python environment, if using virtualenv, e.g.,  source opencv/bin/activate


def detectArenaCoordinates (fileName, minRadius=180, maxRadius=220, numFrames=100, blur=11, circle='min', videoDisplay=False, return_img=False):
    count = 0
    cap = cv2.VideoCapture(fileName)

    l = []

    ## loop throught the frames and detect the arnea
    ## save the x,y of center and radius in the list l

    while cap.isOpened() and count < numFrames :
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(img,blur)
            cimg = cv2.cvtColor(blurred,cv2.COLOR_GRAY2BGR)
            circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,50,
                                      param1=70,param2=50,minRadius=minRadius,maxRadius=maxRadius)
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
                l.append(i)
        except:
            pass
            
        if videoDisplay:
            cv2.imshow('frame', cimg)
            if cv2.waitKey(1) == ord('q'):
                 break
             
        count=count+1

    ## clean up
    cap.release()
    if videoDisplay :
        cv2.destroyAllWindows()

    # Select circle
    l = np.asarray(l)
    if l.shape[0]==0 :
        print("problem with circle detection, array size 0")
    
    if circle == 'min':
        ## selected the smallest detected circle
        coordinates=l[np.argmin(l[:,2]),:]   # coordinates = [x, y, r]
    elif circle == 'median':
        coordinates=l[np.argmin(np.abs(l[:,2] - np.median(l[:,2]))),:]   # coordinates = [x, y, r]


    # Return
    if return_img:
        return coordinates, img
    else:
        return coordinates



# Execute if run as an executable file (not imported by other script)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("videoFile",help="path to a video")
    parser.add_argument("minRadius",help="minimal radius",type=int)
    parser.add_argument("maxRadius",help="maximal radius",type=int)
    parser.add_argument("numFrames",help="number of frames to analyze",type=int)
    parser.add_argument("blur",help="odd number setting the amount of blur in the image before doing the circle detection",type=int)
    parser.add_argument("-v","--videoDisplay", help="display the detection of arena for each frame",action="store_true")

    args = parser.parse_args()

    # set some variables for this trial
    fileName=args.videoFile
    minRadius=args.minRadius
    maxRadius=args.maxRadius
    numFrames=args.numFrames
    blur=args.blur
    videoDisplay=args.videoDisplay

    if os.path.isfile(fileName)== False:
        print(fileName + " does not exist")
        quit()
    
    if minRadius < 0 :
         print("minRadius < 0")
         quit()
    if maxRadius < minRadius :
        print("maxRadius smaller than minRadius")
        quit()
    if numFrames < 1 :
        print("number of frames is smaller than 1")
        quit()

    # Detect coordinates
    x, img = detectArenaCoordinates (fileName, minRadius, maxRadius, numFrames, blur, circle='median', videoDisplay=videoDisplay, return_img=True)

    ## draw the selected circle and save in a file called arenaDetection.png
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.circle(cimg,(x[0],x[1]),x[2],(0,255,0),2)
    cv2.circle(cimg,(x[0],x[1]),2,(0,0,255),3)
    # save the last frame with detected circle
    fn=os.path.dirname(os.path.abspath(fileName))+'/arenaDetection.png'
    print("detected arena shown in " + fn)
    cv2.imwrite(fn,cimg)

    ## save the circle x,y,r values in a text file
    fn=os.path.dirname(os.path.abspath(fileName))+'/arenaCoordinates'
    print("coordinates saved in " + fn)
    print(x)
    np.savetxt(fn,x, fmt='%d')
