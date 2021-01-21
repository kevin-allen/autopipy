# Deeplabcut models

We use deeplabcut to detect objects (mouse, lever, bridge) in the video or images. 
Below is a list of models that we are currently using.

In the laboratory, they are stored in `/adata/models/`

The config file for these models is the `config.yaml` file in the directories listed below.

* arena_top-Allen-2019-10-30: detect the mouse and the lever in 480 x 480 videos
* detectBridgeDLC/arena_top-Allen-2020-08-20: detect bridge in 640 x 480 videos
* bridgeDetection_480_480-Allen-2020-11-06: detect bridge in 480 x 480 videos
* leverDetector-Allen-2020-09-29: detect the lever in 640 x 480 videos


You can use the classes defined in dlcObjectDetectors.py to use these models for inference.

```
configFile="/adata/models/arena_top-Allen-2019-10-30/config.yaml"
videoFile="/adata/electro/mn4656/mn4656-03102019-1510/output.avi"
leverMouseOutputVideoFile = "/adata/electro/mn4656/mn4656-03102019-1510/output_leverMouse.avi"

# I created output.avi for testing 
# ffmpeg -ss 00:02:00.0 -i mn4656-03102019-1510.arena_top.avi -c copy -t 00:00:30.0 output.avi

mouseLeverD = mouseLeverDetector(pathConfigFile=configFile)
#mouseLeverD.inferenceVideo(pathVideoFile=videoFile)
#mouseLeverD.loadPositionData(pathVideoFile=videoFile)
#mouseLeverD.labelVideo(pathVideoFile=videoFile)
#mouseLeverD.getDataFrameOut(pathVideoFile=videoFile)
mouseLeverD.positionOientationFromFile(pathVideoFile=videoFile)
mouseLeverD.labelVideoMouseLever(pathVideoFile=videoFile,pathOutputFile=leverMouseOutputVideoFile)
```
