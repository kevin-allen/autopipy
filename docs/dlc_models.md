# Deeplabcut models

We use convolutional deep neural network trained with deeplabcut to detect objects (mouse, lever, bridge) in the video or images. 
You can use the documentation from deeplabcut to create and train your models.

In autopipy, we only do inference/predictions with already trained models.

There is a dlc class that is a wrapper around deeplabcut functions. There is a class that inherits from dlb that is specific for the detection of specific objects.

Below is a list of models that we routinely use.

In the laboratory, they are stored in `/adata/models/`

The config file for these models is the `config.yaml` file in the directories listed below.

* **arena_top-Allen-2019-10-30**: detect the mouse and the lever in 480 x 480 videos
* **detectBridgeDLC_640_480/arena_top-Allen-2020-08-20**: detect bridge in 640 x 480 videos
* **bridgeDetection_480_480-Allen-2020-11-06**: detect bridge in 480 x 480 videos
* **leverDetector-Allen-2020-09-29**: detect the lever in 640 x 480 videos

These directories normally contains all the data required to re-train and evaluate the models. 

To run inference with autopipy, all you really need is the `config.yaml` file and the `dlc-models` directory.


## Retraining

autopipy has nothing to do with training deeplabcut models, but if you need to retrain one, have a look at the jupyter notebook in the model directories.

## autopipy classes for object detection

You can use the classes defined in dlcObjectDetectors.py to use these models for inference. Some of these classes are listed below.

* leverDetector
* mouseLeverDetector
* bridgeDetector

Tips: The arenaDetector class is not using dlc and is stored in cvObjectDetectors.py


```
configFile="/adata/models/arena_top-Allen-2019-10-30/config.yaml"
videoFile="/adata/electro/mn4656/mn4656-03102019-1510/output.avi"
mouseLeverOutputVideoFile = "/adata/electro/mn4656/mn4656-03102019-1510/output_mouseLever.avi"

# I created output.avi for testing 
# ffmpeg -ss 00:02:00.0 -i mn4656-03102019-1510.arena_top.avi -c copy -t 00:00:30.0 output.avi

mouseLeverD = mouseLeverDetector(pathConfigFile=configFile)
#mouseLeverD.inferenceVideo(pathVideoFile=videoFile)
#mouseLeverD.loadPositionData(pathVideoFile=videoFile)
#mouseLeverD.labelVideo(pathVideoFile=videoFile)
#mouseLeverD.getDataFrameOut(pathVideoFile=videoFile)
mouseLeverD.positionOientationFromFile(pathVideoFile=videoFile)
mouseLeverD.labelVideoMouseLever(pathVideoFile=videoFile,pathOutputFile=mouseLeverOutputVideoFile)
```
