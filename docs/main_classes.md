# Main classes in autopipy

We will develop classes to represent the main components of the task.

We will also have classes to perform specific analysis, for example for video detection.

## List of classes

* **project**: Representing a research project or experiment containing a list of sessions.
* **session**: Representing a single session. This class can check that we have all the needed files for analysis. It also get trial times from the log file.
* **trial**: Representing a single trial. This takes care of extracting the trial features and make trial related stuff (videos).
* **dlc**: Class to run DeepLabCut inference
* **leverDetector**: Detect the lever in video, inherits from dlc
* **mouseLeverDetector**: Detect the mouse and lever in video, inherits from dlc
* **bridgeDetector**: Detect the bridge in video, inherits from dlc
* **arenaDetector**: Detect the arena in video
