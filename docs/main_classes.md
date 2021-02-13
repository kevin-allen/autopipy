# Main classes in autopipy

We will develop classes to represent the main components of the task.

We will also have classes to perform specific analysis, for example for video detection.

## List of classes

* **project**: Represents a research project or experiment containing a list of sessions.
* **session**: Represents a single session. This class can check that we have all the needed files for analysis. It also get trial times from the log file.
* **trial**: Represents a single trial. This takes care of extracting the trial features and make trial related stuff (videos).
* **navPath**: Represents a path of an animal. This will extract different variables from it (e.g. distance, directional vector length).
* **dlc**: Class to run DeepLabCut inference.
* **leverDetector**: Detects the lever in video, inherits from dlc.
* **mouseLeverDetector**: Detects the mouse and lever in video, inherits from dlc.
* **bridgeDetector**: Detects the bridge in video, inherits from dlc.
* **arenaDetector**: Detects the arena in video.
