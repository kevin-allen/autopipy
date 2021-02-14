# Main classes in autopipy

We will develop classes to represent the main components of the task.

We will also have classes to perform specific analysis, for example for video detection.

## List of classes

* **Project**: Represents a research project or experiment containing a list of sessions.
* **Session**: Represents a single session. This class can check that we have all the needed files for analysis. It also get trial times from the log file.
* **Trial**: Represents a single trial. This takes care of extracting the trial features and make trial related stuff (videos).
* **NavPath**: Represents a path of an animal. This will extract different variables from it (e.g. distance, directional vector length).
* **Dlc**: Class to run DeepLabCut inference.
* **LeverDetector**: Detects the lever in video, inherits from dlc.
* **MouseLeverDetector**: Detects the mouse and lever in video, inherits from dlc.
* **BridgeDetector**: Detects the bridge in video, inherits from dlc.
* **ArenaDetector**: Detects the arena in video.
