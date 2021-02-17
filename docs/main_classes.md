# Main classes in autopipy

We use classes to organize the code in a logical structure that closely maps onto the experiment.  



## List of classes

### Classes to perform position tracking of mouse, lever and arena

* **Dlc**: Class to run DeepLabCut inference.
* **LeverDetector**: Detects the lever in video, inherits from dlc.
* **MouseLeverDetector**: Detects the mouse and lever in video, inherits from dlc.
* **BridgeDetector**: Detects the bridge in video, inherits from dlc.
* **ArenaDetector**: Detects the arena in video. The arena is detected using the opencv library.

### Other classes
* **Project**: Represents a research project or experiment containing a list of sessions.
* **Session**: Represents a single session. This class can check that we have all the needed files for analysis. It also get trial times from the log file.
* **Trial**: Represents a single trial. This takes care of extracting the trial features and make trial related stuff (videos).
* **NavPath**: Represents a path of an animal. This will extract different variables from it (e.g. distance, directional vector length).
* **Lever**: Represents the lever location in a trial. The class can be used to know if the animal is in the lever zone (right next to the lever).
