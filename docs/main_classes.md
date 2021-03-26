# Main classes in autopipy

We use classes to organize the code in a logical structure that closely maps onto the experiment.  



## List of classes


### Data analysis

Here is a list of most classes used to perform data analysis of your project. The important relationships between classes are shown in the figure below.

* **Project**: Represents a research project or experiment containing a list of `Session` objects.
* **Session**: Represents a single session. This class can check that we have all the needed files for analysis. It also get trial times from the log file. It can have a list of `Trial` objects.
* **Trial**: Represents a single trial. From the door opening to the door closing. This takes care of extracting key events within a trial. You can use it to create a video or a plot for a trial. It finds the beginning and end points for each journey. 
* **Journey**: A jouney is an excursion on the circular arena by the mouse. It might or might not include a lever press. The journey contains a dictionary of `NavPath` objects. In this dictionary, you will get different searching and homing paths for the journey.
* **NavPath**: Represents a path of an animal. This extracts many variables associated with a path (e.g. distance, directional vector length).
* **Lever**: Represents the lever location in a trial. The class is be used mainly to test if the animal is in the lever zone (right next to the lever).

![classes](classes.png)

### Video analysis (object detection)

* **Dlc**: Class to run DeepLabCut inference.
* **LeverDetector**: Detects the lever in video, inherits from dlc.
* **MouseLeverDetector**: Detects the mouse and lever in video, inherits from dlc.
* **BridgeDetector**: Detects the bridge in video, inherits from dlc.
* **ArenaDetector**: Detects the arena in video. The arena is detected using the opencv library.

