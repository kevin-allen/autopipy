# Main classes in autopipy

We will develop classes to represent the main components of the task.

We will also have classes to perform specific analysis, for example for video detection.

## Started

* session: Representing a single session
* trial: Representing a single trial
* dlc: Class to run DeepLabCut inference
* leverDetector: Detect the lever in video, inherits from dlc
* arenaDetector: Detect the arena in video

## To be written

* calculate an arena arenaMask to remove outside of arena
* crop image for neural network
* bridgeDetector

