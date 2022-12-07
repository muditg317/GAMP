# Mo-GaS: Motion & Gaze Selection for Playing Atari Games
## CS 7648 Team 1 Final Project
### Team Members
* Krishanu Agrawal
* Ezra Ameperosa
* Mudit Gupta
* Himanshu Vairagade

### Project Overview
We've created a variation/combination of the Selective Eye-gaze Augmentation and the Coverage-base Gazed Loss approaches for playing Atari games. Read the following for more details:
* [project proposal](../CS7648-Team1-Proposal-Gaze-and-Motion-Prioritization.pdf)
* [project update](../CS7648-Team1-Project_Update-Motion-and-Gaze-Selection.pdf)
* [final report](../CS7648-Team1-Final_Report-Motion-and-Gaze-Selection.pdf)

## Project Usage
### Dependencies
* Python 3, PyTorch, OpenCV, NumPy, Matplotlib, SciPy, and other standard libraries
  * see the [conda environment file](./Mo-GaS.yml) for a list of dependencies
The `configure_enviroment.sh` script will handle the installation of the dependencies.

### Setup Instructions
1. Clone the repository
2. Run `./configure_environment.sh` to create a conda environment with the required dependencies
3. Download data, preprocess the data, train the gaze model, train the action selection network

> **_NOTE:_** Make sure all commands are run from the root of the repository (this folder, `Mo-GaS/`)

### Data
#### Downloading
```bash
python src/data/data_setup.py
```