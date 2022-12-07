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
#### Configuration
> **_ATTENTION:_** Check the `GAMES_FOR_TRAINING` key in the `src/config.yaml` files to make sure it only has the games for which you want to download/process/train data. The default is just `breakout` for testing purposes.

#### Downloading
```bash
python src/data/download.py
python src/data/process.py
```