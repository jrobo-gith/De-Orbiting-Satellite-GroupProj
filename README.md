# Group Project - De-orbiting Satellite 

## The Project
The task is to predict where a de-orbiting satellite will hit the surface of the earth.

## The situation
A satellite is orbiting earth with a decaying orbit. Around its perigee, its orbital motion has started to
become affected by atmospheric drag. The satellite is no longer controllable. It is observed by a
number of ground-based radar stations, which can get periodic fixes on its position. Aerodynamic
drag will cause the satellite to slow down until it impacts the ground. For obvious reasons it would
be interesting to know where this will happen.

## The task
Create a simulation of the orbital decay of a satellite around earth. This will consist of two parts. The
predictor assimilates measurement data from the radar stations, and thus updates the predictions
on the final landing site. As there is no radar data publicly available, we will also create a simulator
that produces the measurement data we feed into our landing site predictor.

## The knowns and unknowns
Here we have to distinguish between the predictor and the simulator. For the simulator we will
have to assume mass, size, shape of the satellite and initial orbital parameters, and use those to
simulate its position in space. The predictor will not “know” these, but infer some information from
prior assumptions and the incoming measurements. The simulator will generate noisy measurement
data to feed to predictor.

## Installation Instructions

To install the latest version of this software, you will need the 'git' software, if you don't have git installed on your PC, [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) is a link which outlines how to install git. Alternatively, you could navigate to the github repository and under the green button `<> Code`, download the zip folder and follow the instructions from creating a python environment after having unzipped and `cd`'d into this folder. 

First, you must create a blank folder on your PC, `cd` into this folder, and type the following commands:
```bash
git clone https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj.git
```
then, once the repository is initialised, enter:
```bash
cd folder
```
then, once it is initialised, enter:
and
```bash
git pull origin main
```
to copy the files from the `main` branch onto the PC. 
<br><br>
Now, in the same file, create a python virtual environment by entering:
```bash
python -m venv venv
```
and activate the environment using:

*for Windows:*
```bash
.\venv\scripts\activate.bat
```

*for Mac and Linux:*
```bash
source venv/bin/activate
```
To run the software, use:

```bash
python master.py
```
which will run the master file, loading the necessary files dynamically.
