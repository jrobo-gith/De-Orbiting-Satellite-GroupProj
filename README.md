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
on the final landing site. As there is no radar data publicly available, you will also create a simulator
that produces the measurement data you feed into your landing site predictor.

## The knowns and unknowns
Here we have to distinguish between the predictor and the simulator. For the simulator you will
have to assume mass, size, shape of the satellite and initial orbital parameters, and use those to
simulate its position in space. The predictor will not “know” these, but infer some information from
prior assumptions and the incoming measurements. The simulator will generate noisy measurement
data to feed to predictor.
