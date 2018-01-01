# Dynamic_obstacle_avoidance_unity

There are 2 goals for this project.
1. Drone avoids moving yellow ball by learning its velocity through DRL.
* Deep reinforcement learning was done using ppo with 7 coordinates(yellow ball's position, velocity and drone's position) input.
* Deep reinforcement learning was done using dqn with drone's 1st view image pixel only as input.
2. Leverage simulation to the real world by using both domain randomization and domain adaptation.

Currently, We have gone through the first part and results are as below.

## Result

### [Experiment 1-1-1. Coordinates as state input, both training and testing at no wind environment 0.99 gamma value ]
1. Performance : 98% Avoidance
<img src = "https://user-images.githubusercontent.com/34183439/34464400-9de9dd8e-eec0-11e7-98f5-4ec50121261a.gif" width="600" height="400">

### [Expriment 1-1-2. Coordinates as state input, training at no wind environment and testing at wind force affects only to yellow balls]
1. Performance : 99% Avoidance(x)--> 97% Avoidance(o)
* We concluded that first 99% result was just by luck. After, doing more trials performance converged to 97%.
* We concluded the reason why performance was tied with experiment 1-1-1. was because gamma value was 0.99. Ball reaches to drone after 60 frames. Since, gamma value is 0.99 drone's q-function is more affected by balls after 30 frames. Thus, drone can still avoid windy ball, even though it changes velocity after 30 frames.
<img src = "https://user-images.githubusercontent.com/34183439/34464401-9ed128e2-eec0-11e7-8740-b77cf687cc3c.gif" width="600" height="400">

2. The average reward increases!
<img src = "https://user-images.githubusercontent.com/34183439/34465199-bcc2d8a2-eee6-11e7-976d-86430b1d90c9.PNG">


### [Experiment 1-2. Observation as state input, training at no wind environment, trained with 0.999 gamma value]
1. Performance : Avoided around 7/10 times. (We need to use cloud. Very slow training)
<img src = "https://user-images.githubusercontent.com/34183439/34464402-9fc4f3a0-eec0-11e7-920a-9ca67c0ea33b.gif" width="600" height="400">
2. The average reward increases!
<img src = "https://user-images.githubusercontent.com/34183439/34465215-ecd67e6c-eee7-11e7-8019-1ef09e2dbc47.PNG">
