# Dynamic_obstacle_avoidance_unity

There are 2 goals for this project.
1. Drone avoids moving yellow ball by learning its velocity through DRL.
* Deep reinforcement learning was done using ppo with 7 coordinates(yellow ball's position, velocity and drone's position) input.
* Deep reinforcement learning was done using dqn with drone's 1st view image pixel(80*80*1) only as input.
2. Leverage simulation to the real world by using both domain randomization and domain adaptation.

Currently, I have gone through the first part and results are as below.

## Result

### [Coordinates as state input, both training and testing at no wind environment ]
#### 98% Avoidance
<img src = "https://user-images.githubusercontent.com/34183439/34464400-9de9dd8e-eec0-11e7-98f5-4ec50121261a.gif" width="600" height="400">

### [Coordinates as state input, training at no wind environment and testing at wind force affects only to yellow balls]
#### 99% Avoidance
<img src = "https://user-images.githubusercontent.com/34183439/34464401-9ed128e2-eec0-11e7-8740-b77cf687cc3c.gif" width="600" height="400">

The average reward increases!
<img src = "https://user-images.githubusercontent.com/34183439/34465199-bcc2d8a2-eee6-11e7-976d-86430b1d90c9.PNG">


### [Observation as state input, training at no wind environment, trained with 0.999 gamma value]
#### Avoided around 7/10 times. Need more training using cloud server.
<img src = "https://user-images.githubusercontent.com/34183439/34464402-9fc4f3a0-eec0-11e7-920a-9ca67c0ea33b.gif" width="600" height="400">
The average reward increases!
<img src = "https://user-images.githubusercontent.com/34183439/34465215-ecd67e6c-eee7-11e7-8019-1ef09e2dbc47.PNG">
