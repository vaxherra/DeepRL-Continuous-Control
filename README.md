# DeepRL-Continuous-Control
Udacity Deep Reinforcement Learning: Continuous Control Project

## Code structure

The code is structured as follows:
- `Continuous_Control.ipynb` is the main notebook for the project. It contains the code for the agent and the training loop.
- `model.py` contains the neural network model for the actor and the critic.
- `agent.py` contains the agent class.
- `utils.py` contains some utility functions for training

- `Report.md` is the report for the project.
- `README.md` is this file.
- `checkpoint_actor.pth` and `checkpoint_critic.pth` are the saved weights for the actor and the critic.
- `rewards.png` is a plot of the rewards during training.

## Project Details
 the project environment details (i.e., the state and action spaces, and when the environment is considered solved).

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that 
the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target 
location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities 
of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. 
Every entry in the action vector should be a number between -1 and 1.

## Getting Started

It is recommended to follow the Udacity DRL ND dependencies [instructions here](https://github.com/udacity/deep-reinforcement-learning#dependencies).
See also folder `python/` for the dependencies. These are installed during notebook execution.

This project utilises [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [NumPy](http://www.numpy.org/) and [PyTorch](https://pytorch.org/) 

A prebuilt simulator is required in be installed. You need only select the environment that matches your operating system:

### Version 1: One (1) Agent
Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

### Version 2: Twenty (20) Agents
Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

The file needs to placed in the root directory of the repository and unzipped.

Next, before starting the environment utilising the corresponding prebuilt app from Udacity  **Before running the code cell in the notebook**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.


## Instructions
How to run the code in the repository, to train the agent.
- Run the cells in `Continuous_Control.ipynb` to train the agent. The weights for the actor and the critic are saved, so
it is not necessary to train the agent again.