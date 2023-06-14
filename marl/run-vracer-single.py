#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from mpi4py import MPI
from env import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--maxGenerations',
    help='Maximum Number of generations to run',
    default=50,
    type=int,
    required=False)    
parser.add_argument(
    '--learningRate',
    help='Learning rate for the selected optimizer',
    default=1e-4,
    type=float,
    required=False)
parser.add_argument(
    '--concurrentWorkers',
    help='Number of concurrent workers / environments',
    default=1,
    type=int,
    required=False)
parser.add_argument(
    '--nx',
    help='Number of gridpoints in x',
    default=16,
    type=int,
    required=False)
parser.add_argument(
    '--nz',
    help='Number of gridpoints in z',
    default=16,
    type=int,
    required=False)
parser.add_argument(
    '--nctrlx',
    help='Number of control in x',
    default=16,
    type=int,
    required=False)
parser.add_argument(
    '--nctrlz',
    help='Number of control in z',
    default=16,
    type=int,
    required=False)


args = parser.parse_args()

print("Running Cartpole example with arguments:")
print(args)
####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Testing Frequency"] = 10

nState = args.nx*args.nz
for i in range(nState):
    e["Variables"][i]["Name"] = "Sensor No. " + str(i)
    e["Variables"][i]["Type"] = "State"

for a in range(nState, nState+args.nctrlx*args.nctrlz):
    e["Variables"][a]["Name"] = "Contro No. " + str(a)
    e["Variables"][a]["Type"] = "Action"
    e["Variables"][a]["Lower Bound"] = -0.04285714285714286
    e["Variables"][a]["Upper Bound"] = +0.04285714285714286
    e["Variables"][a]["Initial Exploration Noise"] = 0.01

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Concurrent Workers"] = args.concurrentWorkers

e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"]= 0.3

e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = args.learningRate
e["Solver"]["Mini Batch"]["Size"] = 32
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 256

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 256

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = args.maxGenerations

### Setting file output configuration

e["File Output"]["Enabled"] = True
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Frequency"] = 10
e["File Output"]["Path"] = "./_korali_vracer_single/"

###  Configuring the distributed conduit

if args.concurrentWorkers > 1:
    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed"
    k["Conduit"]["Ranks Per Worker"] = 1

### Running Experiment

k.run(e)
