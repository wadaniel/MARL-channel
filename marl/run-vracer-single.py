#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from mpi4py import MPI
from env import *
import shutil
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
    '--episodeLength',
    help='Length of sim in steps',
    default=500,
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
parser.add_argument(
    '--compression',
    help='Field ocmpression factor in one dim',
    default=2,
    type=int,
    required=False)
parser.add_argument(
    '--run',
    help='Run tag',
    default=0,
    type=int,
    required=False)


args = parser.parse_args()
args.workDir = "./../bin" #f"./_korali_vracer_single_{args.run}/"

print("Running Flow control with arguments:")
print(args)

srcDir = './../bin/'

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda s : env(s, args)
e["Problem"]["Testing Frequency"] = 10
e["Problem"]["Policy Testing Episodes"] = 1

nState = 2*args.nx*args.nz//(args.compression**2)
for i in range(nState):
    e["Variables"][i]["Name"] = "Sensor No. " + str(i)
    e["Variables"][i]["Type"] = "State"

for a in range(nState, nState+args.nctrlx*args.nctrlz):
    e["Variables"][a]["Name"] = "Contro No. " + str(a)
    e["Variables"][a]["Type"] = "Action"
    e["Variables"][a]["Lower Bound"] = -0.04285714285714286
    e["Variables"][a]["Upper Bound"] = +0.04285714285714286
    e["Variables"][a]["Initial Exploration Noise"] = 0.05

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = args.concurrentWorkers
e["Solver"]["Concurrent Workers"] = args.concurrentWorkers

e["Solver"]["Experience Replay"]["Start Size"] = 5*args.episodeLength*args.concurrentWorkers #131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 524288
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"]= 0.3

e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Learning Rate"] = args.learningRate
e["Solver"]["Mini Batch"]["Size"] = 128
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = args.maxGenerations

### Setting file output configuration

e["File Output"]["Enabled"] = True
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Frequency"] = 10
e["File Output"]["Path"] = args.workDir
e["Console Output"]["Verbosity"] = "Detailed"

###  Configuring the distributed conduit

if args.concurrentWorkers > 1:
    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed"
    k["Conduit"]["Ranks Per Worker"] = 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#if rank == 0:
    #print(f'[korali_optimize] rank 0 copying files to {args.workDir}')
    #os.makedirs(args.workDir, exist_ok=True)
    #shutil.copy(srcDir + "bla.i", args.workDir)
    #shutil.copy(srcDir + "bla_16x65x16_1", args.workDir)
MPI.COMM_WORLD.Barrier()


### Running Experiment

k.run(e)
