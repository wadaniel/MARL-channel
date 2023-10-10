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
    '--maxExperiences',
    help='Maximum Number of experiencecs to run',
    default=1000000,
    type=int,
    required=False)
parser.add_argument(
    '--episodeLength',
    help='Length of sim in steps',
    default=1000,
    type=int,
    required=False)
parser.add_argument(
    '--pol',
    help='Policy type (Normal, Clipped Normal, ..)',
    default="Clipped Normal",
    type=str,
    required=False)    
parser.add_argument(
    '--learningRate',
    help='Learning rate for the selected optimizer',
    default=1e-4,
    type=float,
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
    default=1,
    type=int,
    required=False)
parser.add_argument(
    '--nagx',
    help='Number of agents in x',
    default=1,
    type=int,
    required=False)
parser.add_argument(
    '--nagz',
    help='Number of agents in z',
    default=1,
    type=int,
    required=False)
parser.add_argument(
    '--ycoords',
    help='Sampling height (alt -0.83146961)',
    default=-0.99880,
    type=float,
    required=False)
parser.add_argument(
    '--run',
    help='Run tag',
    default=0,
    type=int,
    required=False)
parser.add_argument(
    '--test',
    help='Eval policy',
    action='store_true',
    required=False)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from rank {rank}")

args = parser.parse_args()
args.workDir = "./../bin" #f"./_korali_vracer_single_{args.run}/"
args.resDir = f"/scratch/wadaniel/MARL-channel/_korali_vracer_multi_{args.ycoords}_{args.run}"
args.concurrentWorkers = comm.Get_size() - 1

if rank == 0:
    print("Running Flow control with arguments:")
    print(args)

srcDir = './../bin/'


####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

found = e.loadState(args.resDir + '/latest')
if found == True:
    print("[Korali] Continuing execution from previous run...\n")

### Defining the Cartpole problem's configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda s : env(s, args)
e["Problem"]["Testing Frequency"] = 5
e["Problem"]["Policy Testing Episodes"] = 1
e["Problem"]["Agents Per Environment"] = args.nagx*args.nagz

assert args.nx/args.compression % args.nagx == 0
assert args.nz/args.compression % args.nagz == 0
assert args.nctrlx/args.compression % args.nagx == 0
assert args.nctrlz/args.compression % args.nagz == 0

nState = 2*args.nx*args.nz//(args.compression**2*args.nagx*args.nagz)

for i in range(nState):
    e["Variables"][i]["Name"] = "Sensor No. " + str(i)
    e["Variables"][i]["Type"] = "State"

maxv = 0.04285714285714286

nAct = args.nx*args.nz//(args.compression**2*args.nagx*args.nagz)

for a in range(nState, nState+nAct):
    e["Variables"][a]["Name"] = "Contro No. " + str(a)
    e["Variables"][a]["Type"] = "Action"
    e["Variables"][a]["Lower Bound"] = -maxv
    e["Variables"][a]["Upper Bound"] = +maxv
    e["Variables"][a]["Initial Exploration Noise"] = 2.*maxv

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Concurrent Workers"] = 1 # set below
e["Solver"]["Episodes Per Generation"] = max(args.concurrentWorkers,1)
e["Solver"]["Multi Agent Relationship"] = "Individual"

e["Solver"]["Experience Replay"]["Start Size"] = 8192
e["Solver"]["Experience Replay"]["Maximum Size"] = 524288
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Serialize"] = True

e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Learning Rate"] = args.learningRate
e["Solver"]["Mini Batch"]["Size"] = 512//(args.nagx*args.nagz)
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
e["Solver"]["Policy"]["Distribution"] = args.pol
#e["Solver"]["Policy"]["Distribution"] = "Normal"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Experiences"] = args.maxExperiences

### Setting file output configuration

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Use Multiple Files"] = False
e["File Output"]["Frequency"] = 10
e["File Output"]["Path"] = f"../_korali_vracer_multi_{args.ycoords}_{args.run}/" #args.resDir

###  Configuring the distributed conduit

if args.test:
    e["Solver"]["Mode"] = "Testing"
    e["Solver"]["Testing"]["Sample Ids"] = [0]

elif args.concurrentWorkers > 1:
    e["Solver"]["Concurrent Workers"] = args.concurrentWorkers

    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed"
    k["Conduit"]["Ranks Per Worker"] = 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

wdir = f"{args.resDir}/sample{rank}/"
print(f'[run_vracer_multi] running with {nState} states and {nAct} actions')

os.makedirs(wdir, exist_ok=True)
print(f'[run_vracer_multi] rank {rank} copying files to {wdir}')

if rank == 0:
    text_file = open(f"{args.resDir}/args.out", "w")
    text_file.write(str(args))
    text_file.close()
    os.system(f"sed 's/SAMPLINGHEIGHT/{args.ycoords}/' {srcDir}bla_macro.i > {wdir}bla_macro.i")
    os.system(f"sed 's/UINIT/init_16x65x16_minchan_00{rank}.u/' {wdir}bla_macro.i > {wdir}bla.i")
    shutil.copy(srcDir + "bla_16x65x16_1", wdir)

if args.concurrentWorkers > 1:
    MPI.COMM_WORLD.Barrier()

### Running Experiment
k.run(e)

print(f'[run_vracer_multi] training finished with args {args}')
