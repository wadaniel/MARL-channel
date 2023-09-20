#! /usr/bin/env bash

if [ $# -lt 1 ] ; then
	echo "Usage: bash ./sbatch-run-vracer-marl.sh RUNNAME"
	exit 1
fi

RUNNAME=$1

# number of agents
NAGENTS=2

# number of workers
NWORKER=32
# NWORKER=1

# number of nodes per worker
NRANKS=1
# NRANKS=9

# number of cores per worker
NUMCORES=12

# number of workers * number of nodes per worker
NNODES=$(( $NWORKER * $NRANKS ))

# setup run directory and copy necessary files
#RUNPATH="${SCRATCH}/MARL_channel/marl/${RUNNAME}"
#mkdir -p ${RUNPATH}
#cp run-irl-swimmer ${RUNPATH}
#cd ${RUNPATH}

cd ../marl/

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s1160
#SBATCH --constraint=gpu
#SBATCH --job-name="channel_marl_${RUNNAME}"
#SBATCH --output=${RUNNAME}_out_%j.txt
#SBATCH --error=${RUNNAME}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=normal
##SBATCH --time=00:30:00
##SBATCH --partition=debug
#SBATCH --nodes=$((NNODES+1))
#SBATCH --mail-user=wadaniel@ethz.ch
#SBATCH --mail-type=END,FAIL

srun --nodes=$((NNODES+1)) --ntasks-per-node=1 --cpus-per-task=$NUMCORES --threads-per-core=1 python ./run-vracer-multi.py --run ${RUNNAME}

EOF
#python -m korali.plot --dir _trainingResults --out vracer_irl.pdf
#python -m korali.rlview --dir _trainingResults --out reward.pdf
#python -m korali.rlview --dir _trainingResults --featureReward --out freward.pdf

echo "Starting with ${NWORKER} simulations each using ${NRANKS} nodes with ${NUMCORES} cores"
echo "----------------------------"

chmod 755 daint_sbatch
sbatch daint_sbatch
