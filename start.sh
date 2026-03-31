#!/bin/bash

#SBATCH --job-name=de_opt
#SBATCH --output=de_opt.out
#SBATCH --time=00:25:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --partition=dev_cpu_il
##SBATCH --exclusive
##SBATCH --dependency=singleton

sh clean.sh
source de
#env | grep -i SLURM > env_log 
ns_node=$SLURMD_NODENAME
pyro5-ns -n $ns_node &
#echo $ns_node >> ns_node 
for i in $(seq 1 $SLURM_NTASKS); do
	srun -n 1 -N 1  python server.py $i $ns_node >server_log 2>&1&
done
python start_de.py $ns_node > log 2>&1

