### Exemplary optimization framework for optimizing the hydroFoil case 


## How to run the optimization 

0. add your python environment to "de" file 

1. run slurm start script: start.sh

2. sacct -j <jobid> to see how job steps counter increases...(job step counter only increases when cpus_per_task > 1 in start.sh)

(3.) observe optimization progress with "tail -f de_opt.log"


## Necessary python packages: 

(pip install...)
- numpy (version: 2.1.2!!)
- pygmo
- Pyro5
- oslo.concurrency
- foamlib
- scikit-learn
(for further information see requirements.txt)


## How to adapt to new optimization problems: ###

1. define new optimization problem and implement function def run_problem(...):

2. import file in server.py and add run_problem() in manager_object

3. implement new pygmo_problem in problem.py (adapt Pyro evaluation to "manager.run_problem()", and change upper and lower bound according to your problem)

4. add this pygmo problem to your archipelago in start_de.py 
