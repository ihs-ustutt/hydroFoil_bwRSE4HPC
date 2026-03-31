import pygmo as pg
from problem import * 
from utils import * 
import json
import os
from random import sample
import Pyro5.api
from oslo_concurrency import lockutils
stateCounter = initialize_statecounter()
logger = import_logger()

lb = np.array([150, 155, 0.01]) ### adapt: your_boundaries
ub = np.array([170, 175, 0.1])

### optimization setting ###
numb_of_gens = 1
n_procs = int(os.environ["SLURM_NTASKS"]) ## == amount of islands
number_of_islands = n_procs
max_simulations = 35000
restart = False
if 'runData' in os.listdir():
    restart = True


### auxiliaries ###
def set_start_pop(pop, pop_size):
    with open('start_db.json','r') as file:
        start_db = json.load(file)
    integer_keys = [int(key) for key in list(start_db.keys())]
    start_db_max = max(integer_keys)
    individuums_list = sample(range(1,start_db_max),pop_size) 
    for individuum in individuums_list:
        obj = start_db[str(individuum)][0]
        obj = np.array(start_db[str(individuum)][0])
        obj = ( obj - lb ) / (ub-lb) #entnormen [0,1] in start_db.json
        pop.push_back(obj, np.array([start_db[str(individuum)][1]]))


def save_archi(archi, archi_dict = {}, island = 'all'):  
    if island == 'all':
        for i, island in enumerate(archi):
            archi_dict.update({str(i): {'pop': island.get_population().get_x().tolist(),
                                        'f':   island.get_population().get_f().tolist()}}) 
    elif isinstance(island, int):
        i = island
        island = archi
        archi_dict.update({str(i): {'pop': island.get_population().get_x().tolist(),
                                    'f':   island.get_population().get_f().tolist()}}) 
    with open('archi_save.json', "w") as file:
        json.dump(archi_dict,file)
    return archi_dict


def set_start_pop_when_restart(pop, island_id, archi_dict_path = 'archi_save.json'):
    with open(archi_dict_path, 'r') as file:
        archi_dict = json.load(file)
    x = archi_dict[str(island_id)]['pop']
    f = archi_dict[str(island_id)]['f']
    for i,obj in enumerate(x):
        pop.push_back(x=obj, f = f[i])



if __name__ == "__main__":
    if restart: 
        logger.info("This is a RESTARTED run.")
    ### SET UP PYGMO STUFF ###
    algo = pg.algorithm(pg.de(gen = numb_of_gens))

    ### Fill Server list ###
    logger.info(f"n_procs: {n_procs}.")
    server_list = []
    check_length = n_procs
    while len(server_list) != check_length:
        logger.info(f"Checking name_server_list: current length {len(server_list)}. Wait...")
        nameserver_host = sys.argv[1] 
        server_list = [key for key in Pyro5.api.locate_ns(host=nameserver_host).list().keys()][1::]
        time.sleep(10)
    logger.info(f" All {check_length} server are ready. ")

    ### Initialize Archipelago ###
    logger.info("Start init archi...")
    archi = pg.archipelago()
    archi.set_topology(pg.fully_connected())
    archi.set_migrant_handling(pg.core.migrant_handling.evict)
    logger.info(f"Initialize Islands with number_of_islands={number_of_islands}.")
    for isleID in range(number_of_islands):
        prob = pg.problem(hydroFoil_problem(server=server_list[isleID]))
        pop = pg.population(prob)
        ### set start population ###
        if not restart:
            set_start_pop(pop, pop_size = 8)
        elif restart:
            set_start_pop_when_restart(pop, isleID)
        archi.push_back( pg.island(algo = algo,pop = pop)) 
    logger.info("Finished Archi init." )
    archi_dict = save_archi(archi, island = 'all')

    #### EVOLVE ####
    logger.info("Start evolve loop with current status:")
    state_max = 0
    while (state_max < max_simulations):
        try:
            archSize = np.size(archi)
            for i in range(archSize):
                if archi[i].status == pg.core.evolve_status.idle:
                  logging.info( 'Archipelago[ %d ] in idle', i )
                  logging.info( 'Evolve archipelago[ %d ] again ...', i )
                  try:
                      state_max = stateCounter.currentMaxId()
                  except ValueError: ###no individuals yet
                      pass
                  try:
                      archi_dict = save_archi(archi[i], archi_dict, island = i)
                  except:
                      logger.info("Not saved archi...")
                  archi[i].evolve()
                  time.sleep(5)
                elif archi[i].status == pg.core.evolve_status.busy_error:
                  logging.warning('Archipelago[ %d ] in busy_error', i )
                  logging.info( "### Archipelago Status Output ### \n%s" % (archi.__str__()) )
                elif archi[i].status == pg.core.evolve_status.idle_error:
                  logging.warning('Archipelago[ %d ] in idle_error', i )
                  #logging.info( "### Archipelago Status Output ### \n%s" % (archi.__str__()) )
                  logging.info( "Call wait_check on island %d" % i )
                  archi[i].wait_check()
                else:
                  time.sleep(5)
        except Exception as inst:
            logging.exception(inst)   
            logging.error("BREAKING EVOLVE LOOP")
            break



    




