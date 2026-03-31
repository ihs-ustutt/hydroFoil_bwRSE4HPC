from pyDtOO import dtClusteredSingletonState as stateCounter
import logging 
import numpy as np




def initialize_statecounter(path = "./runData", case_properties = ['eta', 'F', 'dH']):
    ### initiailize statecounter ###
    pathRunData = path
    stateCounter.DATADIR = pathRunData 
    stateCounter.PREFIX = 'DE'
    stateCounter.CASE = 'tbd'
    stateCounter.ADDDATA = case_properties + ['history']
    stateCounter.ADDDATADEF = list(np.zeros(len(case_properties)))  + ['{}']
    return stateCounter


def import_logger(filename = 'de_opt.log', level = logging.DEBUG):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    if not logger.handlers:
        filehandler = logging.FileHandler(filename)
        filehandler.setLevel(level) 
        filehandler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s : %(message)s'))
        logger.addHandler(filehandler)
    return logger










        
