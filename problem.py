import sys
from utils import *
import pygmo as pg
from hydroFoil import * 
import Pyro5.api
logger = import_logger()
stateCounter = initialize_statecounter()


class hydroFoil_problem():
    """
    Custom Problem for hydroFoil 
    """
    def __init__(self,server):
        self.server = server

    def reach_out_to_pyro(self, x):
        state = stateCounter(defObj = x, defFit = 1e6).state() 
        logger.info(f"Start proxy for {state} on {self.server}:")
        try:
            with Pyro5.api.Proxy(f"PYRONAME:{self.server}") as manager_object: 
                fit, fit_extra, state, history = manager_object.run_hydroFoil({'x':x.tolist(), 'state':state}) ### adapt: run_your_case
                self.update_stateCounter(fit, fit_extra, state, history)
                logger.info(f"Finished Proxy.run_tistos() for {state}.")
        except Pyro5.errors.NamingError:
            logger.debug(f"server hasn't started yet: {self.server}" )
        except Exception:
            logger.warning(f"Pyro Traceback Error on {self.server}.")
            logger.warning("".join(Pyro5.errors.get_pyro_traceback()))
        return fit, state

    def update_stateCounter(self, fit, fit_extra, state, history):
        s = stateCounter(state)        
        s.update('fitness', fit)
        s.update('history', history)
        s.update('dH', fit_extra['dHMean'])
        s.update('F', fit_extra['FMean'])
        s.update('eta', fit_extra['eta'])

    def fitness(self,x):
        lb = np.array([150, 155, 0.01]) ### adapt: your_boundaries
        ub = np.array([170, 175, 0.1])
        x_u = np.zeros(3)
        x_u[0] = lb[0] + (ub[0]-lb[0]) * x[0]
        x_u[1] = lb[1] + (ub[1]-lb[1]) * x[1]
        x_u[2] = lb[2] + (ub[2]-lb[2]) * x[2]
        fit, state = self.reach_out_to_pyro(x_u)
        return np.array([fit])

    def get_bounds(self): 
            ### Set Problem Bounds ###
            min_vec = [0, 0, 0]
            max_vec = [1, 1, 1]
            return (min_vec, max_vec)

