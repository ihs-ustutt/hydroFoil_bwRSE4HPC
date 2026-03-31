import Pyro5.api
from utils import *
from hydroFoil import *
import os
import socket
import numpy
import time

host = socket.gethostname()
env_id = sys.argv[1]
nameserver_node = sys.argv[2]
Pyro5.config.HOST = host
name = host + "_" + str(env_id) + ".manager"


@Pyro5.api.expose
class manager(object):
    def run_hydroFoil(self, input_dict):
        x = input_dict['x']
        state = input_dict['state']
        fit, fit_extra, state, history = runHydFoil(x, state)
        return fit, fit_extra, state, history

### REGISTER SERVER ###
daemon = Pyro5.server.Daemon(host)
ns = Pyro5.api.locate_ns(host=nameserver_node)
uri = daemon.register(manager)    # register the greeting maker as a Pyro object
ns.register(name, uri)
daemon.requestLoop()              # start the event loop of the server to wait for calls

