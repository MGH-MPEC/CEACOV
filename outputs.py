# -*- coding: utf-8 -*-
"""
COVID-19 Microsimulation Model

@author: Chris Panella (cpanella@mgh.harvard.edu)
"""

import numpy as np
from enums import *

class Outputs:
    def __init__(self, inputs):
        self.daily_transmission = np.zeros((inputs.time_horizon, SUBPOPULATIONS_NUM), dtype=float)
        self.daily_states = np.zeros((inputs.time_horizon, SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=int)
        self.daily_mortality = np.zeros((inputs.time_horizon, SUBPOPULATIONS_NUM), dtype=int)
        self.daily_costs = np.zeros((inputs.time_horizon, SUBPOPULATIONS_NUM), dtype=float)
        self.inputs = inputs
    # record statistics at end of day, called in step
    def log_daily_state(self, day, states, transmissions, mortality):
        self.daily_states[day, :, :] = states
        self.daily_transmission[day, :] = transmissions
        self.daily_mortality[day, :] = mortality

    def add_cost(self, cost, day):
        self.daily_costs[day, patient[SUBPOPULATION]] += 1

    def write_outputs(self, file):
        outcomes = ["day#"] +  list(DISEASE_STATE_STRS) + ["dead", "exposures"]
        header = "\t".join(outcomes)
        data = np.zeros((self.inputs.time_horizon, len(outcomes)), dtype=float)
        data[:,0] = np.arange(np.size(data[:,0]))
        data[:,1:1+DISEASE_STATES_NUM] = np.sum(self.daily_states[:,:,:], axis=1)
        data[:,-2] = np.cumsum(np.sum(self.daily_mortality, axis=1))
        data[:,-1] = np.sum(self.daily_transmission, axis=1)
        np.savetxt(file, data, fmt="%.6f", delimiter="\t", header=header)
