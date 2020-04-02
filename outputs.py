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
        self.daily_new_infections = np.zeros(inputs.time_horizon, dtype=int)
        self.daily_interventions = np.zeros((inputs.time_horizon, INTERVENTIONS_NUM), dtype=int)
        self.daily_tests = np.zeros(inputs.time_horizon, dtype=int)
#        self.daily_costs = np.zeros((inputs.time_horizon, SUBPOPULATIONS_NUM), dtype=float)
        self.inputs = inputs
    # record statistics at end of day, called in step
    def log_daily_state(self, day, states, transmissions, mortality, infections, interventions, tests):
        self.daily_states[day, :, :] = states
        self.daily_transmission[day, :] = transmissions
        self.daily_mortality[day, :] = mortality
        self.daily_tests[day] = tests
        self.daily_interventions[day,:] = interventions
        self.daily_new_infections[day] = infections

    def add_cost(self, cost, day):
        self.daily_costs[day, patient[SUBPOPULATION]] += 1

    def write_outputs(self, file):
        outcomes = DAILY_OUTCOME_STRS
        header = "\t".join(outcomes)
        index = {outcome: index for index, outcome in enumerate(outcomes)}
        data = np.zeros((self.inputs.time_horizon, len(outcomes)), dtype=float)
        data[:,index["day#"]] = np.arange(np.size(data[:,0]))
        data[:,1:1+DISEASE_STATES_NUM] = np.sum(self.daily_states[:,:,:], axis=1)
        data[:,index["infections"]] = np.cumsum(self.daily_new_infections)
        data[:,index["dead"]] = np.cumsum(np.sum(self.daily_mortality, axis=1))
        data[:,index["exposures"]] = np.sum(self.daily_transmission, axis=1)
        data[:,index["no intervention"]:index["no intervention"] + INTERVENTIONS_NUM] = self.daily_interventions
        data[:,index["tests"]] = np.cumsum(self.daily_tests, axis=1)
        np.savetxt(file, data, fmt="%.6f", delimiter="\t", header=header)
