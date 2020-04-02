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
        self.cumulative_states = np.zeros((inputs.time_horizon, DISEASE_STATES_NUM), dtype = int)
        self.daily_mortality = np.zeros((inputs.time_horizon, SUBPOPULATIONS_NUM), dtype=int)
        self.daily_interventions = np.zeros((inputs.time_horizon, INTERVENTIONS_NUM), dtype=int)
        self.daily_tests = np.zeros(inputs.time_horizon, dtype=int)
#        self.daily_costs = np.zeros((inputs.time_horizon, SUBPOPULATIONS_NUM), dtype=float)
        self.inputs = inputs
    # record statistics at end of day, called in step
    def log_daily_state(self, day, states, cumulative, transmissions, mortality, interventions, tests):
        self.daily_states[day, :, :] = states
        self.daily_transmission[day, :] = transmissions
        self.daily_mortality[day, :] = mortality
        self.daily_tests[day] = tests
        self.daily_interventions[day,:] = interventions
        self.cumulative_states[day,:] = cumulative

    def add_cost(self, cost, day):
        self.daily_costs[day, patient[SUBPOPULATION]] += 1

    def write_outputs(self, file):
        outcomes = DAILY_OUTCOME_STRS
        header = "\t".join(outcomes)
        index = {outcome: index for index, outcome in enumerate(outcomes)}
        data = np.zeros((self.inputs.time_horizon, len(outcomes)), dtype=float)
        data[:,index["day#"]] = np.arange(np.size(data[:,0]))
        data[:,1:1+DISEASE_STATES_NUM] = np.sum(self.daily_states[:,:,:], axis=1)
        data[:,index["cumulative mild"]:index["cumulative mild"] + 4] = self.cumulative_states[:,MILD:MILD+4]
        # data[:,index["cumulative infections"]] = np.full(self.inputs.time_horizon, self.inputs.cohort_size) - self.daily_states[SUSCEPTABLE] - self.daily_states[RECOVERED] - np.cumsum(self.mortality, axis=1)
        data[:,index["dead"]] = np.cumsum(np.sum(self.daily_mortality, axis=1))
        data[:,index["exposures"]] = np.sum(self.daily_transmission, axis=1)
        data[:,index["no intervention"]:index["no intervention"] + INTERVENTIONS_NUM] = self.daily_interventions
        data[:,index["tests"]] = self.daily_tests
        np.savetxt(file, data, fmt="%.6f", delimiter="\t", header=header)
