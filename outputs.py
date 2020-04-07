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
        self.daily_new_infections = np.zeros(inputs.time_horizon, dtype=int)
        self.daily_resource_utilization = np.zeros(inputs.time_horizon, RESOURCES_NUM, dtype=int)
        self.inputs = inputs
    # record statistics at end of day, called in step
    def log_daily_state(self, day, states, cumulative, transmissions, infections, mortality, interventions, tests, resources):
        self.daily_states[day, :, :] = states
        self.daily_transmission[day, :] = transmissions
        self.daily_new_infections[day] = infections
        self.daily_mortality[day, :] = mortality
        self.daily_tests[day] = tests
        self.daily_interventions[day,:] = interventions
        self.cumulative_states[day,:] = cumulative
        self.daily_resource_utilization[day,:] = resources

    def log_costs(self, test_costs, intervention_costs, mortality_costs):
        self.daily_costs[day, patient[SUBPOPULATION]] += 1

    def write_outputs(self, file):
        outcomes = DAILY_OUTCOME_STRS
        header = "\t".join(outcomes)
        index = {outcome: index for index, outcome in enumerate(outcomes)}
        data = np.zeros((self.inputs.time_horizon, len(outcomes)), dtype=float)
        data[:,index["day#"]] = np.arange(np.size(data[:,0]))
        data[:,1:1+DISEASE_STATES_NUM] = np.sum(self.daily_states[:,:,:], axis=1)
        data[:,index["cumulative asymptomatic"]:index["cumulative asymptomatic"] + 4] = self.cumulative_states[:,ASYMP:ASYMP+4]
        data[:,index["new infections"]] = self.daily_new_infections
        data[:,index["cumulative infections"]] = np.full(self.inputs.time_horizon, self.inputs.cohort_size, dtype=int) - np.sum(self.daily_states[:,:,SUSCEPTABLE], axis=1)
        data[:,index["dead"]] = np.cumsum(np.sum(self.daily_mortality, axis=1))
        data[:,index["exposures"]] = np.sum(self.daily_transmission, axis=1)
        data[:,index["no intervention"]:index["no intervention"] + INTERVENTIONS_NUM] = self.daily_interventions
        data[:,index["tests"]] = self.daily_tests
        data[:,-RESOURCES_NUM] = self.daily_resource_utilization
        np.savetxt(file, data, fmt="%.6f", delimiter="\t", header=header)
