# -*- coding: utf-8 -*-
"""
COVID-19 Microsimulation Model

@author: Chris Panella (cpanella@mgh.harvard.edu)
"""

import numpy as np
from enums import *

class Outputs:
    def __init__(self, inputs):
        self.daily_transmission = np.zeros((inputs.time_horizon, TRANSMISSION_GROUPS_NUM, TRANSMISSION_GROUPS_NUM), dtype=float)
        self.daily_states = np.zeros((inputs.time_horizon, SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=int)
        self.cumulative_states = np.zeros((inputs.time_horizon, DISEASE_STATES_NUM), dtype = int)
        self.daily_mortality = np.zeros((inputs.time_horizon, SUBPOPULATIONS_NUM, INTERVENTIONS_NUM), dtype=int)
        self.daily_interventions = np.zeros((inputs.time_horizon, INTERVENTIONS_NUM, DISEASE_STATES_NUM), dtype=int)
        self.daily_tests = np.zeros((inputs.time_horizon, TESTS_NUM, 2), dtype=int)
        self.daily_new_infections = np.zeros((inputs.time_horizon, TRANSMISSION_GROUPS_NUM), dtype=int)
        self.non_covid_presenting = np.zeros(inputs.time_horizon, dtype=int)
        self.daily_resource_utilization = np.zeros((inputs.time_horizon, RESOURCES_NUM), dtype=int)
        self.costs = np.zeros((inputs.time_horizon, 3), dtype=float)
        self.inputs = inputs
    # record statistics at end of day, called in step
    def log_daily_state(self, day, states, cumulative, transmissions, infections, mortality, interventions, tests, resources, non_covid, costs):
        self.daily_states[day, :, :] = states
        self.daily_transmission[day, :] = transmissions
        self.daily_new_infections[day, :] = infections
        self.daily_mortality[day, :, :] = mortality
        self.daily_tests[day,:] = tests
        self.daily_interventions[day,:,:] = interventions
        self.cumulative_states[day,:] = cumulative
        self.daily_resource_utilization[day,:] = resources
        self.non_covid_presenting[day] = non_covid
        self.costs[day] = costs

    def write_outputs(self, file, state_detail=None):
        outcomes = DAILY_OUTCOME_STRS
        header = "\t".join(outcomes)
        index = {outcome: index for index, outcome in enumerate(outcomes)}
        data = np.zeros((self.inputs.time_horizon, len(outcomes)), dtype=float)
        data[:,index["day#"]] = np.arange(np.size(data[:,0]))
        data[:,1:1+DISEASE_STATES_NUM] = np.sum(self.daily_states[:,:,:], axis=1)
        data[:,index["cumulative asymptomatic"]:index["cumulative asymptomatic"] + 4] = self.cumulative_states[:,ASYMP:ASYMP+4]
        data[:,index["tn_group 0 new infections"]:index["tn_group 0 new infections"] + TRANSMISSION_GROUPS_NUM] = self.daily_new_infections
        data[:,index["cumulative infections"]] = np.full(self.inputs.time_horizon, self.inputs.cohort_size, dtype=int) - np.sum(self.daily_states[:,:,SUSCEPTABLE], axis=1)
        data[:,index["dead"]] = np.cumsum(np.sum(self.daily_mortality, axis=(1,2)))
        data[:,index["dead"] + 1 : index["dead"] + 1 + SUBPOPULATIONS_NUM] = np.sum(self.daily_mortality, axis=2)
        data[:,index["FoI tn_group 0 -> tn_group 0"]:index["FoI tn_group 0 -> tn_group 0"] + (TRANSMISSION_GROUPS_NUM**2)] = np.reshape(np.swapaxes(self.daily_transmission, 1, 2), (-1, TRANSMISSION_GROUPS_NUM**2))
        data[:,index["non-covid presenting"]] = self.non_covid_presenting        
        data[:,index["intervention 0"]:index["intervention 0"] + INTERVENTIONS_NUM] = np.sum(self.daily_interventions,axis=2)
        data[:,index["test 0 (-)"]:index["test 0 (-)"] + (2*TESTS_NUM)] = np.reshape(self.daily_tests, (-1, 2*TESTS_NUM))
        data[:,index["test costs"]:index["test costs"]+3] = self.costs
        data[:,index["test costs"]+3:index["test costs"]+3+RESOURCES_NUM] = self.daily_resource_utilization
        data[:, -INTERVENTIONS_NUM:] = np.sum(self.daily_mortality, axis=1)
        np.savetxt(file, data, fmt="%.6f", delimiter="\t", header=header)
        if state_detail:
            state_header = "\t".join(["day #"] + [f"{intv} while {dstate}" for intv in INTERVENTION_STRS for dstate in DISEASE_STATE_STRS])
            state_data = np.zeros((self.inputs.time_horizon, (INTERVENTIONS_NUM*DISEASE_STATES_NUM)+1), dtype=int)
            state_data[:,0] = np.arange(np.size(data[:,0]))
            state_data[:,1:] = np.reshape(self.daily_interventions,(self.inputs.time_horizon, (INTERVENTIONS_NUM*DISEASE_STATES_NUM)))
            np.savetxt(state_detail, state_data, fmt="%.6f", delimiter="\t", header=state_header)

