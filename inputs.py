# -*- coding: utf-8 -*-
"""
COVID-19 Microsimulation Model

@author: Chris Panella (cpanella@mgh.harvard.edu)
"""

import numpy as np
import json
from enums import *

def dict_to_array(d):
    if type(d) is dict:
        return [dict_to_array(v) for v in list(d.values())]
    else:
        return d

def generate_simulation_inputs():
    sim_in = {
        "cohort_size": 1000,
        "time_horizon": 180
    }
    return sim_in

def generate_initialization_inputs():

    init_in = {
        "subpopulation dist":  dict(zip(SUBPOPULATION_STRS, [1] + ([0] * (SUBPOPULATIONS_NUM - 1)))),
        "initial disease dist": dict(zip(DISEASE_STATE_STRS, [1] + ([0] * (DISEASE_STATES_NUM - 1))))
    }
    return init_in

def generate_transition_inputs():
    state_trans_in = {
    "daily progression probability": {"for " + subpop: {"from " + prevstate: {"to " + newstate: 0
                for newstate in DISEASE_STATE_STRS[MILD:RECOVERED]}
            for prevstate in DISEASE_STATE_STRS[INCUBATION:RECOVERED]}
        for subpop in SUBPOPULATION_STRS},
    "daily recovery probability": {" for " + subpop: {"while " + dstate: 0
            for dstate in DISEASE_STATE_STRS[MILD:RECOVERED]}
        for subpop in SUBPOPULATION_STRS},
    "daily mortality probability while critical": {"for " + subpop: 0
        for subpop in SUBPOPULATION_STRS}
    }
    return state_trans_in


def generate_transmission_inputs():
    transm_in = {
    "daily transmission probability": {"for " + subpop: {"while " + dstate: 0
            for dstate in DISEASE_STATE_STRS[MILD:RECOVERED]}
        for subpop in SUBPOPULATION_STRS}
    }
    return transm_in

def generate_intervention_input():
    treatm_in = {
    "resource constraint": 0,
    "require positive test": False,
    "availible for disease state": {dstate: False for dstate in DISEASE_STATE_STRS[MILD:RECOVERED]},
    "disease progression rate multipliers": {"for " + subpop: {"while " + dstate: 1.0
            for dstate in DISEASE_STATE_STRS[MILD:RECOVERED]}
        for subpop in SUBPOPULATION_STRS},
    "recovery rate multipliers": {"for " + subpop: {"while " + dstate: 1.0
            for dstate in DISEASE_STATE_STRS[MILD:RECOVERED]}
        for subpop in SUBPOPULATION_STRS},
    "transmisson rate multipliers": {"while " + dstate: 1.0
        for dstate in DISEASE_STATE_STRS[MILD:RECOVERED]},
    "mortality rate multipliers": {"for " + subpop: 1.0
        for subpop in SUBPOPULATION_STRS},
    "INTERVENTION costs": {"for " + subpop: {"while " + dstate: 0
            for dstate in DISEASE_STATE_STRS[MILD:RECOVERED]}
        for subpop in SUBPOPULATION_STRS}
    }
    return treatm_in

# generate imput format
def generate_input_dict():
    inputs = {}
    # Simulation Params
    inputs["simulation parameters"] = generate_simulation_inputs()
    # Initial State
    inputs["initial state"] = generate_initialization_inputs()
    # Disease State Transitions
    inputs["state transitions"] = generate_transition_inputs()
    # Transmissions
    inputs["transmissions"] = generate_transmission_inputs()
    """
    # INTERVENTION
    inputs["INTERVENTIONs"] = {INTERVENTION: generate_INTERVENTION_input() for INTERVENTION in INTERVENTION_STRS}
    """
    return inputs

class Inputs():
    def __init__(self):
        # simulation parameters
        self.cohort_size = 1000
        self.time_horizon = 180
        # initialization inputs
        self.subpop_dist = np.zeros((SUBPOPULATIONS_NUM), dtype=float)
        self.dstate_dist = np.zeros((DISEASE_STATES_NUM), dtype=float)
        # transition inputs
        self.progression_probs = np.zeros((SUBPOPULATIONS_NUM, DISEASE_STATES_NUM, DISEASE_STATES_NUM), dtype=float)
        self.mortality_probs = np.zeros((SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        #transmission inputs
        self.trans_prob = np.zeros((SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        """
        #INTERVENTION inputs
        self.treatm_constraints = np.zeros(INTERVENTIONS_NUM, dtype=int)
        self.treatm_requires_test = np.zeros((INTERVENTIONS_NUM), dtype=bool)
        self.treatm_eligible_states = np.zeros((INTERVENTIONS_NUM, DISEASE_STATES_NUM), dtype=bool)
        self.treatm_prog_mults = np.zeros((INTERVENTIONS_NUM, SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        self.treatm_recovery_mults = np.zeros((INTERVENTIONS_NUM, SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        self.treatm_mort_mults = np.zeros((INTERVENTIONS_NUM, SUBPOPULATIONS_NUM), dtype=float)
        self.treatm_trans_mults = np.zeros((INTERVENTIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        """
    def read_inputs(self, param_dict):
        # simulation parameters
        sim_params = param_dict["simulation parameters"]
        self.cohort_size = sim_params["cohort_size"]
        self.time_horizon = sim_params["time_horizon"]
        # initialization inputs
        init_params = param_dict["initial state"]
        self.subpop_dist = np.asarray(dict_to_array(init_params["subpopulation dist"]), dtype=float)
        self.dstate_dist = np.asarray(dict_to_array(init_params["initial disease dist"]), dtype=float)
        # transition inputs
        trans_params = param_dict["state transitions"]
        self.progression_probs[:,INCUBATION:RECOVERED,MILD:RECOVERED] = np.asarray(dict_to_array(trans_params["daily progression probability"]), dtype=float)
        self.progression_probs[:,MILD:RECOVERED,RECOVERED] = np.asarray(dict_to_array(trans_params["daily recovery probability"]), dtype=float)
        self.mortality_probs = np.asarray(dict_to_array(trans_params["daily mortality probability while critical"]), dtype=float)
        #transmission inputs
        transm_params = param_dict["transmissions"]
        self.trans_prob[:,MILD:RECOVERED] = np.asarray(dict_to_array(transm_params["daily transmission probability"]), dtype=float)
        """
        #INTERVENTION inputs
        treatm_params = param_dict["INTERVENTIONs"]
        for i in range(INTERVENTIONS_NUM):
            treatm = treatm_params[INTERVENTION_STRS[i]]
            self.treatm_constraints[i] = treatm["resource constraint"]
            self.treatm_requires_test[i] = treatm["require positive test"]
            self.treatm_eligible_states[i,MILD:RECOVERED] = np.asarray(dict_to_array(treatm["availible for disease state"]), dtype=bool)
            self.treatm_recovery_mults[i,:,MILD:RECOVERED] = np.asarray(dict_to_array(treatm["recovery rate multipliers"]), dtype=float)
            self.treatm_prog_mults[i,:,MILD:RECOVERED] = np.asarray(dict_to_array(treatm["disease progression rate multipliers"]), dtype=float)
            self.treatm_mort_mults[i,:] =  np.asarray(dict_to_array(treatm["mortality rate multipliers"]), dtype=float)
            self.treatm_trans_mults[i,MILD:RECOVERED] = np.asarray(dict_to_array(treatm["transmisson rate multipliers"]), dtype=float)
        """

def create_input_file(file):
    with open(file, 'w') as f:
        text = json.dumps(generate_input_dict(), indent=2)
        f.write(text)


# takes a filepath and returns
def read_inputs(file):
    with open(file, 'r') as f:
        text = f.read()
        # parse JSON
        param_dict = json.loads(text)
        # create inputs object and populate fields
        inputs = Inputs()
        inputs.read_inputs(param_dict)
        return inputs

