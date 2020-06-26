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

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def generate_simulation_inputs():
    sim_in = {
        "cohort size": 1000,
        "time horizon": 180,
        "fixed seed": True,
        "detailed state outputs": False
    }
    return sim_in


def generate_initialization_inputs():
    init_in = {
        "transmission group dist": dict(zip(TRANSMISSION_GROUP_STRS, [1] + ([0] * (TRANSMISSION_GROUPS_NUM - 1)))),
        "risk category dist":  {f"for {tgroup}": dict(zip(SUBPOPULATION_STRS, [1] + ([0] * (SUBPOPULATIONS_NUM - 1))))
            for tgroup in TRANSMISSION_GROUP_STRS},
        "initial disease dist": {f"for {tgroup}": dict(zip(DISEASE_STATE_STRS, [1] + ([0] * (DISEASE_STATES_NUM - 1))))
            for tgroup in TRANSMISSION_GROUP_STRS},
        "severity dist": {f"for {subpop}": dict(zip(DISEASE_PROGRESSION_STRS, [1] + ([0] * (DISEASE_PROGRESSIONS_NUM - 1))))
            for subpop in SUBPOPULATION_STRS},
        "start intervention": {f"for {tgroup}": 0
            for tgroup in TRANSMISSION_GROUP_STRS}
    }
    return init_in


def generate_progression_inputs():
    prog_in = {f"daily disease progression probability for {INTERVENTION_STRS[intv]}":
            {f"for severity = {DISEASE_PROGRESSION_STRS[severity]}":
                {f"from {DISEASE_STATE_STRS[dstate]} to {DISEASE_STATE_STRS[PROGRESSION_PATHS[severity][dstate]]}": 0
                for dstate in DISEASE_STATES[INCUBATION:RECOVERED] if PROGRESSION_PATHS[severity][dstate]}
            for severity in DISEASE_PROGRESSIONS}
        for intv in INTERVENTIONS}
    prog_in["daily prob recovery from severe state in critical path"] = {f"for {INTERVENTION_STRS[intv]}": 0
        for intv in INTERVENTIONS}
    prog_in["initial immunity on recovery probability"] = {f"for {INTERVENTION_STRS[intv]}": {f"for severity = {severity}": 1
            for severity in DISEASE_PROGRESSIONS}
        for intv in INTERVENTIONS}
    return prog_in


def generate_mortality_inputs():
    mort_in = {f"daily mortality prob for {subpop}": {f"on {intv}": {f"while {dstate}": 0
                for dstate in DISEASE_STATE_STRS[SEVERE:CRITICAL+1]}
            for intv in INTERVENTION_STRS}
        for subpop in SUBPOPULATION_STRS}
    return mort_in


def generate_transmission_inputs():
    transm_in = {
    "transmission rate": {f"for {intv}": {f"while {dstate}": 0
            for dstate in DISEASE_STATE_STRS[ASYMP:RECOVERED]}
        for intv in INTERVENTION_STRS},
    "rate multiplier thresholds": {f"t{threshold}": (10 + 10*threshold)
        for threshold in range(T_RATE_PERIODS_NUM - 1)},
    "rate multipliers": {threshold: {"for " + intv: 1
            for intv in INTERVENTION_STRS}
        for threshold in ["for 0 <= day# < t0"] +
                          [f"for t{i} <= day# < t{i+1}" for i in range(T_RATE_PERIODS_NUM-2)] +
                          [f"day# > t{T_RATE_PERIODS_NUM-1}"]},
    "contact matrix":{f"for {intv}": {f"from {tgroup}": {f"to {igroup}": 1
                for igroup in TRANSMISSION_GROUP_STRS}
            for tgroup in TRANSMISSION_GROUP_STRS}
        for intv in INTERVENTION_STRS}
    }
    return transm_in


def generate_test_inputs():
    testing_in = {
    f"test {test}": {
        "result return time": 0,
        "probability of positive result": {f"for {dstate}": 0.0
            for dstate in DISEASE_STATE_STRS},
        "delay to test": 0
        }
        for test in TESTS}
    return testing_in


def generate_testing_strat_inputs():
    test_strat_in = { 
    INTERVENTION_STRS[n]: {
        "probability of presenting to care": {f"while {dstate}": 0
            for dstate in DISEASE_STATE_STRS},
        "switch to intervention on positive test result": {f"if observed {symstate}": n
            for symstate in OBSERVED_STATE_STRS},
        "switch to intervention on negative test result": {f"if observed {symstate}": n
            for symstate in OBSERVED_STATE_STRS},
        "test number": {f"if observed {symstate}": 0
            for symstate in OBSERVED_STATE_STRS},
        "testing frequency": {f"if observed {symstate}": 1
            for symstate in OBSERVED_STATE_STRS},
        "probability receive test": {f"if observed {symstate}": {f"for {subpop}": 0
                for subpop in SUBPOPULATION_STRS}
            for symstate in OBSERVED_STATE_STRS}
        }
        for n in INTERVENTIONS}
    return test_strat_in

def generate_cost_inputs():
    cost_in = {
        "testing costs": {f"test {test}": 0.0
            for test in TESTS},
        "daily intervention costs": {intervention: {f"if observed {symstate}": 0.0
                    for symstate in OBSERVED_STATE_STRS}
            for intervention in INTERVENTION_STRS},                                                    
        "mortality costs": {intervention: 0.0
            for intervention in INTERVENTION_STRS}
        }
    return cost_in

def generate_resource_inputs():
    rsc_in = {
    "resource availabilities": {resource: 0
        for resource in RESOURCE_STRS},
    "resource requirements":{f"for {intervention}":{f"if observed {symstate}": []
            for symstate in OBSERVED_STATE_STRS}
        for intervention in INTERVENTION_STRS},
    "back-up interventions": {f"for {intervention}":{f"if observed {symstate}": 0
            for symstate in OBSERVED_STATE_STRS}
        for intervention in INTERVENTION_STRS}
    }
    return rsc_in

def generate_non_covid_inputs():
    no_co_in = {
    "daily prob present non-covid":{f"with {symstate} symptoms":{f"for {subpop}": 0.0
            for subpop in SUBPOPULATION_STRS}
        for symstate in OBSERVED_STATE_STRS[SYMP_MODERATE:SYMP_CRITICAL+1]},
    "non covid symptom duration":{f"for {symstate} symptoms":{f"for {subpop}": 0
            for subpop in SUBPOPULATION_STRS}
        for symstate in OBSERVED_STATE_STRS[SYMP_MODERATE:SYMP_CRITICAL+1]},
    }
    return no_co_in


# generate imput format
def generate_input_dict():
    inputs = {}
    # Model Version
    inputs["model version"] = MODEL_VERSION
    # Simulation Params
    inputs["simulation parameters"] = generate_simulation_inputs()
    # Initial State
    inputs["initial state"] = generate_initialization_inputs()
    # Disease Progression
    inputs["disease progression"] = generate_progression_inputs()
    # Mortality
    inputs["disease mortality"] = generate_mortality_inputs()
    # Transmissions
    inputs["transmissions"] = generate_transmission_inputs()
    # Tests
    inputs["tests"] = generate_test_inputs()
    # Intervention Strategies
    inputs["testing strategies"] = generate_testing_strat_inputs()
    # Costs
    inputs["costs"] = generate_cost_inputs()
    # Resources
    inputs["resources"] = generate_resource_inputs()
    # Non-COVID RI
    inputs["non-covid illness"] = generate_non_covid_inputs()

    return inputs

class Inputs():
    def __init__(self):
        # simulation parameters
        self.cohort_size = 1000
        self.time_horizon = 180
        self.fixed_seed = True
        self.state_detail = False
        # initialization inputs
        self.tgroup_dist = np.zeros((TRANSMISSION_GROUPS_NUM), dtype=float)
        self.subpop_dist = np.zeros((TRANSMISSION_GROUPS_NUM, SUBPOPULATIONS_NUM), dtype=float)
        self.dstate_dist = np.zeros((TRANSMISSION_GROUPS_NUM, DISEASE_STATES_NUM), dtype=float)
        self.severity_dist = np.zeros((SUBPOPULATIONS_NUM, DISEASE_PROGRESSIONS_NUM), dtype=float)
        self.start_intvs = np.zeros((TRANSMISSION_GROUPS_NUM), dtype=int)
        # transition inputs
        self.progression_probs = np.zeros((INTERVENTIONS_NUM, DISEASE_PROGRESSIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        self.severe_kludge_probs = np.zeros((INTERVENTIONS_NUM), dtype=float)
        self.initial_prob_immunity = np.zeros((INTERVENTIONS_NUM, DISEASE_PROGRESSIONS_NUM), dtype=float)
        self.mortality_probs = np.zeros((SUBPOPULATIONS_NUM, INTERVENTIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        #transmission inputs
        self.trans_rate_thresholds = np.zeros(T_RATE_PERIODS_NUM-1, dtype=int)
        self.trans_prob = np.zeros((T_RATE_PERIODS_NUM, INTERVENTIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        # transmissions from, transmissions to
        self.contact_matrices = np.zeros((INTERVENTIONS_NUM, TRANSMISSION_GROUPS_NUM, TRANSMISSION_GROUPS_NUM), dtype=float)
        # test inputs
        self.test_return_delay = np.zeros(TESTS_NUM, dtype=int)
        self.test_characteristics = np.zeros((TESTS_NUM, DISEASE_STATES_NUM), dtype=float)
        self.test_lag = np.zeros((TESTS_NUM), dtype=int)
        # intervention
        self.prob_present = np.zeros((INTERVENTIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        self.switch_on_test_result = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM, 2), dtype=int)
        self.test_number = np.zeros((INTERVENTIONS_NUM,OBSERVED_STATES_NUM), dtype=int)
        self.testing_frequency = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=int)
        self.prob_receive_test = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM, SUBPOPULATIONS_NUM), dtype=float)
        # cost inputs
        self.testing_costs = np.zeros(TESTS_NUM, dtype=float)
        self.intervention_daily_costs = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=float)
        self.mortality_costs = np.zeros((DISEASE_STATES_NUM, INTERVENTIONS_NUM), dtype=float)
        # resource inputs
        self.resource_base_availability = np.zeros(RESOURCES_NUM, dtype=int)
        self.resource_requirements = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=np.uint8)
        self.fallback_interventions = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=int)
        # non-covid RI
        self.prob_present_non_covid = np.zeros((OBSERVED_STATES_NUM, SUBPOPULATIONS_NUM), dtype=float)
        self.non_covid_ri_durations = np.zeros((OBSERVED_STATES_NUM, SUBPOPULATIONS_NUM), dtype=float)

    def read_inputs(self, param_dict):
        if param_dict["model version"] != MODEL_VERSION:
            raise InvalidParamError("Inputs do not match model version")
    
        # simulation parameters
        sim_params = param_dict["simulation parameters"]
        self.cohort_size = sim_params["cohort size"]
        self.time_horizon = sim_params["time horizon"]
        self.fixed_seed = sim_params["fixed seed"]
        self.state_detail = sim_params["detailed state outputs"]

        # initialization inputs
        init_params = param_dict["initial state"]
        self.tgroup_dist = np.asarray(dict_to_array(init_params["transmission group dist"]), dtype=float)
        self.subpop_dist = np.asarray(dict_to_array(init_params["risk category dist"]), dtype=float)
        self.dstate_dist = np.asarray(dict_to_array(init_params["initial disease dist"]), dtype=float)
        self.severity_dist = np.asarray(dict_to_array(init_params["severity dist"]), dtype=float)
        self.start_intvs = np.asarray(dict_to_array(init_params["start intervention"]), dtype=int)
        if np.any((self.start_intvs < 0) | (self.start_intvs >= INTERVENTIONS_NUM)):
            raise UserWarning("start intervention inputs must be valid intervention numbers")

        # transition inputs
        # kinda gross - consider reworking
        prog_array = dict_to_array(param_dict["disease progression"])
        for intv in INTERVENTIONS:
            for severity in DISEASE_PROGRESSIONS:
                for dstate in DISEASE_STATES:
                    if PROGRESSION_PATHS[severity][dstate]:
                        self.progression_probs[intv][severity][dstate] = prog_array[intv][severity][dstate - INCUBATION]
        self.severe_kludge_probs[:] = np.asarray(dict_to_array(param_dict["disease progression"]["daily prob recovery from severe state in critical path"]), dtype=float)
        self.initial_prob_immunity = np.asarray(dict_to_array(param_dict["disease progression"]["initial immunity on recovery probability"]), dtype=float)
        self.mortality_probs[:,:,SEVERE:CRITICAL+1] = np.asarray(dict_to_array(param_dict["disease mortality"]), dtype=float)
        
        # transmission inputs
        transm_params = param_dict["transmissions"]
        trans_mults = np.asarray(dict_to_array(transm_params["rate multipliers"]), dtype=float)
        self.trans_prob[:,:,ASYMP:RECOVERED] = dict_to_array(transm_params["transmission rate"])
        self.contact_matrices = normalized(np.asarray(dict_to_array(transm_params["contact matrix"]), dtype=float), axis=2, order=1)
      
        # apply transmission mults
        for i in range(T_RATE_PERIODS_NUM):
            for j in range(INTERVENTIONS_NUM):
                self.trans_prob[i,j,:] *= trans_mults[i][j]

        self.trans_rate_thresholds = np.asarray(dict_to_array(transm_params["rate multiplier thresholds"]))

        # test inputs
        test_inputs = dict_to_array(param_dict["tests"])
        for test in TESTS:
            self.test_return_delay[test] = test_inputs[test][0]
            self.test_characteristics[test,:] = test_inputs[test][1]
            self.test_lag[test] = test_inputs[test][2]

        # intervention strategies
        intv_strat_inputs =  param_dict["testing strategies"]
        for i in INTERVENTIONS:
            strat_dict = intv_strat_inputs[INTERVENTION_STRS[i]]
            self.prob_present[i,:] = dict_to_array(strat_dict["probability of presenting to care"])
            self.switch_on_test_result[i,:,0] = dict_to_array(strat_dict["switch to intervention on negative test result"])
            self.switch_on_test_result[i,:,1] = dict_to_array(strat_dict["switch to intervention on positive test result"])
            self.test_number[i,:] = dict_to_array(strat_dict["test number"])
            self.testing_frequency[i,:] = dict_to_array(strat_dict["testing frequency"])
            self.prob_receive_test[i,:,:] = dict_to_array(strat_dict["probability receive test"])
        if np.any((self.switch_on_test_result < 0) | (self.switch_on_test_result >= INTERVENTIONS_NUM)):
            raise UserWarning("switch on test return inputs must be valid intervention numbers")

        # costs
        cost_inputs = param_dict["costs"]
        self.testing_costs = np.asarray(dict_to_array(cost_inputs["testing costs"]))
        self.intervention_daily_costs = np.asarray(dict_to_array(cost_inputs["daily intervention costs"]))
        self.mortality_costs = np.asarray(dict_to_array(cost_inputs["mortality costs"]))

        # resources
        rsc_inputs = param_dict["resources"]
        self.resource_base_availability = np.asarray(dict_to_array(rsc_inputs["resource availabilities"]))
        requirements = dict_to_array(rsc_inputs["resource requirements"])
        for intervention in INTERVENTIONS:
            for symstate in OBSERVED_STATES:
                rscs = requirements[intervention][symstate]
                self.resource_requirements[intervention][symstate] = np.packbits([1 if i in rscs else 0 for i in range(8)])
        self.fallback_interventions = np.asarray(dict_to_array(rsc_inputs["back-up interventions"]))

        # non-covid RI
        non_covid_inputs = param_dict["non-covid illness"]
        self.prob_present_non_covid[SYMP_MODERATE:SYMP_CRITICAL+1, :] = np.asarray(dict_to_array(non_covid_inputs["daily prob present non-covid"]))
        self.non_covid_ri_durations[SYMP_MODERATE:SYMP_CRITICAL+1, :] = np.asarray(dict_to_array(non_covid_inputs["non covid symptom duration"]))

#creates blank input file template
def create_input_file(file):
    with open(file, 'w') as f:
        text = json.dumps(generate_input_dict(), indent=2)
        f.write(text)


# takes a filepath and returns inputs object
def read_inputs(file):
    with open(file, 'r') as f:
        text = f.read()
        # parse JSON
        param_dict = json.loads(text)
        # create inputs object and populate fields
        inputs = Inputs()
        inputs.read_inputs(param_dict)
        return inputs

