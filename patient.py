# -*- coding: utf-8 -*-
"""
COVID-19 Microsimulation Model

@author: Chris Panella (cpanella@mgh.harvard.edu)
"""

import numpy as np
import math
from enums import *
from outputs import Outputs

# returns an index drawn from the specified distribution
def draw_from_dist(dist):
    cumulative_dist = 0
    draw = np.random.random()
    for i in range(dist.size):
        cumulative_dist += dist[i]
        if cumulative_dist > draw:
            return i
        i += 1
    raise InvalidParamError("probabilites must sum to 1")


# utility functions to work with exponential rates and probabilities

def apply_rate_mult(prob, rate_mult):
    prob = 1.0 - ((1.0 - prob) ** rate_mult)
    return prob

def rate_to_prob(rate):
    return 1.0 - math.exp(-rate)

def prob_to_rate(prob):
    return -math.log(1.0-prob)

# helper functions for updating patient state (in loop, might get turned to Cython)

def roll_for_incidence(patient, transmissions, inputs):
    # need to convert to probability!
    prob_exposure = rate_to_prob(transmissions / inputs.cohort_size)
    if np.random.random() < prob_exposure:
        patient[FLAGS] = patient[FLAGS] | IS_INFECTED
        return True
    else:
        return False


def roll_for_transition(patient, state_tracker, inputs):
    intv = patient[INTERVENTION]
    dstate = patient[DISEASE_STATE]
    severity = patient[DISEASE_PROGRESSION]
    prob_trans = inputs.progression_probs[intv, severity, dstate]
    if np.random.random() < prob_trans: # state changed
        new_state = PROGRESSION_PATHS[severity][dstate]
        state_tracker[new_state] += 1
        patient[DISEASE_STATE] = new_state
        patient[FLAGS] = patient[FLAGS] & (~PRESENTED_THIS_DSTATE)
        if new_state == RECOVERED:
            patient[FLAGS] = patient[FLAGS] & ~(IS_INFECTED)
        return True
    else:
        return False


def roll_for_mortality(patient, inputs):
    prob_mort = inputs.mortality_probs[patient[SUBPOPULATION], patient[DISEASE_STATE]]
    if np.random.random() < prob_mort:
        patient[FLAGS] = patient[FLAGS] & (~IS_ALIVE)
        return True
    else:
        return False


def roll_for_presentation(patient, inputs):
    if np.random.random() < inputs.prob_present[patient[INTERVENTION],patient[DISEASE_STATE]]:
        patient[FLAGS] = patient[FLAGS] | PRESENTED_THIS_DSTATE
        patient[FLAGS] = patient[FLAGS] & (~NON_COVID_RI)
        patient[OBSERVED_STATE_TIME] = 0
        if patient[DISEASE_STATE] in DISEASE_STATES[MODERATE:RECUPERATION+1]:
            patient[OBSERVED_STATE] = patient[DISEASE_STATE]-MODERATE+1
        else:
            patient[OBSERVED_STATE] = SYMP_ASYMP
        return True
    else:
        return False


def roll_for_testing(patient, inputs):
    if np.random.random() < inputs.prob_receive_test[patient[INTERVENTION],patient[OBSERVED_STATE],patient[SUBPOPULATION]]:
        num = inputs.test_number[patient[INTERVENTION],patient[OBSERVED_STATE]]
        patient[FLAGS] = patient[FLAGS] | HAS_PENDING_TEST
        # bitwise nonesense to set pending result flags
        if np.random.random() < inputs.test_characteristics[num,patient[DISEASE_STATE]]:
            patient[FLAGS] = patient[FLAGS] | PENDING_TEST_RESULT
        else:
            patient[FLAGS] = patient[FLAGS] & ~(PENDING_TEST_RESULT)
        patient[TIME_TO_TEST_RETURN] = inputs.test_return_delay[num]
        return True
    else:
        return False

def switch_intervention(patient, inputs, intervention, resource_utilization):
    obs = patient[OBSERVED_STATE]
    old_intv = patient[INTERVENTION]
    # return resources to the pool
    resource_utilization -= np.unpackbits(inputs.resource_requirements[old_intv, obs])
    # loop through until requirements are met
    new_intv = intervention
    new_req = inputs.resource_requirements[new_intv, obs]
    count = 0
    while np.packbits([(0 >= inputs.resource_base_availability - resource_utilization)]) & new_req: # does not meet requirment
        temp = inputs.fallback_interventions[old_intv, obs]
        old_intv = new_intv
        new_intv = temp
        new_req = inputs.resource_requirements[new_intv, obs]
        count += 1
        if count > 2*INTERVENTIONS_NUM:
            raise UserWarning("No intervention available for the given resource constraints")
    resource_utilization += np.unpackbits(new_req)
    patient[INTERVENTION] = new_intv
    return True

# Main simulation class
class SimState():
    def __init__(self, inputs):
        self.day = 0
        self.cohort = np.zeros((inputs.cohort_size, NUM_STATE_VARS), dtype=np.intc)
        self.transmissions = 0
        self.inputs = inputs
        self.outputs = Outputs(inputs)
        self.cumulative_state_tracker = np.zeros(DISEASE_STATES_NUM, dtype=int)
        self.test_costs = np.zeros(TESTS_NUM, dtype=float)
        self.intervention_costs = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=float)
        self.mortality_costs = 0
        self.resource_utilization = np.zeros(RESOURCES_NUM, dtype=int)
        self.non_covids = 0

    def initialize_cohort(self):
        # save relevant dists as locals
        subpop_dist = self.inputs.subpop_dist
        disease_dist = self.inputs.dstate_dist
        # iterate over patient_array
        for patient in self.cohort:
            patient[FLAGS] = patient[FLAGS] | IS_ALIVE
            # draw demographics
            subpop = draw_from_dist(subpop_dist)
            patient[SUBPOPULATION] = subpop
            # draw disease state
            dstate = draw_from_dist(self.inputs.dstate_dist)
            patient[DISEASE_STATE] = dstate
            # initialize paths (even for susceptables)
            progression = draw_from_dist(self.inputs.severity_dist[subpop])
            patient[DISEASE_PROGRESSION] = progression
            
            # dstate specific updates
            if dstate == SUSCEPTABLE:
                self.cumulative_state_tracker[SUSCEPTABLE] += 1
            
            if dstate == RECOVERED:
                self.cumulative_state_tracker[0:progression+3] += 1
                self.cumulative_state_tracker[RECOVERED] += 1

            elif dstate == INCUBATION:
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                self.cumulative_state_tracker[0:INCUBATION+1] += 1
            
            elif dstate < RECUPERATION: # Asymptomatic, Mild/Moderate, Severe, Critical
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                if progression > (dstate - ASYMP):
                    patient[DISEASE_PROGRESSION] = (dstate - ASYMP)
                self.cumulative_state_tracker[0:dstate+1] += 1
            
            elif dstate == RECUPERATION:
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                patient[DISEASE_PROGRESSION] = TO_CRITICAL
 
            else:
                raise UserWarning("Patient disease state is in unreachable state")

            # transmissions for day 0
            self.transmissions += self.inputs.trans_prob[patient[INTERVENTION],dstate]
            # resources in use at init
            self.resource_utilization += np.unpackbits(self.inputs.resource_requirements[patient[INTERVENTION], patient[OBSERVED_STATE]])

    def step(self):
        """performs daily patient updates"""
        # print(f"simulating day {self.day}")
        # local variables for quick access
        newtransmissions = np.zeros(SUBPOPULATIONS_NUM, dtype=float)
        state_tracker = np.zeros((SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        mort_tracker = np.zeros(SUBPOPULATIONS_NUM, dtype=int)
        intv_tracker = np.zeros(INTERVENTIONS_NUM, dtype=int)
        non_covid_present_dist = np.zeros(OBSERVED_STATES_NUM, dtype=float)
        self.non_covids = 0
        daily_tests = 0
        new_infections = 0
        inputs = self.inputs
        # loop over cohort
        for patient in self.cohort:
            if not (patient[FLAGS] & IS_ALIVE):     # if patient is dead, nothing to do
                continue
            
            # Non-COVID RI presentation
            if ~patient[FLAGS] & NON_COVID_RI: # Don't already have non-COVID RI
                if ~patient[FLAGS] & IS_INFECTED: # Don't have COVID
                    non_covid_present_dist[:] = inputs.prob_present_non_covid[:,patient[SUBPOPULATION]]
                    if np.random.random() < np.sum(non_covid_present_dist):
                        patient[FLAGS] = patient[FLAGS] | (NON_COVID_RI + PRESENTED_THIS_DSTATE)
                        patient[OBSERVED_STATE_TIME] = 0
                        non_covid_present_dist[0] = non_covid_present_dist / np.sum(non_covid_present_dist)
                        patient[OBSERVED_STATE] = draw_from_dist(non_covid_present_dist)
                        switch_intervention(patient, inputs, patient[INTERVENTION], self.resource_utilization)
            else:
                if OBSERVED_STATE_TIME >= inputs.non_covid_ri_durations[patient[OBSERVED_STATE], patient[SUBPOPULATION]]: # Time is up
                    patient[OBSERVED_STATE] = SYMP_ASYMP
                    patient[FLAGS] = patient[FLAGS] & ~(NON_COVID_RI + PRESENTED_THIS_DSTATE)
                    switch_intervention(patient, inputs, patient[INTERVENTION], self.resource_utilization)


            # update treatment/testing state
            if patient[FLAGS] & PRESENTED_THIS_DSTATE or (roll_for_presentation(patient, inputs) and switch_intervention(patient, inputs, patient[INTERVENTION], self.resource_utilization)):
                if (not patient[FLAGS] & HAS_PENDING_TEST) and  (patient[OBSERVED_STATE_TIME] % inputs.testing_frequency[patient[INTERVENTION]][patient[OBSERVED_STATE]] == 0):
                    if roll_for_testing(patient, inputs):
                        daily_tests += 1
                        test = inputs.test_number[patient[INTERVENTION],patient[OBSERVED_STATE]]
                        self.test_costs[test] += inputs.testing_costs[test]
            if patient[FLAGS] & HAS_PENDING_TEST:
                if patient[TIME_TO_TEST_RETURN] == 0:
                    # perform test teturn updates
                    patient[FLAGS] = patient[FLAGS] & (~HAS_PENDING_TEST)
                    result = bool(patient[FLAGS] & PENDING_TEST_RESULT)
                    old_intv = patient[INTERVENTION]
                    new_intv = inputs.switch_on_test_result[old_intv,patient[OBSERVED_STATE],int(result)]
                    if new_intv != old_intv:
                        switch_intervention(patient, inputs, new_intv, self.resource_utilization)

                else:
                    patient[TIME_TO_TEST_RETURN] -= 1
            patient[OBSERVED_STATE_TIME] += 1

            # update disease state
            if patient[DISEASE_STATE] == SUSCEPTABLE:
                if roll_for_incidence(patient, self.transmissions, inputs):
                    patient[DISEASE_STATE] = INCUBATION
                    patient[FLAGS] = patient[FLAGS] & (~PRESENTED_THIS_DSTATE)
                    self.cumulative_state_tracker[INCUBATION] += 1
                    new_infections += 1
            else:
                roll_for_transition(patient, self.cumulative_state_tracker, inputs)
            # roll for mortality
            if (patient[DISEASE_STATE] == CRITICAL) or (patient[DISEASE_PROGRESSION] == TO_SEVERE and patient[DISEASE_STATE] == SEVERE):
                roll_for_mortality(patient, inputs)
            # update patient state tracking
            if patient[FLAGS] & IS_ALIVE:
                # calculate tomorrow's exposures
                newtransmissions[patient[SUBPOPULATION]] += inputs.trans_prob[patient[INTERVENTION],patient[DISEASE_STATE]]
                state_tracker[patient[SUBPOPULATION],patient[DISEASE_STATE]] += 1
                intv_tracker[patient[INTERVENTION]] += 1
                self.intervention_costs[patient[INTERVENTION],patient[OBSERVED_STATE]] += inputs.intervention_daily_costs[patient[INTERVENTION],patient[OBSERVED_STATE]]
                self.non_covids += int(patient[FLAGS] & NON_COVID_RI)

            else: # must have died this month
                mort_tracker[patient[SUBPOPULATION]] += 1
                self.mortality_costs += inputs.mortality_costs[patient[INTERVENTION]]

        costs = np.asarray([np.sum(self.test_costs), np.sum(self.intervention_costs), self.mortality_costs], dtype=float)
        self.outputs.log_daily_state(self.day, state_tracker, self.cumulative_state_tracker, newtransmissions, new_infections, mort_tracker, intv_tracker, daily_tests, self.resource_utilization, self.non_covids, costs)
        self.transmissions = np.sum(newtransmissions)
        self.day += 1

    def run(self):
        if self.inputs.fixed_seed:
            np.random.seed(0)
        else:
            np.random.seed()
        self.initialize_cohort()
        for i in range(self.inputs.time_horizon):
            self.step()


