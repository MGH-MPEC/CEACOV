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
    return (np.random.random() < prob_exposure)


def roll_for_transition(patient, state_tracker, inputs):
    intv = patient[INTERVENTION]
    dstate = patient[DISEASE_STATE]
    severity = patient[DISEASE_PROGRESSION]
    prob_trans = inputs.progression_probs[intv, severity, dstate]
    if np.random.random() < prob_trans: # state changed
        new_state = PROGRESSION_PATHS[severity][dstate]
        state_tracker[new_state] += 1
        patient[DISEASE_STATE] = PROGRESSION_PATHS[severity][dstate]
        patient[FLAGS] = patient[FLAGS] & ~(PRESENTED_THIS_DSTATE)

def roll_for_mortality(patient, inputs):
    subpop = patient[SUBPOPULATION]
    prob_mort = inputs.mortality_probs[subpop]
    if np.random.random() < prob_mort:
        patient[FLAGS] = patient[FLAGS] & (~IS_ALIVE)
        return True
    else:
        return False

def roll_for_presentation(patient, inputs):
    if np.random.random() < inputs.prob_present[patient[INTERVENTION]][patient[DISEASE_STATE]]:
        patient[FLAGS] = patient[FLAGS] | PRESENTED_THIS_DSTATE
        if patient[DISEASE_STATE] in DISEASE_STATES[MILD:RECUPERATION+1]:
            patient[OBSERVED_STATE] = patient[DISEASE_STATE]-MILD+1
            patient[OBSERVED_STATE_TIME] = 0
        else:
            patient[OBSERVED_STATE] = SYMP_NONE
        return True
    else:
        return False

def roll_for_testing(patient, inputs):
    if np.random.random() < inputs.prob_receive_test[patient[INTERVENTION]][patient[OBSERVED_STATE]]:
        num = inputs.test_number[patient[INTERVENTION]][patient[OBSERVED_STATE]]
        patient[FLAGS] = patient[FLAGS] | HAS_PENDING_TEST
        # bitwise nonesense to set pending result flags
        patient[FLAGS] | (np.random.random() < inputs.test_characteristics[num][patient[DISEASE_STATE]]) << PENDING_TEST_RESULT
        patient[TIME_TO_TEST_RETURN] = inputs.test_return_delay[num]
        return True
    else:
        return False

# Main simulation class
class SimState():
    def __init__(self, inputs):
        self.day = 0
        self.cohort = np.zeros((inputs.cohort_size, NUM_STATE_VARS), dtype=np.intc)
        self.transmissions = 0
        self.inputs = inputs
        self.outputs = Outputs(inputs)
        self.cumulative_state_tracker = np.zeros(DISEASE_STATES_NUM, dtype=int)

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
            if (SUSCEPTABLE < dstate) and (dstate < RECOVERED):
                if dstate == RECUPERATION:
                    patient[DISEASE_PROGRESSION] = TO_CRITICAL
                elif progression > (dstate - MILD):
                    patient[DISEASE_PROGRESSION] = (dstate - MILD)
            # transmissions for day 0
            self.transmissions += self.inputs.trans_prob[patient[INTERVENTION],patient[dstate]]
            # cumulative state tracking
            if dstate < RECOVERED:
                self.cumulative_state_tracker[0:dstate+1] += 1
            else:
                self.cumulative_state_tracker[0:progression+3] += 1
                self.cumulative_state_tracker[-1] += 1

    def step(self):
        """performs daily patient updates"""
        # print(f"simulating day {self.day}")
        # local variables for quick access
        newtransmissions = np.zeros(SUBPOPULATIONS_NUM, dtype=float)
        state_tracker = np.zeros((SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        mort_tracker = np.zeros(SUBPOPULATIONS_NUM, dtype=int)
        intv_tracker = np.zeros(INTERVENTIONS_NUM, dtype=int)
        daily_tests = 0
        inputs = self.inputs
        # loop over cohort
        for patient in self.cohort:
            if not (patient[FLAGS] & IS_ALIVE):     # if patient is dead, nothing to do
                continue
            # update treatment/testing state
            if patient[FLAGS] & PRESENTED_THIS_DSTATE or roll_for_presentation(patient, inputs):
                if patient[OBSERVED_STATE_TIME] % inputs.testing_frequency[patient[INTERVENTION]][patient[OBSERVED_STATE]] == 0:
                    if roll_for_testing(patient, inputs):
                        daily_tests += 1
            if patient[FLAGS] & HAS_PENDING_TEST:
                if patient[TIME_TO_TEST_RETURN] == 0:
                    # perform test teturn updates
                    patient[FLAGS] = patient[FLAGS] & (~HAS_PENDING_TEST)
                    result = bool(patient[FLAGS] & PENDING_TEST_RESULT)
                    patient[INTERVENTION] = inputs.switch_on_test_result[patient[INTERVENTION]][patient[OBSERVED_STATE]][int(result)]
                else:
                    patient[TIME_TO_TEST_RETURN] -= 1
            patient[OBSERVED_STATE_TIME] += 1

            # update disease state
            if patient[DISEASE_STATE] == SUSCEPTABLE:
                if roll_for_incidence(patient, self.transmissions, inputs):
                    patient[DISEASE_STATE] = INCUBATION
                    self.cumulative_state_tracker[INCUBATION] += 1
            else:
                roll_for_transition(patient, self.cumulative_state_tracker, inputs)
            # roll for mortality
            if patient[DISEASE_STATE] == CRITICAL:
                roll_for_mortality(patient, inputs)
            # update patient state tracking
            if patient[FLAGS] & IS_ALIVE:
                # calculate tomorrow's exposures
                newtransmissions[patient[SUBPOPULATION]] += inputs.trans_prob[patient[INTERVENTION]][patient[DISEASE_STATE]]
                state_tracker[patient[SUBPOPULATION],patient[DISEASE_STATE]] += 1
                intv_tracker[patient[INTERVENTION]] += 1
            else: # must have died this month
                mort_tracker[patient[SUBPOPULATION]] += 1

        self.outputs.log_daily_state(self.day, state_tracker, self.cumulative_state_tracker, newtransmissions, mort_tracker, intv_tracker, daily_tests)
        self.transmissions = np.sum(newtransmissions)
        self.day += 1

    def run(self):
        np.random.seed(0)
        self.initialize_cohort()
        for i in range(self.inputs.time_horizon):
            self.step()


