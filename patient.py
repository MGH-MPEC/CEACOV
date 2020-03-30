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


def roll_for_transition(patient, inputs):
    intv = patient[INTERVENTION]
    dstate = patient[DISEASE_STATE]
    severity = patient[DISEASE_PROGRESSION]
    prob_trans = inputs.progression_probs[intv, severity, dstate]
    if np.random.random() < prob_trans: # state changed
        patient[DISEASE_STATE] = PROGRESSION_PATHS[severity][dstate]
        patient[D_STATE_TIME] = 0

def roll_for_mortality(patient, inputs):
    subpop = patient[SUBPOPULATION]
    prob_mort = inputs.mortality_probs[subpop]
    if np.random.random() < prob_mort:
        patient[FLAGS] = patient[FLAGS] & (~IS_ALIVE)
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
            if (SUSCEPTABLE < dstate) and (dstate < RECOVERED):
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                progression = draw_from_dist(self.inputs.severity_dist[subpop])
                patient[DISEASE_PROGRESSION] = progression if progression >= (dstate - MILD) else dstate
                self.transmissions += calculate_transmissions(patient, self.inputs) 

    def step(self):
        """performs daily patient updates"""
        print(f"simulating day {self.day}")
        # local variables for quick access
        newtransmissions = np.zeros(SUBPOPULATIONS_NUM, dtype=float)
        oldtransmissions = self.transmissions
        state_tracker = np.zeros((SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        mort_tracker = np.zeros(SUBPOPULATIONS_NUM, dtype=int)
        # loop over cohort
        for patient in self.cohort:
            if not (patient[FLAGS] & IS_ALIVE):     # if patient is dead, nothing to do
                continue
            # update disease state
            if patient[DISEASE_STATE] == SUSCEPTABLE:
                if roll_for_incidence(patient, oldtransmissions, self.inputs):
                    patient[DISEASE_STATE] = INCUBATION
                    patient[D_STATE_TIME] = 0
                    patient[DISEASE_PROGRESSION] = draw_from_dist(self.inputs.severity_dist[patient[SUBPOPULATION]])
            else:
                roll_for_transition(patient, self.inputs)
            # roll for mortality
            if patient[DISEASE_STATE] == CRITICAL:
                roll_for_mortality(patient, self.inputs)
            # update patient state tracking
            if patient[FLAGS] & IS_ALIVE:
                # calculate tomorrow's exposures
                if (patient[FLAGS] & IS_INFECTED):
                    newtransmissions[patient[SUBPOPULATION]] += inputs.trans_prob[patient[INTERVENTION], patient[DISEASE_STATE]]
                state_tracker[patient[SUBPOPULATION],patient[DISEASE_STATE]] += 1
            else: # must have died this month
                mort_tracker[patient[SUBPOPULATION]] += 1
            


        self.outputs.log_daily_state(self.day, state_tracker, oldtransmissions, mort_tracker)
        self.transmissions = np.sum(newtransmissions)
        self.day += 1

    def run(self):
        self.initialize_cohort()
        for i in range(self.inputs.time_horizon):
            self.step()


