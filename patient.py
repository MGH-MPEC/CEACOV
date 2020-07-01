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

def roll_for_incidence(patient, transmissions, t_group_sizes):
    # need to convert to probability!
    tgroup = patient[TRANSM_GROUP]
    prob_exposure = rate_to_prob(transmissions[tgroup] / t_group_sizes[tgroup])
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
    # gross case for severe->recovered in critical path. Hopefully temporary...
    if (severity == TO_CRITICAL) and (dstate == SEVERE):
        prob_critical = inputs.progression_probs[intv, severity, dstate]
        prob_recovery = inputs.severe_kludge_probs[intv]
        trans = draw_from_dist(np.array([(1-prob_critical-prob_recovery), prob_recovery, prob_critical]))
        if trans:
            new_state = RECOVERED if (trans == 1) else CRITICAL
            state_tracker[new_state] += 1
            patient[DISEASE_STATE] = new_state
            patient[FLAGS] = patient[FLAGS] & (~PRESENTED_THIS_DSTATE)
            if new_state == RECOVERED:
                patient[FLAGS] = patient[FLAGS] & ~(IS_INFECTED)
            return True
        else:
            return False
    else:
        prob_trans = inputs.progression_probs[intv, severity, dstate]
        if np.random.random() < prob_trans: # state changed
            new_state = PROGRESSION_PATHS[severity, dstate]
            state_tracker[new_state] += 1
            patient[DISEASE_STATE] = new_state
            patient[FLAGS] = patient[FLAGS] & (~PRESENTED_THIS_DSTATE)
            if new_state == RECOVERED:
                patient[FLAGS] = patient[FLAGS] & ~(IS_INFECTED)
            return True
        else:
            return False


def roll_for_mortality(patient, inputs, resource_utilization):
    prob_mort = inputs.mortality_probs[patient[SUBPOPULATION], patient[INTERVENTION], patient[DISEASE_STATE]]
    if np.random.random() < prob_mort:
        patient[FLAGS] = patient[FLAGS] & (~IS_ALIVE)
        resource_utilization -= np.unpackbits(inputs.resource_requirements[patient[INTERVENTION], patient[OBSERVED_STATE]])
        return True
    else:
        return False


def roll_for_presentation(patient, resource_use, inputs):
    if np.random.random() < inputs.prob_present[patient[INTERVENTION],patient[DISEASE_STATE]]:
        patient[FLAGS] = patient[FLAGS] | PRESENTED_THIS_DSTATE
        patient[FLAGS] = patient[FLAGS] & (~NON_COVID_RI)
        patient[OBSERVED_STATE_TIME] = 0
        if patient[DISEASE_STATE] in DISEASE_STATES[MODERATE:RECUPERATION+1]:
            new_obs_state = patient[DISEASE_STATE]-MODERATE+1
            switch_intervention(patient, inputs, patient[INTERVENTION], new_obs_state, resource_use)
            patient[OBSERVED_STATE] = new_obs_state
        else:
            new_obs_state = SYMP_ASYMP
            switch_intervention(patient, inputs, patient[INTERVENTION], new_obs_state, resource_use)
            patient[OBSERVED_STATE] = new_obs_state
        return True
    else:
        return False


def roll_for_testing(patient, test_counter, inputs):
    if np.random.random() < inputs.prob_receive_test[patient[INTERVENTION],patient[OBSERVED_STATE],patient[SUBPOPULATION]]:
        num = inputs.test_number[patient[INTERVENTION],patient[OBSERVED_STATE]]
        patient[FLAGS] = patient[FLAGS] | HAS_PENDING_TEST
        # get test characteristic time-from-infection interval
        time_period = TEST_SENS_THRESHOLDS_NUM + 1
        time_infected = patient[TIME_INFECTED]
        for i in range(TEST_SENS_THRESHOLDS_NUM):
            if time_infected < inputs.test_sens_thresholds[num,i]:
                time_period = i
                break
        if np.random.random() < inputs.test_characteristics[num,time_period]:
            patient[FLAGS] = patient[FLAGS] | PENDING_TEST_RESULT
            test_counter[num,1] += 1
        else:
            patient[FLAGS] = patient[FLAGS] & ~(PENDING_TEST_RESULT)
            test_counter[num,0] += 1
        patient[TIME_TO_TEST_RETURN] = inputs.test_return_delay[num]
        return True
    else:
        return False

def switch_intervention(patient, inputs, new_intervention, new_obs_state, resource_utilization):
    # return resources to the pool
    resource_utilization -= np.unpackbits(inputs.resource_requirements[patient[INTERVENTION], patient[OBSERVED_STATE]])
    # loop through until requirements are met
    new_intv = new_intervention
    new_req = inputs.resource_requirements[new_intv, new_obs_state]
    count = 0
    while np.packbits(0 >= inputs.resource_base_availability - resource_utilization) & new_req: # does not meet requirment
        new_intv = inputs.fallback_interventions[new_intv, new_obs_state]
        new_req = inputs.resource_requirements[new_intv, new_obs_state]
        count += 1
        if count >= INTERVENTIONS_NUM:
            raise UserWarning("No intervention available for the given resource constraints")
    resource_utilization += np.unpackbits(new_req)
    patient[INTERVENTION] = new_intv
    return True

# Main simulation class
class SimState():
    def __init__(self, inputs):
        self.day = 0
        self.cohort = np.zeros((inputs.cohort_size, NUM_STATE_VARS), dtype=np.intc)
        # to transmission group
        self.transmissions = np.zeros(TRANSMISSION_GROUPS_NUM, dtype=float)
        self.inputs = inputs
        self.outputs = Outputs(inputs)
        self.cumulative_state_tracker = np.zeros(DISEASE_STATES_NUM, dtype=int)
        self.test_costs = np.zeros(TESTS_NUM, dtype=float)
        self.intervention_costs = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=float)
        self.mortality_costs = 0
        self.resource_utilization = np.zeros(RESOURCES_NUM, dtype=int)
        self.non_covids = 0
        self.transmission_groups = np.zeros(TRANSMISSION_GROUPS_NUM, dtype=int)

    def initialize_cohort(self):
        # save relevant dists as locals
        tgroup_dist = self.inputs.tgroup_dist
        subpop_dist = self.inputs.subpop_dist
        disease_dist = self.inputs.dstate_dist
        severity_dist = self.inputs.severity_dist
        intv_dist = self.inputs.start_intvs
        # iterate over patient_array
        for patient in self.cohort:
            patient[FLAGS] = patient[FLAGS] | IS_ALIVE
            # draw demographics
            tgroup = draw_from_dist(tgroup_dist)
            patient[TRANSM_GROUP] = tgroup
            self.transmission_groups[tgroup] += 1

            subpop = draw_from_dist(subpop_dist[tgroup])
            patient[SUBPOPULATION] = subpop

            patient[INTERVENTION] = intv_dist[tgroup]
            # draw disease state
            dstate = draw_from_dist(disease_dist[tgroup])
            patient[DISEASE_STATE] = dstate
            # initialize paths (even for susceptables)
            progression = draw_from_dist(self.inputs.severity_dist[subpop])
            patient[DISEASE_PROGRESSION] = progression

            # simulate time since infection
            patient[TIME_INFECTED] = 1
            
            # dstate specific updates
            if dstate == SUSCEPTABLE:
                self.cumulative_state_tracker[SUSCEPTABLE] += 1
                patient[TIME_INFECTED] = 0
            
            elif dstate == RECOVERED:
                self.cumulative_state_tracker[0:progression+3] += 1
                self.cumulative_state_tracker[RECOVERED] += 1

            elif dstate == INCUBATION:
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                self.cumulative_state_tracker[0:INCUBATION+1] += 1
            
            elif dstate < RECUPERATION: # Asymptomatic, Mild/Moderate, Severe, Critical
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                if progression < (dstate - ASYMP):
                    patient[DISEASE_PROGRESSION] = (dstate - ASYMP)
                self.cumulative_state_tracker[0:dstate+1] += 1
            
            elif dstate == RECUPERATION:
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                patient[DISEASE_PROGRESSION] = TO_CRITICAL
 
            else:
                raise UserWarning("Patient disease state is in unreachable state")


            # transmissions for day 0
            foi_contribution = self.inputs.trans_prob[0,patient[INTERVENTION],dstate] * self.inputs.contact_matrices[patient[INTERVENTION],patient[TRANSM_GROUP],:]
            self.transmissions[:] += foi_contribution
            # resources in use at init
            self.resource_utilization += np.unpackbits(self.inputs.resource_requirements[patient[INTERVENTION], patient[OBSERVED_STATE]])

    def step(self):
        """performs daily patient updates"""
        # print(f"simulating day {self.day}")
        # local variables for quick access
        # From group, to group
        newtransmissions = np.zeros((TRANSMISSION_GROUPS_NUM,TRANSMISSION_GROUPS_NUM), dtype=float)
        state_tracker = np.zeros((SUBPOPULATIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        mort_tracker = np.zeros((SUBPOPULATIONS_NUM, INTERVENTIONS_NUM), dtype=int)
        intv_tracker = np.zeros((INTERVENTIONS_NUM, DISEASE_STATES_NUM), dtype=int)
        non_covid_present_dist = np.zeros(OBSERVED_STATES_NUM, dtype=float)
        daily_tests = np.zeros((TESTS_NUM, 2), dtype=int)
        new_infections = np.zeros((TRANSMISSION_GROUPS_NUM), dtype=int)
        inputs = self.inputs
        # time period for transm rate:
        trans_period = T_RATE_PERIODS_NUM - 1
        for i in range(T_RATE_PERIODS_NUM - 1):
            if self.day <= inputs.trans_rate_thresholds[i]:
                trans_period = i
                break
        # loop over cohort
        for patient in self.cohort:
            if not (patient[FLAGS] & IS_ALIVE):     # if patient is dead, nothing to do
                continue
            
            # Non-COVID RI presentation
            if ~patient[FLAGS] & NON_COVID_RI: # Don't already have non-COVID RI
                if ~patient[FLAGS] & IS_INFECTED: # Don't have COVID
                    non_covid_present_dist[:] = inputs.prob_present_non_covid[:,patient[SUBPOPULATION]]
                    if np.random.random() < np.sum(non_covid_present_dist):
                        non_covid_present_dist = non_covid_present_dist / np.sum(non_covid_present_dist)
                        symptoms = draw_from_dist(non_covid_present_dist)
                        switch_intervention(patient, inputs, patient[INTERVENTION], symptoms, self.resource_utilization)
                        patient[NON_COVID_TIME] = inputs.non_covid_ri_durations[symptoms, patient[SUBPOPULATION]]
                        patient[OBSERVED_STATE] = symptoms
                        patient[OBSERVED_STATE_TIME] = 0
                        patient[FLAGS] = patient[FLAGS] | (NON_COVID_RI + PRESENTED_THIS_DSTATE)
                        self.non_covids += 1
            else:# has non-covid RI
                patient[NON_COVID_TIME] -= 1 # countdown non-covid symptom duration
                if (patient[NON_COVID_TIME] == 0) or (patient[FLAGS] & IS_INFECTED): #non-covid duration is up or *newly* infected COVID
                    switch_intervention(patient, inputs, patient[INTERVENTION], SYMP_ASYMP, self.resource_utilization)
                    patient[OBSERVED_STATE] = SYMP_ASYMP
                    patient[OBSERVED_STATE_TIME] = 0
                    patient[FLAGS] = patient[FLAGS] & ~(NON_COVID_RI + PRESENTED_THIS_DSTATE)
                    self.non_covids -= 1

            # update treatment/testing state
            if (patient[FLAGS] & PRESENTED_THIS_DSTATE or roll_for_presentation(patient, self.resource_utilization, inputs)) and (not patient[FLAGS] & HAS_PENDING_TEST):
                test = inputs.test_number[patient[INTERVENTION],patient[OBSERVED_STATE]]                
                if ((patient[OBSERVED_STATE_TIME] - inputs.test_lag[test]) % inputs.testing_frequency[patient[INTERVENTION]][patient[OBSERVED_STATE]] == 0):
                    if roll_for_testing(patient, daily_tests, inputs):
                        self.test_costs[test] += inputs.testing_costs[test]
            if patient[FLAGS] & HAS_PENDING_TEST:
                if patient[TIME_TO_TEST_RETURN] == 0:
                    # perform test teturn updates
                    patient[FLAGS] = patient[FLAGS] & (~HAS_PENDING_TEST)
                    result = bool(patient[FLAGS] & PENDING_TEST_RESULT)
                    old_intv = patient[INTERVENTION]
                    new_intv = inputs.switch_on_test_result[old_intv,patient[OBSERVED_STATE],int(result)]
                    if new_intv != old_intv:
                        switch_intervention(patient, inputs, new_intv, patient[OBSERVED_STATE], self.resource_utilization)
                        patient[OBSERVED_STATE_TIME] = -1
                else:
                    patient[TIME_TO_TEST_RETURN] -= 1
            
            patient[OBSERVED_STATE_TIME] += 1

            # update disease state
            if patient[TIME_INFECTED] > 0:
                patient[TIME_INFECTED] +=  1

            if patient[DISEASE_STATE] == SUSCEPTABLE:
                if roll_for_incidence(patient, self.transmissions, self.transmission_groups):
                    patient[DISEASE_STATE] = INCUBATION
                    patient[FLAGS] = patient[FLAGS] & (~PRESENTED_THIS_DSTATE)
                    patient[TIME_INFECTED] = 1
                    self.cumulative_state_tracker[INCUBATION] += 1
                    new_infections[patient[TRANSM_GROUP]] += 1
            else:
                roll_for_transition(patient, self.cumulative_state_tracker, inputs)
            # roll for mortality
            if (patient[DISEASE_STATE] == CRITICAL) or (patient[DISEASE_PROGRESSION] == TO_SEVERE and patient[DISEASE_STATE] == SEVERE):
                roll_for_mortality(patient, inputs, self.resource_utilization)
            # update patient state tracking
            if patient[FLAGS] & IS_ALIVE:
                # calculate tomorrow's exposures
                foi_contribution = inputs.trans_prob[trans_period, patient[INTERVENTION],patient[DISEASE_STATE]] * inputs.contact_matrices[patient[INTERVENTION],patient[TRANSM_GROUP],:]
                newtransmissions[patient[TRANSM_GROUP], :] += foi_contribution
                state_tracker[patient[SUBPOPULATION],patient[DISEASE_STATE]] += 1
                intv_tracker[patient[INTERVENTION],patient[DISEASE_STATE]] += 1
                self.intervention_costs[patient[INTERVENTION],patient[OBSERVED_STATE]] += inputs.intervention_daily_costs[patient[INTERVENTION],patient[OBSERVED_STATE]]

            else: # must have died this month
                mort_tracker[patient[SUBPOPULATION], patient[INTERVENTION]] += 1
                self.mortality_costs += inputs.mortality_costs[patient[INTERVENTION]]

        costs = np.asarray([np.sum(self.test_costs), np.sum(self.intervention_costs), self.mortality_costs], dtype=float)
        self.outputs.log_daily_state(self.day, state_tracker, self.cumulative_state_tracker, newtransmissions, new_infections, mort_tracker, intv_tracker, daily_tests, self.resource_utilization, self.non_covids, costs)
        self.transmissions = np.sum(newtransmissions, axis=0)
        self.day += 1

    def run(self):
        if self.inputs.fixed_seed:
            np.random.seed(0)
        else:
            np.random.seed()
        self.initialize_cohort()
        for i in range(self.inputs.time_horizon):
            self.step()


