# -*- coding: utf-8 -*-
"""
COVID-19 Microsimulation Model
@author: Chris Panella (cpanella@mgh.harvard.edu)
"""

import numpy as np
import math
from enums import *
from outputs import Outputs


# returns the index of the bin to which a value belongs
# linear search - use np.digitize for longer arrays
def digitize_linear(x, thresholds):
    for i in range(len(thresholds)):
        if x < thresholds[i]:
            return i
    return len(thresholds)


# utility functions to work with exponential rates and probabilities

def apply_rate_mult(prob, rate_mult):
    prob = 1.0 - ((1.0 - prob) ** rate_mult)
    return prob


def rate_to_prob(rate):
    return 1.0 - math.exp(-rate)


def prob_to_rate(prob):
    return -math.log(1.0 - prob)


# helper functions for updating patient state

def get_symptoms(patient):
    """returns the symptoms corresponding to the patient's disease state"""
    if patient[DISEASE_STATE] in DISEASE_STATES[MODERATE:RECUPERATION+1]:
        return patient[DISEASE_STATE]-MODERATE+1
    else:
        return SYMP_ASYMP


def roll_for_incidence(patient, transmissions, t_group_sizes, baseline_exposure):
    """returns True if the patient becomes infected with COVID"""
    # need to convert to probability!
    tgroup = patient[TRANSM_GROUP]
    prob_exposure = rate_to_prob(transmissions[tgroup] / t_group_sizes[tgroup])
    prob_exposure = prob_exposure + baseline_exposure - (prob_exposure * baseline_exposure)
    if np.random.random() < prob_exposure:
        patient[FLAGS] = patient[FLAGS] | IS_INFECTED
        return True
    else:
        return False


def roll_for_transition(patient, state_tracker, inputs):
    """handles transitions except for incidence including immune transtion"""
    intv = patient[INTERVENTION]
    dstate = patient[DISEASE_STATE]
    severity = patient[DISEASE_PROGRESSION]
    new_state = None
    if dstate in {SUSCEPTABLE, IMMUNE}:
        # roll to transition immunity
        if np.random.random() < inputs.daily_prob_lose_immunity[patient[IMMUNE_STATE], patient[SUBPOPULATION]]:
            new_state = SUSCEPTABLE
    elif patient[FLAGS] & PRE_RECOVERY:
        # roll for pre-recovery -> immune
        if np.random.random() < inputs.expedited_recovery_probs[intv, severity, dstate]:
            new_state = IMMUNE
            patient[FLAGS] = patient[FLAGS] & ~PRE_RECOVERY
    else:
        # roll for normal progression or pre-recovery
        trans_type = np.random.choice(PROG_TYPES_NUM, p=inputs.progression_probs[intv, severity, dstate])
        if trans_type == PROG_NORMAL:
            new_state = PROGRESSION_PATHS[severity, dstate]
        elif trans_type == PROG_PRE_REC:
            # transitioning to pre-rec, no state change
            patient[FLAGS] = patient[FLAGS] | PRE_RECOVERY

    if new_state is not None:
        if new_state in {IMMUNE, SUSCEPTABLE}:
            if new_state == IMMUNE: # recovered from disease
                patient[FLAGS] = patient[FLAGS] & ~IS_INFECTED
                new_istate = RECOVERED
            else: # waning immunity
                new_istate = inputs.immunity_transition[patient[IMMUNE_STATE]]
            if new_istate != -1:  # included in case users do not want immune state transition on loss of full immunity
                patient[IMMUNE_STATE] = new_istate
                # re-roll severity
                patient[DISEASE_PROGRESSION] = np.random.choice(DISEASE_PROGRESSIONS_NUM, p=inputs.severity_dist[new_istate, patient[SUBPOPULATION]])
                # check to see if full immunity "sticks"
                if (np.random.random() < inputs.prob_full_immunity[new_istate, patient[SUBPOPULATION]]):
                    new_state = IMMUNE
                else:
                    new_state = SUSCEPTABLE

        # update patient
        patient[DISEASE_STATE] = new_state
        state_tracker[new_state] += 1
        patient[FLAGS] = patient[FLAGS] & ~PRESENTED_THIS_DSTATE
        return True
    else:  # no transition
        return False


def roll_for_mortality(patient, inputs, resource_utilization):
    """returns True and updates resource utilization if the patient dies"""
    prob_mort = inputs.mortality_probs[patient[SUBPOPULATION], patient[INTERVENTION]]
    if np.random.random() < prob_mort:
        patient[FLAGS] = patient[FLAGS] & (~IS_ALIVE)
        resource_utilization -= np.unpackbits(inputs.resource_requirements[patient[INTERVENTION], patient[OBSERVED_STATE]])
        return True
    else:
        return False


def roll_for_presentation(patient, resource_use, resource_availability, state_tracker, inputs):
    """if the patient presents to care, returns True and updates their observed state and related values"""
    if np.random.random() < inputs.prob_present[patient[INTERVENTION], patient[DISEASE_STATE]]:
        patient[FLAGS] = patient[FLAGS] | PRESENTED_THIS_DSTATE
        patient[FLAGS] = patient[FLAGS] & (~NON_COVID_RI)
        patient[OBSERVED_STATE_TIME] = 0
        new_obs_state = get_symptoms(patient)
        switch_intervention(patient, inputs, patient[INTERVENTION], new_obs_state, resource_use, resource_availability, state_tracker)
        patient[OBSERVED_STATE] = new_obs_state
        return True
    else:
        return False


def roll_for_testing(patient, test_num, test_counter, inputs):
    """returns True, determines the test result, and sets the delay to test return if the patient is tested"""
    if ((patient[TIME_TO_CONF_TEST] == -1 and (np.random.random() < inputs.prob_receive_test[patient[INTERVENTION], patient[OBSERVED_STATE], patient[SUBPOPULATION]]) ) 
        or (patient[TIME_TO_CONF_TEST] == 0 and np.random.random() < inputs.prob_confirmatory_test[patient[INTERVENTION]])):
        patient[FLAGS] = patient[FLAGS] | HAS_PENDING_TEST
        # get test characteristic time-from-infection interval
        time_period = TEST_SENS_THRESHOLDS_NUM + 1
        time_infected = patient[TIME_INFECTED]
        for i in range(TEST_SENS_THRESHOLDS_NUM + 1):
            if time_infected < inputs.test_sens_thresholds[test_num, i]:
                time_period = i
                break
        # roll for test result, with probability of testing positive based on time infected
        if np.random.random() < inputs.test_characteristics[test_num, time_period]:
            patient[FLAGS] = patient[FLAGS] | PENDING_TEST_RESULT
            test_counter[test_num, 1] += 1
        else:
            patient[FLAGS] = patient[FLAGS] & ~PENDING_TEST_RESULT
            test_counter[test_num, 0] += 1
        patient[TIME_TO_TEST_RETURN] = inputs.test_return_delay[test_num]
        return True
    else:
        return False


def switch_intervention(patient, inputs, new_intervention, new_obs_state, resource_utilization, resource_availability, state_tracker):
    """updates patient intervention and resource utilization"""
    rec_reqs = inputs.resource_requirements
    # return resources to the pool
    resource_utilization -= np.unpackbits(rec_reqs[patient[INTERVENTION], patient[OBSERVED_STATE]])
    # loop through until requirements are met
    new_intv = new_intervention
    new_req = rec_reqs[new_intv, new_obs_state]
    count = 0
    while np.packbits(0 >= resource_availability - resource_utilization) & new_req:  # does not meet requirement
        new_intv = inputs.fallback_interventions[new_intv, new_obs_state]
        new_req = rec_reqs[new_intv, new_obs_state]
        count += 1
        if count >= INTERVENTIONS_NUM:
            raise UserWarning("No intervention available for the given resource constraints")
    resource_utilization += np.unpackbits(new_req)
    if patient[INTERVENTION] != new_intv:
        patient[INTERVENTION] = new_intv
        vax = inputs.vaccination[new_intv]
        if (0 <= vax < VACCINES_NUM) and not (patient[FLAGS] & IS_INFECTED):
            new_istate = vax + 2
            # set new immune state (linked to transmission)
            patient[IMMUNE_STATE] = new_istate
            # re-roll severity
            patient[DISEASE_PROGRESSION] = np.random.choice(DISEASE_PROGRESSIONS_NUM, p=inputs.severity_dist[new_istate, patient[SUBPOPULATION]])
            # roll for full immunity
            if np.random.random() < inputs.prob_full_immunity[new_istate, patient[SUBPOPULATION]]:
                patient[DISEASE_STATE] = IMMUNE


class SimState:
    """main simulation class"""

    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = Outputs(inputs)
        self.initialize_cohort()

    def initialize_cohort(self):

        # state variables
        self.day = 0
        self.cohort = np.zeros((self.inputs.cohort_size, NUM_STATE_VARS), dtype=np.intc)
        self.transmissions = np.zeros(TRANSMISSION_GROUPS_NUM, dtype=float)
        self.resource_utilization = np.zeros(RESOURCES_NUM, dtype=int)
        self.non_covids = 0
        self.transmission_groups = np.zeros(TRANSMISSION_GROUPS_NUM, dtype=int)

        # accumulators
        self.cumulative_state_tracker = np.zeros(DISEASE_STATES_NUM, dtype=int)
        self.cumulative_vax_mortality = np.zeros(IMMUNE_STATES_NUM, dtype=int)
        self.screen_costs = np.zeros(OBSERVED_STATES_NUM, dtype=float)
        self.test_costs = np.zeros(TESTS_NUM, dtype=float)
        self.intervention_costs = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=float)
        self.mortality_costs = 0

        # save relevant dists as locals
        tgroup_dist = self.inputs.tgroup_dist
        istate_dist = self.inputs.istate_dist
        subpop_dist = self.inputs.subpop_dist
        disease_dist = self.inputs.dstate_dist
        severity_dist = self.inputs.severity_dist
        intv_dist = self.inputs.start_intvs

        # iterate over patient_array
        for patient in self.cohort:
            patient[FLAGS] = patient[FLAGS] | IS_ALIVE
            # draw demographics
            tgroup = np.random.choice(TRANSMISSION_GROUPS_NUM, p=tgroup_dist)
            patient[TRANSM_GROUP] = tgroup
            self.transmission_groups[tgroup] += 1
            istate = np.random.choice(IMMUNE_STATES_NUM, p=istate_dist[tgroup])
            patient[IMMUNE_STATE] = istate
            subpop = np.random.choice(SUBPOPULATIONS_NUM, p=subpop_dist[tgroup])
            patient[SUBPOPULATION] = subpop
            intv = intv_dist[tgroup]
            patient[INTERVENTION] = intv
            patient[TIME_TO_CONF_TEST] = -1
            # draw disease state
            dstate = np.random.choice(DISEASE_STATES_NUM, p=disease_dist[tgroup, istate])
            patient[DISEASE_STATE] = dstate
            # initialize paths (even for susceptables)
            progression = np.random.choice(DISEASE_PROGRESSIONS_NUM, p=severity_dist[istate, subpop])
            patient[DISEASE_PROGRESSION] = progression

            # dstate specific updates
            if dstate == SUSCEPTABLE:
                self.cumulative_state_tracker[SUSCEPTABLE] += 1
                # patient[TIME_INFECTED] is already set to 0

            elif dstate == IMMUNE:
                self.cumulative_state_tracker[0:progression+3] += 1
                self.cumulative_state_tracker[IMMUNE] += 1
                patient[TIME_INFECTED] += np.random.randint(45)  # spread out initial time of infection

            elif dstate == INCUBATION:
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                self.cumulative_state_tracker[0:INCUBATION+1] += 1
                patient[TIME_INFECTED] = 1

            elif dstate < RECUPERATION:  # Asymptomatic, Mild/Moderate, Severe, Critical
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                if progression < (dstate - ASYMP):
                    patient[DISEASE_PROGRESSION] = (dstate - ASYMP)
                self.cumulative_state_tracker[0:dstate+1] += 1

            elif dstate == RECUPERATION:
                patient[FLAGS] = patient[FLAGS] | IS_INFECTED
                patient[DISEASE_PROGRESSION] = TO_CRITICAL

            else:
                raise UserWarning("Patient disease state is in unreachable state")

            # initialize time infected
            if dstate > INCUBATION:
                patient[TIME_INFECTED] += 1  # patients start one day into current dstate
                for ds in range(dstate):
                    if PROGRESSION_PATHS[progression][ds] != -1:
                        p = self.inputs.progression_probs[intv, progression, ds]
                        if ds == CRITICAL:  # can't forget surviorship bias!
                            p = 1 - ((1 - p) * (1 - self.inputs.mortality_probs[subpop, intv]))
                        try:
                            patient[TIME_INFECTED] += np.random.geometric(p)
                        except ValueError:  # in case of 0 prob progression in debug runs, just ignore initialization
                            pass

            # transmissions for day 0
            foi_contribution = self.inputs.trans_prob[0, patient[INTERVENTION], tgroup, dstate]
            foi_contribution *= self.inputs.immunity_transm_mult[istate]
            foi_contribution *= self.inputs.exposure_matrices[intv, tgroup, :]
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
        vax_tracker = np.zeros((IMMUNE_STATES_NUM, DISEASE_STATES_NUM+1), dtype=int)  # the +1 column is for mortality
        non_covid_present_dist = np.zeros(OBSERVED_STATES_NUM, dtype=float)
        daily_screens = np.zeros(2, dtype=int)
        daily_tests = np.zeros((TESTS_NUM, 2), dtype=int)
        new_infections = np.zeros(TRANSMISSION_GROUPS_NUM, dtype=int)
        inputs = self.inputs
        # calculate time period for model-time stratified inputs
        trans_period = digitize_linear(self.day, inputs.trans_rate_thresholds)
        resource_availability = inputs.resource_availabilities[digitize_linear(self.day, inputs.resource_availability_thresholds), :]        
        test_availability = inputs.test_availabilities[digitize_linear(self.day, inputs.test_availability_thresholds), :]

        # loop over cohort
        for patient in self.cohort:
            if not (patient[FLAGS] & IS_ALIVE):     # if patient is dead, nothing to do
                continue

            # Non-COVID RI presentation
            if ~patient[FLAGS] & NON_COVID_RI:  # Don't already have non-COVID RI
                if ~patient[FLAGS] & IS_INFECTED:  # Don't have COVID
                    non_covid_present_dist[:] = inputs.prob_present_non_covid[:, patient[SUBPOPULATION]]
                    if np.random.random() < np.sum(non_covid_present_dist):
                        non_covid_present_dist = non_covid_present_dist / np.sum(non_covid_present_dist)
                        symptoms = np.random.choice(OBSERVED_STATES_NUM, p=non_covid_present_dist)
                        switch_intervention(patient, inputs, patient[INTERVENTION], symptoms, self.resource_utilization, resource_availability, state_tracker)
                        patient[NON_COVID_TIME] = inputs.non_covid_ri_durations[symptoms, patient[SUBPOPULATION]]
                        patient[OBSERVED_STATE] = symptoms
                        patient[OBSERVED_STATE_TIME] = 0
                        patient[FLAGS] = patient[FLAGS] | (NON_COVID_RI + PRESENTED_THIS_DSTATE)
                        self.non_covids += 1
            else:  # has non-covid RI
                patient[NON_COVID_TIME] -= 1 # countdown non-covid symptom duration
                if (patient[NON_COVID_TIME] == 0) or (patient[FLAGS] & IS_INFECTED): # non-covid duration is up or *newly* infected COVID
                    switch_intervention(patient, inputs, patient[INTERVENTION], SYMP_ASYMP, self.resource_utilization, resource_availability, state_tracker)
                    patient[OBSERVED_STATE] = SYMP_ASYMP
                    patient[OBSERVED_STATE_TIME] = 0
                    patient[FLAGS] = patient[FLAGS] & ~(NON_COVID_RI + PRESENTED_THIS_DSTATE)
                    self.non_covids -= 1
            # roll for symptom screen
            if (~patient[FLAGS] & WAS_SX_SCREENED):
                symptoms = get_symptoms(patient)
                if (np.random.random() < inputs.prob_sx_screen[patient[INTERVENTION], symptoms, patient[SUBPOPULATION]]) and (self.day % inputs.screening_frequency[patient[INTERVENTION], symptoms] == 0):
                    patient[FLAGS] = patient[FLAGS] | WAS_SX_SCREENED
                    # roll for symptom screen result, with probability of positive screen based on symptoms
                    if np.random.random() < inputs.screen_characteristics[symptoms]:
                        patient[FLAGS] = patient[FLAGS] | PENDING_SCREEN_RESULT
                        daily_screens[1] += 1
                        # if they screened positive and are not already waiting for a confirmatory test, set the number of days until they will be tested
                        if patient[TIME_TO_CONF_TEST] == -1:
                            patient[TIME_TO_CONF_TEST] = inputs.confirmatory_test_lag[patient[INTERVENTION]]
                    else:
                        patient[FLAGS] = patient[FLAGS] & ~PENDING_SCREEN_RESULT
                        daily_screens[0] += 1
                    patient[TIME_TO_SCREEN_RETURN] = inputs.screen_return_delay
                    self.screen_costs[symptoms] += inputs.screening_costs[symptoms]
                if patient[FLAGS] & WAS_SX_SCREENED:
                    if patient[TIME_TO_SCREEN_RETURN] == 0:
                        result = bool(patient[FLAGS] & PENDING_SCREEN_RESULT)
                        old_intv = patient[INTERVENTION]
                        new_intv = inputs.switch_on_screen_result[patient[INTERVENTION], symptoms, int(result)]
                        if new_intv != old_intv:
                            switch_intervention(patient, inputs, new_intv, patient[OBSERVED_STATE], self.resource_utilization, resource_availability, state_tracker)
                        patient[FLAGS] = patient[FLAGS] & ~WAS_SX_SCREENED
                    else:
                        patient[TIME_TO_SCREEN_RETURN] -= 1
            # update treatment/testing state             
            if (patient[FLAGS] & PRESENTED_THIS_DSTATE or roll_for_presentation(patient, self.resource_utilization, resource_availability, state_tracker, inputs)) and (not patient[FLAGS] & HAS_PENDING_TEST):
                test = inputs.test_number[patient[INTERVENTION], patient[OBSERVED_STATE]]
                time_past_delay = patient[OBSERVED_STATE_TIME] - inputs.test_lag[test]
                # check whether they are eligible, either for the regular testing or the confirmatory test, based on delays, frequency, and availability
                if (((time_past_delay >= 0 and time_past_delay % inputs.testing_frequency[patient[INTERVENTION]][patient[OBSERVED_STATE]] == 0) or patient[TIME_TO_CONF_TEST] == 0) and 
                (test_availability[test] > np.sum(daily_tests[test]))):
                    # roll for testing if eligible
                    if roll_for_testing(patient, test, daily_tests, inputs):
                        self.test_costs[test] += inputs.testing_costs[test]
                # continue the countdown to a confirmatory test if not tested today
                if patient[TIME_TO_CONF_TEST] >= 0:
                    patient[TIME_TO_CONF_TEST] -= 1

            if patient[FLAGS] & HAS_PENDING_TEST:
                if patient[TIME_TO_TEST_RETURN] == 0:
                    # perform test return updates
                    patient[FLAGS] = patient[FLAGS] & (~HAS_PENDING_TEST)
                    result = bool(patient[FLAGS] & PENDING_TEST_RESULT)
                    old_intv = patient[INTERVENTION]
                    new_intv = inputs.switch_on_test_result[old_intv, patient[OBSERVED_STATE], int(result)]
                    # switch intereventions if necessary and reset the observed state time
                    if new_intv != old_intv:
                        switch_intervention(patient, inputs, new_intv, patient[OBSERVED_STATE], self.resource_utilization, resource_availability, state_tracker)
                        patient[OBSERVED_STATE_TIME] = -1
                else:
                    patient[TIME_TO_TEST_RETURN] -= 1

            patient[OBSERVED_STATE_TIME] += 1

            # update disease state
            if patient[TIME_INFECTED] > 0:
                patient[TIME_INFECTED] += 1

            if patient[DISEASE_STATE] == SUSCEPTABLE:
                if roll_for_incidence(patient, self.transmissions, self.transmission_groups, self.inputs.baseline_infection_probs[patient[TRANSM_GROUP]]):
                    patient[DISEASE_STATE] = INCUBATION
                    patient[FLAGS] = patient[FLAGS] & (~PRESENTED_THIS_DSTATE)
                    patient[TIME_INFECTED] = 1
                    self.cumulative_state_tracker[INCUBATION] += 1
                    new_infections[patient[TRANSM_GROUP]] += 1
            else:
                roll_for_transition(patient, self.cumulative_state_tracker, inputs)
            # roll for mortality
            if patient[DISEASE_STATE] == CRITICAL:
                roll_for_mortality(patient, inputs, self.resource_utilization)
            # update patient state tracking
            if patient[FLAGS] & IS_ALIVE:
                # calculate tomorrow's exposures
                foi_contribution = inputs.trans_prob[trans_period, patient[INTERVENTION], patient[TRANSM_GROUP], patient[DISEASE_STATE]]
                foi_contribution *= inputs.immunity_transm_mult[patient[IMMUNE_STATE]]
                foi_contribution *= inputs.exposure_matrices[patient[INTERVENTION], patient[TRANSM_GROUP], :]
                newtransmissions[patient[TRANSM_GROUP], :] += foi_contribution

                state_tracker[patient[SUBPOPULATION], patient[DISEASE_STATE]] += 1
                intv_tracker[patient[INTERVENTION], patient[DISEASE_STATE]] += 1
                vax_tracker[patient[IMMUNE_STATE], patient[DISEASE_STATE]] += 1
                self.intervention_costs[patient[INTERVENTION], patient[OBSERVED_STATE]] += inputs.intervention_daily_costs[patient[INTERVENTION], patient[OBSERVED_STATE]]

            else:  # must have died this month
                mort_tracker[patient[SUBPOPULATION], patient[INTERVENTION]] += 1
                self.mortality_costs += inputs.mortality_costs[patient[INTERVENTION]]
                self.cumulative_vax_mortality[patient[IMMUNE_STATE]] += 1

        costs = np.asarray([np.sum(self.test_costs), np.sum(self.intervention_costs), self.mortality_costs], dtype=float)
        vax_tracker[:,-1] = self.cumulative_vax_mortality  # keep track of cumulative mortality by vax status

        self.outputs.log_daily_state(self.day, state_tracker, self.cumulative_state_tracker, newtransmissions, new_infections, mort_tracker, intv_tracker, vax_tracker, daily_screens, daily_tests, self.resource_utilization, self.non_covids, costs)
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
