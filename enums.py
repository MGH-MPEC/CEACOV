# -*- coding: utf-8 -*-
"""
COVID-19 Microsimulation Model

@author: Chris Panella (cpanella@mgh.harvard.edu)
"""

import numpy as np


# Exeptions

class InvalidParamError(Exception):
    """ class for invalid input errors"""
    pass

# Meta Params

MODEL_VERSION = "v0.9_multTests"


# State var indices

NUM_STATE_VARS = 13

STATE_VARS = FLAGS, NON_COVID_TIME, SUBPOPULATION, TRANSM_GROUP, OBSERVED_STATE, OBSERVED_STATE_TIME, DISEASE_STATE, DISEASE_PROGRESSION, INTERVENTION, TIME_TO_SCREEN_RETURN, TIME_TO_CONF_TEST, TIME_TO_TEST_RETURN, TIME_INFECTED = range(NUM_STATE_VARS)


# Flags (bitwise flags in powers of 2

FLAGS_NUM = 9

IS_ALIVE, IS_INFECTED, PRESENTED_THIS_DSTATE, WAS_SX_SCREENED, PENDING_SCREEN_RESULT, HAS_PENDING_TEST, PENDING_TEST_RESULT, NON_COVID_RI, PRE_RECOVERY = map(lambda x: 2 ** x, range(FLAGS_NUM))


# Demographic State

AGE_CATEGORY_STRS = ("0-19y", "20-59y", ">=60y")

SUBPOPULATIONS_NUM = 3

SUBPOPULATIONS = range(SUBPOPULATIONS_NUM)

SUBPOPULATION_STRS = AGE_CATEGORY_STRS

TRANSMISSION_GROUPS_NUM = 4

TRANSMISSION_GROUPS = range(TRANSMISSION_GROUPS_NUM)

TRANSMISSION_GROUP_STRS = [f"tn_group {n}" for n in range(TRANSMISSION_GROUPS_NUM)]

T_RATE_PERIODS_NUM = 5


# Covid Disease State

DISEASE_STATES_NUM = 8

DISEASE_STATES = SUSCEPTABLE, INCUBATION, ASYMP, MODERATE, SEVERE, CRITICAL, RECUPERATION, RECOVERED = range(DISEASE_STATES_NUM)

DISEASE_STATE_STRS = ("susceptible", "pre-infectious incubation", "asymptomatic", "mild/moderate", "severe", "critical", "recuperation", "recovered")

DISEASE_PROGRESSIONS_NUM = 4

DISEASE_PROGRESSIONS = TO_ASYMP, TO_MODERATE, TO_SEVERE, TO_CRITICAL = range(DISEASE_PROGRESSIONS_NUM)

DISEASE_PROGRESSION_STRS = ("asymptomatic", "mild/moderate", "severe", "critical")

PROGRESSION_PATHS = np.array([[-1, ASYMP, RECOVERED, -1, -1, -1, -1, SUSCEPTABLE],
                              [-1, ASYMP, MODERATE, RECOVERED, -1, -1, -1, SUSCEPTABLE],
                              [-1, ASYMP, MODERATE, SEVERE, RECOVERED, -1, -1, SUSCEPTABLE],
                              [-1, ASYMP, MODERATE, SEVERE, CRITICAL, RECUPERATION, RECOVERED, SUSCEPTABLE]], dtype=int)

HAS_PRE_RECOVERY_STATE = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 1, 0, 0, 0]], dtype=bool)

PROG_TYPES_NUM = 3

PROG_TYPES = (PROG_NONE, PROG_NORMAL, PROG_PRE_REC) = range(PROG_TYPES_NUM)

# Resources

RESOURCES_NUM = 8

RESOURCE_STRS = [f"resource {i}" for i in range(0, RESOURCES_NUM)]

RESOURCES_PERIODS_NUM = 5

# Interventions

INTERVENTIONS_NUM = 18

INTERVENTIONS = range(INTERVENTIONS_NUM)

INTERVENTION_STRS = tuple([f"intervention {i}" for i in range(INTERVENTIONS_NUM)])


OBSERVED_STATES_NUM = 5

OBSERVED_STATES = SYMP_ASYMP, SYMP_MODERATE, SYMP_SEVERE, SYMP_CRITICAL, SYMP_RECUPERATION = range(OBSERVED_STATES_NUM)

OBSERVED_STATE_STRS = ("no symptoms", "mild/moderate", "severe", "critical", "recuperation")

# Prophs

PROPH_DROPOUT_PERIODS_NUM = 3

# Testing

TESTS_NUM = 8

TEST_AVAILABILITY_PERIODS_NUM = 5

TEST_SENS_THRESHOLDS_NUM = 4

TEST_CHAR_THRESHOLD_STRS = ["never infected", "1 <= time_infected < t0", "t0 <= time_infected < t1",
                            "t1 <= time_infected < t2", "t2 <= time_infected < t3", "time_infected >= t3"]

TESTS = range(TESTS_NUM)

# Costs

COST_STRS = ("screen costs", "test costs", "intervention costs", "mortality costs")

# Outcomes

DAILY_OUTCOME_STRS = ["day#"] + list(DISEASE_STATE_STRS) + [f"cumulative {state}" for state in DISEASE_PROGRESSION_STRS] + \
                     [f"{igroup} new infections" for igroup in TRANSMISSION_GROUP_STRS] + ["cumulative infections", "dead"] + [f"mortality for {subpop}" for subpop in SUBPOPULATION_STRS] + \
                     [f"FoI {tgroup} -> {igroup}" for igroup in TRANSMISSION_GROUP_STRS for tgroup in TRANSMISSION_GROUP_STRS] + \
                     ["non-covid presenting"] + list(INTERVENTION_STRS) + [f"screen ({status})" for status in ("-","+")] +  [f"test {n} ({status})" for n in TESTS for status in ("-","+")] + list(COST_STRS) + \
                     [f"resource untilization {rsc}" for rsc in range(RESOURCES_NUM)] + [f"mortality on {intv}" for intv in INTERVENTION_STRS]

