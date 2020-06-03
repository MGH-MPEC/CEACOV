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

MODEL_VERSION = "v0.6"

# State var indices
NUM_STATE_VARS = 8

STATE_VARS = FLAGS, SUBPOPULATION, OBSERVED_STATE, OBSERVED_STATE_TIME, DISEASE_STATE, DISEASE_PROGRESSION, INTERVENTION, TIME_TO_TEST_RETURN = range(NUM_STATE_VARS)

# Flags (bitwise flags in powers of 2)

FLAGS_NUM = 7

IS_ALIVE, IS_INFECTED, PRESENTED_THIS_DSTATE, HAS_PENDING_TEST, PENDING_TEST_RESULT, NON_COVID_RI, EVER_TESTED_POSITIVE = map(lambda x: 2 ** x, range(FLAGS_NUM))

# Demographic State

# GENDER_STRS = ("female", "male")

AGE_CATEGORY_STRS = ("0-19y", "20-59y", ">=60y")

# HIV_STATUS_STRS = ("HIV+", "no HIV")

# COMORBIDITY_STRS = ("no comorbidities", "some comorbidities")


SUBPOPULATIONS_NUM = 3

SUBPOPULATIONS = range(SUBPOPULATIONS_NUM)

SUBPOPULATION_STRS = AGE_CATEGORY_STRS


# Covid Disease State

DISEASE_STATES_NUM = 8

DISEASE_STATES = SUSCEPTABLE, INCUBATION, ASYMP, MODERATE, SEVERE, CRITICAL, RECUPERATION, RECOVERED = range(DISEASE_STATES_NUM)

DISEASE_STATE_STRS = ("susceptible", "pre-infectious incubation", "asymptomatic", "mild/moderate", "severe", "critical", "recuperation", "recovered")

DISEASE_PROGRESSIONS_NUM = 4

DISEASE_PROGRESSIONS = TO_ASYMP, TO_MODERATE, TO_SEVERE, TO_CRITICAL = range(DISEASE_PROGRESSIONS_NUM)

DISEASE_PROGRESSION_STRS = ("asymptomatic", "mild/moderate", "severe", "critical")

PROGRESSION_PATHS = np.array([[0, ASYMP, RECOVERED, 0, 0, 0, 0, 0],
         					  [0, ASYMP, MODERATE, RECOVERED, 0, 0, 0, 0],
          					  [0, ASYMP, MODERATE, SEVERE, RECOVERED, 0, 0, 0],
          					  [0, ASYMP, MODERATE, SEVERE, CRITICAL, RECUPERATION, RECOVERED, 0]], dtype=int)

# Resources

RESOURCES_NUM = 8

RESOURCE_STRS = [f"resource {i}" for i in range(0,RESOURCES_NUM)]

# Interventions

INTERVENTIONS_NUM = 8

INTERVENTIONS = range(INTERVENTIONS_NUM)

INTERVENTION_STRS = tuple(["no intervention"] + [f"intervention {i}" for i in range(1,INTERVENTIONS_NUM)])


OBSERVED_STATES_NUM = 5

OBSERVED_STATES = SYMP_ASYMP, SYMP_MODERATE, SYMP_SEVERE, SYMP_CRITICAL, SYMP_RECUPERATION = range(OBSERVED_STATES_NUM)

OBSERVED_STATE_STRS = ("no symptoms", "mild/moderate", "severe", "critical", "recuperation")


T_RATE_PERIODS_NUM = 5

# Testing

TESTS_NUM = 8

TESTS = range(TESTS_NUM)

# Costs

COST_STRS = ("test costs", "intervention costs", "mortality costs")

# Outcomes

DAILY_OUTCOME_STRS = ["day#"] + list(DISEASE_STATE_STRS) + [f"cumulative {state}" for state in DISEASE_PROGRESSION_STRS] + \
					 ["new infections", "cumulative infections", "dead"] + [f"mortality for {subpop}" for subpop in SUBPOPULATION_STRS] + \
					 ["exposures", "non-covid presenting"] + list(INTERVENTION_STRS) + \
					 [f"test {n} ({status})" for n in TESTS for status in ("-","+")] + list(COST_STRS) + \
					 [f"resource untilization {rsc}" for rsc in range(RESOURCES_NUM)] + [f"mortality on {intv}" for intv in INTERVENTION_STRS]
