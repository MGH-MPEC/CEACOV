# -*- coding: utf-8 -*-
"""
COVID-19 Microsimulation Model

@author: Chris Panella (cpanella@mgh.harvard.edu)
"""

# Exeptions

class InvalidParamError(Exception):
	""" class for invalid input errors"""
	pass

# Meta Params

# State var indices
NUM_STATE_VARS = 5

STATE_VARS = FLAGS, SUBPOPULATION, DISEASE_STATE, D_STATE_TIME, INTERVENTION = range(NUM_STATE_VARS)

# Patient State

SUBPOPULATIONS_NUM = 6

SUBPOPULATIONS = FEMALE_YOUTH, MALE_YOUTH, FEMALE_ADULT, MALE_ADULT, FEMALE_SENIOR, MALE_SENIOR = range(SUBPOPULATIONS_NUM)

SUBPOPULATION_STRS = ("female 0-15y", "male 0-15y", "female 15-55y", "male 15-55y", "female >55y", "male >55y")

RISK_FACTORS_NUM = 4

RISK_FACTORS = (0, 1, 2, 3)

# Flags (bitwise flags in powers of 2)

FLAGS_NUM = 4

IS_ALIVE, IS_INFECTED, IS_DIAGNOSED, HAS_INTERVENTION = map(lambda x: 2 ** x, range(FLAGS_NUM))

# Covid Disease State

DISEASE_STATES_NUM = 7

DISEASE_STATES = SUSCEPTABLE, INCUBATION, MILD, MODERATE, SEVERE, CRITICAL, RECOVERED = range(DISEASE_STATES_NUM)

DISEASE_STATE_STRS = ("susceptable", "incubation", "mild", "moderate", "severe", "critical", "recovered")

# Interventions

INTERVENTIONS_NUM = 3

INTERVENTION_FLAGS = map(lambda x: 2 ** x, range(INTERVENTIONS_NUM))

INTERVENTION_STRS = ("isolation", "hospitalization", "intensive care")

# Outcomes

# DAILY_OUTCOMES_NUM = 