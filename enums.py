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

# State var indices
NUM_STATE_VARS = 6

STATE_VARS = FLAGS, SUBPOPULATION, DISEASE_STATE, D_STATE_TIME, DISEASE_PROGRESSION, INTERVENTION = range(NUM_STATE_VARS)

# Flags (bitwise flags in powers of 2)

FLAGS_NUM = 3

IS_ALIVE, IS_INFECTED, IS_DIAGNOSED = map(lambda x: 2 ** x, range(FLAGS_NUM))

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

DISEASE_STATES = SUSCEPTABLE, INCUBATION, MILD, MODERATE, SEVERE, CRITICAL, RECUPERATION, RECOVERED = range(DISEASE_STATES_NUM)

DISEASE_STATE_STRS = ("susceptable", "incubation", "mild", "moderate", "severe", "critical", "recuperation", "recovered")

DISEASE_PROGRESSIONS_NUM = 4

DISEASE_PROGRESSIONS = TO_MILD, TO_MODERATE, TO_SEVERE, TO_CRITICAL = range(DISEASE_PROGRESSIONS_NUM)

DISEASE_PROGRESSION_STRS = DISEASE_STATE_STRS[MILD:RECUPERATION]

PROGRESSION_PATHS = np.array([[0, MILD, RECOVERED, 0, 0, 0, 0, 0],
         					  [0, MILD, MODERATE, RECOVERED, 0, 0, 0, 0],
          					  [0, MILD, MODERATE, SEVERE, RECOVERED, 0, 0, 0],
          					  [0, MILD, MODERATE, SEVERE, CRITICAL, RECUPERATION, RECOVERED, 0]], dtype=int)

# Resources

RESOURCES_NUM = 3

RESOURCES = HOSPITAL_BEDS, ICU_BEDS, VENTILATORS = range(RESOURCES_NUM)

RESOURCE_STRS = ("hospital beds", "ICU beds", "Ventilators")

# Interventions

INTERVENTIONS_NUM = 6

INTERVENTIONS = range(INTERVENTIONS_NUM)

INTERVENTION_STRS = tuple(["no intervention"] + [f"intervention {i}" for i in range(1,INTERVENTIONS_NUM)])

# Outcomes

# DAILY_OUTCOMES_NUM = 
