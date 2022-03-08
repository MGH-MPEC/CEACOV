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
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def generate_simulation_inputs():
    sim_in = {
        "cohort size": 1000,
        "time horizon": 180,
        "fixed seed": True,
        "detailed state outputs": False,
        "detailed vaccination outputs": False
    }
    return sim_in


def generate_initialization_inputs():
    init_in = {
        "transmission group dist": {
            TRANSMISSION_GROUP_STRS[tgroup]: 1 if tgroup == 0 else 0
            for tgroup in TRANSMISSION_GROUPS
        },
        "risk category dist": {
            f"for {tgroup}": {
                SUBPOPULATION_STRS[subpop]: 1 if subpop == 0 else 0
                for subpop in SUBPOPULATIONS
            } for tgroup in TRANSMISSION_GROUP_STRS
        },
        "immune states dist": {
            f"for {tgroup}": {
                IMMUNE_STATE_STRS[istate]: 1 if istate == 0 else 0
                for istate in range(IMMUNE_STATES_NUM)
            } for tgroup in TRANSMISSION_GROUP_STRS
        },
        "initial disease dist": {
            f"for {tgroup}": {
                f"for {istate}": {
                    DISEASE_STATE_STRS[dstate]: 1 if dstate == 0 else 0
                    for dstate in DISEASE_STATES
                } for istate in IMMUNE_STATE_STRS
            } for tgroup in TRANSMISSION_GROUP_STRS
        },
        "covid naive severity dist": {
            f"for {subpop}": {
                DISEASE_PROGRESSION_STRS[dprog]: 1 if dprog == 0 else 0
                for dprog in DISEASE_PROGRESSIONS
            } for subpop in SUBPOPULATION_STRS
        },
        "start intervention": {
            f"for {tgroup}": 0
            for tgroup in TRANSMISSION_GROUP_STRS
        }
    }
    return init_in


def generate_progression_inputs():
    prog_in = {
        INTERVENTION_STRS[intv]: {
            f"for severity = {DISEASE_PROGRESSION_STRS[severity]}": {
                "daily disease progression probability": {
                    f"from {DISEASE_STATE_STRS[dstate]} to {DISEASE_STATE_STRS[PROGRESSION_PATHS[severity][dstate]]}": 0
                    for dstate in DISEASE_STATES if (PROGRESSION_PATHS[severity][dstate] != -1)
                },
                "daily switch to pre-recovery probability": {
                    f"from {DISEASE_STATE_STRS[dstate]}": 0
                    for dstate in DISEASE_STATES if HAS_PRE_RECOVERY_STATE[severity][dstate]
                },
                "daily expedited recovery probability": {
                    f"from pre-recovery: {DISEASE_STATE_STRS[dstate]}": 0
                    for dstate in DISEASE_STATES if HAS_PRE_RECOVERY_STATE[severity][dstate]
                }
            } for severity in DISEASE_PROGRESSIONS
        } for intv in INTERVENTIONS
    }
    return prog_in


def generate_immunity_inputs():
    immunity_in = {
        "vaccination to be given (-1 for no vaccine)": {
            f"for {intv}": -1
            for intv in INTERVENTION_STRS
        }
    }
    immunity_in.update({
        f"immunity parameters for {istate}": {
            "initial prob full immunity": {
                f"for {subpop}": 1
                for subpop in SUBPOPULATION_STRS
            },
            "daily prob loss full immunity": {
                f"for {subpop}": 0
                for subpop in SUBPOPULATION_STRS
            },
            "partial immunity severity dist": {
                f"for {subpop}": {
                    DISEASE_PROGRESSION_STRS[dprog]: 1 if dprog == 0 else 0
                    for dprog in DISEASE_PROGRESSIONS
                } for subpop in SUBPOPULATION_STRS
            },
            "partial immunity transmission rate multiplier": 1,
            "immune state transition on loss of full immunity": -1,
            "immune state priority": 0
        } for istate in IMMUNE_STATE_STRS[RECOVERED:]
    })
    return immunity_in


def generate_mortality_inputs():
    mort_in = {
        f"daily mortality prob for {subpop} while critical": {
            f"on {intv}": 0
            for intv in INTERVENTION_STRS
        } for subpop in SUBPOPULATION_STRS
    }
    return mort_in


def generate_transmission_inputs():
    transm_in = {
        INTERVENTION_STRS[n]: {
            "baseline daily infection probability": {
                f"for {tgroup}": 0
                for tgroup in TRANSMISSION_GROUP_STRS
            },
            "transmission probability per exposure": {
                f"for {tgroup}": {
                    f"while {dstate}": 0
                    for dstate in DISEASE_STATE_STRS[ASYMP:IMMUNE]
                }
                for tgroup in TRANSMISSION_GROUP_STRS
            },
            "transmission rate multipliers": {
                threshold: 1
                for threshold in ["for 0 <= day# < t0"] +
                                 [f"for t{i} <= day# < t{i+1}" for i in range(T_RATE_PERIODS_NUM-2)] +
                                 [f"day# > t{T_RATE_PERIODS_NUM-2}"]
            },
            "exposure matrix": {
                f"from {tgroup}": {
                    f"to {igroup}": 1
                    for igroup in TRANSMISSION_GROUP_STRS if igroup >= tgroup
                }
                for tgroup in TRANSMISSION_GROUP_STRS
            }
        } for n in INTERVENTIONS
    }
    
    transm_in["transmission multiplier time thresholds"] = {
        f"t{threshold}": (10 + (10 * threshold))
        for threshold in range(T_RATE_PERIODS_NUM-1)
    }
    
    return transm_in


def generate_test_inputs():
    # general testing parameters
    testing_in = {
        "symptom screen result return time": 0,
        "probability of positive symptom screen result": {
            f"if {symstate}": 0.0
            for symstate in OBSERVED_STATE_STRS
        },
        "test availability thresholds": {
            f"t{threshold}": (10 + (10 * threshold))
            for threshold in range(TEST_AVAILABILITY_PERIODS_NUM-1)
        },
        "daily test availabilities": {
            threshold: {
                f"test {test}": 0
                for test in TESTS
            } for threshold in ["for 0 <= day# < t0"] +
                               [f"for t{i} <= day# < t{i+1}" for i in range(TEST_AVAILABILITY_PERIODS_NUM-2)] +
                               [f"day# > t{TEST_AVAILABILITY_PERIODS_NUM-2}"]}
    }
    # specific test characteristics
    testing_in.update({
        f"test {test}": {
            "result return time": 0,
            "probability of positive result": {
                interval: 0.0
                for interval in TEST_CHAR_THRESHOLD_STRS
            },
            "sensitivity thresholds": {
                f"t{threshold}": (5 + (5 * threshold))
                for threshold in range(TEST_SENS_THRESHOLDS_NUM)},
            "delay to test": 0
        } for test in TESTS
    })
    return testing_in


def generate_testing_strat_inputs():
    test_strat_in = {
        INTERVENTION_STRS[n]: {
            "switch to intervention on positive symptom screen result": {
                f"if {symstate}": n
                for symstate in OBSERVED_STATE_STRS
            },
            "switch to intervention on negative symptom screen result": {
                f"if {symstate}": n
                for symstate in OBSERVED_STATE_STRS
            },
            "screening interval": {
                f"if {symstate}": 1
                for symstate in OBSERVED_STATE_STRS
            },
            "probability receive symptom screen": {
                f"if {symstate}": {
                    f"for {subpop}": 0
                    for subpop in SUBPOPULATION_STRS
                }
                for symstate in OBSERVED_STATE_STRS
            },
            "probability receive confirmatory test": 0,
            "delay to confirmatory test": 0,
            "probability of presenting to care": {
                f"while {dstate}": 0
                for dstate in DISEASE_STATE_STRS
            },
            "switch to intervention on positive test result": {
                f"if observed {symstate}": n
                for symstate in OBSERVED_STATE_STRS
            },
            "switch to intervention on negative test result": {
                f"if observed {symstate}": n
                for symstate in OBSERVED_STATE_STRS
            },
            "test number": {
                f"if observed {symstate}": 0
                for symstate in OBSERVED_STATE_STRS
            },
            "testing interval": {
                f"if observed {symstate}": 1
                for symstate in OBSERVED_STATE_STRS
            },
            "probability receive test": {
                f"if observed {symstate}": {
                    f"for {subpop}": 0
                    for subpop in SUBPOPULATION_STRS
                }
                for symstate in OBSERVED_STATE_STRS
            }
        } for n in INTERVENTIONS
    }
    return test_strat_in


def generate_cost_inputs():
    cost_in = {
        "screening cost": {
            f"if {symstate}": 0.0
            for symstate in OBSERVED_STATE_STRS
        }, 
        "testing costs": {
            f"test {test}": 0.0
            for test in TESTS
        },
        "daily intervention costs": {
            intervention: {
                f"if observed {symstate}": 0.0
                for symstate in OBSERVED_STATE_STRS
            } for intervention in INTERVENTION_STRS
        },
        "mortality costs": {
            intervention: 0.0
            for intervention in INTERVENTION_STRS
        }
    }
    return cost_in


def generate_resource_inputs():
    rsc_in = {
        "resource availability thresholds": {
            f"t{threshold}": (10 + (10 * threshold))
            for threshold in range(RESOURCES_PERIODS_NUM - 1)},
        "resource availabilities": {
            threshold: {
                resource: 0
                for resource in RESOURCE_STRS}
            for threshold in ["for 0 <= day# < t0"] +
                             [f"for t{i} <= day# < t{i+1}" for i in range(RESOURCES_PERIODS_NUM-2)] +
                             [f"day# > t{RESOURCES_PERIODS_NUM-2}"]
        },
        "resource requirements": {
            f"for {intervention}": {
                f"if observed {symstate}": []
                for symstate in OBSERVED_STATE_STRS}
            for intervention in INTERVENTION_STRS},
        "back-up interventions": {
            f"for {intervention}": {
                f"if observed {symstate}": 0
                for symstate in OBSERVED_STATE_STRS}
            for intervention in INTERVENTION_STRS}
    }
    return rsc_in


def generate_non_covid_inputs():
    no_co_in = {
        "daily prob present non-covid": {
            f"with {symstate} symptoms": {
                f"for {subpop}": 0.0
                for subpop in SUBPOPULATION_STRS}
            for symstate in OBSERVED_STATE_STRS[SYMP_MODERATE:SYMP_CRITICAL+1]
        },
        "non covid symptom duration": {
            f"for {symstate} symptoms": {
                f"for {subpop}": 0
                for subpop in SUBPOPULATION_STRS}
            for symstate in OBSERVED_STATE_STRS[SYMP_MODERATE:SYMP_CRITICAL+1]},
    }
    return no_co_in


def generate_prophylaxis_inputs():
    proph_in = {
        "enable prophylaxis module": False,
        "prophylaxis efficacy": 1,
        "target coverage": 1,
        "initial coverage": 0,
        "time of target coverage": 20,
        "probability of dropout time thresholds": {
            f"t{threshold}": (10 + (10 * threshold))
            for threshold in range(PROPH_DROPOUT_PERIODS_NUM-1)
        },
        "probability of dropout": {
            threshold: 0
            for threshold in ["for 0 <= day# < t0"] +
                             [f"for t{i} <= day# < t{i+1}" for i in range(PROPH_DROPOUT_PERIODS_NUM-2)] + 
                             [f"day# > t{PROPH_DROPOUT_PERIODS_NUM-2}"]
        }
    }
    return proph_in


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
    # Immunity
    inputs["immunity"] = generate_immunity_inputs()
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
    # Prophyaxis
    inputs["Prophyaxis"] = generate_prophylaxis_inputs()

    return inputs


class Inputs():
    def __init__(self):
        # simulation parameters
        self.cohort_size = 1000
        self.time_horizon = 180
        self.fixed_seed = True
        self.state_detail = False
        self.vax_detail = False
        # initialization inputs
        self.tgroup_dist = np.zeros((TRANSMISSION_GROUPS_NUM), dtype=float)
        self.subpop_dist = np.zeros((TRANSMISSION_GROUPS_NUM, SUBPOPULATIONS_NUM), dtype=float)
        self.istate_dist = np.zeros((TRANSMISSION_GROUPS_NUM, IMMUNE_STATES_NUM), dtype=float)
        self.dstate_dist = np.zeros((TRANSMISSION_GROUPS_NUM, IMMUNE_STATES_NUM, DISEASE_STATES_NUM), dtype=float)
        self.start_intvs = np.zeros((TRANSMISSION_GROUPS_NUM), dtype=int)
        # transition inputs
        self.progression_probs = np.zeros((INTERVENTIONS_NUM, DISEASE_PROGRESSIONS_NUM, DISEASE_STATES_NUM, PROG_TYPES_NUM), dtype=float)
        self.expedited_recovery_probs = np.zeros((INTERVENTIONS_NUM, DISEASE_PROGRESSIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        self.mortality_probs = np.zeros((SUBPOPULATIONS_NUM, INTERVENTIONS_NUM), dtype=float)
        # immunity inputs
        self.vaccination = np.zeros(INTERVENTIONS_NUM, dtype=int)
        self.prob_full_immunity = np.ones((IMMUNE_STATES_NUM, SUBPOPULATIONS_NUM), dtype=float)
        self.daily_prob_lose_immunity = np.zeros((IMMUNE_STATES_NUM, SUBPOPULATIONS_NUM), dtype=float)
        self.severity_dist = np.zeros((IMMUNE_STATES_NUM, SUBPOPULATIONS_NUM, DISEASE_PROGRESSIONS_NUM), dtype=float)
        self.immunity_transm_mult = np.ones(IMMUNE_STATES_NUM, dtype=float)
        self.immunity_transition = np.full(IMMUNE_STATES_NUM, -1, dtype=int)
        self.immunity_priority = np.full(IMMUNE_STATES_NUM, 0, dtype=int)
        # transmission inputs
        self.baseline_infection_probs = np.zeros(TRANSMISSION_GROUPS_NUM, dtype=float)
        self.trans_rate_thresholds = np.zeros(T_RATE_PERIODS_NUM-1, dtype=int)
        self.trans_prob = np.zeros((T_RATE_PERIODS_NUM, INTERVENTIONS_NUM, TRANSMISSION_GROUPS_NUM, DISEASE_STATES_NUM), dtype=float)
        # transmissions from, transmissions to
        self.exposure_matrices = np.zeros((INTERVENTIONS_NUM, TRANSMISSION_GROUPS_NUM, TRANSMISSION_GROUPS_NUM), dtype=float)
        # test inputs, starting with the symptom screen and continuing to regular tests
        self.prob_sx_screen = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM, SUBPOPULATIONS_NUM), dtype=float)
        self.screen_characteristics = np.zeros((OBSERVED_STATES_NUM), dtype=float)
        self.switch_on_screen_result = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM, 2), dtype=int)
        self.screening_frequency = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=int)
        self.test_availability_thresholds = np.zeros((TEST_AVAILABILITY_PERIODS_NUM-1), dtype=int)
        self.test_availabilities = np.zeros((TEST_AVAILABILITY_PERIODS_NUM, TESTS_NUM), dtype=int)
        self.test_return_delay = np.zeros(TESTS_NUM, dtype=int)
        self.test_characteristics = np.zeros((TESTS_NUM, len(TEST_CHAR_THRESHOLD_STRS)), dtype=float)
        self.test_sens_thresholds = np.ones((TESTS_NUM, TEST_SENS_THRESHOLDS_NUM + 1), dtype=int)
        self.test_lag = np.zeros((TESTS_NUM), dtype=int)
        # intervention
        self.prob_present = np.zeros((INTERVENTIONS_NUM, DISEASE_STATES_NUM), dtype=float)
        self.switch_on_test_result = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM, 2), dtype=int)
        self.test_number = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=int)
        self.testing_frequency = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=int)
        self.prob_receive_test = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM, SUBPOPULATIONS_NUM), dtype=float)
        self.prob_confirmatory_test = np.zeros((INTERVENTIONS_NUM), dtype=float)
        self.confirmatory_test_lag = np.zeros((INTERVENTIONS_NUM), dtype=int)
        # cost inputs
        self.screening_costs = np.zeros(OBSERVED_STATES_NUM, dtype=float)
        self.testing_costs = np.zeros(TESTS_NUM, dtype=float)
        self.intervention_daily_costs = np.zeros((INTERVENTIONS_NUM, OBSERVED_STATES_NUM), dtype=float)
        self.mortality_costs = np.zeros((DISEASE_STATES_NUM, INTERVENTIONS_NUM), dtype=float)
        # resource inputs
        self.resource_availability_thresholds = np.zeros((RESOURCES_PERIODS_NUM-1), dtype=int)
        self.resource_availabilities = np.zeros((RESOURCES_PERIODS_NUM, RESOURCES_NUM), dtype=int)
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
        self.vax_detail = sim_params["detailed vaccination outputs"]

        # initialization inputs
        init_params = param_dict["initial state"]
        self.tgroup_dist = np.asarray(dict_to_array(init_params["transmission group dist"]), dtype=float)
        self.subpop_dist = np.asarray(dict_to_array(init_params["risk category dist"]), dtype=float)
        self.istate_dist = np.asarray(dict_to_array(init_params["immune states dist"]), dtype=float)
        self.dstate_dist = np.asarray(dict_to_array(init_params["initial disease dist"]), dtype=float)
        self.severity_dist[0] = np.asarray(dict_to_array(init_params["covid naive severity dist"]), dtype=float)
        self.start_intvs = np.asarray(dict_to_array(init_params["start intervention"]), dtype=int)
        if np.any((self.start_intvs < 0) | (self.start_intvs >= INTERVENTIONS_NUM)):
            raise UserWarning("start intervention inputs must be valid intervention numbers")

        # transition inputs
        # really gross - consider reworking
        prog_array = dict_to_array(param_dict["disease progression"])
        for intv in INTERVENTIONS:
            for severity in DISEASE_PROGRESSIONS:
                for dstate in DISEASE_STATES:
                    if (PROGRESSION_PATHS[severity,dstate] != -1):
                        if dstate == IMMUNE:
                            self.progression_probs[intv, severity, IMMUNE, PROG_NORMAL] = prog_array[intv][severity][0][-1]
                        else:                        
                            self.progression_probs[intv,severity,dstate,PROG_NORMAL] = prog_array[intv][severity][0][dstate-INCUBATION]
                            if (dstate >= ASYMP) and (dstate - ASYMP < severity):
                                self.progression_probs[intv,severity,dstate,PROG_PRE_REC] = prog_array[intv][severity][1][dstate-ASYMP]
                                self.expedited_recovery_probs[intv,severity,dstate] = prog_array[intv][severity][2][dstate-ASYMP]
        self.progression_probs[:,:,:,PROG_NONE] = 1 - np.sum(self.progression_probs, axis=3)

        self.mortality_probs[:,:] = np.asarray(dict_to_array(param_dict["disease mortality"]), dtype=float)

        # immunity inputs
        imty_params = dict_to_array(param_dict["immunity"])
        self.vaccination = np.asarray(imty_params[0])
        for i_status in range(RECOVERED, IMMUNE_STATES_NUM):
            self.prob_full_immunity[i_status,:] = imty_params[i_status][0]
            self.daily_prob_lose_immunity[i_status,:] = imty_params[i_status][1]
            self.severity_dist[i_status,:,:] = imty_params[i_status][2]
            self.immunity_transm_mult[i_status] = imty_params[i_status][3]
            try:
                vax_num = imty_params[i_status][4]
                self.immunity_transition[i_status] = (vax_num + 2) if (vax_num >= 0) else -1
                self.immunity_priority[i_status] = imty_params[i_status][5]
            except IndexError:
                self.immunity_transition[i_status] = -1
                self.immunity_priority[i_status] = 0

        # transmission inputs
        transm_params = param_dict["transmissions"]
        self.trans_rate_thresholds = np.asarray(dict_to_array(transm_params["transmission multiplier time thresholds"]))
        tgroups = self.tgroup_dist.copy()
        tgroups[tgroups == 0] = 1
        group_size_correction = np.outer(1 / tgroups, tgroups)
        for i in INTERVENTIONS:
            strat_dict = transm_params[INTERVENTION_STRS[i]]
            try:
                self.baseline_infection_probs = dict_to_array(strat_dict["baseline daily infection probability"])
            except KeyError:
                pass
            self.trans_prob[:,i,:,ASYMP:IMMUNE] = dict_to_array(strat_dict["transmission probability per exposure"])
            trans_mults = np.asarray(dict_to_array(strat_dict["transmission rate multipliers"]), dtype=float)
            # apply transmission mults
            for j in range(T_RATE_PERIODS_NUM):
                self.trans_prob[j,i,:,:] *= trans_mults[j]
            exposure_array = dict_to_array(strat_dict["exposure matrix"])
            for tgroup in TRANSMISSION_GROUPS:
                self.exposure_matrices[i, tgroup, tgroup:] = exposure_array[tgroup]
                self.exposure_matrices[i,tgroup:,tgroup] = exposure_array[tgroup] * group_size_correction[tgroup:,tgroup]
                # for igroup in TRANSMISSION_GROUPS:
                #     if igroup >= tgroup:  # already read from inputs
                #         break
                #     else:  # calculate under symmetric contact assumption
                #         self.exposure_matrices[i, tgroup, igroup] = self.exposure_matrices[i, igroup, tgroup] * group_size_correction[tgroup, igroup]

        # test inputs
        test_inputs = dict_to_array(param_dict["tests"])
        self.screen_return_delay = test_inputs[0]
        self.screen_characteristics[:] = test_inputs[1]
        self.test_availability_thresholds[:] = test_inputs[2]
        self.test_availabilities[:,:] = test_inputs[3]
        for test in TESTS:
            self.test_return_delay[test] = test_inputs[test+4][0]
            self.test_characteristics[test,:] = test_inputs[test+4][1]
            self.test_sens_thresholds[test,1:] = test_inputs[test+4][2]
            self.test_lag[test] = test_inputs[test+4][3]

        # intervention strategies
        intv_strat_inputs =  param_dict["testing strategies"]
        for i in INTERVENTIONS:
            strat_dict = intv_strat_inputs[INTERVENTION_STRS[i]]
            self.switch_on_screen_result[i,:,0] = dict_to_array(strat_dict["switch to intervention on negative symptom screen result"])
            self.switch_on_screen_result[i,:,1] = dict_to_array(strat_dict["switch to intervention on positive symptom screen result"])
            self.screening_frequency[i,:] = dict_to_array(strat_dict["screening interval"])
            self.prob_sx_screen[i,:,:] = dict_to_array(strat_dict["probability receive symptom screen"])
            self.prob_confirmatory_test[i] = strat_dict["probability receive confirmatory test"]
            self.confirmatory_test_lag[i] = strat_dict["delay to confirmatory test"]
            self.prob_present[i,:] = dict_to_array(strat_dict["probability of presenting to care"])
            self.switch_on_test_result[i,:,0] = dict_to_array(strat_dict["switch to intervention on negative test result"])
            self.switch_on_test_result[i,:,1] = dict_to_array(strat_dict["switch to intervention on positive test result"])
            self.test_number[i,:] = dict_to_array(strat_dict["test number"])
            self.testing_frequency[i,:] = dict_to_array(strat_dict["testing interval"])
            self.prob_receive_test[i,:,:] = dict_to_array(strat_dict["probability receive test"])

        if np.any((self.switch_on_test_result < 0) | (self.switch_on_test_result >= INTERVENTIONS_NUM)):
            raise UserWarning("switch on test return inputs must be valid intervention numbers")

        # costs
        cost_inputs = param_dict["costs"]
        self.screening_costs = np.asarray(dict_to_array(cost_inputs["screening cost"]))
        self.testing_costs = np.asarray(dict_to_array(cost_inputs["testing costs"]))
        self.intervention_daily_costs = np.asarray(dict_to_array(cost_inputs["daily intervention costs"]))
        self.mortality_costs = np.asarray(dict_to_array(cost_inputs["mortality costs"]))

        # resources
        rsc_inputs = param_dict["resources"]
        self.resource_availability_thresholds = np.asarray(dict_to_array(rsc_inputs["resource availability thresholds"]), dtype=int)
        self.resource_availabilities = np.asarray(dict_to_array(rsc_inputs["resource availabilities"]), dtype=int)
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
