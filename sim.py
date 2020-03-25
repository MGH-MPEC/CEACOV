# -*- coding: utf-8 -*-
"""
COVID-19 Microsimulation Model

@author: Chris Panella (cpanella@mgh.harvard.edu)
"""

from enums import *
from inputs import *
from patient import *
from outputs import *
import numpy as np
import os, sys
from glob import glob



if __name__ == "__main__" or __name__ == "builtins":
    
    input_files = []
    # run in local directory (with python file)
    if len(sys.argv) == 1:
        folder = os.path.dirname(os.path.realpath(__file__))
        input_files = glob("*.json")
    # search specified directory for json
    elif os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        input_files = glob(os.path.join(folder, "*.json"))
    # raise error for invalid arguments
    else:
        raise UserWarning("invalid call, unexpected arguments detected")
    # generate template file when run on empty directory
    if not input_files:
        template = os.path.join(sys.argv[1], "covid_input_template.json")
        print("No input files detected, generating input template")
        create_input_file("template.json")
    else: 
        # create results folder
        results_directory = os.path.join(folder, "results")
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        # run the sim!
        for input_file in input_files:            
            sim_state = SimState(read_inputs(input_file))
            out_file = os.path.splitext(os.path.split(input_file)[1])[0] + ".csv"
            print("running " + input_file)
            sim_state.run()
            sim_state.outputs.write_outputs(os.path.join(results_directory, out_file))

