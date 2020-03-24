# CEACOV Microsimulation Model

The **C**linical and **E**conomic **A**nalysis of **COV**ID-19 Interventions (CEACOV) model is a dynamic Monte Carlo microsimulation model incorporating aspects of COVID-19 disease natural history, SARS-CoV-2 transmission, and vaccination to project epidemic growth, clinical outcomes, health care resource utilization, and cost effectiveness.

## Project Overview

### Requirements

Requires python 3.6+ and numpy

### sim.py

Top level logic to first locate the input file(s), then initialize and step through the simulation(s)

To run from the command line call `python sim.py "your/directory"` on a directory containing input JSON files. Running on a directory that does not contain JSON files will automatically generate a blank input JSON template.

### inputs.py

`Inputs` container class for user-specified parameters along with utilities to generate and read parameters from JSON files

### patient.py

Contains the main simulation class - **you probably want to start here**

This file contains the simulation logic in the two methods of the `SimState` class: `initialize_cohort()` and `step()`. There are also several helper functions found at the top of the file containing various simulation subroutines.

### outputs.py

`Outputs` class to accumulate outcomes and write them to TSV files

### enums.py

Constants, meta-parameters, enumerations, and utility functions

## License
[MIT](https://choosealicense.com/licenses/mit/)


## Disclaimer

The CEACOV model source code is provided by the Medical Practice Evaluation Center (MPEC) at Massachusetts General Hospital in Boston, MA, USA for interested readers and reviewers. This repository does not include data sources, populated project-specific input files, or comprehensive instructions on how to derive input parameters. We are not responsible or liable for third party use of this model and cannot endorse any results obtained through CEACOV by users not associated with the MPEC.

## Acknowledgements

This project was funded in part by the National Institutes of Health [R37 AI058736-16S1, K24 AR057827, and T32 AI007433] and by a fellowship from the Royal Society and Wellcome Trust [210479/Z/18/Z].

The funding sources had no role in the design, implementation, or publication of this software.
