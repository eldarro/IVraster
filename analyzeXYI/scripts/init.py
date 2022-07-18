# -*- coding: utf-8 -*-
'''
@author: lucasd
this script is used to setup the default environment for the data management program
it can also be imported into other scripts to load the management environment
'''

import os

# Function to setup the local environment, used by parse, icq
def envr():
    # THE DEFUALT ENVIRONMENT
    SCRIPTS = os.getcwd() # It is anticipated that this script will run from the script directory
    HOME = os.path.dirname(SCRIPTS)
    DATA = os.path.join(HOME,'data')
    
    # SETUP DIRECTORIES NEEDED FOR SCRIPTS
    ARCHIVE = os.path.join(DATA,'archive')
    TEMP = os.path.join(DATA,'temp')
    DEV = os.path.join(DATA,'dev')
    PROC = os.path.join(DATA,'proc')
    PLOTS = os.path.join(HOME,'plots')
    REPORTS = os.path.join(HOME,'reports')
    
    # ADD THE DIRECTORY PATHS TO A LIST
    directories = [SCRIPTS,HOME,DATA,ARCHIVE,TEMP,DEV,PROC,PLOTS,REPORTS]
    
    # CHECK IF THE PATH TO THE DIRECTORIES EXISTS
    # IF THE PATH DOES NOT EXIST, MAKE IT
    for DIR in directories:
        if not os.path.exists(DIR):
            os.mkdir(DIR)
    return directories

# Function to pass the expected header names for the IV data, used by parse
def header():
    return ['Repeat','VAR2','Point','CH1_Voltage','CH1_Current','CH1_Time','CH1_Source']

if __name__ == "__main__":
    envr() # Setup the local environment