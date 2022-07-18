# -*- coding: utf-8 -*-
'''
@author: jagath, lucasd
this script is used to parse the iv file into seperate files
and to average the repeated measurements from each file
it can also call the analysis script
and it is callable from the icq script
'''

import init
import numpy as np
import os as os
import csv
import analysis

##############################################################################
# FUNCTIONS GO HERE

# COMPARE THE COLUMN NAMES FROM THE DATA FILE AGAINST THE EXPECTED NAMES
# IF THE EXPECTED NAMES ARE NOT FOUND, THE DATA FILE WAS FORMATTED WRONG
# IF THE EXPECTED NAMES ARE NOT FOUND, THIS FUNCTION WILL THROW AN ERROR
def check_names(dnames, expected):
    for name in expected:
        if not name in dnames:
            print('Expected name "%s" not found'%name)
            
# LOAD THE DATA USING THE NUMPY DATA LOAD FUNCTION
# DATA LOAD TAKES THE PATH TO THE CSV FILE AS THE ARGUEMENT
# DATA LOAD RETURNS THE DATA AS A NUMPY ARRAY, CALLABLE BY THE NAME ARGUMENT    
def load_data(csv):
    return np.genfromtxt(csv, delimiter = ',', comments='#', names = True)

# CLEAR THE FILES PRESENT IN A TARGET DIRECTORY
def clear_dir(target_dir):
    for file in os.listdir(target_dir):
        os.remove(os.path.join(target_dir,file))

# LOAD A CSV FILE CONTAINING IV MEASUREMENTS FROM MULTIPLE CHANNELS (CAN BE ONE)
# SEPERATE THE DATA BY CHANNEL AND WRITE THE SPLICED DATA INTO NEW FILES
def seperate(newcsv,TEMP,expected_names):
    data = load_data(newcsv)
    # SETUP AN ARRAY OF SEQUENTIALLY MEASURED CHANNELS
    # THE VALUES ARE UNIQUE IN A SEQUENTIAL SENSE
    # E.G. [1,1,1,2,3,2,2] -> [1,2,3,2]
    channels = []
    for i in range(len(data)):
        if not channels:
            channels.append(data['Repeat'][i])
        if data['Repeat'][i] != channels[-1]:
            channels.append(data['Repeat'][i])
    for ch in channels: # Iterating through each repeat(categorized by their number)
        rows = [] # Empty list meant to contain the rows of the parsed files
        for i in range(len(data)):
            if data['Repeat'][i] == ch:
                rows.append(data[i]) # Filling up the list
        filename = os.path.basename(newcsv)[:-4] + "_" + str(int(ch)) + ".csv" # Assembling the file name for the parsed files per repeat
        #print(filename)
        parsecsv = os.path.join(TEMP,filename)
        print(expected_names)
        with open(parsecsv, "w", newline = '') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(expected_names)
            csvwriter.writerows(rows)  # Creating the parsed file using the extracted header and parsed rows
        csvfile.close()
        
# LOAD A CSV FILE CONTAINING IV MEASUREMENTS FROM ONE CHANNEL
# AVERAGE THE REPEATED MEASUREMENTS AND CALCULATE THE STATISTICAL ERROR
# WRITE A NEW CSV FILE WITH THE CALCULATED VALUES
def average(sepfile,TEMP,expected_names):
    data = load_data(sepfile)
    #print(sepfile)
    #print(data.dtype.names)
    # SETUP AN ARRAY OF SEQUENTIALLY MEASURED SOURCE VALUES
    # THE VALUES ARE UNIQUE IN A SEQUENTIAL SENSE
    # E.G. [1,1,1,2,3,2,2] -> [1,2,3,2]
    source = []
    for i in range(len(data)):
        if not source:
            source.append(data['CH1_Source'][i])
        if data['CH1_Source'][i] != source[-1]:
            source.append(data['CH1_Source'][i])
    #print(source)
    # SETUP THE HEADERS AND FILENAMES FOR NEW AVERAGED DATA FILE
    volt, volt_err, current, current_err, vset, ch, n = [], [], [], [], [], [], []
    filename = os.path.basename(sepfile)[:-4] + "_proc.csv" # Assembling the file name for the averaged files
    #print(filename)
    averagecsv = os.path.join(TEMP,filename)
    # MAIN LOOP TO FILL NEW DATA FILE
    with open(averagecsv, "w", newline = "") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['volt', 'volt_err', 'current', 'current_err', 'vsource', 'ch', "n"])
        # THIS LOOP IS USED TO ASSEMBLE THE REPEATED MEASUREMENTS, I.E., THE SAME SOURCE V
        for sc in source:
            #print(sc)
            rows = [] # The repeated measurements are stored in this list
            for i in range(len(data)):
                if data['CH1_Source'][i] == sc: # Pt increments when a new V source is used
                    rows.append([data['CH1_Voltage'][i], data['CH1_Current'][i], data['CH1_Time'][i], data['CH1_Source'][i]])
            rows = np.array(rows)
            #print(len(rows))
            # ADD THE DATA TO A NEW ARRAY FOR AVERAGING
            # THE INPUT NUMBERS COME FROM THE ORDERING OF THE ROWS ARRAY
            volt.append(np.mean(rows[:,0])) # Mean Voltage
            volt_err.append(np.std(rows[:,0]) / np.sqrt(len(rows))) # The standard error
            current.append(np.mean(rows[:,1])) # Mean Current
            current_err.append(np.std(rows[:,1]) / np.sqrt(len(rows))) # The standard error
            vset.append(rows[0,2]) # Setpoint voltage
            ch.append(data['Repeat'][0]) # The channel number
            n.append(len(rows)) # The number of data points averaged   
        # WRITE THE DATA TO A CSV FILE
        for i in range(len(volt)):
            csvwriter.writerow([volt[i], volt_err[i], current[i], current_err[i], vset[i], ch[i], n[i]])
    csvfile.close() # Close the file
    #os.remove(os.path.join(sepfile)) # Remove the old file

# THE PARSE FUNCTION EXPECTS CSV FILES
# THE PARSE FUNCTION CALLS THREE FUNCTIONS
# THE FIRST FUNCTION TAKES A CSV FILE CONTAINING MULTIPLE IV MEASUREMENTS AND PARSES IT BASED ON THE CHANNEL SCANNED
# THE SECOND FUNCTION TAKES THE SPLICED CSV FILES AND AVERAGES THE REPEATED MEASUREMENTS AND CALCULATES THE STATISTICAL ERROR
# THE THIRD FUNCTION PERFORMS AN ANALYSIS OF THE IV MEASUREMENTS AND WRITES THE RESULTS TO THE PLOTS AND REPORTS DIRECTORIES
def parse(newcsv,TEMP):
    # THESE ARE THE NAMES EXPECTED IN THE DATA FILE
    expected_names = init.header()
    clear_dir(TEMP) # Clear the temporary directory before starting to use it
    seperate(newcsv,TEMP,expected_names) # Parses the IV measurements by channel, write the results to temp
    temp_files = os.listdir(TEMP) # List the new csv files from temp
    for file in temp_files: # Iterate 
        if file[-4:] == ".csv": # Check if the file is a csv file
            average(os.path.join(TEMP,file),TEMP, expected_names) # Averages the repeated measurements from each IV measurement and writes as an output, deletes the input file
    #temp_files = os.listdir(TEMP) # List the new files output from the average function
    #for file in temp_files: # Iterate
        #if file[-8:] == "proc.csv": # Check if the file is the output from the average function (proc) and if its a csv
            #analysis.analyze(os.path.join(TEMP,file)) # Analyzes the data and writes plots/reports

##############################################################################
# MAIN LOOP GOES HERE
# THIS IS MOSTLY USED FOR DEVELOPMENT, BUT CAN ALSO BE USED TO PARSE FUNCTIONS OUTSIDE OF THE ICQ SCRIPT
if __name__ == "__main__":
    #print(__name__)
    SCRIPTS,HOME,DATA,ARCHIVE,TEMP,DEV,PLOTS,REPORTS = init.envr() # Setup the local environment
    file_loc = os.path.join(ARCHIVE,'test.csv') # Where is the test file located
    data = load_data(file_loc) # Load data for debugging
    #print(data.dtype.names) # Load data names for debugging
    #expected_names = init.header()
    #check_names(data.dtype.names, expected_names) # Compare the data names to the expected names
    parse(file_loc,TEMP)

##############################################################################
# FUNCTION LOOP GOES HERE
# THIS IS USED IN THE ICQ FUNCTION
if __name__ == os.path.basename(__file__):
    #print(__name__)
    SCRIPTS,HOME,DATA,ARCHIVE,TEMP,DEV,PLOTS,REPORTS = init.envr() # Setup the local environment