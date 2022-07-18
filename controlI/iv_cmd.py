#!/usr/bin/python3
import argparse
import time
import subprocess
import datetime

'''
Examples for SiPM-FE-Rev1-A 

FORWARD IV:
iv_cmd_Rev1-A.py --start_v -5.2 --stop_v -0.1 --step_v 0.3 --only_chans 1,2,3,4,5,6,7,8 --adc_rate 4 --num_readings 6 --file test.csv
iv_cmd_Rev1-A.py --start_v -5.2 --stop_v -0.1 --step_v 0.3 --only_chans 9,10,11,12,13,14,15,16 --adc_rate 4 --num_readings 6 --file test.csv

REVERS IV:
iv_cmd_Rev1-A.py --start_v 47 --stop_v 57 --step_v 0.5 --only_chans 1,2,3,4,5,6,7,8 --adc_rate 4 --num_readings 6 --file test.csv
iv_cmd_Rev1-A.py --start_v 47 --stop_v 57 --step_v 0.5 --only_chans 9,10,11,12,13,14,15,16 --adc_rate 4 --num_readings 6 --file test.csv
=======
Examples:
./lolx_iv_cmd.py --start_v -6.1 --stop_v -1 --step_v 0.3 --adc_rate 4 --num_readings 6 --file test_fw.csv
./lolx_iv_cmd.py --start_v 47 --stop_v 57 --step_v 0.5 --adc_rate 4 --num_readings 6 --file test_rv.csv
'''

# input channel -> DAC add, DAC ch.
input_chan_to_dac_addr = {
    1: (5, 0),
    2: (5, 1),
    3: (5, 2),
    4: (5, 3),
    5: (5, 4),
    6: (5, 5),
    7: (5, 6),
    8: (5, 7),
    9: (2, 0),
    10: (2, 1),
    11: (2, 2),
    12: (2, 3),
    13: (2, 4),
    14: (2, 5),
    15: (2, 6),
    16: (2, 7)
}

# There are 4 Quad-ADCs for channels 1..16 -> ADC add 6,4,3,1
input_chan_to_adc_addr = {
    1: "6 0 1",
    2: "6 4 5",
    3: "6 6 7",
    4: "6 2 3",
    5: "4 0 1",
    6: "4 4 5",
    7: "4 6 7",
    8: "4 2 3",
    9: "3 0 1",
    10: "3 4 5",
    11: "3 6 7",
    12: "3 2 3",
    13: "1 0 1",
    14: "1 4 5",
    15: "1 6 7",
    16: "1 2 3"  # input channel 16
}

def run_cmd(cmd, num_tries=3):
    """
    Lowest-level function for running a subprocess command (generally dactest
    or adctest).

    Args:
        * cmd (str) - The command to run
        * num_tries (int) - Maximum number of attempts to try the command, if
            we get a non-zero exit code from it the first time
    """
    print(cmd)
    try:
        return subprocess.run(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        if num_tries == 1:
            print("FAILED to run command `%s`" % cmd)
            print("Return code was `%s` (we expected 0)" % e.returncode)
            print("Output was:\n```\n%s```" % e.output.decode("ascii"))
            exit(1)

        return run_cmd(cmd, num_tries-1)

def run_hv(cmd, num_tries=3):
    """
    Run an adctest command and extract the readback voltage from stdout.

    Expected output from cmd is like:

    ```
    Gain = 1
    Rate = 5 samples/second
    Measuring between channels 3 and 2
    ADC value: 4596384
    Converted: 68.4915 V
    ```

    Returns:
        float
    """
    sub = run_cmd(cmd, num_tries)

    retval = None
    """    for line in sub.stdout.decode("ascii").split("\n"):
        if line.startswith("Converted:"):
            retval = line.split(" ")[1]
    """
    for line in sub.stdout.decode("ascii").split("\n"):
        if cmd.find("-l") == -1 and line.startswith("Converted:"):
            retval = line.split(" ")[1]
        if cmd.find("-l") != -1 and line.startswith("Converted:"):
            retval = line.split("mean:")[1].split(",")[0]

    if retval is None:
        raise RuntimeError("Unexpected output running %s:\n\n%s\n\n%s" % (cmd, sub.stdout.decode("ascii"), sub.stderr.decode("ascii")))

    if retval.find("nV") != -1:
        retval = float(retval.replace("nV", "")) * 1e-9
    elif retval.find("uV") != -1:
        retval = float(retval.replace("uV", "")) * 1e-6
    elif retval.find("mV") != -1:
        retval = float(retval.replace("mV", "")) * 1e-3
    elif retval.find(" V") != -1:
        retval = float(retval.replace(" V", ""))
    else:
        try:
            retval = float(retval)
        except Exception:
            print(sub.stdout.decode("ascii"))
            raise ValueError("Unexpected value when reading voltage: %s" % retval)

    return retval

def run_current(cmd, num_tries=3):
    """
    Run an adctest command and extract the readback current from stdout.

    Expected output is like this if -l is NOT an argument in cmd:

    ```
    Gain = 1
    Rate = 5 samples/second
    Measuring between channels 0 and 1
    ADC value: 59941
    Converted: 0.728061uA
    ```

    Expected output is like this if -l IS an argument in cmd:

    ```
    Gain = 1
    Rate = 5 samples/second
    Measuring between channels 0 and 1
    2096
    1693
    1358
    1162
    1049
    Results:
    TimeDiff (us):    mean: 200035, stdev: 8.04285
    ADCValue:         mean: 1471.6, stdev: 381.114
    Current:         mean: 17.8745nA, stdev: 4.62913nA
    ```

    Returns:
        float
    """
    sub = run_cmd(cmd, num_tries)

    retval = None

    for line in sub.stdout.decode("ascii").split("\n"):
        if cmd.find("-l") == -1 and line.startswith("Converted:"):
            retval = line.split(" ")[1]
        if cmd.find("-l") != -1 and line.startswith("Converted:"):
            retval = line.split("mean:")[1].split(",")[0]

    if retval is None:
        raise RuntimeError("Unexpected output running %s:\n\n%s\n\n%s" % (cmd, sub.stdout.decode("ascii"), sub.stderr.decode("ascii")))

    if retval.find("fA") != -1:
        retval = float(retval.replace("fA","")) * 1e-15
    elif retval.find("pA") != -1:
        retval = float(retval.replace("pA", "")) * 1e-12
    elif retval.find("nA") != -1:
        retval = float(retval.replace("nA", "")) * 1e-9
    elif retval.find("uA") != -1:
        retval = float(retval.replace("uA", "")) * 1e-6
    elif retval.find("mA") != -1:
        retval = float(retval.replace("mA", "")) * 1e-3
    elif retval.find(" A") != -1:
        retval = float(retval.replace(" A", ""))
    else:
        print(sub.stdout.decode("ascii"))
        raise ValueError("Unexpected value when reading current: %s" % retval)

    return retval

def run_dacout(cmd, num_tries=3):
    """
    Run a dactest command and extract the setting that will be set.

    Expected output from cmd is like:

    ```
    setting: 32000
    ```

    Returns:
        int
    """
    sub = run_cmd(cmd, num_tries)

    retval = None

    for line in sub.stdout.decode("ascii").split("\n"):
        if line.startswith("Closest"):
            retval = int(line.split(" ")[-1])

    if retval is None:
        raise RuntimeError("Unexpected output running %s:\n\n%s\n\n%s" % (cmd, sub.stdout.decode("ascii"), sub.stderr.decode("ascii")))

    return retval

def run_dacrdb(cmd, num_tries=3):
    """
    Run a dactest command and extract the readback numbers.

    Expected output from cmd is like:

    ```
    V[0] = 48000
    V[1] = 32000
    V[2] = 32000
    V[3] = 32000
    V[4] = 32000
    V[5] = 32000
    V[6] = 32000
    V[7] = 32000
    ```

    Returns:
        array of int
    """
    sub = run_cmd(cmd, num_tries)

    retval = []

    for line in sub.stdout.decode("ascii").split("\n"):
        if line.startswith("V["):
            retval.append(int(line.split(" ")[-1]))

    if retval is None:
        raise RuntimeError("Unexpected output running %s:\n\n%s\n\n%s" % (cmd, sub.stdout.decode("ascii"), sub.stderr.decode("ascii")))

    return retval

def set_all_dac(voltage):
    """
    Set all the DACouts to the given voltage.

    Args:
        * voltage (float)
    """
    # -1 sets all DACouts. We need to do it on 2 and 5.
    print("Setting all DAC to %s" % voltage)
    setting = run_dacout("dactest -v 2 -1 %s" % voltage)
    time.sleep(0.05)
    readback = run_dacrdb("dactest -r 2")

    for i,r in enumerate(readback):
        if r != setting:
            raise RuntimeError("Incorrect readback of cselect 2 on DACout %s: %s vs expected %s" % (i, r, setting))

    setting = run_dacout("dactest -v 5 -1 %s" % voltage)
    time.sleep(0.05)
    readback = run_dacrdb("dactest -r 5")

    for i,r in enumerate(readback):
        if r != setting:
            raise RuntimeError("Incorrect readback of cselect 5 on DACout %s: %s vs expected %s" % (i, r, setting))

def set_hvout(voltage, sleepV):
    """
    Set the HVout voltage, and return the readback voltage.

    Args:
        * voltage (float)

    Returns:
        float
    """
    run_cmd("dactest -v hv 1 %s" % voltage)
    time.sleep(sleepV)

    timeout = datetime.datetime.now() + datetime.timedelta(seconds=10)
    tolerance_v_abs = 2.5

    while True:
        vrdb = read_hvout()

        if abs(vrdb - voltage) < tolerance_v_abs :
	        break

        if datetime.datetime.now() > timeout:
            raise RuntimeError("HV didn't get close to %s, latest reading is %s" % (voltage, vrdb))
        time.sleep(0.05)

    return vrdb

def get_hvout_ref(v_step_err):
    """
    Get HVout read back for reference before stepping HVout 

    Args:
    * v_step (float)

    Returns:
        float
    """

    timeout = datetime.datetime.now() + datetime.timedelta(seconds=20)
    v_old = -999

    while True:
        rdb = read_hvout()

        if abs(rdb - v_old) < v_step_err :
            rdb_ref = (v_old+rdb)/2 
            break

        if datetime.datetime.now() > timeout:
            raise RuntimeError("HVout unstable reading, last read back is %s, error is %s" % (rdb, rdb - v_old))
        time.sleep(0.5)
        if v_old != -999 :
            print("HVout_ref read back is %s, error is %s" % (rdb, rdb - v_old))
        v_old = rdb
    return rdb_ref

def check_hvout_step(v_old,v_step,v_step_err):
    """
    Get the HVout read back and check it with old value, return the readback voltage.

    Args:
        * v_old (float)
        * v_step (float)

    Returns:
        float
    """

    timeout = datetime.datetime.now() + datetime.timedelta(seconds=15)
    while True:
        rdb = read_hvout()

        if abs(rdb - v_old - v_step) < v_step_err :
            break

        if datetime.datetime.now() > timeout:
            raise RuntimeError("HV didn't step close to next value %s, previous value %s, latest reading is %s, error is %s" % (v_old + v_step, v_old, rdb, rdb - v_old - v_step ))
        time.sleep(0.5)
        print("HVout read back is %s, error is %s" % (rdb, rdb - v_old - v_step))
    return rdb

def read_hvout():
    """
    Read the HVout voltage.

    Returns:
        float
    """
    return run_hv("adctest -l 1000 -r 8 hv 1 0")

def set_dac(channel, voltage):
    """
    Set the DACout of a single channel.

    Args:
        * channel (int) - 1-16
        * voltage (float)
    """
    dac_addr, dac_chan = input_chan_to_dac_addr[channel]
    setting = run_dacout("dactest -v %s %s %s" % (dac_addr, dac_chan, voltage))
    readback = run_dacrdb("dactest -r %s" % dac_addr)

    if readback[dac_chan] != setting:
        raise RuntimeError("Unable to set DAC %s[%s] to %s; readback is %s" % (dac_addr, dac_chan, setting, readback[dac_chan]))

def get_current(channel, num_readings, adc_rate):
    """
    Get the current of a channel (in Amps).

    Args:
        * channel (int) - 1-16
        * num_readings (float) - Number of current readings to average before returning

    Returns:
        float
    """
    print("running on Ch", channel)
    return run_current("adctest -l %s -r %s %s" % (num_readings, adc_rate, input_chan_to_adc_addr[channel]))

def do_print(f, line):
    """
    Print to screen, and optionally to file.

    Args:
        * f (file or None) - File to print to
        * line (str) - String to print
    """
    print(line)
    if f:
        f.write(line + "\n")

def main():
    """
    Main function for running the scan loop.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_v", type=float, required=True, help="Start point of scan (V)")
    parser.add_argument("--stop_v", type=float, required=True, help="End point of scan (V)")
    parser.add_argument("--step_v", type=float, required=True, help="Increment between scan points (V)")
    parser.add_argument("--only_chans", type=str, help="Only read current of certain channels. Provide a comma-separated list (e.g. 1,3,5). Default is to measure all 16 channels.")
    parser.add_argument("--num_readings", type=int, default=5, help="Number of current measurements to average at each scan point. Default is 5")
    parser.add_argument("--adc_rate", type=int, default=0, help="Sample rate ADC for Isipm. Default is 0 = 5SPS")
    parser.add_argument("--filename", type=str, help="Output CSV file name. Default is to just print to screen.")
    args = parser.parse_args()

    # for customizing wait times (all in seconds) and no averaging, Austin
    # all major (greater than 0.05s) sleep commands are in this main code, other than one call in set_hvout
    sleepHighVoltage = 5 #same as old code
    sleepAllDAC = 1 #new, DACs only wait 0.05 s
    sleepNumReadings = 0.05 #new, time to wait between each reading for same Voltage

    # changed numReadings = 1 in getCurrent(), and repeat several times
    # removed std. dev from measurement

    num_steps = int((args.stop_v - args.start_v) / args.step_v) + 1
    hv_list = [args.start_v + args.step_v * i for i in range(num_steps)]

    if args.only_chans:
        chan_list = [int(x) for x in args.only_chans.split(",")]
    else:
        chan_list = list(range(1,17))

    # Set Global Voltage of the Q pump (Set and enable)
    # adc Qpump between 3,2
    run_cmd("dactest -v hv 1 0")              # pre-setting HVout close to 0V
    run_cmd("dactest -v hv 0 70")             # pre-setting QPout close to 70V
    run_cmd("setPin ENQP hi")                 # enabling charge pump
    run_cmd("dactest hv 3 65000")            # enabling HVout
    time.sleep(1)
    qp = run_hv("adctest -l 20 -r 3 hv 3 2")        # reading back Charge Pump QP+
    print("Charge Pump QP+ is %s" % qp)
    qp = run_hv("adctest -l 20 -r 3 hv 4 5")        # reading back Charge Pump QP-
    print("Charge Pump QP- is %s" % qp)

    # Set all DACs Address 2 and 5 to 2V
    dac = 0.5 if hv_list[0] < 0 else 2
    set_all_dac(dac)
    time.sleep(sleepAllDAC)

    # Preset HVout to start value - v_step
    hv_rdb_old = set_hvout(hv_list[0] - args.step_v, sleepHighVoltage)
    print("Initial HVout read back is %s" % hv_rdb_old)
    tolerance_v_step = 0.004 + 0.4 * args.step_v / 100  #4mV + 0.4% of step
    print("V-step_toleranse will be %s" % tolerance_v_step)
    print("ADC rate is %s     Number of samples is %s" % (args.adc_rate, args.num_readings))
    R_total = 59984.9   #used for calculation of Vdrop across total output resistance of HV source
    print("R_total is %s ohms" % R_total)

    if args.filename:
        f = open(args.filename, "w")
        print("Output is being written to %s" % args.filename)
    else:
        f = None
        print("Output is being written to screen only")

    #do_print(f, "Set voltage (V),Readback voltage (V),%s," % ",".join("Current chan %i (A)" % i for i in chan_list))
    do_print(f, "Start V, Stop V, Step V, Num Readings, ADC Rate, Sleep HV, Sleep DAC, Sleep Readings")
    do_print(f, "%s, %s, %s, %s, %s, %s, %s, %s" % (args.start_v, args.stop_v, args.step_v, args.num_readings, args.adc_rate, sleepHighVoltage, sleepAllDAC, sleepNumReadings))
    do_print(f, "HV_set (V),HV_rdb (V), HV_err (V),%s,%s" % (",".join("VsCH_%i (V)" % i for i in chan_list),",".join("IsCH_%i (A)" % i for i in chan_list)))

    # Loop through the voltages and channels
    for hv_v in hv_list:
        set_hvout(hv_v, sleepHighVoltage)
        hv_v_rdb = check_hvout_step(hv_rdb_old, args.step_v, tolerance_v_step )
        hv_err = hv_v_rdb - hv_rdb_old - args.step_v
 	hv_rdb_old = hv_v_rdb
        
	# Set DAC to 0.5V when setpoint is negative; 2V when positive
        if hv_v < 0 and dac != 0.5:
            dac = 0.5
            print("in set DAC loop 1")
            set_all_dac(dac)
            time.sleep(sleepAllDAC)
        if hv_v > 0 and dac != 2:
            dac = 2
            print("in set DAC loop 2")
            set_all_dac(dac)
            time.sleep(sleepAllDAC)
        
        currents = []
        Vsipm = []

        #read many times and write all (no averaging)
        for i in range(args.num_readings):
            
            for chan in chan_list:
                curr = get_current(channel = chan, num_readings = 1, adc_rate = args.adc_rate)
                currents.append(curr)
                Vsipm.append( hv_v_rdb + dac - curr * R_total)

            #do_print(f, "%s,%s,%s" % (hv_v,hv_v_rdb,",".join("%.5e" % x for x in currents)))
            #do_print(f, "%s,%s,%s,%s" % (hv_v,hv_v_rdb,"," .join("%.5e" % x for x in currents),",".join("%.5e" % x for x in Vsipm)))
            do_print(f, "%s,%s,%.5f,%s,%s" % (hv_v,hv_v_rdb,hv_err,",".join("%.5e" % x for x in Vsipm) ,",".join("%.5e" % x for x in currents)))
            
            #sleep for each reading of all channels
            time.sleep(sleepNumReadings)
            currents = []
            Vsipm = []
            hv_v_rdb = read_hvout()

    # turn off HV
    # Setting it to 0V sets HV out to hiZ.
    run_cmd("dactest -v hv 3 0")

if __name__ == "__main__":
    main()
