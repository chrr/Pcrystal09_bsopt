#!/home/chr/sys/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2013, Christoph Reimann 

crystalExe = '/home/chr/sys/bin/Pcrystal09'
crystalVersion = 'CRYSTAL09 v2.0.1'
mpi = '/home/chr/sys/openmpi-1.4.1/bin/mpirun -np 8'
copyCmd = 'cp'
grepCmd = 'grep'
# basis set restrictions
exponentLowerLimit = 0.2

import minuit
import subprocess as sp
import re
import sys
import os
import shutil
from optparse import OptionParser

# collect parameters in jobConfig dict
jobConfig = {
    'mpi'                : mpi,
    'executable'         : crystalExe, 
    'program'            : crystalVersion,
    'template'           : None,
    'required_files'     : [],
    'exponentLowerLimit' : exponentLowerLimit,
    'grep'               : grepCmd,
    'defaultInputFile'   : 'INPUT',
    'defaultOutputFile'  : 'OUTPUT',
    }

def cleanup(scratchdir, status, job_config):
    # remove scratch directories
    print('Removing temporary directory %s...' % scratchdir)
    print('  rm -rf %s' % scratchdir)
    sp.call(['rm', '-r', '-f', scratchdir])

def igrep(text, filename):
    pipe = sp.Popen([grepCmd, "-i", text, filename], stdout=sp.PIPE, stderr=sp.PIPE)
    pipe.wait()
    return pipe.stdout.readlines()

class BSOptimizer(object):
    "Base class for basis set optimizer"
    parameters = {}

    def __init__(self, job_config):
        self.template_file = job_config['template']
        self.job_config = job_config
        self.N_runs = 0
        self.initial_value = 0.0
        self.lowest_value = 0.0
        self.last_value = 0.0
        self.cmd = self.job_config['mpi'].split(' ')+self.job_config['executable'].split(' ')

    # succession is important within the parameters, so sort them first according to their name
    def get_keys(self):
        return sorted(self.parameters.keys())

    def get_parameters(self):
        keys = self.get_keys()
        return [self.parameters[k] for k in keys]

    def run_calculation(self, *new_parameters):
        # default: return 0.0 - all suitable energies will be negative
        retval = 0.0
        self.N_runs += 1
        print("Optimization run %d:" % self.N_runs)
        self.write_input(new_parameters, open(self.job_config['defaultInputFile'], "w"))
        print("Parameters (initial +++ best +++ current):")
        for index, k in enumerate(self.get_keys()):
            print("    %-5s   %12.7f    %12.7f    %12.7f" % (k, self.parameters[k], self.best_params[k], new_parameters[index]))
        # call external program
        output = open(self.job_config['defaultOutputFile'], 'w')
        try:
            returncode = sp.check_call(args=self.cmd, stdout=output, stderr=output)
        except sp.CalledProcessError, e:
            returncode = e.returncode
        output.close()
        return returncode
# fin: class BSOptimizer

class crystalOptimizer(BSOptimizer):
    "Basis set optimizer for use with CRYSTAL"

    def __init__(self, job_config):
        BSOptimizer.__init__(self, job_config)
        # regular expressions
        # template parameter are specified according to
        #  exponents: ${as1= 1.0}
        #  coefficients: ${ds1= 1.0
        self.parameterKeyValueRegexp = re.compile("\${([^=}]+)=([^}]+)}")
        self.energyRegexp = re.compile("-?\d*\.\d*E(?:\+|-)\d*")
        self.cyclesRegexp = re.compile("CYCLES\s*(\d+)")
        self.parameterKeyRegexp = re.compile("\${([^=}]+)}")
        self.parameterRegexp = re.compile("\${([^}]+)}")
        # preparse template file
        self.template = self._load_initial_parameters(self.template_file)
        self.best_params = dict(self.parameters)
        
    # load initial parameters from the input template
    def _load_initial_parameters(self, input_template):
        input = open(input_template)
        template = input.read()
        input.close()
        # determine initial parameters
        regexp = self.parameterKeyValueRegexp
        while(1):
            m = regexp.search(template)
            if m is None:
                break
            else:
                key = m.groups()[0]
                if self.parameters.has_key(key):
                    print("Error: Parameter '%s' is defined more than once!" % key)
                    sys.exit(1)
                template = template.replace("%s=%s" % m.groups(), key)
                self.parameters[key] = float(m.groups()[1])
        # fin: while(1)
        # prepare internal variables
        keys = self.get_keys()
        if len(keys) == 0:
            print("Error: No parameters to optimize defined in '%s'!" % str(input_template))
            sys.exit(1)
        # hack: the run_stub function returns a function with the proper number of arguments
        arguments = reduce(lambda x,y: "%s, %s" % (x,y), keys)
        exec("""def run_stub(obj): 
    def func(%s):
        return obj.run_calculation(%s)
    return func
""" % (arguments, arguments))
        self.run = run_stub(self)
        return template

    def write_template_file(self):
        name = os.path.splitext(self.template_file)[0] + ".new_template"
        print("\nWriting best performing parameter set to new template file %s..." % name)
        template_file = open(name, "w")
        for line in self.template.split('\n'):
            while(1):
                # FIXME - use self.parameterRegexp here instead?
                m = self.parameterKeyRegexp.search(line)
                if m is None:
                    break
                else:
                    key = m.groups()[0]
                    line = line.replace('${%s}' % key, "${%s=%14.8f}" % (key, float(self.best_params[key])))
            template_file.write("%s\n" % line)
        template_file.close()
        print("Writing new input file INPUT.best...")
        parameters = [self.best_params[k] for k in self.get_keys()]
        self.write_input(parameters, open("%s.best" % self.job_config['defaultInputFile'], "w"))
        print("Copying %s to %s.best..." % (self.job_config['defaultOutputFile'], self.job_config['defaultInputFile']))
        sp.call([copyCmd, '%s' % self.job_config['defaultOutputFile'], '%s.best' % self.job_config['defaultOutputFile']])
                
    def write_input(self, parameters, input):
        template = self.template
        keys = sorted(self.parameters.keys())
        # look for parameter declarations
        while(1):
            m = self.parameterRegexp.search(template)
            if m is None:
                break
            else:
                key = m.groups()[0]
                template = template.replace('${%s}' % key, "%14.8f" % float(parameters[keys.index(key)]))
        input.write(template)
        input.close()

    def run_calculation(self, *new_parameters):
        if self.N_runs == 0:
            # provide initial guess
            if not os.path.exists('fort.20'):
                shutil.copyfile(self.job_config['fort.20'], 'fort.20')
        returncode = BSOptimizer.run_calculation(self, *new_parameters)
        if returncode != 0:
            print("  --> %s returncode: %s" % (self.job_config['program'], str(returncode)))
        # get the final total energy
        lines = igrep("SCF ENDED", self.job_config['defaultOutputFile'])
        if len(lines) != 1:
            print("  ==> Calculation failed!!")
        else:
            m = self.energyRegexp.search(lines[0])
            if m is not None:
                total_energy = float(m.group(0))
                if self.N_runs == 1:
                    self.initial_value = total_energy
                # determine no. of SCF cycles for statistics
                m = self.cyclesRegexp.search(lines[0])
                cycles = "XX" if m is None else m.group(1)
                print("  --> Total energy: %12.7f (%s cycles)" % (total_energy, cycles))
                retval = total_energy
            else:        
                print("  ==> Calculation failed: %s" % lines[0])
                print("Mysterious error when looking for total energy!?")
                sys.exit(1)
        # check that the actual result is not too different from the last one
        # this is necessary due to a "feature" in CRYSTAL that sometimes leads to unrealistically large steps
        if abs(retval - self.last_value) > abs(self.last_value/10.0):
            # ignore special case: start condition
            if self.last_value != 0.0:
                retval = 0.0
                print("  ==> WARNING: Result is too far off and will be ignored")
        else:
            self.last_value = retval
        # save only the best result
        if retval < self.lowest_value:
            self.lowest_value = retval
            shutil.copyfile("fort.9", "fort.20")
            self.best_params = {}
            for index, k in enumerate(self.get_keys()):
                self.best_params[k] = new_parameters[index]
            self.write_template_file()
        print("")
        return retval
# fin: class crystalOptimizer

class optimizationController(object):
    name = 'Basis set optimization'
    version = 'July 2013'

    def __init__(self, optimizer):
        print(self.name)
        print("Version: %s\n" % self.version)
        self.optimizer = optimizer
    
    # function that starts the optimization 
    def run(self):
        # prepare limits and initial errors
        # max_exponent: get the largest exponent and double it
        max_exponent = max(self.optimizer.best_params.values())*2
        steps = {}
        for k in self.optimizer.get_keys():
            steps[k] = self.optimizer.parameters[k]
            # start with huge step sizes
            steps['err_%s' % k] = self.optimizer.best_params[k]/100.0*40
            # in crystal, orbital exponents should not fall below exponentLowerLimit!
            if k.startswith('a'):
                steps['limit_%s' % k] = (exponentLowerLimit, max_exponent)
            # uncomment in case restrictions are wanted for the coefficients as well:
            # if k.startwith('d'):
            #     steps['limit_%s' % k] = (-10, 10)

        steps.update(self.optimizer.parameters)
        m = minuit.Minuit(self.optimizer.run, strategy=0, **steps)        
        print("Initial values (parameter +++ stepsize +++ limits):")
        for k in self.optimizer.get_keys():
            v = steps[k]
            e = steps["err_%s" % k]
            lim = None
            if steps.has_key('limit_%s' % k):
                lim = steps['limit_%s' % k]
            if lim is None:
                print("% 5s  %12.7f %12.7f" % (k, v, e))
            else:
                print("% 5s  %12.7f %12.7f     (%f,%f) " % (k, v, e, lim[0], lim[1]))
        print('')
        # perform the optimization
        try:
            m.migrad()
        except Exception, e:
            print("Exception raised: %s" % str(e))
            print("Please check your results carefully!")
        print("")
        print("Initial total energy: %.7f" % self.optimizer.initial_value)
        print("Lowest total energy:  %.7f" % self.optimizer.lowest_value)
# fin: class optimizationController


if __name__ == '__main__':
    options = {
        '-t' : ('--template=', 'template file with initial parameters', 
                { 'dest' : 'template' } ),
        '-i' : ('--generate-input', 'generate a crystal input from the template and print it', 
                { 'dest' : 'generate_input', 'action' : 'store_true', 'default' : False } ),
        '-g' : ('--guess=', 'file to use as fort.20 required for GUESSP directive',
                { 'dest' : 'guess', 'default' : None } ),
        '-l' : ('--lowerlimit=', 'lower limit to employ during optimization of exponents (default: %.2f)' % exponentLowerLimit,
                { 'dest' : 'lower_limit', 'default' : exponentLowerLimit, 'type' : 'float' } ),
        }
    parser = OptionParser(usage='usage: %s [options] files' % sys.argv[0])
    for k,v in options.iteritems():
        parser.add_option(k, v[0], help=v[1], **v[2])
    opts, args = parser.parse_args(sys.argv)
    if opts.template is None:
        print("Error: You must supply a template file that will be used to create the CRYSTAL input!")
        sys.exit(1)
    else:
        jobConfig['template'] = opts.template

    files = set(args[1:]+[opts.template])
    jobConfig['required_files'] = list(files)
    jobConfig['cmdline_options'] = opts
    jobConfig['cmdline_arguments'] = args
    jobConfig['exponentLowerLimit'] = opts.lower_limit

    c = optimizationController(crystalOptimizer(jobConfig))
    if opts.generate_input == True:
        c.optimizer.write_input(c.optimizer.get_parameters(), sys.stdout)
        sys.exit(0)

    # check for GUESSP keyword    
    if 0 == len(igrep('guessp', jobConfig['template'])):
        print("Warning: The template file does not contain the GUESSP keyword. It is recommended to restart from fort.20 in order to accelerate the optimization!")
    # check for SCFDIR keyword
    if 0 != len(igrep('scfdir', jobConfig['template'])):
        print("Warning: The template file contains the SCFDIR keyword. However if the system is small enough it is recommended to perform a conventional SCF.")
    # check for OPTGEOM
    if 0 != len(igrep('optgeom', jobConfig['template'])):
        print("Warning: The template file contains the OPTGEOM keyword. Using structure relaxation in combination with the basis set optimization is not supported and thus a waste of time.")
        
    # check if fort.20 file has been specified
    if opts.guess is None:
        print("Error: You must supply a file containing wave function data that will be used as the initial guess!")
        sys.exit(1)
    else:
        jobConfig['fort.20'] = opts.guess
    # make sure that all required files are available
    for i in jobConfig['required_files'] + [opts.guess]:
        if not os.path.exists(i):
            print("Error: Required file '%s' is not accessible" % i)
            sys.exit(11)
    # start the optimization
    print("Performing basis set optimization using the template '%s'.\n" % jobConfig['template'])
    c.run()
