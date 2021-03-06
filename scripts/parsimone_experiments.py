#!/usr/bin/env python

##
# @file parsimone_experiments.py
# @brief Script for experimenting with ParsiMoNe.
# @author Ankit Srivastava <asrivast@gatech.edu>
#
# Copyright 2020 Georgia Institute of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict
from itertools import product
import os
import os.path
from os.path import join

from utils import get_hostfile, get_mpi_configurations, read_dataset, write_dataset, get_experiment_datasets, get_runtime


big_datasets = OrderedDict([
    #(name        , (-f, -n, -m, -s, -c, -v, -i)),
    ('yeast'      , ('data/yeast/yeast_microarray_expression.tsv', 5716, 2577, '\t', True, True, True)),
    ('development', ('data/athaliana/athaliana_development_exp.tsv', 18373, 5102, ' ', True, True, True)),
    ('complete'   , ('data/athaliana/athaliana_complete_exp.tsv', 18380, 16838, ' ', True, True, True)),
    ])

dataset_groups = dict([
    ('big',       big_datasets),
    ])

all_datasets = OrderedDict(list(big_datasets.items()))

all_algorithms = [
    'lemontree',
    'genomica',
    ]

all_processes = [
    ]
for power in range(0, 11):
    all_processes.append(2 ** power)

ppn_mappings = OrderedDict([
    (16, '1:2:3:4:5:6:7:8:13:14:15:16:17:18:19:20'),
    (18, '1:2:3:4:5:6:7:8:9:13:14:15:16:17:18:19:20:21'),
    (20, '1:2:3:4:5:6:7:8:9:10:13:14:15:16:17:18:19:20:21:22'),
    (22, '1:2:3:4:5:6:7:8:9:10:11:13:14:15:16:17:18:19:20:21:22:23'),
    (24, '0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23'),
    ])

NUM_REPEATS = 5


def parse_datasets(args):
    '''
    Get datasets to be used for the experiments.
    '''
    experiment_datasets = []
    if args.dataset is None:
        args.dataset = list(big_datasets.keys())[:1]
    for d in args.dataset:
        if d in all_datasets:
            experiment_datasets.append(d)
        elif d in dataset_groups:
            experiment_datasets.extend(list(dataset_groups[d].keys()))
        else:
            raise RuntimeError('Dataset %s is not recognized' % d)
    args.dataset = experiment_datasets


def parse_args():
    '''
    Parse command line arguments.
    '''
    import argparse
    from multiprocessing import cpu_count
    from os.path import expanduser, realpath

    parser = argparse.ArgumentParser(description='Run scaling experiments')
    parser.add_argument('-b', '--basedir', metavar='DIR', type=str, default=realpath(join(expanduser('~'), 'ParsiMoNe')), help='Base directory for running the experiments')
    parser.add_argument('-s', '--scratch', metavar='DIR', type=str, default=realpath(join(expanduser('~'), 'scratch')), help='Scratch directory, visible to all the nodes')
    parser.add_argument('-d', '--dataset', metavar='NAME', type=str, nargs='*', help='Datasets (or groups of datasets) to be used')
    parser.add_argument('-n', '--variables', metavar='N', type=int, nargs='*', help='Number of variable(s) to be used')
    parser.add_argument('-m', '--observations', metavar='M', type=int, nargs='*', help='Number of observation(s) to be used')
    parser.add_argument('-a', '--algorithm', metavar='NAME', type=str, nargs='*', default=[all_algorithms[0]], help='Algorithms to be used')
    parser.add_argument('-g', '--arguments', metavar='ARGS', type=str, help='Arguments to be passed to the underlying script')
    parser.add_argument('-p', '--process', metavar='P', type=int, nargs='*', default=all_processes, help='Processes to be used')
    parser.add_argument('--ppn', metavar='PPN', type=int, nargs='*', default=[cpu_count()], help='Number of processes per node to be used')
    parser.add_argument('-r', '--repeat', metavar='N', type=int, default=NUM_REPEATS, help='Number of times the experiments should be repeated')
    parser.add_argument('--mpi-arguments', metavar='ARGS', type=str, help='Arguments to be passed to mpirun')
    parser.add_argument('--hostfile', metavar='HOSTS', type=str, help='Hostfile to be used for the runs')
    parser.add_argument('--lemontree', action='store_true', help='Flag for running lemon-tree instead of our implementation')
    parser.add_argument('--results', metavar = 'FILE', type=str, default='results_%s' % os.environ.get('PBS_JOBID', 0), help='Name of the csv file to which results will be written')
    parser.add_argument('--exec-suffix', type=str, default='', help='Suffix to add to the executable')
    parser.add_argument('--output-suffix', type=str, default='', help='Suffix to add to the output directory')
    args = parser.parse_args()
    parse_datasets(args)
    return args


def get_executable_configurations(executable, datasets, algorithms, arguments, lemontree):
    boolean_args = ['-c', '-v', '-i']
    default_parsimone_args = ['-r', '--warmup', '--hostnames']
    configurations = []
    for name, algorithm in product(datasets, algorithms):
        dataset_args = all_datasets[name]
        script_args = [executable]
        script_args.append('-a %s' % algorithm)
        script_args.append('-f %s -n %d -m %d -s \'%s\'' % tuple(dataset_args[:4]))
        script_args.extend(b for i, b in enumerate(boolean_args) if dataset_args[4 + i])
        if arguments is not None:
            script_args.append(arguments)
        if not lemontree:
            script_args.extend(default_parsimone_args)
        configurations.append((name, algorithm, ' '.join(script_args)))
    return configurations


def parse_runtimes(output):
    # optional runtimes
    warmup = get_runtime('warming up MPI', output, required=False)
    reading = get_runtime('reading the file', output, required=False)
    ganesh = get_runtime('the GaneSH run', output, required=False)
    consensus = get_runtime('consensus clustering', output, required=False)
    trees = get_runtime('learning module trees', output, required=False)
    candidates = get_runtime('learning candidate splits', output, required=False)
    choose = get_runtime('choosing splits', output, required=False)
    sync = get_runtime('synchronizing the modules', output, required=False)
    parents = get_runtime('learning module parents', output, required=False)
    modules = get_runtime('learning the modules', output, required=False)
    writing = get_runtime('writing the files', output, required=False)
    # required runtime
    network = get_runtime('getting the network', output, required=True)
    return [warmup, reading, ganesh, consensus, trees, candidates, choose, sync, parents, modules, network, writing]


def run_experiment(basedir, scratch, config, outsuffix, repeat, lemontree, compare):
    from datetime import datetime
    import subprocess
    import sys
    from tempfile import TemporaryDirectory

    MAX_TRIES = 5
    dirname = join(scratch, config[0])
    if lemontree:
        dirname += '_lemontree'
    dirname += outsuffix
    outdir = dirname if not os.path.exists(dirname) else TemporaryDirectory(dir=scratch).name
    r = 0
    t = 0
    while r < repeat:
        arguments = config[-1] + ' -o %s' % outdir
        print('Started the run at', datetime.now().strftime('%c'))
        print(arguments)
        sys.stdout.flush()
        output = ''
        process = subprocess.Popen(arguments, shell=True, stdout=subprocess.PIPE)
        for line in iter(process.stdout.readline, b''):
            line = line.decode('utf-8')
            output += line
            print(line, end='')
        process.communicate()
        if process.returncode != 0:
            t += 1
            if t == MAX_TRIES:
                print('ERROR: Run failed %d times' % t)
                yield None
            print('Run failed. Retrying.')
            continue
        else:
            t = 0
        sys.stdout.flush()
        print('Finished the run at', datetime.now().strftime('%c'))
        print()
        if compare:
            print('Comparing generated files in %s with %s' % (outdir, dirname))
            sys.stdout.flush()
            compare_args = [join(basedir, 'common', 'scripts', 'compare_lemontree.py'), outdir, dirname]
            try:
                subprocess.check_call(' '.join(compare_args), shell=True)
            except subprocess.CalledProcessError as ce:
                print(ce)
                print('ERROR: Comparison failed')
        yield parse_runtimes(output)
        r += 1


def main():
    '''
    Main function.
    '''
    args = parse_args()
    datasets = args.dataset
    if args.variables or args.observations:
        datasets = get_experiment_datasets(args.basedir, datasets, args.variables, args.observations, args.scratch, all_datasets)
    else:
        for dataset in datasets:
            ds = list(all_datasets[dataset])
            all_datasets[dataset] = tuple([join(args.basedir, ds[0])] + ds[1:])
    executable = join(args.basedir, 'parsimone' + args.exec_suffix) if not args.lemontree else join(args.basedir, 'common', 'scripts', 'parsimone_lemontree.py')
    exec_configs = get_executable_configurations(executable, datasets, args.algorithm, args.arguments, args.lemontree)
    if not args.lemontree:
        mpi_configs = get_mpi_configurations(args.scratch, args.process, args.ppn, args.mpi_arguments, args.hostfile)
        all_configs = list((executable[0], executable[1], mpi[0], mpi[-1] + ' ' + executable[-1]) for executable, mpi in product(exec_configs, mpi_configs))
    else:
        if args.process or args.ppn:
            print('WARNING: Ignoring the -p and --ppn arguments while running Lemon-Tree')
        all_configs = []
        for config in exec_configs:
            all_configs.append(tuple(list(config[:-1]) + [1, config[-1]]))
    print('*** Writing the results to', os.path.abspath(args.results), '***\n\n')
    with open(args.results, 'w') as results:
        results.write('# warmup,reading,ganesh,consensus,trees,candidates,choose,sync,parents,modules,network,writing\n')
        for config in all_configs:
            comment = 'runtime for dataset=%s using algorithm=%s on processors=%d' % tuple(config[:3])
            results.write('# %s %s\n' % ('our' if not args.lemontree else 'lemontree', comment))
            for rt in run_experiment(args.basedir, args.scratch, config, args.output_suffix, args.repeat, args.lemontree, True):
                if rt is not None:
                    results.write(','.join(str(t) for t in rt) + '\n')
                    results.flush()


if __name__ == '__main__':
    main()
