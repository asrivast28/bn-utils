#!/usr/bin/env python

##
# @file mnets_experiments.py
# @brief Script for experimenting with mnets.
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
from os.path import join, basename
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

from discretize import read_dataset, write_dataset


big_datasets = OrderedDict([
    #(name        , (-f, -n, -m, -s, -c, -v, -i)),
    ('yeast'      , ('data/yeast/yeast_microarray_expression.tsv', 5716, 2577, '\t', True, True, True)),
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
        args.dataset = list(big_datasets.keys())
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
    from os.path import expanduser, realpath

    parser = argparse.ArgumentParser(description='Run scaling experiments')
    parser.add_argument('-b', '--basedir', metavar='DIR', type=str, default=realpath(join(expanduser('~'), 'mnets')), help='Base directory for running the experiments.')
    parser.add_argument('-s', '--scratch', metavar='DIR', type=str, default=realpath(join(expanduser('~'), 'scratch')), help='Scratch directory, visible to all the nodes.')
    parser.add_argument('-d', '--dataset', metavar='NAME', type=str, nargs='*', help='Datasets (or groups of datasets) to be used.')
    parser.add_argument('-n', '--variables', metavar='N', type=int, nargs='*', help='Number of variable(s) to be used.')
    parser.add_argument('-m', '--observations', metavar='M', type=int, nargs='*', help='Number of observation(s) to be used.')
    parser.add_argument('-a', '--algorithm', metavar='NAME', type=str, nargs='*', default=[all_algorithms[0]], help='Algorithms to be used.')
    parser.add_argument('-g', '--arguments', metavar='ARGS', type=str, help='Arguments to be passed to the underlying script.')
    parser.add_argument('-p', '--process', metavar='P', type=int, nargs='*', default=all_processes, help='Processes to be used.')
    parser.add_argument('--ppn', metavar='PPN', type=int, nargs='*', default=[list(ppn_mappings.keys())[0]], help='Number of processes per node to be used.')
    parser.add_argument('-r', '--repeat', metavar='N', type=int, default=NUM_REPEATS, help='Number of times the experiments should be repeated.')
    parser.add_argument('--lemontree', action='store_true', help='Flag for running lemon-tree instead of our implementation.')
    parser.add_argument('--results', metavar = 'FILE', type=str, default='results_%s' % os.environ.get('PBS_JOBID', 0), help='Name of the csv file to which results will be written.')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to add to the executable.')
    args = parser.parse_args()
    parse_datasets(args)
    return args


def get_experiment_datasets(basedir, datasets, variables, observations, scratch):
    experiment_datasets = []
    if not variables:
        variables = [None]
    if not observations:
        observations = [None]
    for dname, n, m in product(datasets, variables, observations):
        if n is None and m is None:
            experiment_datasets.append(dname)
        else:
            dataset = all_datasets[dname]
            read = read_dataset(join(basedir, dataset[0]), dataset[3], dataset[4], dataset[5], dataset[6])
            n = n if n is not None else dataset[1]
            m = m if m is not None else dataset[2]
            exp_dname = '%s_n%d_m%d' % (dname, n, m)
            exp_dataset = list(dataset)
            exp_dataset[0] = join(scratch, '%s%s' % (exp_dname, os.path.splitext(dataset[0])[1]))
            exp_dataset[1] = n
            exp_dataset[2] = m
            if not os.path.exists(exp_dataset[0]):
                write_dataset(read.iloc[:m,:n], exp_dataset[0], exp_dataset[3], exp_dataset[4], exp_dataset[5], exp_dataset[6])
            all_datasets.update([(exp_dname, tuple(exp_dataset))])
            experiment_datasets.append(exp_dname)
    return experiment_datasets


def get_executable_configurations(executable, datasets, algorithms, arguments, lemontree):
    boolean_args = ['-c', '-v', '-i']
    default_mnets_args = ['-r', '--warmup', '--hostnames']
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
            script_args.extend(default_mnets_args)
        configurations.append((name, algorithm, ' '.join(script_args)))
    return configurations


def get_hostfile(scratch, ppn):
    nodefile = os.environ['PBS_NODEFILE']
    seen = set()
    hosts = []
    with open(nodefile, 'r') as nf:
        for n in nf.readlines():
            if n not in seen:
                hosts.append(n.strip() + ':%d' % ppn)
            seen.add(n)
    with NamedTemporaryFile(mode='w', suffix='.hosts', dir=scratch, delete=False) as hf:
        hf.write('\n'.join(hosts) + '\n')
    return hf.name


def get_mpi_configurations(scratch, processes, ppns):
    default_mpi_args = ['-env MV2_SHOW_CPU_BINDING 1']
    configurations = []
    ppn_hostfiles = dict((ppn, get_hostfile(scratch, ppn)) for ppn in ppns)
    for p, ppn in product(processes, ppns):
        mpi_args = ['mpirun -np %d -hostfile %s -env MV2_CPU_MAPPING %s' % (p, ppn_hostfiles[ppn], ppn_mappings[ppn])]
        mpi_args.extend(default_mpi_args)
        configurations.append((p, ' '.join(mpi_args)))
    return configurations


def get_runtime(action, output, required=True):
    import re

    float_pattern = r'((?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)'
    pattern = 'Time taken in %s: %s' % (action, float_pattern)
    match = re.search(pattern, output)
    if required:
        return float(match.group(1))
    else:
        return float(match.group(1) if match is not None else 0)


def parse_runtimes(output):
    # optional runtimes
    warmup = get_runtime('warming up MPI', output, required=False)
    ganesh = get_runtime('the GaneSH run', output, required=False)
    consensus = get_runtime('consensus clustering', output, required=False)
    trees = get_runtime('learning the module trees', output, required=False)
    candidates = get_runtime('learning candidate splits', output, required=False)
    choose = get_runtime('choosing splits', output, required=False)
    sync = get_runtime('synchronizing the modules', output, required=False)
    parents = get_runtime('learning the module parents', output, required=False)
    modules = get_runtime('learning the modules', output, required=False)
    writing = get_runtime('writing the files', output, required=True)
    # required runtimes
    reading = get_runtime('reading the file', output, required=True)
    network = get_runtime('getting the network', output, required=True)
    return [warmup, reading, ganesh, consensus, trees, candidates, choose, sync, parents, modules, network, writing]


def run_experiment(basedir, scratch, config, repeat, lemontree, compare):
    import subprocess

    MAX_TRIES = 5
    dirname = join(scratch, config[0])
    if lemontree:
        dirname += '_lemontree'
    outdir = dirname if not os.path.exists(dirname) else TemporaryDirectory(dir=scratch).name
    r = 0
    t = 0
    while r < repeat:
        arguments = config[-1] + ' -o %s' % outdir
        print(arguments)
        sys.stdout.flush()
        try:
            output = ''
            process = subprocess.Popen(arguments, shell=True, stdout=subprocess.PIPE)
            for line in iter(process.stdout.readline, b''):
                line = line.decode('utf-8')
                output += line
                print(line, end='')
        except subprocess.CalledProcessError:
            t += 1
            if t == MAX_TRIES:
                raise
            print('Run failed. Retrying.')
            continue
        else:
            t = 0
        sys.stdout.flush()
        if compare:
            print('Comparing generated files in %s with %s' % (outdir, dirname))
            sys.stdout.flush()
            compare_args = [join(basedir, 'common', 'scripts', 'compare_lemontree.py'), outdir, dirname]
            subprocess.check_call(' '.join(compare_args), shell=True)
        yield parse_runtimes(output)
        r += 1


def main():
    '''
    Main function.
    '''
    args = parse_args()
    datasets = args.dataset
    if args.variables or args.observations:
        datasets = get_experiment_datasets(args.basedir, datasets, args.variables, args.observations, args.scratch)
    else:
        for dataset in datasets:
            ds = list(all_datasets[dataset])
            all_datasets[dataset] = tuple([join(args.basedir, ds[0])] + ds[1:])
    executable = join(args.basedir, 'mnets' + args.suffix) if not args.lemontree else join(args.basedir, 'common', 'scripts', 'mnets_lemontree.py')
    exec_configs = get_executable_configurations(executable, datasets, args.algorithm, args.arguments, args.lemontree)
    if not args.lemontree:
        mpi_configs = get_mpi_configurations(args.scratch, args.process, args.ppn)
        all_configs = list((executable[0], executable[1], mpi[0], mpi[-1] + ' ' + executable[-1]) for executable, mpi in product(exec_configs, mpi_configs))
    else:
        all_configs = []
        for config, p, ppn in product(exec_configs, args.process, args.ppn):
            par_config = '--nprocs %d --ppn %d' % (p, ppn)
            all_configs.append(tuple(list(config[:-1]) + [p, config[-1] + ' ' + par_config]))
    print('*** Writing the results to', os.path.abspath(args.results), '***\n\n')
    with open(args.results, 'w') as results:
        results.write('# warmup,reading,ganesh,consensus,trees,candidates,choose,sync,parents,modules,network,writing\n')
        for config in all_configs:
            comment = 'runtime for dataset=%s using algorithm=%s on processors=%d' % tuple(config[:3])
            results.write('# %s %s\n' % ('our' if not args.lemontree else 'lemontree', comment))
            for rt in run_experiment(args.basedir, args.scratch, config, args.repeat, args.lemontree, True):
                results.write(','.join(str(t) for t in rt) + '\n')
                results.flush()


if __name__ == '__main__':
    main()
