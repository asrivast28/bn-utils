#!/usr/bin/env python

##
# @file utils.py
# @brief Common scripting utilities.
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

from itertools import product


def get_hostfile(scratch, ppn):
    import os
    from tempfile import NamedTemporaryFile

    nodefile = os.environ.get('PBS_NODEFILE')
    if nodefile is None:
        return None
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


def get_mpi_configurations(scratch, processes, ppns, extra_mpi_args, hostfile):
    from collections import OrderedDict

    custom_ppn_mappings = OrderedDict([
        (16, '1:2:3:4:5:6:7:8:13:14:15:16:17:18:19:20'),
        (18, '1:2:3:4:5:6:7:8:9:13:14:15:16:17:18:19:20:21'),
        (20, '1:2:3:4:5:6:7:8:9:10:13:14:15:16:17:18:19:20:21:22'),
        (22, '1:2:3:4:5:6:7:8:9:10:11:13:14:15:16:17:18:19:20:21:22:23'),
        ])
    default_mpi_args = ['-env MV2_SHOW_CPU_BINDING 1', '-env MV2_HYBRID_ENABLE_THRESHOLD 8192']
    configurations = []
    ppn_hostfiles = dict((ppn, get_hostfile(scratch, ppn) if hostfile is None else hostfile) for ppn in ppns)
    for p, ppn in product(processes, ppns):
        cpu_mapping = custom_ppn_mappings.get(ppn, ':'.join(str(p) for p in range(ppn)))
        mpi_args = ['mpirun -np %d' % p]
        if ppn_hostfiles[ppn] is not None:
            mpi_args.append('-hostfile %s' % ppn_hostfiles[ppn])
        mpi_args.append('-env MV2_CPU_MAPPING %s' % cpu_mapping)
        mpi_args.extend(default_mpi_args)
        if extra_mpi_args is not None:
            mpi_args.append(extra_mpi_args)
        configurations.append((p, ' '.join(mpi_args)))
    return configurations


def read_dataset(name, sep, colobs, varnames, indices):
    '''
    Read the dataset from the given CSV file.
    '''
    import pandas

    header = None
    index = False
    if colobs:
        if indices:
            header = 0
        if varnames:
            index = 0
    else:
        if varnames:
            header = 0
        if indices:
            index = 0
    dataset = pandas.read_csv(name, sep=sep, header=header, index_col=index)
    if colobs:
        dataset = dataset.T
    return dataset


def write_dataset(dataset, name, sep, colobs, varnames, indices):
    '''
    Write the given pandas dataset as a CSV file.
    '''
    header = False
    index = False
    if colobs:
        dataset = dataset.T
        if indices:
            header = True
        if varnames:
            index = True
    else:
        if varnames:
            header = True
        if indices:
            index = True
    dataset.to_csv(name, sep=sep, header=header, index=index)


def get_experiment_datasets(basedir, datasets, variables, observations, scratch, all_datasets):
    import os
    from os.path import join

    experiment_datasets = []
    if not variables:
        variables = [None]
    if not observations:
        observations = [None]
    for dname, n, m in product(datasets, variables, observations):
        dataset = all_datasets[dname]
        n = n if n is not None else dataset[1]
        m = m if m is not None else dataset[2]
        exp_dname = '%s_n%d_m%d' % (dname, n, m)
        exp_dataset = list(dataset)
        exp_dataset[0] = os.path.join(scratch, '%s%s' % (exp_dname, os.path.splitext(dataset[0])[1]))
        exp_dataset[1] = n
        exp_dataset[2] = m
        if not os.path.exists(exp_dataset[0]):
            read = read_dataset(join(basedir, dataset[0]), dataset[3], dataset[4], dataset[5], dataset[6])
            write_dataset(read.iloc[:m,:n], exp_dataset[0], exp_dataset[3], exp_dataset[4], exp_dataset[5], exp_dataset[6])
        all_datasets.update([(exp_dname, tuple(exp_dataset))])
        experiment_datasets.append(exp_dname)
    return experiment_datasets


def get_runtime(action, output, required=True):
    from re import search

    float_pattern = r'((?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)'
    pattern = 'Time taken in %s: %s' % (action, float_pattern)
    match = search(pattern, output)
    if required:
        return float(match.group(1))
    else:
        return float(match.group(1) if match is not None else 0)
