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


def get_hostfile(scratch, ppn):
    import os
    from tempfile import NamedTemporaryFile

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


def get_mpi_configurations(scratch, processes, ppns, extra_mpi_args):
    from collections import OrderedDict
    from itertools import product

    custom_ppn_mappings = OrderedDict([
        (16, '1:2:3:4:5:6:7:8:13:14:15:16:17:18:19:20'),
        (18, '1:2:3:4:5:6:7:8:9:13:14:15:16:17:18:19:20:21'),
        (20, '1:2:3:4:5:6:7:8:9:10:13:14:15:16:17:18:19:20:21:22'),
        (22, '1:2:3:4:5:6:7:8:9:10:11:13:14:15:16:17:18:19:20:21:22:23'),
        ])
    default_mpi_args = ['-env MV2_SHOW_CPU_BINDING 1', '-env MV2_HYBRID_ENABLE_THRESHOLD 8192']
    configurations = []
    ppn_hostfiles = dict((ppn, get_hostfile(scratch, ppn)) for ppn in ppns)
    for p, ppn in product(processes, ppns):
        cpu_mapping = custom_ppn_mappings.get(ppn, ':'.join(str(p) for p in range(ppn)))
        mpi_args = ['mpirun -np %d -hostfile %s -env MV2_CPU_MAPPING %s' % (p, ppn_hostfiles[ppn], cpu_mapping)]
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


def get_runtime(action, output, required=True):
    from re import search

    float_pattern = r'((?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)'
    pattern = 'Time taken in %s: %s' % (action, float_pattern)
    match = search(pattern, output)
    if required:
        return float(match.group(1))
    else:
        return float(match.group(1) if match is not None else 0)