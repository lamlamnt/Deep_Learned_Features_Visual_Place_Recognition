#!/usr/bin/env python

'''
    This script can be used to download data from the UTIAS-Multiseason Dataset
    and the UTIAS In The Dark dataset.

    This code is built on the download script provided at: 
    http://robots.engin.umich.edu/nclt/
'''

from __future__ import print_function
import sys
import os
import subprocess
import argparse

base_dir = 'ftp://asrl3.utias.utoronto.ca/2020-vtr-dataset'

dataset_dir = {'multiseason': '%s/UTIAS-Multiseason' % (base_dir), 
                 'inthedark': '%s/UTIAS-In-The-Dark' % (base_dir)}

dataset_name = {'multiseason': 'UTIAS Multiseason Dataset', 
                  'inthedark': 'UTIAS In The Dark Dataset'}

num_runs = {'inthedark': 39, 'multiseason': 136}

def main (args):

    getopt = argparse.ArgumentParser(description='Download VT&R dataset')
    getopt.add_argument('--season', action='store_true',
            help='Download data form the UTIAS Multiseason Dataset')
    getopt.add_argument('--dark', action='store_true',
            help='Download data form the UTIAS In The Dark Dataset')
    getopt.add_argument('--all', action='store_true',
            help='Download all runs form the given dataset')
    getopt.add_argument('--runs', type=int, nargs='+',
            help='Download runs listed as integers, e.g. --runs 2 21 35')
    getopt.add_argument('--runs_range', type=int, nargs=2,
            help='Download runs in range (inclusive) given as two integers, \
                  e.g. --runs_range 3 20')

    args = getopt.parse_args()

    if not args.season and not args.dark:
        print('Dataset has not been specified. Use --help for options.')
        return 0

    if not args.all and args.runs is None and args.runs_range is None:
        print('Runs have not been specified. Use --help for options.')
        return 0

    dataset = 'multiseason' if args.season else 'inthedark'

    print('Downloading data from %s\n' % (dataset_name[dataset]))

    runs = []
    if args.all:
        runs = list(range(num_runs[dataset]))
    elif args.runs is not None:
        runs = args.runs
    elif args.runs_range is not None:
        runs = list(range(args.runs_range[0], args.runs_range[1] + 1))

    if len(runs) == 0:
        print('An empty or invalid range of runs were specified.' \
              ' Please choose a different set of runs.')
        return 0 

    print('Preparing to download the following runs: \n%s\n' % (runs))

    for run in runs:

        if run not in range(num_runs[dataset]):
            print('Run %d not in dataset' % (run))
            continue

        cmd = ['wget', '--continue', 
               '%s/run_%06d.zip' % (dataset_dir[dataset], run)]
        subprocess.call(cmd)

    return 0

if __name__ == '__main__':
    sys.exit (main(sys.argv))
