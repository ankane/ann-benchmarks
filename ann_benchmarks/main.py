from __future__ import absolute_import
import argparse
import os
import random
import sys
import shutil

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.results import get_results
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.definitions import get_definitions, list_algorithms
from ann_benchmarks.runner import run_subprocess


def positive_int(s):
    i = None
    try:
        i = int(s)
    except ValueError:
        pass
    if not i or i < 1:
        raise argparse.ArgumentTypeError("%r is not a positive integer" % s)
    return i

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            '--dataset',
            metavar='NAME',
            help='the dataset to load training points from',
            default='glove-100-angular')
    parser.add_argument(
            "-k", "--count",
            default=10,
            type=positive_int,
            help="the number of near neighbours to search for")
    parser.add_argument(
            '--definitions',
            metavar='FILE',
            help='load algorithm definitions from FILE',
            default='algos.yaml')
    parser.add_argument(
            '--algorithm',
            metavar='NAME',
            help='run only the named algorithm',
            default=None)
    parser.add_argument(
            '--list-algorithms',
            help='print the names of all known algorithms and exit',
            action='store_true',
            default=argparse.SUPPRESS)
    parser.add_argument(
            '--force',
            help='''re-run algorithms even if their results already exist''',
            action='store_true')
    parser.add_argument(
            '--runs',
            metavar='COUNT',
            type=positive_int,
            help='run each algorithm instance %(metavar)s times and use only the best result',
            default=3)
    parser.add_argument(
            '--timeout',
            type=int,
            help='Timeout (in seconds) for each individual algorithm run, or -1 if no timeout should be set',
            default=-1)

    args = parser.parse_args()
    if args.timeout == -1:
        args.timeout = None

    if hasattr(args, "list_algorithms"):
        list_algorithms(args.definitions)
        sys.exit(0)

    # Nmslib specific code
    # Remove old indices stored on disk
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    # TODO(erikbern): deal with this later
    #algos_already_run = set()
    #if not args.force:
    #    for res in get_results(args.dataset, args.count):
    #        print(res)
    #        algos_already_run.add((res.attrs["library"], res.attrs["name"]))

    dataset = get_dataset(args.dataset)
    dimension = len(dataset['train'][0]) # TODO(erikbern): ugly
    point_type = 'float' # TODO(erikbern): should look at the type of X_train
    distance = dataset.attrs['distance']
    definitions = get_definitions(args.definitions, dimension, point_type, distance, args.count)

    if args.algorithm:
        print('running only', args.algorithm)
        definitions = [d for d in definitions if d.algorithm == args.algorithm]

    random.shuffle(definitions)
    print('order:', definitions)

    for definition in definitions:
        print(definition, '...')

        run_subprocess(definition, args.dataset, args.count, args.runs)