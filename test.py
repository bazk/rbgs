#!/usr/bin/env python2

import os, sys
import subprocess
import re
import argparse

def err(msg, code=1):
    print >> sys.stderr, "error: " + msg
    sys.exit(code)

def algorithm(val):
    if (val == "j") or (val == "g"):
        return val
    else:
        raise argparse.ArgumentTypeError("%s is not a valid algorithm" % val)

parser = argparse.ArgumentParser()
parser.add_argument("-n",                       help="list of N (# of discretization intervals)", type=int, nargs='+', default=[32, 33, 1024, 1025, 2048, 2049])
parser.add_argument("-t",                       help="list of T (# of threads)", type=int, nargs='+', default=[1, 2, 4, 6, 8])
parser.add_argument("-i",                       help="number of tests to run for each configuration", type=int, default=30)
parser.add_argument("-c",                       help="number of iterations per test", type=int, default=200)
parser.add_argument("-o", "--output",           help="output file", type=argparse.FileType('w'), default=sys.stdout)
parser.add_argument("-m", "--algorithm",        help="number of iterations per test", type=algorithm, default="g")
args = parser.parse_args()

try:
    subprocess.check_call(["make", "-s"])
except subprocess.CalledProcessError:
    err("build failed")

args.output.write('N/T ' + ' '.join(str(t) for t in args.t) + '\n')

total = len(args.n) * len(args.t) * args.i
completed = 0

for n in args.n:
    args.output.write(str(n) + ' ')

    for t in args.t:
        avg = 0

        cmd = "likwid-pin -c 0-%d ./rbgs %d %d %d %d %s" % (t-1, n, n, t, args.c, args.algorithm)
        print >> sys.stderr, "[%2d%%] %s" % (int(completed * 100 / float(total)), cmd)

        for i in range(args.i):
            try:
                out = subprocess.check_output(cmd.split())
                match = re.search('.*processamento: ([0-9\.]+) segundos.*', out)
                avg += float(match.group(1))
                completed += 1
            except subprocess.CalledProcessError:
                err("execution failed")

        args.output.write(str(avg / args.i) + ' ')

    args.output.write('\n')

args.output.close()

    # speedup = open("speedup_"+args.algorithm+".txt", "w")
    # speedup.write('Threads ' + ' '.join("N="+str(n) for n in N) + '\n')

    # for t in args.t:
    #     speedup.write(str(t) + ' ')

    #     for n in args.n:
    #         speedup.write(str(results[n][1] / results[n][t]) + ' ')

    #     speedup.write('\n')

    # speedup.close()
