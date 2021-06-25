#!/bin/sh

g++ -DMYTHREADS=24 -o livd_exe -O3 liver.cc -lgomp -fopenmp -lm


