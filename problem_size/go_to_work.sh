#!/bin/sh
./compile.sh
cp results.txt  old_results_"$(date +"%Y%m%d_%H%M%S")".csv
rm results.txt
qsub run.sh
