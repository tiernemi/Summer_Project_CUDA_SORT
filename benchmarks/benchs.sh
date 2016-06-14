#!/bin/bash

rm plots/*
rm bench_data/*
./b/bench "/home/users/mschpc/2015/tiernemi/project/data/testDataP2.txt"
#./b/bench "/home/users/mschpc/2015/tiernemi/project/data/smallData.txt"
gnuplot -e "load \"benchs.gp\""
evince plots/*.eps

