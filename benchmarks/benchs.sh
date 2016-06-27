#!/bin/bash

rm plots/*
rm bench_data/*
./b/bench "/home/users/mschpc/2015/tiernemi/project/data/TVS2_TransparentGeometryData_2016_06_21_00001.txt"
#./b/bench "/home/users/mschpc/2015/tiernemi/project/data/smallData.txt"
gnuplot -e "load \"benchs.gp\""
evince plots/*.eps

