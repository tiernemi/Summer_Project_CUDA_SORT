#!/bin/bash

./b/bench "/home/users/mschpc/2015/tiernemi/project/data/testData.txt"
gnuplot -e "load \"benchs.gp\""
evince plots/*.eps

