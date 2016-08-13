set terminal postscript enhanced color
set output "./plots/sortMultCameraComparisonTot.eps"
set title "Sort Rate Vs Camera ID For Multiple Algorithms : Total CPU Time"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Camera ID"
datFiles = system('ls bench_data/times*')
namesFiles =  system("ls bench_data/times* | gawk 'match($0, /times([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 9 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultCameraComparisonTransInc.eps"
set title "Sort Rate Vs Camera ID For Multiple Algorithms : Sort+Transform"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Camera ID"
datFiles = system('ls bench_data/times*')
namesFiles =  system("ls bench_data/times* | gawk 'match($0, /times([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 7 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultCameraComparisonSortOnly.eps"
set title "Sort Rate Vs Camera ID For Multiple Algorithms : Sort Only"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Camera ID"
datFiles = system('ls bench_data/times*')
namesFiles =  system("ls bench_data/times* | gawk 'match($0, /times([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 6 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultCameraComparisonSumTimes.eps"
set title "Sort Rate Vs Camera ID For Multiple Algorithms : Summed Event Times"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Camera ID"
datFiles = system('ls bench_data/times*')
namesFiles =  system("ls bench_data/times* | gawk 'match($0, /times([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 8 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set logscale x 2
set terminal postscript enhanced color
set output "./plots/sortMultSizeComparisonSumTimes.eps"
set title "Sort Rate Vs Array Size For Multiple Algorithms : Summed Event Times"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size"
datFiles = system('ls bench_data/size*')
namesFiles =  system("ls bench_data/size* | gawk 'match($0, /times([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 1:8 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultSizeComparisonSortOnly.eps"
set title "Sort Rate Vs Array Size For Multiple Algorithms : Sort Only"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size"
datFiles = system('ls bench_data/size*')
namesFiles =  system("ls bench_data/size* | gawk 'match($0, /times([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 1:6 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultSizeComparisonTransInc.eps"
set title "Sort Rate Vs Array Size For Multiple Algorithms : Sort + Transform"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size"
datFiles = system('ls bench_data/size*')
namesFiles =  system("ls bench_data/size* | gawk 'match($0, /times([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 1:7 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultSizeComparisonTot.eps"
set title "Sort Rate Vs Array Size For Multiple Algorithms : Total CPU time"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size"
datFiles = system('ls bench_data/size*')
namesFiles =  system("ls bench_data/size* | gawk 'match($0, /times([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 1:9 lt i with linespoints title sprintf("%s", word(namesFiles,i))






#unset logscale y
#set terminal postscript enhanced color
#set output "./plots/sortSpeedUpCameraComparison.eps"
#set title "SpeedUp Of Algorithms Vs Camera ID"
#set ylabel "SpeedUp"
#set xtics rotate out

#set yrange [*:*]
#set xrange [*:*]
#set xlabel "Camera ID"
#datFiles = system('ls bench_data/speedup*')
#namesFiles =  system("ls bench_data/speedup* | gawk 'match($0, /speedup([A-Z]*[a-z]*)_([A-Z][a-z]*)_([A-Z][a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]\ a[3]a[4]}' ")

#plot for [i=1:words(datFiles)] word(datFiles,i) using 4 lt i with linespoints title sprintf("%s", word(namesFiles,i))

