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

unset logscale y
set terminal postscript enhanced color
set output "./plots/sortSpeedUpCameraComparison.eps"
set title "SpeedUp Of Algorithms Vs Camera ID"
set ylabel "SpeedUp"
#set xtics rotate out

set yrange [*:*]
set xrange [*:*]
set xlabel "Camera ID"
datFiles = system('ls bench_data/speedup*')
namesFiles =  system("ls bench_data/speedup* | gawk 'match($0, /speedup([A-Z]*[a-z]*)_([A-Z][a-z]*)_([A-Z][a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]\ a[3]a[4]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 4 lt i with linespoints title sprintf("%s", word(namesFiles,i))


## Select histogram data
#set style data histogram
# Give the bars a plain fill pattern, and draw a solid line around them.
#set style fill solid border
#set key left
#set yrange [0:100]

#set style histogram clustered
#plot for [COL=2:10] 'histogramC.dat' using COL:xticlabels(1) title columnheader

