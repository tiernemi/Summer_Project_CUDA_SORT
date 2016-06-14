set terminal postscript enhanced color
set output "./plots/sortMultCameraComparison.eps"
set title "Sort Rate Vs Camera ID For Multiple Algorithms"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Camera ID"
set yrange [0:100]
set xrange [0:55]
datFiles = system('ls bench_data/times*')
namesFiles =  system("ls bench_data/times* | gawk 'match($0, /times([A-Z][a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 3 lc i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultCameraComparisonRateScr.eps"
set title "Sort Rate Vs Camera ID For Multiple Algorithms"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Camera ID"
set yrange [*:*]
set xrange [*:*]
datFiles = system('ls bench_data/times*')
namesFiles =  system("ls bench_data/times* | gawk 'match($0, /times([A-Z][a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using 4:3 lc i with points title sprintf("%s", word(namesFiles,i))

## Select histogram data
#set style data histogram
# Give the bars a plain fill pattern, and draw a solid line around them.
#set style fill solid border
#set key left
#set yrange [0:100]

#set style histogram clustered
#plot for [COL=2:10] 'histogramC.dat' using COL:xticlabels(1) title columnheader

