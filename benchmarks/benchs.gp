
set linetype  1 lc rgb "blue" lw 1
set linetype  2 lc rgb "red" lw 1
set linetype  3 lc rgb "orange" lw 1
set linetype  4 lc rgb "gray50" lw 1
set linetype  5 lc rgb "black" lw 1
set linetype  6 lc rgb "green" lw 1
set linetype  7 lc rgb "#e51e10" lw 1
set linetype  8 lc rgb "black"   lw 1
set linetype  9 lc rgb "gray50"  lw 1
set linetype cycle  9

set key below
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

set terminal postscript enhanced color
set output "./plots/sortMultUniSizeComparisonSumTimes.eps"
set title "Sort Rate Vs Array Size For Uniform Distribution : Summed Event Times"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizetimesuni*')
namesFiles =  system("ls bench_data/sizetimesuni* | gawk 'match($0, /timesuni([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):8 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultUniSizeComparisonSortOnly.eps"
set title "Sort Rate Vs Array Size Uniform Distribution : Sort Only"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizetimesuni*')
namesFiles =  system("ls bench_data/sizetimesuni* | gawk 'match($0, /timesuni([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):6 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultUniSizeComparisonTransInc.eps"
set title "Sort Rate Vs Array Size Uniform Distribution : Sort + Transform"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizetimesuni*')
namesFiles =  system("ls bench_data/sizetimesuni* | gawk 'match($0, /timesuni([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):7 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultUniSizeComparisonTot.eps"
set title "Sort Rate Vs Array Size Uniform Distribution : Total CPU time"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizetimesuni*')
namesFiles =  system("ls bench_data/sizetimesuni* | gawk 'match($0, /timesuni([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):9 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set linetype  5 lc rgb "green" lw 1
set terminal postscript enhanced color
set output "./plots/speedupuniSTLSizeComparisonTot.eps"
set title "SpeedUp Over STL Vs Array Size (Uniform) : Total CPU time"
set ylabel "SpeedUp"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizespeeduni*_*_STL_Sort*')
namesFiles =  system("ls bench_data/sizespeeduni*_*_STL_Sort* | gawk 'match($0, /speeduni([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):5 lt i with linespoints title sprintf("%s", word(namesFiles,i))


set terminal postscript enhanced color
set output "./plots/speedupuniSTLSizeComparisonSort.eps"
set title "SpeedUp Over STL Vs Array Size (Uniform) : Sort Only"
set ylabel "SpeedUp"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizespeeduni*_*_STL_Sort*')
namesFiles =  system("ls bench_data/sizespeeduni*_*_STL_Sort* | gawk 'match($0, /speeduni([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):2 lt i with linespoints title sprintf("%s", word(namesFiles,i))


set linetype  1 lc rgb "blue" lw 1
set linetype  2 lc rgb "red" lw 1
set linetype  3 lc rgb "orange" lw 1
set linetype  4 lc rgb "gray50" lw 1
set linetype  5 lc rgb "black" lw 1
set linetype  6 lc rgb "green" lw 1
set linetype  7 lc rgb "#e51e10" lw 1
set linetype  8 lc rgb "black"   lw 1
set linetype  9 lc rgb "gray50"  lw 1
set linetype cycle  9


set key below
set terminal postscript enhanced color
set output "./plots/sortMultNormalSizeComparisonSumTimes.eps"
set title "Sort Rate Vs Array Size For Normal Distribution : Summed Event Times"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizetimesnormal*')
namesFiles =  system("ls bench_data/sizetimesnormal* | gawk 'match($0, /timesnormal([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):8 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultNormalSizeComparisonSortOnly.eps"
set title "Sort Rate Vs Array Size Normal Distribution : Sort Only"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizetimesnormal*')
namesFiles =  system("ls bench_data/sizetimesnormal* | gawk 'match($0, /timesnormal([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):6 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultNormalSizeComparisonTransInc.eps"
set title "Sort Rate Vs Array Size Normal Distribution : Sort + Transform"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizetimesnormal*')
namesFiles =  system("ls bench_data/sizetimesnormal* | gawk 'match($0, /timesnormal([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):7 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set terminal postscript enhanced color
set output "./plots/sortMultNormalSizeComparisonTot.eps"
set title "Sort Rate Vs Array Size Normal Distribution : Total CPU time"
set ylabel "Sort-Rate (M/s)"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizetimesnormal*')
namesFiles =  system("ls bench_data/sizetimesnormal* | gawk 'match($0, /timesnormal([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):9 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set yrange [0:52]
set linetype  5 lc rgb "green" lw 1
set terminal postscript enhanced color
set output "./plots/speedupnormalSTLSizeComparisonTot.eps"
set title "SpeedUp Over STL Vs Array Size (Normal) : Total CPU time"
set ylabel "SpeedUp"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizespeednormal*_*_STL_Sort*')
namesFiles =  system("ls bench_data/sizespeednormal*_*_STL_Sort* | gawk 'match($0, /speednormal([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):5 lt i with linespoints title sprintf("%s", word(namesFiles,i))

set yrange [0:*]
set terminal postscript enhanced color
set output "./plots/speedupnormalSTLSizeComparisonSort.eps"
set title "SpeedUp Over STL Vs Array Size (Normal) : Sort Only"
set ylabel "SpeedUp"
#set xtics rotate out

set xlabel "Array Size (Power of 2)"
datFiles = system('ls bench_data/sizespeednormal*_*_STL_Sort*')
namesFiles =  system("ls bench_data/sizespeednormal*_*_STL_Sort* | gawk 'match($0, /speednormal([A-Z]*[a-z]*)_([A-Z][a-z]*)/, a) {print a[1]a[2]}' ")

plot for [i=1:words(datFiles)] word(datFiles,i) using (log10($1)/log10(2)):2 lt i with linespoints title sprintf("%s", word(namesFiles,i))


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

