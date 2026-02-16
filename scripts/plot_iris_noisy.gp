# gnuplot script (no external Python deps)
# Usage: gnuplot -persist scripts/plot_iris_noisy.gp
set terminal pngcairo size 900,600
set output "iris_noisy.png"

set datafile separator ","
# set key left bottom
set xlabel "Noise percent"
set ylabel "Accuracy"
# set yrange [0:1]
set grid
set key outside
set yrange [0.6:1.0]
set style line 1 lw 3 pt 7
set style line 2 lw 3 pt 5
set style line 3 lw 3 pt 9

plot "iris_noisy.csv" using 1:2 with linespoints ls 1 title "Tree (test)", \
     "" using 1:3 with linespoints ls 2 title "Rules (no prune)", \
     "" using 1:4 with linespoints ls 3 title "Rules (post-prune)"


#plot "iris_noisy.csv" using 1:2 with linespoints title "Tree (test)", \
 #    "iris_noisy.csv" using 1:3 with linespoints title "Rules (no prune, test)", \
  #   "iris_noisy.csv" using 1:4 with linespoints title "Rules (post-prune, test)"
