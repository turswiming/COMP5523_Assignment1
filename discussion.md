# part2 step 2
the gaussian filter can reduce noise caused by tree, grass and so on, with propered sigma (i.e sigma = 1) it can improve the quality of edgemap
# part2 step 5
high thresholds decide with should must appear on the final edge map. low thresholds decide how much weak edges can be considered as edge once connected with strong edges.
To extend exiting lines, we can lower the "low thresholds", to obtain new lines, we can lower the "high thresholds"