Reinforcement Learning - demonstrate the difficulties that sample-average 
methods have for nonstationary problems. 
Luxi Liu
Sep 2022

To compile: 
py a1.py

Parameters of the experiment can be set by the last line in the program:
runExperiment(num_steps=10000, num_experiments=1000, arms=10, var=1,
              rand_walk_var=0.01, alpha=0.2, epsilon=0.2)
I chose not to read parameters from input since there are many parameters,
I think it would be more straightforward to see them listed together.