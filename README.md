# Evolutionary-Search-Strategies
Evolutionary search based optimization strategies for function minimization. Can be currently used for non-constrained continuous function optimization. Well-suited for teaching, benchmarking etc. An example to call the solvers is given in the main() in RCGA.py. 

1. Available solvers

  Currently, the package contains the following solvers:
  
    1. RCGA (Real-coded genetic algorithm)
    
2. Descriptions of the strategies
 
 2.1. RCGA
 
The RCGA implements the real-coded genetic algorithm. The package uses numpy to provide simple data structures. The RCGA makes use of Arithmetic crossover and Mäkinen-Periaux-Toivonen mutation strategies. The mutation probability is updated deterministically within generations.


 

