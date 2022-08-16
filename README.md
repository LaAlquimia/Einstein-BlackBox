# Einstein-BlackBox


Einstein-BlackBox is a machine learning project thought for mathematical model inference given a set of data points,
you will never had to propose math models to predict data, Einstein-BackBox will do it for you.

Since math structure of formulas and fine tuning of parameters.

Einsten-BlackBox is based on Genetic Programming, where the mathematical formulas fight and evolve to predict data, creating random mathematical models, the sub algorithm is called Symbolic Regression, this is the first package where core is coded in C++ and you can use it directly in python.


Code example to use the module:

```
import sys
sys.path.append("build/temp.linux-x86_64-3.9")
import LinearSymbolicRegressor

import numpy as np

symreg_params = {
    # Individual Parameters
    "initial_depth": 1,
    "max_depth": 10,
    "p_arity_1": 0.,
    "p_arity_2": 1,
    "p_arity_3": 0.,

    # Mutation probabilities
    "p_xover": 0.5,
    "p_mutation": 0.4,
    "p_mutation_insert_node": 0.4,
    "p_mutation_delete_node": 0.1,
    "p_mutation_replication": 0.1,

    # Symbolic Regressor Parameters
    "n_best": 3,
    "n_programs": 5,
    "n_generations": 5,
    "stop_after_times": 30,
} 

model = LinearSymbolicRegressor.LinearSymbolicRegressor(symreg_params)

# LOAD YOUR DATASET (NUMPY ARRAYS)
x_train = np.array(range(0,10))
y_train = x_train * x_train

model.fit(y_train, x_train)

#THIS WILL SHOW YOU THE SOLUTIONS
for program in model.best_programs:
    program.print_program()
    print(program.fitness)
```