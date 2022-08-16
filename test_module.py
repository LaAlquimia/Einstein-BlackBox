import sys
sys.path.append("build/temp.linux-x86_64-3.9")
import numpy as np
import LinearSymbolicRegressor

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
print("INIT MODEL")
for program in model.programs:
    program.print_program()

    for gen in program.genes:
        print(gen.operation)

x_train = np.array(range(0,10))
# Simulate experiment points
y_train = x_train * x_train

x_train = np.reshape(x_train,(10,1))
y_train = np.reshape(y_train,(10,1))

print(x_train, y_train)
print(x_train.shape, y_train.shape)
print(type(x_train), type(y_train))

model.fit(y_train, x_train)

for program in model.best_programs:
    program.print_program()
    print(program.fitness)
