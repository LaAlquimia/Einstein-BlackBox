from email import header
import sys
sys.path.append("build/temp.linux-x86_64-3.9")
import numpy as np
import LinearSymbolicRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from parameters import symreg_params

df = pd.read_fwf('gr_MD.dat', header = None)
df.columns = ['x' , 'y', 'z']
df.to_csv("TuringBot.csv")
print(df)


model = LinearSymbolicRegressor.LinearSymbolicRegressor(symreg_params)

# Load points
x_train = df.x.values
y_train = df.y.values
# x_train = np.reshape(x_train,(10,1))
# y_train = np.reshape(y_train,(10,1))

# print(x_train, y_train)
# print(x_train.shape, y_train.shape)
# print(type(x_train), type(y_train))

model.fit(y_train, x_train)
print()

for program in model.best_programs:
    program.print_program()
    print(program.fitness)
    plt.clf()
    plt.plot(df.x, df.y)
    plt.plot(x_train, program.compute_program(x_train), label = program.print_program())
    plt.show()
