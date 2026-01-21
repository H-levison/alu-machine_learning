#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Data initialization (Rows: apples, bananas, oranges, peaches | Cols: Farrah, Fred, Felicia)
fruit = np.array([
    [10, 20, 30],
    [20, 15, 10],
    [5, 10, 15],
    [20, 5, 10]
])

# your code here
persons = ['Farrah', 'Fred', 'Felicia']
fruit_labels = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Initialize the bottom for the stacking
bottom_val = np.zeros(len(persons))

# Loop through each fruit row to create the stack
for i in range(len(fruit)):
    plt.bar(persons, fruit[i], width=0.5, bottom=bottom_val, color=colors[i], label=fruit_labels[i])
    # Update the bottom for the next layer
    bottom_val += fruit[i]

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.yticks(np.arange(0, 81, 10))
plt.legend()
plt.show()
