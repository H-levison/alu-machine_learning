#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

x = np.arange(0, 11)  # Create x values from 0 to 10
plt.plot(x, y, color='red', linestyle='-')  # Plot as a solid red line
plt.xlim(0, 10)  # Ensure the x-axis ranges from 0 to 10
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Line Graph of y = x^3')
plt.savefig('line_plot.png') # Save the plot as an image
