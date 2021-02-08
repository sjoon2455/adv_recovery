import matplotlib.pyplot as plt

x1 = [1, 2, 3]
y1 = [0.62, 0.47, 0.17]
# plotting the line 1 points
plt.plot(x1, y1, 'go--', color='black', label="Original")
# line 2 points
x2 = [1, 2, 3]
y2 = [0.95, 0.86, 0.40]
plt.plot(x2, y2, 'go--', color='blue', label="Task 1")

x3 = [1, 2, 3]
y3 = [1.0, 0.87, 0.49]
plt.plot(x3, y3, 'go--', color='purple', label="Task2")

# plotting the line 2 points
plt.xlabel('epsilon')
# Set the y axis label of the current axis.
plt.ylabel('Accuracy')

plt.xticks([1, 2, 3], [0.01, 0.1, 0.8])

# Set a title of the current axes.
plt.title('Accuracy of each model')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
