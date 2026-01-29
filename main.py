from perceptron import Perceptron
import matplotlib.pyplot as plt

samples = [
    # class +1
    [[6, 5],  1],
    [[5, 6],  1],
    [[7, 6],  1],
    [[6, 7],  1],
    [[8, 6],  1],
    [[7, 5],  1],
    [[5, 7],  1],
    [[8, 7],  1],
    [[6, 8],  1],
    [[7, 7],  1],

    # class -1
    [[5, 3], -1],
    [[2, 1], -1],
    [[1, 2], -1],
    [[2, 2], -1],
    [[3, 2], -1],
    [[2, 3], -1],
    [[3, 1], -1],
    [[1, 3], -1],
    [[4, 4], -1],
    [[3, 3], -1],
]


X_pos = [x for (x, y), label in samples if label == 1]
Y_pos = [y for (x, y), label in samples if label == 1]

X_neg = [x for (x, y), label in samples if label == -1]
Y_neg = [y for (x, y), label in samples if label == -1]

plt.scatter(X_pos, Y_pos)
plt.scatter(X_neg, Y_neg)

plt.xlabel("x1")
plt.ylabel("x2")

# perceptron object and training
p = Perceptron()
p.train(samples)

# plot the seperating hyperplane
print(p.weights)
w1, w2, b = p.weights

x_vals = [0, 9]
y_vals = [-(w1*x + b)/w2 for x in x_vals]

plt.plot(x_vals, y_vals, color='red')
plt.show()
