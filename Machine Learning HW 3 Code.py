# -*- coding: utf-8 -*-
# Problem 1:
t = -0.5:0.001:0.5;
p = 4*exp(-200.*t.*t);
plot(t,p);
ylim([-1,5]);
title('Probabilistic Upper Bound');
xlabel('Epsilon');
ylabel('Bounding function');

# Problem 2:
t1 = 0:0.001:20;
t2 = -20:0.001:0;
t3 = -10:0.001:2;
t4 = 2:0.001:4;
t5 = 4:0.001:10;

f1 = exp(-t1);
f2 = 0 * ones(1, length(t2));
f3 = 0 * ones(1, length(t3));
f4 = 0.5 * ones(1, length(t4));
f5 = 0 * ones(1, length(t5));

hold on
plot(t3,f3);
plot(t4,f4);
plot(t5,f5);
title('PDF $$f_{X|Y}(x|1)$$','interpreter','latex');
xlabel('$$x$$','interpreter','latex');
ylabel('Probability: $$f_{X|Y}(x|1)$$','interpreter','latex');
ylim([-0.1,1]);
xlim([-7,7]);
\end{lstlisting}

\begin{lstlisting}
## Problem 3:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

np.random.seed(2020) # Set random seed so results are repeatable

n = 5000 # number of training points
k = 65 # number of neighbors to consider 

## Generate a simple 2D dataset
X, y = datasets.make_moons(n,'True',0.3)

## Create instance of KNN classifier
classifier = neighbors.KNeighborsClassifier(k,'uniform')
classifier.fit(X, y)

## Plot the decision boundary. 
# Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
h = .02  # step size in the mesh
x_delta = (X[:, 0].max() - X[:, 0].min())*0.05 # add 5% white space to border
y_delta = (X[:, 1].max() - X[:, 1].min())*0.05
x_min, x_max = X[:, 0].min() - x_delta, X[:, 0].max() + x_delta
y_min, y_max = X[:, 1].min() - y_delta, X[:, 1].max() + y_delta
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

## Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-NN classifier trained on %i data points" % (k,n))

## Show the plot
plt.show()

