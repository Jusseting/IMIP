import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import random as rd
import seaborn as sns
import numpy as np

def C(X):
    K=[]
    KK=[]
    for x in X:
        K.append(x[0])
        KK.append(x[1])
    return K,KK

inter=[[-0.5,0.5],[-0.5,0.5]]
n=100
X=[]
y=[]
X1=[]
X2=[]

def f(x):
    if x[0]*(x[1]-0.1)*(x[1]+0.5)>0:
        return 1
    else:
        return 0
    
for i in range(n):
    X.append([rd.random()-0.5,rd.random()-0.5])
    y.append(f(X[i]))

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12,8)
sns.scatterplot(x = C(X)[0], y=C(X)[1], hue=y)
plt.show()

k=20
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each

c1=C(X)[0]
c2=C(X)[1]
x_min, x_max = min(c1) - 0.1, max(c1) + 0.1
y_min, y_max = min(c2) - 0.1, max(c2) + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)
plt.plot(c1, c2, "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "Répartition des clusters"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

index=kmeans.predict(X)

clusters_points=[[] for i in range(k)]

for i in range(len(X)):
    clusters_points[index[i]].append(X[i])
    
clusters=[]

for cluster in clusters_points:
    if len(cluster)>1:
        clusters.append([min(C(cluster)[0]),max(C(cluster)[0]),min(C(cluster)[1]),max(C(cluster)[1])])

def test_cluster(clusters,f,N):
    score=[]
    for cluster in clusters:
        succes=0
        for i in range(N):
            if f([rd.uniform(cluster[0],cluster[1]),rd.uniform(cluster[2],cluster[3])])==1:
                succes+=1
        score.append(succes)
    return clusters[np.argmax(score)]

print(test_cluster(clusters,f,10000))