A Python implementation of Deep Belief Networks based on the scikit-learn library

### Installation ###
pip install git+https://github.com/leoclassic/deep-belief-network-sklearn.git

### Example ###
from dbn_sklearn import DBN\
from sklearn.datasets import load_iris\
from sklearn.preprocessing import MinMaxScaler

data = load_iris()\
X = data.data\
T = data.target

X = MinMaxScaler().fit_transform(X)

model = DBN(hidden_layer_sizes=(100,100))\
model.fit(X,T)\
print(model.score(X,T))
