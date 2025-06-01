from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
# >>> type(iris)
# <class 'sklearn.utils._bunch.Bunch'>

# Feature, label split
X, y = iris.data[:, :2], iris.target
# >>> type(X), type(y)
# (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>)
# >>> len(X), len(y)
# (150, 150)

# Training-test set split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
# >>> len(X_train), len(X_test), len(y_train), len(y_test)
# (112, 38, 112, 38)

# Standardization / normalization
scaler = preprocessing.StandardScaler().fit(X_train)
# >>> scaler
# StandardScaler()

# Normalization of features
# >>> X_train[1], X_test[1]
# (array([4.9, 3.1]), array([6.7, 3.1]))
X_train_stz = scaler.transform(X_train)
X_test_stz = scaler.transform(X_test)
# >>> X_train_stz[1], X_test_stz[1]
# (array([-1.0271058 ,  0.08448757]), array([1.06445511, 0.08448757]))

# KNN instantiation, k=5
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
# >>> knn
# KNeighborsClassifier()

# KNN training / fit
knn.fit(X_train_stz, y_train)
# KNeighborsClassifier()

y_pred = knn.predict(X_test_stz)
# >>> y_pred
# array([1, 1, 0, 2, 1, 2, 0, 0, 1, 2, 2, 0, 2, 1, 2, 2, 1, 0, 2, 2, 0, 0,
#        1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1])
# >>> len(y_pred)
# 38

accuracy_score(y_test, y_pred)
# >>> accuracy_score(y_test, y_pred)
# 0.631578947368421