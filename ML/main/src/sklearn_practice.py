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

# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

# K-Means
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0)

# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)

# Support Vector Machines (SVM)
from sklearn.svm import SVC
svc = SVC(kernel='linear')

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# KNN
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

### Model Fitting
# Supervised learning
lr.fit(X, y)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
# Unsupervised Learning
k_means.fit(X_train)
pca_model = pca.fit_transform(X_train)

### Prediction
# Supervised Estimators
y_pred = svc.predict(np.random.random((2,5)))
y_pred = lr.predict(X_test)
y_pred = knn.predict_proba(X_test)
# Unsupervised Estimators
y_pred = k_means.predict(X_test)

# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)

# Normalization
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)

### Classification metrics
# Accuracy Score
knn.score(X_test, y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

### Regression metrics
# Mean Absolute Error
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2]
mean_absolute_error(y_true, y_pred)
#  Mean Squared Error
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
# R² Score
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)