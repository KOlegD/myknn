import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[7.9, 1.9, 3, 0.4]])
prediction = knn.predict(X_new)
print(f"Прогноз: {prediction}")
print(f"Спрогнозированная метка: {iris_dataset['target_names'][prediction]}")
knnPickle = open('knnpickle_file', 'wb')
pickle.dump(knn, knnPickle)

# loaded_model = pickle.load(open('knnpickle_file', 'rb'))
# result = loaded_model.predict(X_test)
