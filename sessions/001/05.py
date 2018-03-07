from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(iris.data, iris.target, test_size=0.4)

from sklearn.tree import DecisionTreeClassifier
treeCLF = DecisionTreeClassifier()
treeCLF.fit(train_data, train_labels)
treePred = treeCLF.predict(test_data)

from sklearn.neighbors import KNeighborsClassifier
neighCLF = KNeighborsClassifier(n_neighbors=8)
neighCLF.fit(train_data, train_labels)
neighPred = neighCLF.predict(test_data)

from sklearn.metrics import accuracy_score
print accuracy_score(treePred, test_labels)
print accuracy_score(neighPred, test_labels)
