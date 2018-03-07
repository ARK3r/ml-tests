from sklearn.tree import DecisionTreeClassifier

train_features = [[140, 1], [130, 1], [150, 0], [170, 0]]
train_labels = [0, 0, 1, 1]

clf = DecisionTreeClassifier()


clf.fit(train_features, train_labels)
test_features = [[160, 0]]
test_lables = [1]

print clf.predict(test_features)
