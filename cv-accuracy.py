train_data2 = random_crossval(iris_data_with_target,10,10)[3][0][0]
train_class2 = np.array(random_crossval(iris_data_with_target,10,10)[3][0][1])

test_data2 = random_crossval(iris_data_with_target,10,10)[3][1][0]
test_class2 = np.array(random_crossval(iris_data_with_target,10,10)[3][1][1])
bbc2 = BlackBoxClassifier(n_neighbors=10)
bbc2.fit(train_data2, train_class2)
acc2 = bbc2.score(test_data2, test_class2)
