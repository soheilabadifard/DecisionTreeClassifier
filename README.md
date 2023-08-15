# DecisionTreeClassifier

In this repository, The Decision Tree Classifier which uses pre pruning is implemented from scratch.
The experiments are on the "Thyroid data set" which is taken from the UCI repository.

The details of this data set are given as follows:
* This data set contains separate training (“ann-train.data”) and test (“ann-test.data”) sets.
* The training set contains 3772 instances and the test set contains 3428 instances.
* There are a total of 3 classes.
* In the data files, each line corresponds to an instance that has 21 features (15 binary and 6 continuous features) and 1 class label.
* In the last part of the code  decision tree is constructed by considering the cost of using (extracting) the features. The cost of using each feature is given in another file (“ann-thyroid.cost”). It does not include the cost of the 21st feature because it is a combination of the other features.
* The 21st feature is defined using the 19th and 20th features. This means that you do not incur additional costs for this feature if the 19th and 20th features have already been extracted. Otherwise, you have to pay for the cost of the unextracted feature(s).
