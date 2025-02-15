# importing the Python module
import sklearn

# importing the dataset
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# looking at the data
print(label_names)

print(labels)

print(feature_names)

print(features)

# importing the function
from sklearn.model_selection import train_test_split

# splitting the data
train, test, train_labels, test_labels = train_test_split(features, labels,
                                       test_size = 0.33, random_state = 42)


# importing the module of the machine learning model
from sklearn.naive_bayes import GaussianNB

# initializing the classifier
gnb = GaussianNB()

# training the classifier
model = gnb.fit(train, train_labels)


# making the predictions
predictions = gnb.predict(test)

# printing the predictions
print(predictions)


# importing the accuracy measuring function
from sklearn.metrics import accuracy_score

# evaluating the accuracy
print(f"{accuracy_score(test_labels, predictions)*100}%")
