import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("iris.csv")

input_data = dataset.iloc[:, :-1]
target = dataset.iloc[:, -1]

encoder = LabelEncoder()
target = encoder.fit_transform(target)

x_train, x_test, y_train, y_test = train_test_split(
    input_data,
    target,
    test_size=0.2,
    random_state=0
)

model = GaussianNB()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("\nActual Output:")
print(encoder.inverse_transform([y_test[0]]))

print("\nPredicted Output:")
print(encoder.inverse_transform([predictions[0]]))

print("\nAccuracy:", accuracy)