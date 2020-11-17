import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('queens.csv')

print(df.head())

dataset = df[['rent', 'bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway']]
labels = df['neighborhood']

x_train, x_test, y_train, y_test =  train_test_split(dataset, labels, test_size=0.2, random_state=100)




#predictions = classifier.predict(x_test)

k_list = list(range(1, 101))
scores = []
for i in k_list:
    classifier = KNeighborsClassifier(n_neighbors =  i)

    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    scores.append(score)

plt.plot(k_list, scores)
plt.xlabel("K")
plt.ylabel("Accuracies")
plt.show()

best_score = max(scores)
idx = scores.index(best_score)

best_k = k_list[idx]

print(best_k)

best_classifier = KNeighborsClassifier(n_neighbors= best_k)
best_classifier.fit(x_train, y_train)

predictions = best_classifier.predict(x_test)
score = best_classifier.score(x_test, y_test)

print(predictions)
print(score)