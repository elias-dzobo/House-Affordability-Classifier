import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('queens.csv')

def classify(col):

    if col <= 4000:
        return 'Affordable'
    else:
        return 'Expensive'

df['class'] = df['rent'].apply(classify)

data = df[['bedrooms', 'bathrooms', 'size_sqft']]
labels = df['class']

k_list = list(range(1, 101))
scores = []

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 100)



for i in k_list:
    classifier = KNeighborsClassifier(n_neighbors= i)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    scores.append(score)

plt.plot(k_list, scores)
plt.show()

best_score = max(scores)
idx = scores.index(best_score)
best_k = k_list[idx]

print(best_k, best_score)

final_classifier = KNeighborsClassifier(n_neighbors= best_k)
final_classifier.fit(x_train, y_train)

prediction = final_classifier.predict(x_test)
print(prediction)