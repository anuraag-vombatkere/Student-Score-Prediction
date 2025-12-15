import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('student_scores.csv')

Hours = data['Hours'].values
Scores = data['Scores'].values

x_train, x_test, y_train, y_test = train_test_split(Hours, Scores, test_size=0.4, random_state=42)

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

plt.scatter(x_test, y_test, marker = 'o', color = 'blue', label='Actual Scores')
plt.plot(x_test, predictions, linewidth=2, color = 'red', label = 'Predicted Scores')
plt.xlabel('Hours student has studied')
plt.ylabel('Score of student')
plt.title('Predicting student\'s score based on number of hours studied')
plt.legend()
plt.show()
