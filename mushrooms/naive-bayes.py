import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

db = pd.read_csv('../agaricus-lepiota.csv', header=None)

descriptive = db.iloc[:, 1:].values
target = db.iloc[:, 0].values

db.loc[db[12] == '?', 11] = 'b'

labelEncoder = LabelEncoder()
descriptive[:, 0] = labelEncoder.fit_transform(descriptive[:, 0])

column_transformer = ColumnTransformer(
	transformers=[(
		'encoder',
		OneHotEncoder(),
		(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
	)],
	remainder='passthrough'
)
descriptive = column_transformer.fit_transform(descriptive)

for percentage in (0.15, 0.30, 0.50):
	descriptive_train, descriptive_test, target_train, target_test = train_test_split(
		descriptive, target, test_size=percentage, random_state=0
	)

	descriptive_train = descriptive_train.toarray()
	descriptive_test = descriptive_test.toarray()

	standard_scaler = StandardScaler()
	descriptive_train[:, :] = standard_scaler.fit_transform(descriptive_train[:, :])
	descriptive_test[:, :] = standard_scaler.transform(descriptive_test[:, :])

	classifier = GaussianNB()
	classifier.fit(descriptive_train, target_train)
	prediction = classifier.predict(descriptive_test)

	accuracy = accuracy_score(target_test, prediction)
	matrix = confusion_matrix(target_test, prediction)

	print(f'Percentage: {percentage}')
	print(f'Accuracy: {accuracy}')
	print(f'Confusion Matrix:\n{matrix}')
