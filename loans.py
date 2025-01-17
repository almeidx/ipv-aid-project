import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

def loans(percentage, with_standard_scaler, with_one_hot_encoder, method='naive_bayes'):
	db = pd.read_csv('./data/loan-data.csv', header=None)

	print(db.describe())

	exit()

	# db.loc[db[11] == '?', 11] = 'b'

	descriptive = db.iloc[:, 1:].values
	target = db.iloc[:, 0].values

	labelEncoder = LabelEncoder()

	if with_one_hot_encoder:
		indices_to_encode = [0, 7, 16, 18]
		for index in indices_to_encode:
			descriptive[:, index] = labelEncoder.fit_transform(descriptive[:, index])

		column_transformer = ColumnTransformer(
			transformers=[(
				'encoder',
				OneHotEncoder(sparse_output=False),
				(1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21)
			)],
			remainder='passthrough'
		)
		descriptive = column_transformer.fit_transform(descriptive)
	else:
		for i in range(descriptive.shape[1]):
			descriptive[:, i] = labelEncoder.fit_transform(descriptive[:, i])

	descriptive_train, descriptive_test, target_train, target_test = train_test_split(
		descriptive, target, test_size=percentage, random_state=0, stratify=target
	)

	# descriptive_train = descriptive_train.toarray() if with_one_hot_encoder else descriptive_train
	# descriptive_test = descriptive_test.toarray() if with_one_hot_encoder else descriptive_test

	if with_standard_scaler:
		standard_scaler = StandardScaler()
		descriptive_train[:, 1:3] = standard_scaler.fit_transform(descriptive_train[:, 1:3])
		descriptive_test[:, 1:3] = standard_scaler.transform(descriptive_test[:, 1:3])

		descriptive_train[:, 5:7] = standard_scaler.fit_transform(descriptive_train[:, 5:7])
		descriptive_test[:, 5:7] = standard_scaler.transform(descriptive_test[:, 5:7])

		descriptive_train[:, 9:15] = standard_scaler.fit_transform(descriptive_train[:, 9:15])
		descriptive_test[:, 9:15] = standard_scaler.transform(descriptive_test[:, 9:15])

		descriptive_train[:, 17:] = standard_scaler.fit_transform(descriptive_train[:, 17:])
		descriptive_test[:, 17:] = standard_scaler.transform(descriptive_test[:, 17:])

	match method:
		case 'naive_bayes':
			from sklearn.naive_bayes import GaussianNB
			classifier = GaussianNB()
		case 'knn':
			from sklearn.neighbors import KNeighborsClassifier
			classifier = KNeighborsClassifier(n_neighbors=51, metric='minkowski', p=2)
		case 'decision_tree':
			from sklearn.tree import DecisionTreeClassifier
			classifier = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=3)
		case 'random_forest':
			from sklearn.ensemble import RandomForestClassifier
			classifier = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0, max_depth=5)

	classifier.fit(descriptive_train, target_train)
	prediction = classifier.predict(descriptive_test)

	accuracy = accuracy_score(target_test, prediction)

	return accuracy
