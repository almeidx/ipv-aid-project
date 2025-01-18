import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix

def loans(percentage, with_standard_scaler, with_one_hot_encoder, method='naive_bayes'):
		df = pd.read_csv('./data/loan-data.csv')

		df.drop('Loan_ID', axis=1, inplace=True)
		df.loc[df.Gender.isnull(), 'Gender'] = df.Gender.mode()[0]
		df.loc[df.Married.isnull(), 'Married'] = df.Married.mode()[0]
		df.loc[df.Dependents.isnull(), 'Dependents'] = df.Dependents.mode()[0]
		df.loc[df.Self_Employed.isnull(), 'Self_Employed'] = df.Self_Employed.mode()[0]
		df.loc[df.Loan_Amount.isnull(), 'Loan_Amount'] = df.Loan_Amount.mean()
		df.loc[df.Loan_Amount_Term.isnull(), 'Loan_Amount_Term'] = df.Loan_Amount_Term.mode()[0]
		df.loc[df.Credit_History.isnull(), 'Credit_History'] = df.Credit_History.mode()[0]

		descriptive = df.iloc[:, 0:11].values
		target = df.iloc[:, 11].values

		labelEncoder = LabelEncoder()
		for i in range(descriptive.shape[1]):
			descriptive[:, i] = labelEncoder.fit_transform(descriptive[:, i])

		if with_one_hot_encoder:
			column_transformer = ColumnTransformer(
				transformers=[('encoder', OneHotEncoder(sparse_output=False), [9])],
				remainder='passthrough'
			)
			descriptive = column_transformer.fit_transform(descriptive)

		descriptive_train, descriptive_test, target_train, target_test = train_test_split(
			descriptive, target, test_size=percentage, random_state=0, stratify=target
		)

		if with_standard_scaler:
			standard_scaler = StandardScaler()
			descriptive_train[:, :] = standard_scaler.fit_transform(descriptive_train[:, :])
			descriptive_test[:, :] = standard_scaler.transform(descriptive_test[:, :])

		match method:
			case 'naive_bayes':
				from sklearn.naive_bayes import GaussianNB
				classifier = GaussianNB()
			case 'knn':
				from sklearn.neighbors import KNeighborsClassifier
				classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
			case 'decision_tree':
				from sklearn.tree import DecisionTreeClassifier
				classifier = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=3)
			case 'random_forest':
				from sklearn.ensemble import RandomForestClassifier
				classifier = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0, max_depth=3)

		classifier.fit(descriptive_train, target_train)
		prediction = classifier.predict(descriptive_test)

		return accuracy_score(target_test, prediction)
