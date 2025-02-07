from tabulate import tabulate
import argparse

valid_datasets = ('mushrooms', 'loans')
valid_methods = ('naive_bayes', 'decision_tree', 'random_forest', 'knn')

parser = argparse.ArgumentParser(description='Run a classification task on a dataset.')
parser.add_argument('-d', '--dataset', choices=valid_datasets, help='Dataset to use')
parser.add_argument('-m', '--method', choices=valid_methods, help='Classification method to use')

args = parser.parse_args()

dataset = args.dataset
method = args.method

if dataset is None:
	dataset = input(f'Enter the dataset to use ({', '.join(valid_datasets)}): ')
	if dataset not in valid_datasets:
		raise ValueError('Invalid dataset')

if method is None:
	method = input(f'Enter the classification method to use ({', '.join(valid_methods)}): ')
	if method not in valid_methods:
		raise ValueError('Invalid classification method')

print(f'Running classification task on {dataset} dataset using {method} method')

percentages = (0.15, 0.3, 0.5)

configurations = [
	('Label Encoder', False, False),
	('Label Encoder + OneHot Encoder', False, True),
	('Label Encoder + Standard Scaler', True, False),
	('Label Encoder + OneHot Encoder + Standard Scaler', True, True),
]
accuracy_table = []
confusion_matrix_table = []

dataset_fn = None
match dataset:
	case 'mushrooms':
		from mushrooms import mushrooms as dataset_fn
	case 'loans':
		from loans import loans as dataset_fn

for config_name, scaler, encoder in configurations:
	accuracy_row = [config_name]
	matrix_row = [config_name]

	for percentage in percentages:
		accuracy, matrix = dataset_fn(percentage, scaler, encoder, method)
		accuracy_row.append(accuracy)
		matrix_row.append(str(matrix))

	accuracy_table.append(accuracy_row)
	confusion_matrix_table.append(matrix_row)

headers = ['Configuration'] + [f'{p * 100:.0f}%' for p in percentages]

print('\nAccuracy Table:')
print(tabulate(accuracy_table, headers=headers, tablefmt='grid'))

print('\nConfusion Matrix Table:')
print(tabulate(confusion_matrix_table, headers=headers, tablefmt='grid'))
