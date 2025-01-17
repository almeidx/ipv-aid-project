from tabulate import tabulate
import argparse

parser = argparse.ArgumentParser(description="Run a classification task on a dataset.")
parser.add_argument("dataset", choices=("mushrooms", "loans"), help="Dataset to use")
parser.add_argument("method", choices=('naive_bayes', 'knn', 'decision_tree', 'random_forest'), help=f"Classification method to use")

args = parser.parse_args()

dataset = args.dataset
method = args.method

print(f'Running classification task on {dataset} dataset using {method} method')

percentages = (0.15, 0.3, 0.5)


configurations = [
	("Label Encoder", False, False),
	("Label Encoder + OneHot Encoder", False, True),
	("Label Encoder + Standard Scaler", True, False),
	("Label Encoder + OneHot Encoder + Standard Scaler", True, True),
]
table = []

if dataset == "mushrooms":
	from mushrooms import mushrooms

	for config_name, scaler, encoder in configurations:
		row = [config_name]
		for percentage in percentages:
			row.append(mushrooms(percentage, scaler, encoder, method))
		table.append(row)
elif dataset == "loan":
	from loans import loans

	for config_name, scaler, encoder in configurations:
		row = [config_name]
		for percentage in percentages:
			row.append(loans(percentage, scaler, encoder, method))
		table.append(row)

headers = ["Configuration"] + [f"{p * 100:.0f}%" for p in percentages]

print(tabulate(table, headers=headers, tablefmt="grid"))
