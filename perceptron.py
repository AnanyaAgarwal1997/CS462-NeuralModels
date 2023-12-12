import pandas as pd
import numpy as np

def perceptron_training(data, alpha=0.1, bias=-0.1):
    for index, row in data.iterrows():
        actual_output = 0.0
        for j in range(1, 7):
            actual_output += data.at[index, f'X{j}'] * data.at[index, f'W{j}']

        if actual_output + bias > 0:
            actual_output = 1
            data.at[index, 'Actual Output'] = 1
        else:
            actual_output = 0
            data.at[index, 'Actual Output'] = 0

        for j in range(1, 7):
            if data.at[index, 'Expected Output'] != actual_output:
                data.at[index, f'New W{j}'] = data.at[index, f'W{j}'] + ((data.at[index, 'Expected Output'] - actual_output) * alpha * data.at[index, f'X{j}'])
            else:
                data.at[index, f'New W{j}'] = data.at[index, f'W{j}']
            if index < 13:
                data.at[index + 1, f'W{j}'] = data.at[index, f'New W{j}']

    return data


training_data = {
    'X1': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'X2': [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    'X3': [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
    'X4': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
    'X5': [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    'X6': [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    'Expected Output': [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
}

for i in range(1, 7):
    col_name = f'W{i}'
    training_data[col_name] = [np.random.uniform(-0.5, 0.5)] + [np.nan] * (14 - 1)

training_data[f'Actual Output'] = [0] + [np.nan] * (14 - 1)

for i in range(1, 7):
    col_name = f'New W{i}'
    training_data[col_name] = [np.random.uniform(-0.5, 0.5)] + [np.nan] * (14 - 1)

df = pd.DataFrame(training_data)

rearranged_final_weights = perceptron_training(df)
print("Final Weights (Rearranged Data):")
print(rearranged_final_weights)
