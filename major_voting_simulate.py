import json
import numpy as np

import os

json_files_string = """
combined_bin_train_complex_aug_8/BERT_Difficulty_type3_5e6_warm1_decay01_bs16_epo15_aug8
train_final_type3_nlpaug/BERT_Difficulty_type3_5e6_warm3_decay01_bs16_epo20_split10_aug
"""
# output_file_path = 'major_voting/results_PubMedBert_Difficulty_type2_all.json'
output_file_path = 'temp.json'

# List of JSON file paths
json_files_dir = json_files_string.split('\n')

json_files = []
for json_f_d in json_files_dir:
    if json_f_d == '':
        continue
    path_real = os.path.join(json_f_d, 'test_results.json')
    json_files.append(path_real)

# Dictionary to store true values and lists of predicted values per instance
data_dict = {}

# Load data from all specified JSON files
for file_path in json_files:
    with open(file_path, 'r') as file:
        data = json.load(file)
        # Process each instance in the file
        for idx, item in enumerate(data):
            if isinstance(item, dict):  # Ensure it's an instance with prediction data
                if idx not in data_dict:
                    data_dict[idx] = {
                        'text': item['text'],
                        'true_value': item['true_value'],
                        'predicted_values': []
                    }
                data_dict[idx]['predicted_values'].append(item['predicted_value'])

# Create new data with averaged predicted values
averaged_data = []

true_values = []
average_predicted_values = []

for idx, values in data_dict.items():
    avg_predicted = np.mean(values['predicted_values'])
    averaged_instance = {
        'text': values['text'],
        'true_value': values['true_value'],
        'predicted_value': avg_predicted,
        'distance': abs(values['true_value'] - avg_predicted)
    }
    averaged_data.append(averaged_instance)
    
    # Collect true values and averaged predictions for RMSE calculation
    true_values.append(values['true_value'])
    average_predicted_values.append(avg_predicted)

# Calculate RMSE
rmse = np.sqrt(np.mean([(true - pred) ** 2 for true, pred in zip(true_values, average_predicted_values)]))

# Save the new JSON file with averaged predicted values
with open(output_file_path, 'w') as outfile:
    json.dump(averaged_data, outfile, indent=4)

# Print the results
print(f"RMSE: {rmse}")
print(f"Averaged results saved to {output_file_path}")