import pandas as pd
import json

# Read the CSV file
csv_filename = 'csv_file/test_final.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_filename)

# Convert each row to a dictionary and store in a list
samples_list = df.to_dict(orient='records')

# Save the list of dictionaries to a JSON file
json_filename = 'test_final.json'  # Replace with your desired JSON file path
with open(json_filename, 'w') as json_file:
    json.dump(samples_list, json_file, indent=4)
