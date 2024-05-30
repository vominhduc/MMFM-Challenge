import os
import json
import csv

# Directory containing JSON files
json_dir = 'inference_results_final/'
# Output CSV file
output_csv = 'merged_output.csv'

# List to hold rows for the CSV
rows = []

# Iterate over all files in the JSON directory
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        json_path = os.path.join(json_dir, filename)
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            for id, pred in data.items():
                rows.append({'id': id, 'pred': pred, 'split': 'public'})

# Write rows to the CSV file
with open(output_csv, 'w', newline='') as csv_file:
    fieldnames = ['id', 'pred', 'split']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(rows)

print(f"Data has been successfully merged into {output_csv}")
