import pandas as pd

# Read the Excel file
excel_file_path = 'xlsx_file/test_final.xlsx'  # Replace with your xlsx file path
df = pd.read_excel(excel_file_path)

# Save the DataFrame to a CSV file
csv_file_path = 'test_final.csv'  # Replace with your desired CSV file path
df.to_csv(csv_file_path, index=False)

print(f"Excel file {excel_file_path} has been converted to {csv_file_path} successfully.")
