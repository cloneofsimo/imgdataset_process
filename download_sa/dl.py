import os
import multiprocessing
import pandas as pd
import subprocess

csv_file = 'cdn_filelists.csv'
target_folder = '/jfs/sam'

# Ensure the target folder exists
os.makedirs(target_folder, exist_ok=True)

# Read the CSV into a DataFrame
df = pd.read_csv(csv_file, sep='\t')
print(df.head())

# Function to download a file using wget
def download_file(row):
    file_name = row['file_name']
    cdn_link = row['cdn_link']
    target_path = os.path.join(target_folder, file_name)
    try:
        subprocess.run(['wget', '-O', target_path, cdn_link], check=True)
        print(f'Successfully downloaded {file_name}')
    except subprocess.CalledProcessError as e:
        print(f'Failed to download {file_name}: {e}')

# Get the list of rows as dictionaries
rows = df.to_dict(orient='records')

# Create a pool of workers and download files
if __name__ == '__main__':
    with multiprocessing.Pool(32) as pool:
        pool.map(download_file, rows)
