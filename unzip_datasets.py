import os
import sys

data_path = "data"

if not os.path.isdir(data_path):
    try:
        from datasets import load_dataset
        print("The 'data' directory does not exist. Downloading the 'hazyresearch/evaporate' dataset...")
        dataset = load_dataset("hazyresearch/evaporate")
        print("Download complete. Please check the 'data' directory.")
    except ImportError:
        print("Error: The 'datasets' library is not installed. Please install it with 'pip install datasets'.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        sys.exit(1)

print(f"Looking for datasets in: {os.path.abspath(data_path)}")

try:
    sub_directories = os.listdir(data_path)
except OSError as e:
    print(f"Error: Could not read directory '{data_path}': {e}")
    sys.exit(1)

unzipped_count = 0
found_archives = 0

for dir_name in sub_directories:
    sub_path = os.path.join(data_path, dir_name)
    if os.path.isdir(sub_path):
        archive_path = os.path.join(sub_path, 'docs.tar.gz')
        
        if os.path.exists(archive_path):
            found_archives += 1
            print(f"Found archive: {archive_path}")
            print(f"Unzipping...")
            
            # Build the unzip command, quoting paths to handle potential spaces
            command = f'tar -xvf "{archive_path}" -C "{sub_path}"'
            
            # result = os.system(command)
            result = os.system(command + " > /dev/null 2>&1")
            
            if result == 0:
                print(f"Successfully unzipped {archive_path}")
                unzipped_count += 1
            else:
                print(f"Error unzipping {archive_path}. tar command returned exit code {result}.")
                print("Please make sure 'tar' is installed and available in your system's PATH.")

if found_archives == 0:
    print(f"No 'docs.tar.gz' files were found in the subdirectories of '{data_path}'.")
else:
    print(f"\nProcessing complete. Found {found_archives} archives, successfully unzipped {unzipped_count}.")
