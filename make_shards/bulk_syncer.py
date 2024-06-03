import os
import shutil
import time

# Fetch environment variables for source and destination directories
source_dir = os.getenv('SOURCE_DIR', './mds_original/')
dest_dir = os.getenv('DEST_DIR', '/jfs/mds_original/')

# Time in seconds to check if the file size is not growing
INACTIVITY_THRESHOLD = 60

def get_file_size(file_path):
    try:
        return os.path.getsize(file_path)
    except OSError:
        return -1

def move_file(src, dst):
    # Create necessary directories for the destination path
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Move the file from src to dst
    shutil.move(src, dst)

def track_and_move_files(src_dir, dst_dir):
    file_sizes = {}

    while True:
        new_file_sizes = {}
        all_files_processed = True

        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, src_dir)
                dest_path = os.path.join(dst_dir, relative_path)
                
                file_size = get_file_size(file_path)
                new_file_sizes[file_path] = file_size

                if file_path not in file_sizes or file_sizes[file_path] != file_size:
                    all_files_processed = False

        if all_files_processed and new_file_sizes:
            for file_path in new_file_sizes:
                relative_path = os.path.relpath(file_path, src_dir)
                dest_path = os.path.join(dst_dir, relative_path)
                move_file(file_path, dest_path)
            break

        file_sizes = new_file_sizes
        time.sleep(INACTIVITY_THRESHOLD)

if __name__ == '__main__':
    track_and_move_files(source_dir, dest_dir)
