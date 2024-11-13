from pathlib import Path

datasets_path = Path("./datasets")

# Check for empty directories or "set" directories with empty "negatives"
for set_dir in datasets_path.iterdir():
    if set_dir.is_dir():  # Check if it's a directory (set directory)
        # Check if the "negatives" directory exists and is empty
        negatives_dir = set_dir / "negatives"
        if negatives_dir.is_dir() and not any(negatives_dir.iterdir()):
            # If "negatives" is empty, delete the entire "set" directory
            try:
                # Delete any remaining files (positive, input, etc.)
                for file in set_dir.iterdir():
                    file.unlink()
                set_dir.rmdir()  # Delete the "set" directory itself
                print(f"Deleted 'set' directory due to empty 'negatives': {set_dir}")
            except Exception as e:
                print(f"Error while deleting {set_dir}: {e}")
        # Delete the directory if it is empty (including checking "negatives" emptiness)
        elif not any(set_dir.iterdir()):
            set_dir.rmdir()
            print(f"Deleted empty directory: {set_dir}")

# Rename directories in sequential order
# List remaining directories and sort by name to ensure sequential renaming
directories = [d for d in datasets_path.iterdir() if d.is_dir()]
for index, dirpath in enumerate(directories, start=0):
    new_name = datasets_path / f"set_{index}"
    dirpath.rename(new_name)
    print(f"Renamed {dirpath} to {new_name}")
