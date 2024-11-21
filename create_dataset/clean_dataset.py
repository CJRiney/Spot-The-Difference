# Written by ChatGPT

from pathlib import Path

datasets_path = Path("./datasets/eval")

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

# Rename directories to temporary unique names
temp_names = []
for index, dirpath in enumerate(sorted(datasets_path.iterdir()), start=0):
    if dirpath.is_dir():
        temp_name = datasets_path / f"temp_set_{index}"
        dirpath.rename(temp_name)
        temp_names.append(temp_name)
        print(f"Temporarily renamed {dirpath} to {temp_name}")

# Rename directories to final sequential names
for index, temp_path in enumerate(temp_names, start=0):
    final_name = datasets_path / f"set_{index}"
    temp_path.rename(final_name)
    print(f"Renamed {temp_path} to {final_name}")

    # Rename the negative images in the "negatives" directory
    negatives_dir = final_name / "negatives"
    if negatives_dir.is_dir():
        for neg_index, neg_file in enumerate(sorted(negatives_dir.iterdir()), start=1):
            if neg_file.is_file():
                new_neg_name = negatives_dir / f"neg{neg_index}{neg_file.suffix}"
                neg_file.rename(new_neg_name)
                print(f"Renamed {neg_file} to {new_neg_name}")
