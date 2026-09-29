import os
from pathlib import Path


def print_tree(root_path):

    root = Path(root_path).resolve()

    print("\n" + "=" * 100)
    print("PROJECT DIRECTORY STRUCTURE")
    print("=" * 100)
    print(f"\nRoot : {root}\n")

    directories = []
    files = []

    for current_path, dir_names, file_names in os.walk(root):

        dir_names.sort()
        file_names.sort()

        relative = Path(current_path).relative_to(root)

        if str(relative) == ".":
            directory = "."
        else:
            directory = relative.as_posix()

        directories.append(directory)

        for file in file_names:

            file_path = Path(current_path) / file

            files.append(file_path.relative_to(root).as_posix())

    print("=" * 100)
    print("DIRECTORIES")
    print("=" * 100)

    for directory in directories:
        print(directory)

    print("\n" + "=" * 100)
    print("FILES")
    print("=" * 100)

    for file in files:
        print(file)

    print("\n" + "=" * 100)
    print(f"Total Directories : {len(directories)}")
    print(f"Total Files       : {len(files)}")
    print("=" * 100)


if __name__ == "__main__":

    print_tree(".")