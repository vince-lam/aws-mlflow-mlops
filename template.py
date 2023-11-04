import os
from pathlib import Path
import logging
import typing

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = "aws-mlflow-mlops"

files_to_make = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "notebooks/experiments.ipynb",
    "templates/index.html",
]


def create_files(files_to_make: typing.List[str]) -> None:
    """
    Create directories and empty files based on a list of file paths.

    This function takes a list of file paths and creates the necessary directories
    and empty files. If the file already exists, it is left unchanged.

    Parameters:
    files_to_make (list[str]): A list of strings representing the file paths for which
                               directories and files should be created.

    Returns:
    None
    """
    for filepath_str in files_to_make:
        filepath = Path(filepath_str)
        filedir, filename = os.path.split(filepath)

        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir} for file: {filename}")

        if not filepath.exists() or filepath.stat().st_size == 0:
            with open(filepath, "w") as f:
                pass
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"File already exists: {filepath}")


if __name__ == "__main__":
    create_files(files_to_make)
