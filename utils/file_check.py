import os


def check_and_build_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
