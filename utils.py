import os
import shutil

def delete_subfolders(target_folder):

    if os.path.exists(target_folder):
        for item in os.listdir(target_folder):
            item_path = os.path.join(target_folder, item)
            try:
                shutil.rmtree(item_path)  # delete directory
            except NotADirectoryError:
                os.unlink(item_path)  # delete files

        print(f"Sub folders of {target_folder} was deleted.")
    else:
        print(f"{target_folder} was not exist.")


def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"{directory_name} created.")
    else:
        print(f"{directory_name} aready exist.")