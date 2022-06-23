import os
import random
import shutil
import pathlib
from PIL import Image

class ForestFireDataSet():
    """
    Class to load the forest fire dataset.
    """
    def __init__(self, data_dir, train_dir = '', test_dir = '', balance_dataset=True, img_height=128, img_width=128):
        if balance_dataset:
            self._balance_dataset(data_dir)
        else:
            self.train_data_dir = train_dir
            self.test_data_dir = test_dir

        self.img_height = img_height
        self.img_width = img_width

    def _balance_dataset(self, data_dir):
        self._clear_dir(os.path.join(data_dir, 'train'))
        self._clear_dir(os.path.join(data_dir, 'test'))

        subdir_file_counts = self._count_files_in_subdirs(data_dir)
        min_file_counts = min(subdir_file_counts.values())
        train_file_counts, test_file_counts = int(min_file_counts * 0.7), int(min_file_counts * 0.3)

        train_dir = self._clear_dir_and_recreate(os.path.join(data_dir, 'train'))
        test_dir = self._clear_dir_and_recreate(os.path.join(data_dir, 'test'))

        for subdir in subdir_file_counts:
            train_subdir = self._clear_dir_and_recreate(os.path.join(train_dir, subdir.name))
            test_subdir = self._clear_dir_and_recreate(os.path.join(test_dir, subdir.name))

            dir_content = list(pathlib.Path(os.path.join(data_dir, subdir.name)).glob("*"))
            dir_content = self._remove_broken_images(dir_content)
            
            random.shuffle(dir_content)

            for img in dir_content[:train_file_counts]:
                shutil.copyfile(str(img), os.path.join(train_subdir, img.name))

            for img in dir_content[:test_file_counts]:
                shutil.copyfile(str(img), os.path.join(test_subdir, img.name))

        self.train_data_dir = train_dir
        self.test_data_dir = test_dir

    def _remove_broken_images(self, dir_content):
        r_dir_content = []
        for fileimage in dir_content:
            try:
                with Image.open(fileimage) as img:
                    img.verify()
                    r_dir_content.append(fileimage)
            except Exception:
                pass

        return r_dir_content

    def _count_files_in_subdirs(self, dir):
        dir = pathlib.Path(dir)
        subdirs = {}
        for subdir in dir.iterdir():
            subdirs.update({
                subdir: len(list(dir.glob(f"{subdir.name}/*")))
            })
        return subdirs
    
    def _clear_dir(self, dir):
        if os.path.exists(dir): 
            shutil.rmtree(dir)

    def _clear_dir_and_recreate(self, dir):
        self._clear_dir(dir)
        return self._create_dir_if_not_exists(dir)

    def _create_dir_if_not_exists(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir