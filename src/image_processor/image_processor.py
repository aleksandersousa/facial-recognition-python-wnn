import pathlib
import os
from file.image_file import ImageFile


class ImageProcessor:
    def __init__(self) -> None:
        self.training_path = str(
            pathlib.Path().absolute()) + '/images/training/'
        self.testing_path = str(pathlib.Path().absolute()) + '/images/testing/'
        self.image_file = ImageFile()

        self.training_path = self.training_path.replace('\\', '/')
        self.testing_path = self.testing_path.replace('\\', '/')

    def shrink_pixels(self, img):
        pass

    def binarize_with_adaptative_threshold(self, img):
        pass

    def binarize_with_basic_threshold(self, img):
        pass

    def sort_func(e):
        return len(e)

    def process_images(self, path):
        for folder, sub_folders, _ in os.walk(path):
            sub_folders.sort(key=self.sort_func)

            for sub_folder in sub_folders:
                sub_folder_path = os.path.join(folder, sub_folder)
                files = os.listdir(sub_folder_path)
                files.sort(key=self.sort_func)

                for special_file in files:
                    img_path = src_path = os.path.join(
                        folder, sub_folder + '/' + special_file).replace("\\", "/")

                    img = self.image_file.read_img(path)
                    img_input = self.image_file.get_img_input(img)
                    rect_points = self.image_file.get_rect_points(
                        img, img_input)
                    crop_img = self.image_file.crop_img(rect_points, img_input)
