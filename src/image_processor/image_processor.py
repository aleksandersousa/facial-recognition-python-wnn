import pathlib
import os
import cv2
from file.image_file import ImageFile


class ImageProcessor:
    def __init__(self) -> None:
        self.image_file = ImageFile()

        self.training_path = str(
            pathlib.Path().absolute()) + '/images/training/'
        self.testing_path = str(pathlib.Path().absolute()) + '/images/testing/'

        self.training_path = self.training_path.replace('\\', '/')
        self.testing_path = self.testing_path.replace('\\', '/')

    def process_training_images(self, shrink_pixels=None, shrink_size=None, adaptative_threshold=None):
        if shrink_pixels:
            assert shrink_size is not None, "shrink_size is mandatory when passing shrink_pixels"

        self.process_images(self.training_path, shrink_pixels,
                            shrink_size, adaptative_threshold)

    def process_testing_images(self, shrink_pixels=None, shrink_size=None, adaptative_threshold=None):
        if shrink_pixels:
            assert shrink_size is not None, "shrink_size is mandatory when passing shrink_pixels"

        self.process_images(self.testing_path, shrink_size,
                            shrink_pixels, adaptative_threshold)

    # private

    def binarize_with_adaptative_threshold(self, img, shrink_pixels=None, shrink_size=None):
        binarized_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 15, 7)

        cv2.imshow('Teste', binarized_img)
        cv2.waitKey(0)

        binarized_arr = binarized_img.flatten()

        if shrink_pixels:
            binarized_arr = self._shrink_pixels(binarized_arr, shrink_size)

        return binarized_arr

    def binarize_with_basic_threshold(self, img_array, shrink_pixels=None, shrink_size=None):
        binarized_arr = []

        if shrink_pixels:
            img_array = self._shrink_pixels(img_array, shrink_size)

        luminance_average = self._average(img_array)

        for pixel in img_array:
            if pixel > luminance_average:
                binarized_arr.append(1)
            else:
                binarized_arr.append(0)

        return binarized_arr

    def process_images(self, path, shrink_pixels=None, shrink_size=None, adaptative_threshold=None):
        def sort_func(e):
            return len(e)

        for folder, sub_folders, _ in os.walk(path):
            sub_folders.sort(key=sort_func)

            for sub_folder in sub_folders:
                sub_folder_path = os.path.join(folder, sub_folder)
                files = os.listdir(sub_folder_path)
                files.sort(key=sort_func)

                for special_file in files:
                    img_path = os.path.join(
                        folder, sub_folder + '/' + special_file).replace("\\", "/")

                    gray_img = self.image_file.get_img(img_path)

                    if adaptative_threshold:
                        print(len(self.binarize_with_adaptative_threshold(
                            gray_img, shrink_pixels, shrink_size)))
                    else:
                        print(len(self.binarize_with_basic_threshold(
                            gray_img.flatten(), shrink_pixels, shrink_size)))

                    break
                break
            break

    def _average(self, arr):
        return sum(arr) / len(arr)

    def _shrink_pixels(self, img_array, shrink_size):
        counter = 1
        reduced_img_arr = []
        pixel_sum = 0

        for pixel_fragment in img_array:
            pixel_sum += pixel_fragment

            if counter == shrink_size:
                reduced_img_arr.append(pixel_sum / shrink_size)
                counter = 1
                pixel_sum = 0
            else:
                counter += 1

        return reduced_img_arr
