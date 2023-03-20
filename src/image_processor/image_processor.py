import pathlib
import os
import cv2
import csv

from file.image_file import ImageFile

CSV_HEADER = ['label', 'input_bits']


class ImageProcessor:
    def __init__(self) -> None:
        self.__image_file = ImageFile()

        self.__training_path = str(
            pathlib.Path().absolute()) + '/images/training/'
        self.__testing_path = str(
            pathlib.Path().absolute()) + '/images/testing/'

        self.__dts_training_dataset_path = str(
            pathlib.Path().absolute()) + '/dataset/training/'
        self.__dts_testing_dataset_path = str(
            pathlib.Path().absolute()) + '/dataset/testing/'

        self.__training_path = self.__training_path.replace('\\', '/')
        self.__testing_path = self.__testing_path.replace('\\', '/')

        self.__dts_training_dataset_path = self.__dts_training_dataset_path.replace(
            '\\', '/')
        self.__dts_testing_dataset_path = self.__dts_testing_dataset_path.replace(
            '\\', '/')

    def process_training_images(self, set_type, shrink_pixels=None, shrink_size=None, adaptative_threshold=None):
        if shrink_pixels:
            assert shrink_size is not None, "shrink_size is mandatory when passing shrink_pixels"

        dts_path = self.__dts_training_dataset_path + 'training_' + set_type + '.csv'

        file_path = pathlib.Path(dts_path)
        if not file_path.is_file():
            self.__process_images(self.__training_path, dts_path,
                                  shrink_pixels, shrink_size, adaptative_threshold)

    def process_testing_images(self, set_type, shrink_pixels=None, shrink_size=None, adaptative_threshold=None):
        if shrink_pixels:
            assert shrink_size is not None, "shrink_size is mandatory when passing shrink_pixels"

        dts_path = self.__dts_testing_dataset_path + 'testing_' + set_type + '.csv'

        file_path = pathlib.Path(dts_path)
        if not file_path.is_file():
            self.__process_images(self.__testing_path, dts_path,
                                  shrink_size, shrink_pixels, adaptative_threshold)

    # private

    def __binarize_with_adaptative_threshold(self, img, shrink_pixels=None, shrink_size=None):
        binarized_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 15, 7)

        binarized_img_arr = binarized_img.flatten()
        binarized_arr = []

        if shrink_pixels:
            binarized_arr = self.__binarize_shrink_image(
                self.__shrink_pixels(binarized_img_arr, shrink_size))
        else:
            binarized_arr = self.__binarize_normal_image(binarized_img_arr)

        return binarized_arr

    def __binarize_with_basic_threshold(self, img, shrink_pixels=None, shrink_size=None):
        _, binarized_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        binarized_arr = binarized_img.flatten()

        if shrink_pixels:
            binarized_arr = self.__binarize_shrink_image(
                self.__shrink_pixels(binarized_arr, shrink_size))
        else:
            binarized_arr = self.__binarize_normal_image(binarized_arr)

        return binarized_arr

    def __process_images(self, path, dts_path, shrink_pixels=None, shrink_size=None, adaptative_threshold=None):
        def sort_func(e):
            return len(e)

        with open(dts_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(CSV_HEADER)

            for folder, sub_folders, _ in os.walk(path):
                sub_folders.sort(key=sort_func)

                for sub_folder in sub_folders:
                    sub_folder_path = os.path.join(folder, sub_folder)
                    files = os.listdir(sub_folder_path)
                    files.sort(key=sort_func)

                    for special_file in files:
                        img_path = os.path.join(
                            folder, sub_folder + '/' + special_file).replace("\\", "/")
                        gray_img = self.__image_file.get_img(img_path)

                        binarized_img_arr = []

                        if adaptative_threshold:
                            binarized_img_arr = self.__binarize_with_adaptative_threshold(
                                gray_img, shrink_pixels, shrink_size)
                        else:
                            binarized_img_arr = self.__binarize_with_basic_threshold(
                                gray_img, shrink_pixels, shrink_size)

                        data = [sub_folder, self.__to_binary_string(
                            binarized_img_arr)]

                        # write the data
                        writer.writerow(data)

    def __average(self, arr):
        return sum(arr) / len(arr)

    def __shrink_pixels(self, img_array, shrink_size):
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

    def __binarize_shrink_image(self, reduced_img_arr):
        binarized_arr = []

        luminance_average = self.__average(reduced_img_arr)

        for pixel in reduced_img_arr:
            if pixel > luminance_average:
                binarized_arr.append(1)
            else:
                binarized_arr.append(0)

        return binarized_arr

    def __binarize_normal_image(self, img_arr):
        return map(lambda x: x if x != 255 else 1, img_arr)

    def __to_binary_string(self, binarized_img_arr):
        return ''.join(map(str, binarized_img_arr))
