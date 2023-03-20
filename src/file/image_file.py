import cv2


class ImageFile:
    def __read_img(self, path):
        img = cv2.imread(path, 0)

        assert img is not None, "file could not be read, check with os.path.exists()"

        return img

    def get_img(self, path):
        return self.__read_img(path)
