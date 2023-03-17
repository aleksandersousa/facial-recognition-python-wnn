# IMPORTING LIBRARIES
import cv2
import pathlib

from file.image_file import ImageFile


path = str(pathlib.Path().absolute()) + '/images/training/s1/1.pgm'
path = path.replace('\\', '/')


def main():
    image_file = ImageFile()

    img = image_file.read_img(path)
    img_input = image_file.get_img_input(img)
    rect_points = image_file.get_rect_points(img, img_input)
    crop_img = image_file.crop_img(rect_points, img_input)

    # show
    cv2.imshow("Image", img)
    cv2.imshow("Image cropped", crop_img)
    cv2.waitKey(0)

# cv2.imwrite('crop_image0.jpg', crop_img)


if __name__ == '__main__':
    main()
