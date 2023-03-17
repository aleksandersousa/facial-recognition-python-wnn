# IMPORTING LIBRARIES
import cv2
import pathlib

from file.file import File


path = str(pathlib.Path().absolute()) + '/images/training/s1/1.pgm'
path = path.replace('\\', '/')


def main():
    file = File()

    img = file.read_img(path)
    img_input = file.get_img_input(img)
    rect_points = file.get_rect_points(img, img_input)
    crop_img = file.crop_img(rect_points, img_input)

    # show
    cv2.imshow("Image", img)
    cv2.imshow("Image cropped", crop_img)
    cv2.waitKey(0)

# cv2.imwrite('crop_image0.jpg', crop_img)


if __name__ == '__main__':
    main()
