from image_processor.image_processor import ImageProcessor


def main():
    image_processor = ImageProcessor()
    image_processor.process_training_images(adaptative_threshold=True)


if __name__ == '__main__':
    main()
