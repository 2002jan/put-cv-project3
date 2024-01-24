import os

from train import ImageDestructionType

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2

from evaluate import load_model, load_img, destroy_image

models_folder = "model_snapshots"


def main():
    os.makedirs("output_images")

    models = os.listdir(models_folder)
    images_paths = os.listdir("data")
    augmentation_types = list(ImageDestructionType)

    for model in models:
        loaded_model = load_model(f"{models_folder}/{model}", False)

        images = []

        for img in np.random.choice(images_paths, size=9, replace=False):
            images.append(load_img(f"data/{img}"))

        images = np.array(images) / 255.0

        i = 0

        augmented_images = []

        for augment_type in augmentation_types:
            for j in range(3):
                augmented_images.append(destroy_image(images[i * 3 + j], augment_type))

            i += 1

        augmented_images = np.array(augmented_images)

        fixed_images = loaded_model.predict(augmented_images)

        examples = []

        for i in range(9):
            examples.append(np.concatenate([images[i], augmented_images[i], fixed_images[i]], axis=1))

        output_img = np.concatenate(examples, axis=0) * 255
        output_img = cv2.cvtColor(output_img.astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"output_images/{model}.jpg", output_img)


if __name__ == '__main__':
    main()
