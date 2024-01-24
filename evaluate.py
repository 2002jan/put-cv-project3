import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

import numpy as np

from train import ImageDestructionType, destroy_image, load_img, ssim_loss

models_folder = "model_snapshots"


def load_model(model_path):
    model_params = model_path.split("/")[-1].split("_")
    optimizer = model_params[1]
    loss = model_params[2] if model_params[2] != 'ssim' else ssim_loss

    print(f"Loaded model: {model_params[0][1:]}, optimizer: {optimizer}, loss: {model_params[2]}")

    with open(f"{model_path}/model.json", "r") as f:
        json_model_string = f.read()

    loaded_model = keras.models.model_from_json(
        json_model_string,
    )

    loaded_model.load_weights(f"{model_path}/model.hdf5")
    loaded_model.compile(loss=loss, optimizer=optimizer, metrics=[ssim_loss, 'mse', 'mae'])

    return loaded_model


def evaluate_model(model, step=200):
    images_paths = os.listdir("data")

    augmentation_types = list(ImageDestructionType)
    losses = {}

    for augment_type in augmentation_types:
        losses[augment_type] = [0, 0, 0, 0]

    iterations = 0

    for i in range(0, len(images_paths), step):
        iterations += 1

        images = []

        for j, img_path in enumerate(images_paths[i:i + step]):
            print("Processing image {}/{}".format(j + 1, step), end="\r")
            images.append(load_img(f"data/{img_path}"))

        print("")

        images = np.array(images) / 255.0

        for augment_type in augmentation_types:
            images_destroyed = []

            for img in images:
                images_destroyed.append(destroy_image(img, augment_type))

            images_destroyed = np.array(images_destroyed) / 255.0

            vals = model.evaluate(images_destroyed, images, verbose=0)

            del images_destroyed

            for k, val in enumerate(vals):
                losses[augment_type][k] += val

        del images

    for augment_type in augmentation_types:
        for i in range(4):
            losses[augment_type][i] /= iterations

    return losses


def main():
    models = os.listdir(models_folder)
    model = load_model(f"{models_folder}/{models[0]}")

    model_losses = evaluate_model(model)
    print(model_losses)


if __name__ == '__main__':
    main()
