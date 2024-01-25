import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

import numpy as np

from train import ImageDestructionType, destroy_image, load_img, ssim_loss

models_folder = "model_snapshots"
evaluate_output = "evaluate_outputs"


def type_to_string(aug_type: ImageDestructionType) -> str:
    match aug_type:
        case ImageDestructionType.EASY:
            return 'Easy'
        case ImageDestructionType.MEDIUM:
            return 'Medium'
        case ImageDestructionType.HARD:
            return 'Hard'


def load_model(model_path, compile=True):
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

    if compile:
        loaded_model.compile(loss=loss, optimizer=optimizer, metrics=[ssim_loss, 'mse', 'mae'])

    return loaded_model


def evaluate_model(model, step=200):
    images_paths = os.listdir("data")

    augmentation_types = list(ImageDestructionType)
    losses = {}

    for augment_type in augmentation_types:
        losses[augment_type] = [0, 0, 0, 0]

    iterations = 0

    all_img = len(images_paths)

    for i in range(0, all_img, step):
        iterations += 1

        images = []

        for j, img_path in enumerate(images_paths[i:i + step]):
            print("Processing image {}/{}".format(i + j + 1, all_img), end="\r")
            images.append(load_img(f"data/{img_path}"))

        images = np.array(images) / 255.0

        for augment_type in augmentation_types:
            images_destroyed = []

            for img in images:
                images_destroyed.append(destroy_image(img, augment_type))

            images_destroyed = np.array(images_destroyed)

            vals = model.evaluate(images_destroyed, images, verbose=0)

            del images_destroyed

            for k, val in enumerate(vals):
                losses[augment_type][k] += val

        del images

    for augment_type in augmentation_types:
        for i in range(4):
            losses[augment_type][i] /= iterations

    print("")

    return losses


def main():
    models = os.listdir(models_folder)

    print("Select model (default: 1)")

    for i, model in enumerate(models):
        print(f"{i + 1}. {model[1:]}")

    selected_model = int(input())

    if selected_model < 1 or selected_model > len(models) + 1:
        selected_model = 0
    else:
        selected_model -= 1

    model_path = models[selected_model]
    model_name = " ".join(model_path[1:].split("_"))

    model = load_model(f"{models_folder}/{models[0]}")

    print(f"Calculating loss for {model_path[1:]}")

    losses = evaluate_model(model)

    os.makedirs(evaluate_output, exist_ok=True)
    model_output_path = f"{evaluate_output}/{model_path[1:]}.txt"

    if os.path.isfile(model_output_path):
        os.remove(model_output_path)

    with open(model_output_path, "w") as output_file:

        output_file.write(f"Losses for {model_name}\n\n")

        for augmentation_type in list(ImageDestructionType):
            output_file.write(f"Losses for difficulty: {type_to_string(augmentation_type)}\n")

            output_file.write(f"SSIM: {losses[augmentation_type][1]}\n")
            output_file.write(f"MSE: {losses[augmentation_type][2]}\n")
            output_file.write(f"MAE: {losses[augmentation_type][3]}\n")
            output_file.write("------------------------\n")


if __name__ == '__main__':
    main()
