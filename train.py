import os
from enum import Enum
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import datetime

from models.model import InpaintingModel
from models.unet import UNetModel
from models.concatenated import Concatenated
from models.unet_extended import UNetExtended
from models.contex_encoder import ContextEncoder


class ImageDestructionType(Enum):
    EASY = 0
    MEDIUM = 1
    HARD = 2


def get_img_destruction_params(destruction_type: ImageDestructionType):
    match destruction_type:
        case ImageDestructionType.EASY:
            return 20, 40, 5, 10
        case ImageDestructionType.MEDIUM:
            return 7, 12, 20, 30
        case ImageDestructionType.HARD:
            return 15, 20, 25, 40


def load_img(path, img_size = (256,256)):
    img = cv2.imread(path)
    img = cv2.resize(img, img_size)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_data(img_size=(256, 256), train_size=1000, validation_size=300):
    images_paths = os.listdir("data")

    if train_size + validation_size > len(images_paths):
        print("Provided too big dataset size")
        exit(1)

    images = []

    print("Loading train set...")
    for i, img_path in enumerate(images_paths[:train_size]):
        print("Processing image {}/{}".format(i + 1, train_size), end="\r")
        images.append(load_img(f"data/{img_path}", img_size))

    print("\n")

    train = np.array(images) / 255.0

    images = []

    print("Loading validation set...")
    for i, img_path in enumerate(images_paths[train_size:train_size + validation_size]):
        print("Processing image {}/{}".format(i + 1, validation_size), end="\r")
        images.append(load_img(f"data/{img_path}", img_size))

    print("\n")

    validation = np.array(images) / 255.0

    return train, validation


def destroy_image(img, destruction_type: ImageDestructionType):
    destroyed_img = img.copy()

    patches_min, patches_max, size_min, size_max = get_img_destruction_params(destruction_type)

    for _ in range(np.random.randint(patches_min, patches_max)):
        patch_size_x = np.random.randint(size_min, size_max)
        patch_size_y = np.random.randint(size_min, size_max)
        x = np.random.randint(1, img.shape[1] - patch_size_x)
        y = np.random.randint(1, img.shape[0] - patch_size_y)
        destroyed_img[y:y + patch_size_y, x:x + patch_size_x, :] = 0.0

    return destroyed_img


def augment_data(dataset):
    destroyed_dataset = []
    destruction_types = list(ImageDestructionType)
    destruction_types_len = len(destruction_types)

    i = 0

    for img in dataset:
        destroyed_dataset.append(destroy_image(img, destruction_types[i]))
        i = (i + 1) % destruction_types_len

    return np.array(destroyed_dataset)


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def train_test(
        model_builder: InpaintingModel,
        epochs=200,
        batch_size=8,
        optimizer='adam',
        loss_name='ssim',
):
    t, v = load_data(train_size=750, validation_size=200)

    t_destroyed, v_destroyed = augment_data(t), augment_data(v)

    model = model_builder.get_model()

    model_name = f"{model_builder.get_name()}_{optimizer}_{loss_name}"

    loss = [loss_name] if loss_name != 'ssim' else [ssim_loss]

    model.compile(optimizer=optimizer, loss=loss, metrics=[ssim_loss, 'mse', 'mae'])

    early = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    reduce = ReduceLROnPlateau(monitor='val_loss', patience=8),

    log_dir = "logs/fit/" + model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(t_destroyed, t,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(v_destroyed, v),
              shuffle=True,
              callbacks=[early, reduce, tensorboard_callback])

    v_regenerated = model.predict(v_destroyed)

    save_path = f"model_snapshots/f{model_name}"

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    save_path_json = f"{save_path}/model.json"
    save_path_hdf5 = f"{save_path}/model.hdf5"

    if os.path.isfile(save_path_json):
        os.remove(save_path_json)

    if os.path.isfile(save_path_hdf5):
        os.remove(save_path_hdf5)

    with open(save_path_json, "w") as model_file:
        model_file.write(model.to_json())

    keras.saving.save_model(
        model,
        save_path_hdf5,
        save_format="h5"
    )

    examples = []

    for i in np.random.choice(len(v_regenerated), size=9, replace=False):
        examples.append(np.concatenate([v[i], v_destroyed[i], v_regenerated[i]], axis=1))

    output_img = np.concatenate(examples, axis=0) * 255
    output_img = cv2.cvtColor(output_img.astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.jpg", output_img)


if __name__ == "__main__":
    train_test(ContextEncoder())
