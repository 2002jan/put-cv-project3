from keras.models import Model
from tensorflow import keras
from keras import layers, models
from models.model import InpaintingModel


class UNetExtended(InpaintingModel):

    def get_name(self) -> str:
        return "unet-extended"

    def get_model(self, input_shape=(256, 256, 3)):
        inputs = keras.Input(shape=input_shape)

        # Encoder
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Middle
        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

        # Decoder
        up1 = layers.UpSampling2D(size=(2, 2))(conv3)
        concat1 = layers.Concatenate(axis=-1)([conv2, up1])
        conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)
        conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

        up2 = layers.UpSampling2D(size=(2, 2))(conv4)
        concat2 = layers.Concatenate(axis=-1)([conv1, up2])
        conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)
        conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

        # Output
        outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv5)

        model = models.Model(inputs=inputs, outputs=outputs)

        return model
