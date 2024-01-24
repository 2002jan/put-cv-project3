from keras.models import Model
from tensorflow import keras
from keras import layers, models
from models.model import InpaintingModel


class UNetModel(InpaintingModel):

    def get_name(self) -> str:
        return "unet"

    def get_model(self, input_shape=(256, 256, 3)):
        inputs = keras.Input(shape=input_shape)

        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        # Decoder
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        up1 = layers.UpSampling2D(size=(2, 2))(conv2)

        # Output
        outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up1)

        model = models.Model(inputs=inputs, outputs=outputs)

        return model
