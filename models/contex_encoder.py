from keras.models import Model
from tensorflow import keras
from keras import layers, models
from models.model import InpaintingModel


class ContextEncoder(InpaintingModel):

    def get_name(self) -> str:
        return "context-encoder"

    def get_model(self, input_shape=(256, 256, 3)):
        inputs = keras.Input(shape=input_shape)

        conv1 = layers.Conv2D(96, (11, 11), activation='relu', padding='same', strides=(4, 4))(inputs)
        conv2 = layers.Conv2D(256, (5, 5), activation='relu', padding='same', strides=(1, 1))(conv1)
        conv3 = layers.Conv2D(384, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv2)
        conv4 = layers.Conv2D(384, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv3)
        conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv4)
        pool5 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv5)

        channel_wise_fc = layers.Conv2D(256, (1, 1), activation='relu', padding='same')(pool5)
        # Decoder
        upconv1 = layers.Conv2DTranspose(256, (5, 5), activation='relu', padding='same', strides=(2, 2))(
            channel_wise_fc)
        upconv2 = layers.Conv2DTranspose(128, (5, 5), activation='relu', padding='same', strides=(2, 2))(upconv1)
        upconv3 = layers.Conv2DTranspose(64, (5, 5), activation='relu', padding='same', strides=(2, 2))(upconv2)
        upconv4 = layers.Conv2DTranspose(32, (5, 5), activation='relu', padding='same', strides=(2, 2))(upconv3)

        # Last upconv should match the input shape
        upconv5 = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', strides=(2, 2))(upconv4)
        model = models.Model(inputs=inputs, outputs=upconv5)

        return model
