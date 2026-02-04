import keras
from keras import layers, models # models - tools to define model structures (Sequential, Model)

def build_model():
    data_augmentation  = keras.Sequential([
        layers.RandomRotation(0.04),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomFlip('horizontal')
    ])

    # Build CNN
    # Sequential model : layers are stacked one after another
    model = models.Sequential([
        layers.Input(shape=(32,32,3)), # input images with 32x32 pixel and 3 color channels
        data_augmentation ,

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),  # 32 filters and each filter size is 3x3
        layers.BatchNormalization(),
        layers.MaxPooling2D(), # reduces image size by taking maximum value in each 2x2 area

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),  # converts 2D feature maps into 1D vector and need to do this before passing data to Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
