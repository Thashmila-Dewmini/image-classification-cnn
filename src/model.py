import keras
from keras import layers, models # models - tools to define model structures (Sequential, Model)

def build_model():

    # Data augmentation pipeline : preprocessing layers
    # this will slightly changes images every time they pass through
    data_augmentation  = keras.Sequential([   # create mini NN, no trainable weights and only job is randomly transform images
        layers.RandomRotation(0.04), # randomly rotate left or right and rotation range is ±4% of 360° : this will help to recognize objects at slightly different angles
        layers.RandomTranslation(0.1, 0.1), # shifts the image left/right by up to 10% of width and up/down by up to 10% of height
        layers.RandomFlip('horizontal') # randomly flips images left <-> right
    ])

    # Build CNN
    # Sequential model : layers are stacked one after another
    model = models.Sequential([
        layers.Input(shape=(32,32,3)), # input images with 32x32 pixel and 3 color channels
        data_augmentation,

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),  # 32 filters and each filter size is 3x3
        layers.BatchNormalization(),
        layers.MaxPooling2D(), # reduces image size by taking maximum value in each 2x2 area

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(), # normailze layer outputs from previous layer
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
