from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_shape=(128, 128, 3), num_classes=2):
    model = keras.Sequential([
        # Normalize pixel values from 0-255 to 0-1
        layers.Rescaling(1.0 / 255, input_shape=input_shape),

        # CNN layer 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # CNN layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # CNN layer 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Convert image features into a vector
        layers.Flatten(),

        # Fully connected layer
        layers.Dense(64, activation='relu'),

        # Dropout for regularization
        layers.Dropout(0.5),

        # Output layer: 2 classes, cats and dogs
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model