import os
import yaml
from model import create_model
from dataset import load_data

if __name__ == "__main__":
    DATASET_PATH = "dataset"
    EPOCHS = 10
    INPUT_SHAPE = (128, 128, 3)

    os.makedirs("checkpoints", exist_ok=True)

    # Load the dataset
    train_ds, test_ds, class_names = load_data(DATASET_PATH)

    # Create the model
    model = create_model(
        input_shape=INPUT_SHAPE,
        num_classes=len(class_names)
    )

    # Train the model
    model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc}")

    # Save the full model
    model.save("checkpoints/cat_dog_classifier.h5")

    # Save the weights only
    weights_path = "checkpoints/cat_dog_classifier.weights.h5"
    model.save_weights(weights_path)

    # Save hyperparameters and metadata to YAML
    metadata = {
        "dataset_path": DATASET_PATH,
        "epochs": EPOCHS,
        "input_shape": list(INPUT_SHAPE),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "weights_file": weights_path,
        "model_file": "checkpoints/cat_dog_classifier.h5"
    }

    with open("checkpoints/metadata.yaml", "w") as yaml_file:
        yaml.dump(metadata, yaml_file, sort_keys=False)

    print("Model, weights, and metadata saved successfully.")