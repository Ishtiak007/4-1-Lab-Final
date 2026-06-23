import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout




LABELS = {
    "mnist": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],


    "fashion": [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ],


    "cifar10": [
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
        'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
    ]
}




def prepare_dataset(name):
    if name == "mnist":
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
        train_x = train_x[..., np.newaxis]
        test_x = test_x[..., np.newaxis]


    elif name == "fashion":
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
        train_x = train_x[..., np.newaxis]
        test_x = test_x[..., np.newaxis]


    elif name == "cifar10":
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
        train_y = train_y.ravel()
        test_y = test_y.ravel()


    else:
        raise ValueError("Invalid dataset name. Use mnist, fashion, or cifar10.")


    train_x = train_x.astype("float32") / 255.0
    test_x = test_x.astype("float32") / 255.0


    return train_x, test_x, train_y, test_y




def create_cnn(input_shape):
    image_input = Input(shape=input_shape)


    x = Conv2D(32, kernel_size=3, padding="same", activation="relu")(image_input)
    x = MaxPooling2D(pool_size=2)(x)


    x = Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)


    x = Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)


    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)


    output = Dense(10, activation="softmax")(x)


    return Model(image_input, output, name="Image_Classification_CNN")




def train_cnn(model, train_x, train_y):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    history = model.fit(
        train_x,
        train_y,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )


    return history




def show_training_curves(history, dataset):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{dataset.upper()} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dataset}_accuracy.png")
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{dataset.upper()} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dataset}_loss.png")
    plt.show()




def show_sample_predictions(model, test_x, test_y, dataset):
    prediction_scores = model.predict(test_x[:10])
    predicted_classes = np.argmax(prediction_scores, axis=1)


    plt.figure(figsize=(12, 5))


    for index in range(10):
        plt.subplot(2, 5, index + 1)


        if test_x.shape[-1] == 1:
            plt.imshow(test_x[index].squeeze(), cmap="gray")
        else:
            plt.imshow(test_x[index])


        actual = LABELS[dataset][test_y[index]]
        predicted = LABELS[dataset][predicted_classes[index]]


        plt.title(f"Pred: {predicted}\nTrue: {actual}", fontsize=8)
        plt.axis("off")


    plt.suptitle(f"{dataset.upper()} Prediction Results")
    plt.tight_layout()
    plt.savefig(f"{dataset}_sample_predictions.png")
    plt.show()




def run_for_dataset(dataset):
    print("\n====================================")
    print(f"Dataset: {dataset.upper()}")
    print("====================================")


    train_x, test_x, train_y, test_y = prepare_dataset(dataset)


    print("Train shape:", train_x.shape)
    print("Test shape:", test_x.shape)


    cnn_model = create_cnn(train_x.shape[1:])
    cnn_model.summary()


    history = train_cnn(cnn_model, train_x, train_y)


    loss, accuracy = cnn_model.evaluate(test_x, test_y, verbose=1)


    print(f"\nTest Loss for {dataset.upper()}: {loss:.4f}")
    print(f"Test Accuracy for {dataset.upper()}: {accuracy:.4f}")


    show_training_curves(history, dataset)
    show_sample_predictions(cnn_model, test_x, test_y, dataset)




def main():
    dataset_list = ["mnist", "fashion", "cifar10"]


    for dataset in dataset_list:
        run_for_dataset(dataset)




if __name__ == "__main__":
    main()
