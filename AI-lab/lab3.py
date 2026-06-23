import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam




IMAGE_SIZE = 96
BATCH = 16
MAX_IMAGES_PER_CLASS = 300


CAT_LABEL = 3
DOG_LABEL = 5


CLASS_NAMES = ["Cat", "Dog"]




def get_cat_dog_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()


    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()


    train_positions = np.where((train_labels == CAT_LABEL) | (train_labels == DOG_LABEL))[0]
    test_positions = np.where((test_labels == CAT_LABEL) | (test_labels == DOG_LABEL))[0]


    train_images = train_images[train_positions]
    train_labels = train_labels[train_positions]


    test_images = test_images[test_positions]
    test_labels = test_labels[test_positions]


    train_labels = np.where(train_labels == CAT_LABEL, 0, 1)
    test_labels = np.where(test_labels == CAT_LABEL, 0, 1)


    cat_samples = np.where(train_labels == 0)[0][:MAX_IMAGES_PER_CLASS]
    dog_samples = np.where(train_labels == 1)[0][:MAX_IMAGES_PER_CLASS]


    chosen_samples = np.concatenate((cat_samples, dog_samples))
    np.random.shuffle(chosen_samples)


    train_images = train_images[chosen_samples]
    train_labels = train_labels[chosen_samples]


    return train_images, train_labels, test_images, test_labels




def process_single_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = preprocess_input(image)


    return image, label




def make_tf_dataset(images, labels, shuffle_data=False):
    data = tf.data.Dataset.from_tensor_slices((images, labels))


    if shuffle_data:
        data = data.shuffle(buffer_size=500)


    data = data.map(process_single_image)
    data = data.batch(BATCH)
    data = data.prefetch(tf.data.AUTOTUNE)


    return data




def create_vgg16_classifier(training_type):
    input_layer = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))


    vgg_base = VGG16(
        weights="imagenet",
        include_top=False,
        input_tensor=input_layer
    )


    if training_type == "frozen":
        vgg_base.trainable = False


    elif training_type == "partial":
        vgg_base.trainable = True


        for layer in vgg_base.layers:
            layer.trainable = False


        for layer in vgg_base.layers:
            if layer.name.startswith("block5"):
                layer.trainable = True


    elif training_type == "whole":
        vgg_base.trainable = True


    else:
        raise ValueError("training_type must be frozen, partial, or whole")


    x = vgg_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)


    output_layer = Dense(1, activation="sigmoid")(x)


    model = Model(
        inputs=input_layer,
        outputs=output_layer,
        name=f"Cat_Dog_VGG16_{training_type}"
    )


    return model




def train_classifier(model, train_data, validation_data, training_type):
    learning_rate = 0.0001 if training_type == "frozen" else 0.00001


    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=3,
        verbose=1
    )


    return history




def draw_training_graphs(history, training_type):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{training_type.upper()} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{training_type}_accuracy_graph.png")
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{training_type.upper()} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{training_type}_loss_graph.png")
    plt.show()




def prepare_images_for_prediction(images):
    processed = []


    for image in images:
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = preprocess_input(image)
        processed.append(image)


    return tf.stack(processed)




def show_prediction_samples(model, test_images, test_labels, training_type):
    sample_images = test_images[:10]
    sample_labels = test_labels[:10]


    ready_images = prepare_images_for_prediction(sample_images)


    prediction_values = model.predict(ready_images)
    predicted_labels = (prediction_values.ravel() > 0.5).astype(int)


    plt.figure(figsize=(12, 5))


    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i])


        predicted_name = CLASS_NAMES[predicted_labels[i]]
        actual_name = CLASS_NAMES[sample_labels[i]]


        plt.title(f"Pred: {predicted_name}\nTrue: {actual_name}", fontsize=8)
        plt.axis("off")


    plt.suptitle(f"{training_type.upper()} Prediction Samples")
    plt.tight_layout()
    plt.savefig(f"{training_type}_prediction_samples.png")
    plt.show()




def split_train_validation(images, labels):
    split_point = int(len(images) * 0.8)


    train_x = images[:split_point]
    train_y = labels[:split_point]


    val_x = images[split_point:]
    val_y = labels[split_point:]


    return train_x, train_y, val_x, val_y




def run_experiment(training_type, train_data, val_data, test_data, test_images, test_labels):
    print("\n======================================")
    print(f"Training Type: {training_type.upper()}")
    print("======================================")


    model = create_vgg16_classifier(training_type)
    model.summary()


    history = train_classifier(model, train_data, val_data, training_type)


    test_loss, test_accuracy = model.evaluate(test_data, verbose=1)


    print(f"{training_type.upper()} Test Loss: {test_loss:.4f}")
    print(f"{training_type.upper()} Test Accuracy: {test_accuracy:.4f}")


    draw_training_graphs(history, training_type)
    show_prediction_samples(model, test_images, test_labels, training_type)


    return test_loss, test_accuracy




def main():
    train_images, train_labels, test_images, test_labels = get_cat_dog_dataset()


    print("Training images:", train_images.shape)
    print("Testing images:", test_images.shape)


    train_x, train_y, val_x, val_y = split_train_validation(train_images, train_labels)


    train_data = make_tf_dataset(train_x, train_y, shuffle_data=True)
    val_data = make_tf_dataset(val_x, val_y, shuffle_data=False)
    test_data = make_tf_dataset(test_images, test_labels, shuffle_data=False)


    final_scores = {}


    for training_type in ["frozen", "partial", "whole"]:
        loss, accuracy = run_experiment(
            training_type,
            train_data,
            val_data,
            test_data,
            test_images,
            test_labels
        )


        final_scores[training_type] = {
            "loss": loss,
            "accuracy": accuracy
        }


    print("\n========== Final Comparison ==========")


    for training_type, score in final_scores.items():
        print(
            f"{training_type.upper()} -> "
            f"Loss: {score['loss']:.4f}, "
            f"Accuracy: {score['accuracy']:.4f}"
        )




if __name__ == "__main__":
    main()
