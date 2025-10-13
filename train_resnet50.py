import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from resnet50 import build_model
from data_loader import load_state_farm_data, load_drowsiness_data


def train_resnet50(task, dataset_path, epochs=50, batch_size=32, model_save_path=None, max_images_per_class=None):
    print(f"=== TREINAMENTO RESNET50 - TAREFA: {task.upper()} ===")

    if task == 'distraction':
        X_train, y_train, X_test, y_test = load_state_farm_data(dataset_path, max_images_per_class)
    elif task == 'drowsiness':
        X_train, y_train, X_test, y_test = load_drowsiness_data(dataset_path, max_images_per_class=max_images_per_class)
    else:
        raise ValueError("task deve ser 'drowsiness' ou 'distraction'")


    input_shape = X_train.shape[1:]
    model = build_model(task_type=task, input_shape=input_shape)

    model.summary()

    callbacks = []
    if model_save_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        )

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n=== AVALIAÇÃO FINAL ===")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)[:2]
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return model, history


# Exemplo de uso
if __name__ == "__main__":
    model_resnet, history_resnet = train_resnet50(
        task='distraction',  # ou 'drowsiness'
        dataset_path='state_farm_driver_detection',
        epochs=10,
        batch_size=16,
        max_images_per_class=100,
        model_save_path='resnet50_distraction_model.h5'
    )
