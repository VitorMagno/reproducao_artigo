import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from vgg_model import build_model
from data_loader import load_state_farm_data, load_drowsiness_data, prepare_data_for_training


def test_vgg16_model(model_path, task, dataset_path, max_images_per_class=None):
    print(f"\n=== TESTE - VGG16 ({task.upper()}) ===")

    if task == 'distraction':
        X_train, y_train, X_test, y_test = load_state_farm_data(dataset_path, max_images_per_class)
    elif task == 'drowsiness':
        X_train, y_train, X_test, y_test = load_drowsiness_data(dataset_path, max_images_per_class=max_images_per_class)
    else:
        raise ValueError("task deve ser 'drowsiness' ou 'distraction'")

    _, _, X_test, y_test = prepare_data_for_training(X_train, y_train, X_test, y_test, task)

    model = tf.keras.models.load_model(model_path)

    print("\n=== AVALIAÇÃO NO CONJUNTO DE TESTE ===")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)[:2]
    print(f"Test Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    y_pred = model.predict(X_test)

    if task == 'distraction':
        y_true = np.argmax(y_test, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_true = y_test
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(y_true, y_pred_classes))


# Exemplo de uso
if __name__ == "__main__":
    test_vgg16_model(
        model_path='vgg16_drowsiness_model.h5',
        task='drowsiness',
        dataset_path='driver_drowsiness_dataset',
        max_images_per_class=100
    )
