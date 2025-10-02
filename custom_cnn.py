import tensorflow as tf
from tensorflow import keras

def custom_cnn(input_shape=(224, 224, 3), task='drowsiness'):
    """
    Cria modelo CNN com 3 blocos convolucionais + 2 camadas densas finais,
    com saída adaptada para:
    - task='drowsiness': classificação binária (sonolento vs. alerta)
    - task='distraction': classificação com 10 classes

    Parâmetros:
        input_shape (tuple): shape da imagem de entrada (altura, largura, canais)
        task (str): 'drowsiness' ou 'distraction'

    Retorna:
        modelo compilado
    """

    inputs = keras.Input(shape=input_shape)

    # Bloco 1 - 32 filtros
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    # Bloco 2 - 64 filtros
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.35)(x)

    # Bloco 3 - 128 filtros
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.45)(x)

    # Flatten + FCs
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    # Saída baseada na tarefa
    if task == 'drowsiness':
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task == 'distraction':
        output = keras.layers.Dense(10, activation='softmax')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    else:
        raise ValueError("task deve ser 'drowsiness' ou 'distraction'.")

    # Modelo final
    model = keras.Model(inputs=inputs, outputs=output, name=f'CNN_{task}_classifier')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=metrics)

    return model


# ===== Exemplo de uso =====
if __name__ == "__main__":
    task = ['distraction', 'drowsiness']
    model = custom_cnn(task=task[0])
    model.summary()
