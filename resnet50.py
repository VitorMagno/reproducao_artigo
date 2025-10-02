import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(task_type="distraction", input_shape=(224, 224, 3)):
    """
    Cria um modelo de classificação com ResNet50 congelada para duas tarefas:
    - 'distraction': classificação com 10 classes
    - 'drowsiness': classificação binária (probabilidade de sonolento)

    Parâmetros:
        task_type (str): 'distraction' ou 'drowsiness'
        input_shape (tuple): shape da imagem de entrada

    Retorna:
        modelo Keras compilado
    """

    # Importa ResNet50 sem a top layer (fully-connected)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Congela todas as camadas da ResNet50

    # Adiciona camadas customizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    if task_type == "distraction":
        output = Dense(10, activation='softmax', name='distraction_output')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    elif task_type == "drowsiness":
        output = Dense(1, activation='sigmoid', name='drowsiness_output')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        raise ValueError("task_type deve ser 'distraction' ou 'drowsiness'.")

    # Constrói o modelo
    model = Model(inputs=base_model.input, outputs=output)

    # Compila o modelo
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=metrics)

    return model


# ===== Exemplo de uso =====
if __name__ == "__main__":
    task = ['distraction', 'drowsiness']
    model = build_model(task_type=task[0])
    model.summary()
