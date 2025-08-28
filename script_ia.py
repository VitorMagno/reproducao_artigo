#%%
import cv2 as cv2
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
### Modelo 
#%%

def criar_modelo_cnn(input_shape=(224, 224, 3), task='drowsiness'):
    """
    Cria modelo CNN com 3 blocos CNN + 1 bloco totalmente conectado
    
    Args:
        input_shape: formato da imagem de entrada (altura, largura, canais)
        task: 'drowsiness' para detecção binária ou 'distraction' para multi-classe
    
    Returns:
        modelo: modelo Keras compilado
    """
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # =============================================================
    # PRIMEIRO BLOCO CNN (32 filtros)
    # =============================================================
    
    # Camada 1: Convolução 2D (32 filtros, 3x3)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), 
                      padding='same', activation='relu',
                      name='conv1_1')(inputs)
    
    # Camada 2: Normalização em lote
    x = tf.keras.layers.BatchNormalization(name='bn1_1')(x)
    
    # Camada 3: Convolução 2D (32 filtros, 3x3)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), 
                      padding='same', activation='relu',
                      name='conv1_2')(x)
    
    # Camada 4: Normalização em lote
    x = tf.keras.layers.BatchNormalization(name='bn1_2')(x)
    
    # Camada 5: MaxPool2D (downsampling 2x2)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    
    # Camada 6: Dropout (overfitting)
    x = tf.keras.layers.Dropout(rate=0.25, name='dropout1')(x)
    
    # =============================================================
    # SEGUNDO BLOCO CNN (64 filtros)
    # =============================================================
    
    # Camada 1: Convolução 2D (64 filtros, 3x3)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), 
                      padding='same', activation='relu',
                      name='conv2_1')(x)
    
    # Camada 2: Normalização em lote
    x = tf.keras.layers.BatchNormalization(name='bn2_1')(x)
    
    # Camada 3: Convolução 2D (64 filtros, 3x3)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), 
                      padding='same', activation='relu',
                      name='conv2_2')(x)
    
    # Camada 4: Normalização em lote
    x = tf.keras.layers.BatchNormalization(name='bn2_2')(x)
    
    # Camada 5: MaxPool2D (downsampling 2x2)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    
    # Camada 6: Dropout (taxa maior)
    x = tf.keras.layers.Dropout(rate=0.35, name='dropout2')(x)
    
    # =============================================================
    # TERCEIRO BLOCO CNN (128 filtros)
    # =============================================================
    
    # Camada 1: Convolução 2D (128 filtros, 3x3)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), 
                      padding='same', activation='relu',
                      name='conv3_1')(x)
    
    # Camada 2: Normalização em lote
    x = tf.keras.layers.BatchNormalization(name='bn3_1')(x)
    
    # Camada 3: Convolução 2D (128 filtros, 3x3)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), 
                      padding='same', activation='relu',
                      name='conv3_2')(x)
    
    # Camada 4: Normalização em lote
    x = tf.keras.layers.BatchNormalization(name='bn3_2')(x)
    
    # Camada 5: MaxPool2D (downsampling 2x2)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    
    # Camada 6: Dropout (taxa ainda maior)
    x = tf.keras.layers.Dropout(rate=0.45, name='dropout3')(x)
    
    # =============================================================
    # BLOCO FINAL - CAMADAS TOTALMENTE CONECTADAS (7 camadas)
    # =============================================================
    
    # Camada 1: Flatten (achatar para 1D)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    
    # Camada 2: Dense (totalmente conectada)
    x = tf.keras.layers.Dense(units=512, activation='relu', name='fc1')(x)
    
    # Camada 3: Normalização em lote
    x = tf.keras.layers.BatchNormalization(name='bn_fc1')(x)
    
    # Camada 4: Dropout
    x = tf.keras.layers.Dropout(rate=0.5, name='dropout_fc1')(x)
    
    # Camada 5: Dense (segunda camada totalmente conectada)
    x = tf.keras.layers.Dense(units=256, activation='relu', name='fc2')(x)
    
    # Camada 6: Normalização em lote
    x = tf.keras.layers.BatchNormalization(name='bn_fc2')(x)
    
    # Camada 7: Camada de saída (varia conforme a tarefa)
    if task == 'drowsiness':
        # Classificação binária: sonolento vs alerta
        outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output_binary')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task == 'distraction':
        # Classificação multi-classe: 10 classes de distração
        outputs = tf.keras.layers.Dense(units=10, activation='softmax', name='output_multiclass')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', 'top_3_accuracy']
    else:
        raise ValueError("task deve ser 'drowsiness' ou 'distraction'")
    
    # Criar modelo
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'CNN_{task}_detection')
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics
    )
    
    return model