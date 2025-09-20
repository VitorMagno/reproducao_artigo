import os
import cv2
import numpy as np
from customCNN import custom_cnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def load_state_farm_data(dataset_path, max_images_per_class=None):
    """
    Carrega o dataset State Farm (10 classes de distração) que já tem divisão train/test
    
    Args:
        dataset_path: caminho para a pasta state_farm_driver_detection
        max_images_per_class: limite de imagens por classe (None = todas)
    
    Returns:
        (X_train, y_train, X_test, y_test): dados de treino e teste
    """
    
    def load_images_from_folder(folder_path, max_per_class=None):
        images = []
        labels = []
        
        # Listar as pastas c0, c1, ..., c9
        class_folders = sorted([f for f in os.listdir(folder_path) if f.startswith('c')])
        
        for class_folder in class_folders:
            class_path = os.path.join(folder_path, class_folder)
            class_label = int(class_folder[1:])  # Extrai número da classe (c0 -> 0, c1 -> 1, etc.)
            
            print(f"Carregando classe {class_label} de {class_path}")
            
            # Listar todos os arquivos de imagem
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limitar número de imagens se especificado
            if max_per_class is not None:
                image_files = image_files[:max_per_class]
                print(f"  -> Limitado a {len(image_files)} imagens por classe")
            
            images_loaded = 0
            for filename in image_files:
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Manter formato BGR conforme artigo (sem conversão RGB)
                    images.append(img)
                    labels.append(class_label)
                    images_loaded += 1
            
            print(f"  -> {images_loaded} imagens carregadas da classe {class_label}")
        
        return np.array(images), np.array(labels)
    
    # Carregar dados de treino
    train_path = os.path.join(dataset_path, 'train')
    X_train, y_train = load_images_from_folder(train_path, max_per_class=max_images_per_class)
    
    # Carregar dados de teste
    test_path = os.path.join(dataset_path, 'test')
    X_test, y_test = load_images_from_folder(test_path, max_per_class=max_images_per_class)
    
    print(f"State Farm - Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def load_drowsiness_data(dataset_path, test_size=0.2, random_state=42, max_images_per_class=None):
    """
    Carrega o dataset de sonolência e faz divisão treino/teste
    
    Args:
        dataset_path: caminho para a pasta driver_drowsiness_dataset
        test_size: proporção para teste (default: 0.2 = 20%)
        random_state: semente para reprodutibilidade
        max_images_per_class: limite de imagens por classe (None = todas)
    
    Returns:
        (X_train, y_train, X_test, y_test): dados de treino e teste
    """
    
    images = []
    labels = []
    
    # Carregar imagens drowsy (classe 1)
    drowsy_path = os.path.join(dataset_path, 'drowsy')
    print(f"Carregando imagens drowsy de {drowsy_path}")
    
    drowsy_files = [f for f in os.listdir(drowsy_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limitar número de imagens se especificado
    if max_images_per_class is not None:
        drowsy_files = drowsy_files[:max_images_per_class]
        print(f"  -> Limitado a {len(drowsy_files)} imagens drowsy")
    
    drowsy_loaded = 0
    for filename in drowsy_files:
        img_path = os.path.join(drowsy_path, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Manter formato BGR conforme artigo (sem conversão RGB)
            images.append(img)
            labels.append(1)  # drowsy = 1
            drowsy_loaded += 1
    
    print(f"  -> {drowsy_loaded} imagens drowsy carregadas")
    
    # Carregar imagens not_drowsy (classe 0)
    not_drowsy_path = os.path.join(dataset_path, 'not_drowsy')
    print(f"Carregando imagens not_drowsy de {not_drowsy_path}")
    
    not_drowsy_files = [f for f in os.listdir(not_drowsy_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limitar número de imagens se especificado
    if max_images_per_class is not None:
        not_drowsy_files = not_drowsy_files[:max_images_per_class]
        print(f"  -> Limitado a {len(not_drowsy_files)} imagens not_drowsy")
    
    not_drowsy_loaded = 0
    for filename in not_drowsy_files:
        img_path = os.path.join(not_drowsy_path, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Manter formato BGR conforme artigo (sem conversão RGB)
            images.append(img)
            labels.append(0)  # not_drowsy = 0
            not_drowsy_loaded += 1
    
    print(f"  -> {not_drowsy_loaded} imagens not_drowsy carregadas")
    
    # Converter para numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Total de imagens carregadas: {len(X)}")
    print(f"Drowsy: {np.sum(y == 1)}, Not Drowsy: {np.sum(y == 0)}")
    
    # Dividir em treino e teste (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Drowsiness - Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def prepare_data_for_training(X_train, y_train, X_test, y_test, task):
    """
    Prepara os dados para treinamento (sem normalização de pixels, seguindo o artigo)
    
    Args:
        X_train, y_train, X_test, y_test: dados brutos
        task: 'drowsiness' ou 'distraction'
    
    Returns:
        (X_train, y_train, X_test, y_test): dados preparados
    """
    
    # IMPORTANTE: Sem normalização de pixels (0-255 mantidos conforme artigo)
    # Apenas converter para float32 para compatibilidade com TensorFlow
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    if task == 'distraction':
        # Para classificação multi-classe: converter para categorical
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
    elif task == 'drowsiness':
        # Para classificação binária: manter como está (0 ou 1)
        pass
    
    print(f"Dados preparados - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Dados preparados - X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Range de pixels mantido: {X_train.min():.0f}-{X_train.max():.0f} (conforme artigo)")
    
    return X_train, y_train, X_test, y_test

def train_model(task, dataset_path, epochs=50, batch_size=32, model_save_path=None, max_images_per_class=None):
    """
    Função principal para treinar o modelo
    
    Args:
        task: 'drowsiness' ou 'distraction'
        dataset_path: caminho para o dataset
        epochs: número de épocas
        batch_size: tamanho do batch
        model_save_path: caminho para salvar o modelo (opcional)
        max_images_per_class: limite de imagens por classe para desenvolvimento/teste (None = todas)
    
    Returns:
        (model, history): modelo treinado e histórico
    """
    
    print(f"=== INICIANDO TREINAMENTO - TAREFA: {task.upper()} ===")
    
    if max_images_per_class:
        print(f"MODO DESENVOLVIMENTO: Limitado a {max_images_per_class} imagens por classe")
    else:
        print("MODO COMPLETO: Usando todas as imagens disponíveis")
    
    # Carregar dados específicos para cada tarefa
    if task == 'distraction':
        X_train, y_train, X_test, y_test = load_state_farm_data(dataset_path, max_images_per_class)
    elif task == 'drowsiness':
        X_train, y_train, X_test, y_test = load_drowsiness_data(dataset_path, max_images_per_class=max_images_per_class)
    else:
        raise ValueError("task deve ser 'drowsiness' ou 'distraction'")
    
    # Preparar dados
    X_train, y_train, X_test, y_test = prepare_data_for_training(
        X_train, y_train, X_test, y_test, task
    )
    
    # Criar modelo
    input_shape = X_train.shape[1:]  # Pega shape das imagens automaticamente
    print(f"Input shape: {input_shape}")
    
    model = custom_cnn(input_shape=input_shape, task=task)
    
    # Mostrar resumo do modelo
    print("\n=== ARQUITETURA DO MODELO ===")
    model.summary()
    
    # Callbacks (sem validação, seguindo o artigo)
    callbacks = []
    
    if model_save_path:
        # Salvar melhor modelo baseado na accuracy de teste
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='loss',  # Como não temos validação, monitora loss de treino
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        )
    
    print(f"\n=== INICIANDO TREINAMENTO ===")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    
    # Treinar modelo (sem validation_data, seguindo especificação do artigo)
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n=== AVALIANDO NO CONJUNTO DE TESTE ===")
    # Avaliar no conjunto de teste
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)[:2]
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return model, history

# Exemplo de uso:
if __name__ == "__main__":
    
    # MODO DESENVOLVIMENTO - Teste rápido com poucos dados
    print("=== MODO DESENVOLVIMENTO - TESTE RÁPIDO ===")
    
    # Treinar modelo de sonolência com apenas 100 imagens por classe
    model_drowsiness_dev, history_dev = train_model(
        task='drowsiness',
        dataset_path='driver_drowsiness_dataset',
        epochs=5,  # Menos épocas para teste
        batch_size=16,  # Batch menor para teste
        max_images_per_class=100,  # Apenas 100 imagens por classe
        model_save_path='dev_drowsiness_model.h5'
    )
    
    print("\n" + "="*80 + "\n")
    
    # MODO COMPLETO - Treinamento final
    print("=== MODO COMPLETO - TREINAMENTO FINAL ===")
    
    # Treinar modelo de detecção de sonolência com dataset completo
    model_drowsiness, history_drowsiness = train_model(
        task='drowsiness',
        dataset_path='driver_drowsiness_dataset',
        epochs=50,
        batch_size=32,
        max_images_per_class=None,  # Usar todas as imagens
        model_save_path='best_drowsiness_model.h5'
    )
    
    print("\n" + "="*80 + "\n")
    
    # Treinar modelo de classificação de distração
    model_distraction, history_distraction = train_model(
        task='distraction',
        dataset_path='state_farm_driver_detection',
        epochs=50,
        batch_size=32,
        max_images_per_class=None,  # Usar todas as imagens
        model_save_path='best_distraction_model.h5'
    )