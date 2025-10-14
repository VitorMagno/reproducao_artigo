import os
import numpy as np
import cv2
import tensorflow as tf
from custom_cnn import custom_cnn
from sklearn.model_selection import train_test_split

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
    drowsy_path = os.path.join(dataset_path, 'Drowsy')
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
    not_drowsy_path = os.path.join(dataset_path, 'Non Drowsy')
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