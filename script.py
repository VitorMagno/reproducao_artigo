#%%
import cv2 as cv2
import os
import sys
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# %%
def redimensionar_imagem(caminho_entrada, caminho_saida, nova_largura, nova_altura):
    """
    Redimensiona uma imagem usando OpenCV
    
    Args:
        caminho_entrada (str): Caminho da imagem original
        caminho_saida (str): Caminho onde salvar a imagem redimensionada
        nova_largura (int): Nova largura desejada
        nova_altura (int): Nova altura desejada
    
    Returns:
        bool: True se sucesso, False se erro
    """
    try:
        # Ler a imagem
        imagem = cv2.imread(caminho_entrada)
        
        if imagem is None:
            print(f"Erro: Não foi possível carregar a imagem {caminho_entrada}")
            return False
        
        # Obter dimensões originais
        altura_original, largura_original = imagem.shape[:2]
        print(f"Dimensões originais: {largura_original}x{altura_original}")
        
        print(f"Novas dimensões: {nova_largura}x{nova_altura}")
        
        # Redimensionar a imagem
        imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
        
        # Salvar a imagem redimensionada
        sucesso = cv2.imwrite(caminho_saida, imagem_redimensionada)
        
        if sucesso:
            print(f"Imagem salva com sucesso em: {caminho_saida}")
            return True
        else:
            print("Erro ao salvar a imagem")
            return False
            
    except Exception as e:
        print(f"Erro: {str(e)}")
        return False

def redimensionar_multiplas_imagens(pasta_entrada, pasta_saida, largura, altura):
    """
    Redimensiona múltiplas imagens de uma pasta
    
    Args:
        pasta_entrada (str): Pasta com as imagens originais
        pasta_saida (str): Pasta onde salvar as imagens redimensionadas
        largura (int): Nova largura
        altura (int): Nova altura
    """
    # Criar pasta de saída se não existir
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    # Extensões de imagem suportadas
    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Processar cada arquivo na pasta
    for arquivo in os.listdir(pasta_entrada):
        if arquivo.lower().endswith(extensoes_validas):
            caminho_entrada = os.path.join(pasta_entrada, arquivo)
            caminho_saida = os.path.join(pasta_saida, f"{arquivo}")
            
            print(f"\nProcessando: {arquivo}")
            redimensionar_imagem(caminho_entrada, caminho_saida, largura, altura)

### Primeira etapa: redimensionar imagens
#%%
state_farm_train = 'datasets/State_farm/imgs/train'
pastas = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
for i in pastas:
    redimensionar_multiplas_imagens(pasta_entrada=f'{state_farm_train}/{i}', pasta_saida=f'{state_farm_train}/redimensionada/{i}', largura=80, altura=80)
#%%
state_farm_test = 'datasets/State_farm/imgs/test'
redimensionar_multiplas_imagens(pasta_entrada=state_farm_test, pasta_saida=f'{state_farm_test}/redimensionada', largura=80, altura=80)

### Segunda etapa: detectar faces e olhos
#%%
img_test = state_farm_test + '/redimensionada/img_1.jpg'
#%%
class HCC:
    def __init__(self):
        """Inicializa os classificadores Haar Cascade"""
        try:
            # Carregar os classificadores pré-treinados
            # Estes arquivos vêm com a instalação do OpenCV
            self.face_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')
            self.face_cascade3 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            self.face_cascade4 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            self.eye_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.eye_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
            
            # Verificar se os classificadores foram carregados corretamente
            if self.face_cascade1.empty():
                raise Exception("Erro ao carregar classificador de faces1")
            if self.face_cascade2.empty():
                raise Exception("Erro ao carregar classificador de faces2")
            if self.face_cascade3.empty():
                raise Exception("Erro ao carregar classificador de faces3")
            if self.face_cascade4.empty():
                raise Exception("Erro ao carregar classificador de faces4")
            if self.eye_cascade1.empty():
                raise Exception("Erro ao carregar classificador de olhos1")
            if self.eye_cascade2.empty():
                raise Exception("Erro ao carregar classificador de olhos2")
                
            print("Classificadores Haar Cascade carregados com sucesso!")
            
        except Exception as e:
            print(f"Erro ao inicializar detectores: {e}")
            sys.exit(1)
    
    def detectar_faces_olhos(self, imagem_path, output_folder, salvar_resultado=True, mostrar_imagem=True):
        """
        Detecta faces e olhos em uma imagem
        
        Args:
            imagem_path (str): Caminho para a imagem
            salvar_resultado (bool): Se True, salva a imagem com detecções
            mostrar_imagem (bool): Se True, exibe a imagem na tela
            
        Returns:
            tuple: (numero_faces, numero_olhos, imagem_processada)
        """
        try:
            # Carregar a imagem
            img = cv2.imread(imagem_path)
            if img is None:
                print(f"Erro: Não foi possível carregar a imagem {imagem_path}")
                return 0, 0, None
            
            # Converter para escala de cinza (necessário para Haar Cascade)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detectar faces // como usar multiplos classificadores?
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,    # Redução da escala a cada iteração
                minNeighbors=5,     # Mínimo de vizinhos para considerar uma detecção
                minSize=(30, 30),   # Tamanho mínimo da face
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            total_olhos = 0
            
            print(f"Faces detectadas: {len(faces)}")
            
            # Processar cada face detectada
            for i, (x, y, w, h) in enumerate(faces):
                # Desenhar retângulo ao redor da face
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Região de interesse (ROI) para buscar olhos apenas dentro da face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                
                # Detectar olhos dentro da face // como usar multiplos classificadores?
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(10, 10)
                )
                
                print(f"  Olhos detectados na face {i+1}: {len(eyes)}")
                
                # Desenhar retângulos ao redor dos olhos
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                total_olhos += len(eyes)
                
            
            # Salvar resultado se solicitado
            if salvar_resultado:
                nome_base = os.path.splitext(imagem_path)[0]
                extensao = os.path.splitext(imagem_path)[1]
                caminho_saida = f"{output_folder}/{nome_base}_detectado{extensao}"
                cv2.imwrite(caminho_saida, img)
                print(f"Imagem com detecções salva em: {caminho_saida}")
            
            # Mostrar imagem se solicitado
            if mostrar_imagem:
                # Redimensionar se a imagem for muito grande
                altura, largura = img.shape[:2]
                if largura > 800 or altura > 600:
                    ratio = min(800/largura, 600/altura)
                    nova_largura = int(largura * ratio)
                    nova_altura = int(altura * ratio)
                    img_display = cv2.resize(img, (nova_largura, nova_altura))
                else:
                    img_display = img
                
                cv2.imshow('Detecção de Faces e Olhos', img_display)
                print("Pressione qualquer tecla para fechar a janela...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return len(faces), total_olhos, img
            
        except Exception as e:
            print(f"Erro durante a detecção: {e}")
            return 0, 0, None
        

def enhance(image_path, output_folder):
    """Recebe uma pasta para fazer a detecção de faces e olhos

    Args:
        image_path (string): path to image folder
        output_folder (string): path to save processed images

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        output_folder: path to processed images
    """
    detector = HCC()
    # Criar pasta de saída se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Extensões de imagem suportadas
    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Processar cada arquivo na pasta
    for arquivo in os.listdir(image_path):
        if arquivo.lower().endswith(extensoes_validas):
            caminho_entrada = os.path.join(image_path, arquivo)
            caminho_saida = os.path.join(output_folder, f"{arquivo}")
            
            print(f"\nProcessando: {arquivo}")
            detector.detectar_faces_olhos(imagem_path=caminho_entrada, output_folder=caminho_saida, salvar_resultado=True, mostrar_imagem=False)
    return caminho_saida
#%%

state_farm_train = 'datasets/State_farm/imgs/train/redimensionada'
state_farm_test = 'datasets/State_farm/imgs/test/redimensionada'
#%%
for i in pastas:
    state_farm_train = enhance(f'{state_farm_train}/{i}', f'{state_farm_train}/enhanced/{i}')
                               
#%%
state_farm_test = enhance(state_farm_test, f'{state_farm_test}/enhanced')


#%%
### terceira etapa: preprocessar as regiões de interesse
#%%
class preprocessamento_img:
    def __init__(self):
        """Inicializa o melhorador de imagem com filtros pré-definidos"""
        
        # Diferentes kernels de sharpening
        self.kernels_sharpening = {
            'basico': np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]]),
            
            'intenso': np.array([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]]),
            
            'suave': np.array([[-1, -1, -1, -1, -1],
                              [-1,  2,  2,  2, -1],
                              [-1,  2,  8,  2, -1],
                              [-1,  2,  2,  2, -1],
                              [-1, -1, -1, -1, -1]]) / 8.0,
            
            'laplaciano': np.array([[0,  -1,  0],
                                   [-1,  4, -1],
                                   [0,  -1,  0]]),
            
            'unsharp_mask': np.array([[-1, -4, -6, -4, -1],
                                     [-4, -16, -24, -16, -4],
                                     [-6, -24, 476, -24, -6],
                                     [-4, -16, -24, -16, -4],
                                     [-1, -4, -6, -4, -1]]) / 256.0
        }
        
        print("Melhorador de Imagem inicializado com kernels:", list(self.kernels_sharpening.keys()))
    
    def ajustar_contraste_brilho(self, imagem: np.ndarray, alfa: float = 1.0, beta: int = 0) -> np.ndarray:
        """
        Ajusta contraste e brilho da imagem usando a fórmula: nova_imagem = alfa * imagem + beta
        
        Args:
            imagem (np.ndarray): Imagem de entrada
            alfa (float): Fator de contraste (1.0 = sem mudança, >1.0 aumenta, <1.0 diminui)
            beta (int): Fator de brilho (0 = sem mudança, >0 aumenta, <0 diminui)
            
        Returns:
            np.ndarray: Imagem com contraste e brilho ajustados
        """
        # Aplicar a transformação linear: nova_imagem = alfa * imagem + beta
        imagem_ajustada = cv2.convertScaleAbs(imagem, alpha=alfa, beta=beta)
        
        return imagem_ajustada
    
    def aplicar_sharpening(self, imagem: np.ndarray, tipo_kernel: str = 'basico', intensidade: float = 1.0) -> np.ndarray:
        """
        Aplica filtro de nitidez à imagem
        
        Args:
            imagem (np.ndarray): Imagem de entrada
            tipo_kernel (str): Tipo de kernel ('basico', 'intenso', 'suave', 'laplaciano', 'unsharp_mask')
            intensidade (float): Multiplicador da intensidade do efeito
            
        Returns:
            np.ndarray: Imagem com nitidez melhorada
        """
        if tipo_kernel not in self.kernels_sharpening:
            print(f"Kernel '{tipo_kernel}' não encontrado. Usando 'basico'.")
            tipo_kernel = 'basico'
        
        # Obter o kernel
        kernel = self.kernels_sharpening[tipo_kernel] * intensidade
        
        # Aplicar filtro
        imagem_nitida = cv2.filter2D(imagem, -1, kernel)
        
        return imagem_nitida
    
    def sharpening_unsharp_mask(self, imagem: np.ndarray, sigma: float = 1.0, strength: float = 1.5, threshold: int = 0) -> np.ndarray:
        """
        Aplica Unsharp Masking - técnica avançada de nitidez
        
        Args:
            imagem (np.ndarray): Imagem de entrada
            sigma (float): Desvio padrão do blur gaussiano
            strength (float): Intensidade do efeito
            threshold (int): Limite mínimo para aplicar o efeito
            
        Returns:
            np.ndarray: Imagem com unsharp masking aplicado
        """
        # Criar versão borrada da imagem
        blur = cv2.GaussianBlur(imagem, (0, 0), sigma)
        
        # Calcular a máscara (diferença entre original e borrada)
        mask = cv2.subtract(imagem, blur)
        
        # Aplicar threshold se especificado
        if threshold > 0:
            mask = np.where(np.abs(mask) < threshold, 0, mask)
        
        # Aplicar a máscara à imagem original
        imagem_sharp = cv2.addWeighted(imagem, 1.0, mask, strength, 0)
        
        return imagem_sharp
    

    def melhorar_imagem_completa(self, 
                                imagem_path: str, 
                                alfa: float = 1.2, 
                                beta: int = 10,
                                tipo_sharpening: str = 'basico',
                                intensidade_sharpening: float = 1.0,
                                usar_unsharp: bool = True,
                                sigma_unsharp: float = 1.0,
                                strength_unsharp: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica todas as melhorias à imagem
        
        Args:
            imagem_path (str): Caminho da imagem
            alfa (float): Contraste
            beta (int): Brilho
            tipo_sharpening (str): Tipo de kernel de nitidez
            intensidade_sharpening (float): Intensidade da nitidez
            usar_unsharp (bool): Se deve usar unsharp masking
            sigma_unsharp (float): Sigma para unsharp masking
            strength_unsharp (float): Força do unsharp masking
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (imagem_original, imagem_melhorada)
        """
        try:
            # Carregar imagem
            imagem_original = cv2.imread(imagem_path)
            if imagem_original is None:
                raise ValueError(f"Não foi possível carregar a imagem: {imagem_path}")
            
            print(f"Processando: {imagem_path}")
            print(f"Dimensões: {imagem_original.shape}")
            
            # Começar com a imagem original
            imagem_melhorada = imagem_original.copy()
            
            # 1. Ajustar contraste e brilho
            print(f"Aplicando contraste (α={alfa}) e brilho (β={beta})")
            imagem_melhorada = self.ajustar_contraste_brilho(imagem_melhorada, alfa, beta)
            
            # 2. Aplicar sharpening
            if usar_unsharp:
                print(f"Aplicando Unsharp Masking (σ={sigma_unsharp}, força={strength_unsharp})")
                imagem_melhorada = self.sharpening_unsharp_mask(imagem_melhorada, sigma_unsharp, strength_unsharp)
            else:
                print(f"Aplicando sharpening '{tipo_sharpening}' (intensidade={intensidade_sharpening})")
                imagem_melhorada = self.aplicar_sharpening(imagem_melhorada, tipo_sharpening, intensidade_sharpening)
            
            return imagem_original, imagem_melhorada
            
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
            return None, None
    
    def comparar_imagens(self, imagem_original: np.ndarray, imagem_melhorada: np.ndarray, 
                        salvar: bool = True, mostrar: bool = True, nome_arquivo: str = None):
        """
        Compara e exibe as imagens original e melhorada lado a lado
        
        Args:
            imagem_original (np.ndarray): Imagem original
            imagem_melhorada (np.ndarray): Imagem melhorada
            salvar (bool): Se deve salvar a comparação
            mostrar (bool): Se deve mostrar na tela
            nome_arquivo (str): Nome do arquivo para salvar
        """
        if imagem_original is None or imagem_melhorada is None:
            print("Erro: Imagens inválidas para comparação")
            return
        
        # Converter de BGR para RGB para matplotlib
        img_orig_rgb = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
        img_melh_rgb = cv2.cvtColor(imagem_melhorada, cv2.COLOR_BGR2RGB)
        
        # Criar figura com subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Imagem original
        ax1.imshow(img_orig_rgb)
        ax1.set_title('Imagem Original', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Imagem melhorada
        ax2.imshow(img_melh_rgb)
        ax2.set_title('Imagem Melhorada', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Salvar comparação
        if salvar:
            if nome_arquivo is None:
                nome_arquivo = 'comparacao_melhorias.png'
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            print(f"Comparação salva em: {nome_arquivo}")
        
        # Mostrar na tela
        if mostrar:
            plt.show()
        
        plt.close()
    
    def aplicar_melhorias_lote(self, pasta_entrada: str, pasta_saida: str, 
                              alfa: float = 1.2, beta: int = 10,
                              tipo_sharpening: str = 'basico'):
        """
        Aplica melhorias a todas as imagens de uma pasta
        
        Args:
            pasta_entrada (str): Pasta com imagens originais
            pasta_saida (str): Pasta para salvar imagens melhoradas
            alfa (float): Contraste
            beta (int): Brilho
            tipo_sharpening (str): Tipo de sharpening
        """
        # Criar pasta de saída
        if not os.path.exists(pasta_saida):
            os.makedirs(pasta_saida)
        
        extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        arquivos_processados = 0
        
        for arquivo in os.listdir(pasta_entrada):
            if arquivo.lower().endswith(extensoes_validas):
                caminho_entrada = os.path.join(pasta_entrada, arquivo)
                
                # Processar imagem
                original, melhorada = self.melhorar_imagem_completa(
                    caminho_entrada, alfa, beta, tipo_sharpening
                )
                
                if melhorada is not None:
                    # Salvar imagem melhorada
                    nome_saida = f"melhorada_{arquivo}"
                    caminho_saida = os.path.join(pasta_saida, nome_saida)
                    cv2.imwrite(caminho_saida, melhorada)
                    
                    arquivos_processados += 1
                    print(f"✓ {arquivo} → {nome_saida}")
        
        print(f"\nProcessamento concluído: {arquivos_processados} imagens melhoradas")
#%%
def aplicar_sharpening(image_path, output_folder):
    """Aplica filtro sharpening usando unsharp_mask em lote

    Args:
        image_path (string): image folder path
        output_folder (string): output folder path

    Returns:
        output_path: image output path
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Extensões de imagem suportadas
    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Processar cada arquivo na pasta
    for arquivo in os.listdir(image_path):
        if arquivo.lower().endswith(extensoes_validas):
            caminho_entrada = os.path.join(image_path, arquivo)
            caminho_saida = os.path.join(output_folder, f"{arquivo}")
            
            print(f"\nProcessando: {arquivo}")
            melhorada = preprocessamento_img.sharpening_unsharp_mask(caminho_entrada)
            if melhorada is not None:
                    # Salvar imagem melhorada
                    nome_saida = f"{arquivo}"
                    caminho_saida = os.path.join(caminho_saida, nome_saida)
                    cv2.imwrite(caminho_saida, melhorada)
                    
                    arquivos_processados += 1
                    print(f"✓ {arquivo} → {nome_saida}")
    return output_folder


state_farm_train = aplicar_sharpening(state_farm_train, f'{state_farm_train}/processed')
state_farm_test = aplicar_sharpening(state_farm_test, f'{state_farm_test}/processed')


#%%
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