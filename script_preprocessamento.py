#%%
import cv2 as cv2
import os
import sys
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

# %%
def redimensionar_imagem(caminho_entrada, caminho_saida, nova_largura, nova_altura):
    """
    Redimensiona uma imagem usando OpenCV
    
    Args:
        caminho_entrada (str): Caminho da imagem original
        caminho_saida (str): Caminho onde salvar a imagem redimensionada
        nova_largura (int): Nova largura desejada
        nova_altura (int): Nova altura desejada
    Não cheguei 
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
state_farm_train = os.path.join('datasets','state-farm-distracted-driver-detection','imgs','train')
#%%
pastas = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
#%%
for i in pastas:
    entrada = os.path.join(state_farm_train, f'{i}')
    saida = os.path.join(state_farm_train, 'redimensioned', f'{i}')
    redimensionar_multiplas_imagens(pasta_entrada=entrada, pasta_saida=saida, largura=80, altura=80)
#%%
state_farm_test = os.path.join('datasets','state-farm-distracted-driver-detection','imgs','test')
saida = os.path.join(state_farm_test, 'redimensioned')
#%%
redimensionar_multiplas_imagens(pasta_entrada=state_farm_test, pasta_saida=saida, largura=80, altura=80)
### Segunda etapa: detectar faces e olhos
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
    
    def detectar_faces_olhos(self, imagem_path, output_folder, salvar_resultado=True):
        try:
            img = cv2.imread(imagem_path)
            if img is None:
                print(f"Erro: Não foi possível carregar a imagem {imagem_path}")
                return 0, 0, None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detectores de face
            face_detections = []
            for face_cascade in [self.face_cascade1, self.face_cascade2, self.face_cascade3, self.face_cascade4]:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=2,
                    minSize=(10, 10),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                face_detections.extend(faces)

            # Eliminar sobreposições usando groupRectangles (requer lista com pelo menos dois itens iguais)
            faces_filtered, _ = cv2.groupRectangles(face_detections * 2, groupThreshold=1, eps=0.2)

            total_olhos = 0
            print(f"Faces detectadas (após filtragem): {len(faces_filtered)}")

            for i, (x, y, w, h) in enumerate(faces_filtered):
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                # Detectores de olhos
                eye_detections = []
                for eye_cascade in [self.eye_cascade1, self.eye_cascade2]:
                    eyes = eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.05,
                        minNeighbors=2,
                        minSize=(10, 10)
                    )
                    eye_detections.extend(eyes)

                # Remover sobreposição entre olhos detectados
                eyes_filtered, _ = cv2.groupRectangles(eye_detections * 2, groupThreshold=1, eps=0.2)
                print(f"  Olhos detectados na face {i+1}: {len(eyes_filtered)}")

                for (ex, ey, ew, eh) in eyes_filtered:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                total_olhos += len(eyes_filtered)

            if salvar_resultado:
                nome_base = os.path.splitext(os.path.basename(imagem_path))[0]
                extensao = os.path.splitext(imagem_path)[1]
                caminho_saida = os.path.join(output_folder, f"{nome_base}_detectado{extensao}")
                cv2.imwrite(caminho_saida, img)
                print(f"Imagem com detecções salva em: {caminho_saida}")

            return len(faces_filtered), total_olhos, img

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
            caminho_saida = output_folder
            
            print(f"\nProcessando: {arquivo}")
            detector.detectar_faces_olhos(imagem_path=caminho_entrada, output_folder=caminho_saida, salvar_resultado=True)
    return caminho_saida
# #%%
# img_test = state_farm_test + '/redimensionada/img_1.jpg'
# #%%
# HCC().detectar_faces_olhos(imagem_path=img_test, output_folder='./')
#%%
#%%
state_farm_train = os.path.join('datasets','state-farm-distracted-driver-detection','imgs','train', 'redimensioned')
#%%
for i in pastas:
    entrada = os.path.join(state_farm_train, f'{i}')
    saida = os.path.join(state_farm_train, 'enhanced', f'{i}')
    enhance(entrada, saida)                            
#%%
state_farm_test = os.path.join('datasets','state-farm-distracted-driver-detection','imgs','test', 'redimensioned')
saida = os.path.join(state_farm_test, 'enhanced')
enhance(state_farm_test, saida)
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
state_farm_train = os.path.join('datasets','state-farm-distracted-driver-detection','imgs','train', 'enhanced')
for i in pastas:
    entrada= os.path.join(state_farm_train, f'{i}')
    saida = os.path.join(state_farm_train, 'processed', f'{i}')
    preprocessamento_img().aplicar_melhorias_lote(pasta_entrada=entrada, pasta_saida=saida, tipo_sharpening='unsharp_mask')
#%%
state_farm_test = os.path.join('datasets','state-farm-distracted-driver-detection','imgs','test', 'enhanced')
saida = os.path.join(state_farm_test, 'processed')
preprocessamento_img().aplicar_melhorias_lote(pasta_entrada=state_farm_test, pasta_saida=saida, tipo_sharpening='unsharp_mask')
