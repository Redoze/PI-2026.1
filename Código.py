import cv2
import numpy as np
import time

imagem_de_entrada = "sao_jose.jpg" # ========== Trocar pelo nome da imagem que deseja analisar ====== AJUSTE - AJUSTE - AJUSTE ==========

nome_utilizado_no_arquivo_da_imagem = imagem_de_entrada.split('.')[0] # Recife.jpg -> "Recife"

imagem_bgr = cv2.imread(imagem_de_entrada)

if imagem_bgr is None:
    print("Imagem não encontrada.")
    exit()

imagem_hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

# ====================================================================================================
# Limiarização

captura_da_hora_atual_inicio = time.time()

print("="*40)
print("Aplicando Limiarização...")

# Padrão base:
# H - Cor: 30 - 45
# S - Saturação: > 40 - 255 (Ignora asfalto e partes brancas)
# V - Brilho: > 45 - 255 (Ignora sombras e asfalto escuro)

# -------------------------- H,  S,  V
limite_inferior = np.array([30, 40, 50]) # =========================================================== AJUSTE - AJUSTE - AJUSTE ==========
limite_superior = np.array([110, 255, 255]) # ======================================================== AJUSTE - AJUSTE - AJUSTE ==========

aplicacao_limiar = cv2.inRange(imagem_hsv, limite_inferior, limite_superior)

# ====================================================================================================
# Morfologia

print("Aplicando limpeza morfológica...")

# Remove pequenos ruídos soltos.
tamanho_filtro_mediana = 7 # ======================================================================== AJUSTE - AJUSTE - AJUSTE ==========
mascara_suavizada = cv2.medianBlur(aplicacao_limiar, tamanho_filtro_mediana)

# Abertura Morfológica (Erosão -> Dilatação) - passo extra para diminuir ruídos.
tamanho_da_abertura = 7 # ============================================================================ AJUSTE - AJUSTE - AJUSTE ==========
elemento_abertura = np.ones((tamanho_da_abertura, tamanho_da_abertura), np.uint8)
mascara_abertura = cv2.morphologyEx(mascara_suavizada, cv2.MORPH_OPEN, elemento_abertura)

# Fechamento Morfológico (Dilatação -> Erosão) - tentar fechar buracos na vegetação.
tamanho_do_fechamento = 19 # ========================================================================= AJUSTE - AJUSTE - AJUSTE ==========
elemento_fechamento = np.ones((tamanho_do_fechamento, tamanho_do_fechamento), np.uint8)
mascara_processada_final = cv2.morphologyEx(mascara_abertura, cv2.MORPH_CLOSE, elemento_fechamento)

captura_da_hora_atual_inicio_final = time.time()
tempo_total_gasto = captura_da_hora_atual_inicio_final - captura_da_hora_atual_inicio

# ====================================================================================================
# Equação

total_pixels_da_imagem = imagem_bgr.shape[0] * imagem_bgr.shape[1]

total_de_pixels_verdes = cv2.countNonZero(mascara_processada_final) # Contagem apenas dos pixels brancos.

porcentagem_de_verde = (total_de_pixels_verdes / total_pixels_da_imagem) * 100

print("="*40)
print(f"RESULTADO DA IMAGEM: '{imagem_de_entrada}'")
print("="*40)
print(f"Índice de Área Verde: {porcentagem_de_verde:.2f}%")
print(f"Tempo de Processamento: {tempo_total_gasto:.2f} segundos")
print("="*40)
print("PARÂMETROS")
print("="*40)
print(f"Limiarização Piso: {limite_inferior}")
print(f"Limiarização Teto: {limite_superior}")
print(f"Mediana:           {tamanho_filtro_mediana}x{tamanho_filtro_mediana}")
print(f"Abertura M.:       {tamanho_da_abertura}x{tamanho_da_abertura}")
print(f"Fechamento M.:     {tamanho_do_fechamento}x{tamanho_do_fechamento}")
print("="*40)
# ====================================================================================================
# Geração visual

# Cria uma imagem vazia (toda preta) do mesmo tamanho da img de entrada.
imagem_verde = np.zeros_like(imagem_bgr)

# Pinta essa imagem de verde apenas onde a máscara encontrou verde na img de entrada
imagem_verde[mascara_processada_final == 255] = [0, 255, 0] 

mapa_final_com_projecao_verde = cv2.addWeighted(imagem_bgr, 1.0, imagem_verde, 0.5, 0)

# ====================================================================================================
# Salvamento do resultado

# Salva as imagens usando o nome base da imagem de entrada
nome_mascara = f"mascara_{nome_utilizado_no_arquivo_da_imagem}.jpg"
nome_mapa = f"mapa_coberto_{nome_utilizado_no_arquivo_da_imagem}.jpg"

cv2.imwrite(nome_mascara, mascara_processada_final)
cv2.imwrite(nome_mapa, mapa_final_com_projecao_verde)
print(f"Arquivos salvos com sucesso: '{nome_mascara}' e '{nome_mapa}'")

# ====================================================================================================
# Exibição dos resultados

# Caso a janela esteja muito grande, diminua o tamanho dos parâmtros abaixo.
def mostrar_imagem_redimensionada(nome_janela, imagem, largura=1024, altura=1024):
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome_janela, largura, altura) 
    cv2.imshow(nome_janela, imagem)


mostrar_imagem_redimensionada("Imagem Original", imagem_bgr)
mostrar_imagem_redimensionada("Projecao da area verde", mapa_final_com_projecao_verde)
mostrar_imagem_redimensionada("Mascara de Limiarizacao", mascara_processada_final)

print("Pressione qualquer tecla em uma das janelas para finalizar.")
print("="*40)

cv2.waitKey(0)
cv2.destroyAllWindows()