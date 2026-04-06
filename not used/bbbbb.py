import cv2
import numpy as np

# --- Configurações ---
nome_arquivo_entrada = 'imgpreta.jpg' # Substitua pelo nome do seu arquivo
nome_arquivo_saida = 'conteudo_recortado.png'    # Nome do arquivo de saída
limiar_preto = 10 # Sensibilidade. Valores menores pegam tons mais escuros como 'conteúdo'
# --------------------

def recortar_conteudo_com_crop(caminho_entrada, caminho_saida):
    # 1. Carregar a imagem
    try:
        imagem = cv2.imread(caminho_entrada)
        if imagem is None:
            print(f"Erro: Não foi possível carregar a imagem em '{caminho_entrada}'")
            return
    except Exception as e:
        print(f"Erro ao abrir a imagem: {e}")
        return

    # 2. Converter para escala de cinza para facilitar o processamento
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # 3. Aplicar limiar (threshold) para separar o conteúdo do fundo
    # Qualquer pixel mais claro que limiar_preto vira branco (255), o resto vira preto (0)
    _, binarizada = cv2.threshold(cinza, limiar_preto, 255, cv2.THRESH_BINARY)

    # Opcional: Operações morfológicas para juntar partes desconexas
    # kernel = np.ones((5,5), np.uint8)
    # binarizada = cv2.dilate(binarizada, kernel, iterations=1)

    # 4. Encontrar contornos
    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        print("Nenhum conteúdo significativo encontrado.")
        return

    # 5. Encontrar o contorno com a maior área
    contorno_principal = max(contornos, key=cv2.contourArea)

    # 6. Obter o retângulo delimitador (Bounding Box) do contorno principal
    x, y, w, h = cv2.boundingRect(contorno_principal)

    # Adicionar uma pequena margem para não cortar no limite exato (opcional, mas recomendado)
    margem = 5
    y_min = max(0, y - margem)
    y_max = min(imagem.shape[0], y + h + margem)
    x_min = max(0, x - margem)
    x_max = min(imagem.shape[1], x + w + margem)

    # 7. Realizar o crop na imagem original
    imagem_recortada = imagem[y_min:y_max, x_min:x_max]

    # 8. Salvar a imagem recortada
    cv2.imwrite(caminho_saida, imagem_recortada)
    print(f"Imagem recortada salva com sucesso como '{caminho_saida}'")

# --- Execução do Script ---
recortar_conteudo_com_crop(nome_arquivo_entrada, nome_arquivo_saida)