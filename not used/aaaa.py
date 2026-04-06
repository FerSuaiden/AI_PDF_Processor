import os
from pdf2image import convert_from_path
from PIL import Image

# === CONFIGURAÇÕES ===
ARQUIVO_PDF = "fuvest2026-fase1-prova-V1.pdf"
PASTA_SAIDA = "banco_imagens_fuvest_final"

# === O TEU JSON INTEGRAL (ADAPTADO) ===
mapeamento_questoes = [
    {"question_number": 5, "page": 3, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 10, "page": 5, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 13, "page": 7, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 14, "page": 7, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 17, "page": 8, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 19, "page": 9, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 22, "page": 10, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 23, "page": 10, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 24, "page": 11, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 25, "page": 11, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 26, "page": 12, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 27, "page": 12, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 28, "page": 13, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 34, "page": 15, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 35, "page": 15, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 36, "page": 16, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 37, "page": 16, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 43, "page": 19, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 44, "page": 20, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 46, "page": 21, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 52, "page": 24, "image_coordinates": [155, 120, 845, 850]},
    {"question_number": 53, "page": 25, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 57, "page": 26, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 66, "page": 28, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 67, "page": 29, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 68, "page": 29, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 71, "page": 30, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 72, "page": 31, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 73, "page": 31, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 74, "page": 32, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 75, "page": 32, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 76, "page": 33, "image_coordinates": [0, 0, 1000, 500]},
    {"question_number": 81, "page": 34, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 84, "page": 35, "image_coordinates": [0, 500, 1000, 1000]},
    {"question_number": 85, "page": 36, "image_coordinates": [0, 0, 1000, 500]}
]
def executar_crop_fuvest(pdf_path, mapeamento, pasta_final):
    if not os.path.exists(pasta_final):
        os.makedirs(pasta_final)
        print(f"Pasta '{pasta_final}' criada.")

    # Agrupa por página para evitar processar a mesma folha várias vezes
    paginas_alvo = sorted(list(set(q["page"] for q in mapeamento)))
    
    print(f"--- Iniciando extração de {len(mapeamento)} imagens ---")

    for num_pag in paginas_alvo:
        print(f"Processando página {num_pag}...")
        
        try:
            # Converte a página para imagem (300 DPI)
            paginas = convert_from_path(pdf_path, first_page=num_pag, last_page=num_pag, dpi=300)
            img_full = paginas[0]
            largura, altura = img_full.size

            # Filtra as questões que pertencem a esta página
            questoes_da_pagina = [q for q in mapeamento if q["page"] == num_pag]

            for q in questoes_da_pagina:
                # ACESSO À CHAVE CORRETA: 'image_coordinates'
                ymin, xmin, ymax, xmax = q["image_coordinates"]
                
                # Conversão proporcional para pixels reais
                left = (xmin / 1000) * largura
                top = (ymin / 1000) * altura
                right = (xmax / 1000) * largura
                bottom = (ymax / 1000) * altura

                # Faz o recorte
                recorte = img_full.crop((left, top, right, bottom))
                
                # Guarda com o número da questão formatado
                nome_ficheiro = f"questao_{q['question_number']:02d}.png"
                caminho_final = os.path.join(pasta_final, nome_ficheiro)
                
                recorte.save(caminho_final)
                print(f"   [SUCESSO] {nome_ficheiro} gerada.")
                
        except Exception as e:
            print(f"   [ERRO] Falha ao processar página {num_pag}: {e}")

if __name__ == "__main__":
    if os.path.exists(ARQUIVO_PDF):
        executar_crop_fuvest(ARQUIVO_PDF, mapeamento_questoes, PASTA_SAIDA)
        print(f"\nConcluído! Imagens em: {PASTA_SAIDA}")
    else:
        print(f"Erro: O ficheiro '{ARQUIVO_PDF}' não foi encontrado.")