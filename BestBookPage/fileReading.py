import fitz  # PyMuPDF
import re

def extrair_numero_pagina(texto):
    """
    Extrai o número da página do texto, procurando-o em qualquer parte da página.
    Retorna o número da página se encontrado; caso contrário, retorna None.
    """
    padrao_pagina = re.compile(r"\b\d+\b")  # Procura por números isolados
    # Busca o primeiro número isolado em qualquer linha do texto
    matches = padrao_pagina.findall(texto)
    if matches:
        return matches[0]  # Retorna o primeiro número encontrado
    return None

def transcrever_pdf_com_pagina_exibida(pdf_path, output_path, start_page=13):
    documento = fitz.open(pdf_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for numero_pagina in range(start_page - 1, documento.page_count):
            pagina = documento.load_page(numero_pagina)
            texto = pagina.get_text()
            
            # Tenta extrair o número exibido na página em qualquer parte do texto
            numero_exibido = extrair_numero_pagina(texto)
            numero_real = numero_exibido if numero_exibido else f"{numero_pagina + 1}"
            
            # Escreve o número da página e o texto no arquivo de saída
            f.write(f"--- Página {numero_real} ---\n")
            f.write(texto)
            f.write("\n\n")
    
    print(f"Transcrição concluída e salva em {output_path}")

# Exemplo de uso:
pdf_path = "humanas.pdf"  # Substitua pelo caminho do seu PDF
output_path = "transcricao_livro_ajustada.txt"  # Nome do arquivo de saída
transcrever_pdf_com_pagina_exibida(pdf_path, output_path, start_page=13)
