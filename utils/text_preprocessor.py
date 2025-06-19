"""
Utilitários para preprocessamento de texto
"""
import re
import string
import pandas as pd


def preprocess_text(text):
    """
    Preprocessa o texto removendo pontuação, números e espaços extras.
    
    Args:
        text: Texto a ser processado
        
    Returns:
        str: Texto processado
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Converter para string e minúsculas
    text = str(text).lower()
    
    # Remover pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remover números
    text = re.sub(r'\d+', '', text)
    
    # Remover espaços extras
    text = ' '.join(text.split())
    
    return text


def validate_comment(comment):
    """
    Valida se o comentário é válido para processamento.
    
    Args:
        comment: Comentário a ser validado
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not comment or not isinstance(comment, str):
        return False, "Comentário deve ser uma string não vazia"
    
    processed = preprocess_text(comment)
    if not processed.strip():
        return False, "Comentário inválido após processamento"
    
    return True, None 