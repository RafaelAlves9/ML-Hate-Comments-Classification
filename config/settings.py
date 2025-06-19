"""
Configurações do projeto
"""
import os
from datetime import datetime
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações do servidor
HOST = '0.0.0.0'
PORT = 5000

# Caminhos dos arquivos
MODEL_PATH = 'hate_speech_classifier_model.pkl'
MODEL_INFO_PATH = 'model_info.json'

# Limites da API
MAX_BATCH_SIZE = 100

# Mensagens de erro padrão
ERROR_MESSAGES = {
    'MODEL_NOT_LOADED': 'O modelo de ML não foi carregado corretamente',
    'INVALID_DATA': 'Requisição deve conter JSON válido',
    'MISSING_COMMENT': 'Campo "comment" é obrigatório',
    'INVALID_COMMENT': 'Campo "comment" deve ser uma string não vazia',
    'INTERNAL_ERROR': 'Erro interno do servidor',
    'MISSING_COMMENTS': 'Campo "comments" é obrigatório',
    'INVALID_FORMAT': 'Campo "comments" deve ser uma lista',
    'TOO_MANY_COMMENTS': 'Máximo de 100 comentários por requisição',
    'NOT_FOUND': 'Endpoint não encontrado',
    'METHOD_NOT_ALLOWED': 'Método não permitido',
    'MODEL_INFO_NOT_AVAILABLE': 'Informações do modelo não disponíveis'
}

# Configurações de resposta
RESPONSE_LABELS = {
    'HATE_SPEECH': 'É discurso de ódio',
    'NOT_HATE_SPEECH': 'Não é discurso de ódio'
} 