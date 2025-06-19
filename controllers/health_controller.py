"""
Controller responsável pelas rotas de health check e informações
"""
from flask import jsonify
from datetime import datetime
from services.model_service import model_service
from config.settings import ERROR_MESSAGES


def home():
    """Endpoint principal com status da API"""
    return jsonify({
        'status': 'API funcionando!',
        'model_loaded': model_service.is_loaded(),
        'model_info': model_service.get_model_info(),
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'batch_predict': '/predict/batch (POST)'
        }
    })


def health_check():
    """Endpoint de health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_service.is_loaded(),
        'timestamp': datetime.now().isoformat()
    })


def handle_404(error):
    """Handler para erro 404"""
    return jsonify({
        'error': 'Endpoint não encontrado',
        'message': ERROR_MESSAGES['NOT_FOUND']
    }), 404


def handle_405(error):
    """Handler para erro 405"""
    return jsonify({
        'error': 'Método não permitido', 
        'message': ERROR_MESSAGES['METHOD_NOT_ALLOWED']
    }), 405


def handle_500(error):
    """Handler para erro 500"""
    return jsonify({
        'error': 'Erro interno do servidor',
        'message': ERROR_MESSAGES['INTERNAL_ERROR']
    }), 500 