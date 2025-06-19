"""
Controller responsável pelas rotas de predição
"""
from flask import jsonify, request
from datetime import datetime
from services.model_service import model_service
from config.settings import ERROR_MESSAGES, MAX_BATCH_SIZE, logger
from utils.text_preprocessor import validate_comment


def predict():
    """Endpoint para predição de um único comentário"""
    try:
        # Verificar se modelo está carregado
        if not model_service.is_loaded():
            return jsonify({
                'error': 'Modelo não carregado',
                'message': ERROR_MESSAGES['MODEL_NOT_LOADED']
            }), 500
        
        # Obter dados da requisição
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Dados inválidos',
                'message': ERROR_MESSAGES['INVALID_DATA']
            }), 400
        
        if 'comment' not in data:
            return jsonify({
                'error': 'Campo obrigatório ausente',
                'message': ERROR_MESSAGES['MISSING_COMMENT']
            }), 400
        
        comment = data['comment']
        
        # Validar comentário
        is_valid, error_msg = validate_comment(comment)
        if not is_valid:
            return jsonify({
                'error': 'Comentário inválido',
                'message': error_msg
            }), 400
        
        # Fazer predição
        result = model_service.predict_single(comment)
        
        if result['error']:
            return jsonify({
                'error': 'Erro na predição',
                'message': result['message']
            }), 500
        
        # Remover campo de erro do resultado
        del result['error']
        
        logger.info(f"Predição realizada: {result['prediction']} (confiança: {result['confidence']}%)")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro no endpoint /predict: {e}")
        return jsonify({
            'error': 'Erro interno do servidor',
            'message': str(e)
        }), 500


def predict_batch():
    """Endpoint para predição de múltiplos comentários"""
    try:
        # Verificar se modelo está carregado
        if not model_service.is_loaded():
            return jsonify({
                'error': 'Modelo não carregado',
                'message': ERROR_MESSAGES['MODEL_NOT_LOADED']
            }), 500
        
        # Obter dados da requisição
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Dados inválidos',
                'message': ERROR_MESSAGES['INVALID_DATA']
            }), 400
        
        if 'comments' not in data:
            return jsonify({
                'error': 'Campo obrigatório ausente',
                'message': 'O campo "comments" é obrigatório'
            }), 400
        
        comments = data['comments']
        
        if not isinstance(comments, list):
            return jsonify({
                'error': 'Formato inválido',
                'message': 'O campo "comments" deve ser uma lista'
            }), 400
        
        if len(comments) == 0:
            return jsonify({
                'error': 'Lista vazia',
                'message': 'A lista de comentários não pode estar vazia'
            }), 400
        
        if len(comments) > MAX_BATCH_SIZE:
            return jsonify({
                'error': 'Muitos comentários',
                'message': f'Máximo de {MAX_BATCH_SIZE} comentários por requisição'
            }), 400
        
        # Fazer predições
        results = model_service.predict_batch(comments)
        
        # Preparar resposta
        response = {
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Predição em lote realizada: {len(results)} comentários processados")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro no endpoint /predict/batch: {e}")
        return jsonify({
            'error': 'Erro interno do servidor',
            'message': str(e)
        }), 500