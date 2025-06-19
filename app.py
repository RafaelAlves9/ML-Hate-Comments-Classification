from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import re
import string
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging


# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permite requisições de qualquer origem (ajuste conforme necessário)

# Variáveis globais para o modelo e informações
model = None
model_info = None

def load_model():
    """Carrega o modelo treinado e suas informações"""
    global model, model_info
    
    try:
        # Carregar o modelo
        if os.path.exists('hate_speech_classifier_model.pkl'):
            model = joblib.load('hate_speech_classifier_model.pkl')
            logger.info("✅ Modelo carregado com sucesso!")
        else:
            raise FileNotFoundError("Arquivo do modelo não encontrado!")
        
        # Carregar informações do modelo
        if os.path.exists('model_info.json'):
            with open('model_info.json', 'r') as f:
                model_info = json.load(f)
            logger.info("✅ Informações do modelo carregadas com sucesso!")
        else:
            logger.warning("⚠️ Arquivo de informações do modelo não encontrado!")
            model_info = {}
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar o modelo: {e}")
        raise e

def preprocess_text(text):
    """
    Função para pré-processar texto (mesma do notebook):
    - Converte para minúsculas
    - Remove pontuação
    - Remove números
    - Remove espaços extras
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

def predict_hate_speech(comment, model):
    """
    Função para predizer se um comentário é discurso de ódio
    Compatível com LinearSVC (sem predict_proba)
    """
    try:
        # Pré-processar o comentário
        processed_comment = preprocess_text(comment)
        
        if not processed_comment.strip():
            return "Comentário inválido", 0.0, "error"
        
        # Fazer predição
        prediction = model.predict([processed_comment])[0]
        
        # Tentar obter confiança
        confidence = 0.0
        confidence_method = "none"
        
        try:
            # Tenta usar predict_proba (para modelos como SVC, Naive Bayes, etc.)
            probability = model.predict_proba([processed_comment])[0]
            confidence = max(probability) * 100
            confidence_method = "probability"
        except AttributeError:
            try:
                # Para LinearSVC e outros que não têm predict_proba
                decision = model.decision_function([processed_comment])[0]
                # Normaliza usando função sigmoide para [0, 1]
                confidence = 100 * (1 / (1 + np.exp(-abs(decision))))
                confidence_method = "decision_function"
            except:
                # Se nenhum método funcionar, usa confiança padrão
                confidence = 50.0
                confidence_method = "default"
        
        # Interpretar resultado
        result = "Não é discurso de ódio" if prediction == 1 else "É discurso de ódio"
        
        return result, confidence, confidence_method
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return "Erro na predição", 0.0, "error"

@app.route('/', methods=['GET'])
def home():
    """Endpoint de status da API"""
    return jsonify({
        'status': 'API funcionando!',
        'model_loaded': model is not None,
        'model_info': model_info if model_info else {},
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'model_info': '/model-info (GET)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Endpoint para obter informações do modelo"""
    if model_info:
        return jsonify(model_info)
    else:
        return jsonify({'error': 'Informações do modelo não disponíveis'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal para classificação de comentários"""
    try:
        # Verificar se o modelo está carregado
        if model is None:
            return jsonify({
                'error': 'Modelo não carregado',
                'message': 'O modelo de ML não foi carregado corretamente'
            }), 500
        
        # Obter dados da requisição
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Dados inválidos',
                'message': 'Requisição deve conter JSON válido'
            }), 400
        
        if 'comment' not in data:
            return jsonify({
                'error': 'Campo obrigatório ausente',
                'message': 'Campo "comment" é obrigatório'
            }), 400
        
        comment = data['comment']
        
        if not comment or not isinstance(comment, str):
            return jsonify({
                'error': 'Comentário inválido',
                'message': 'Campo "comment" deve ser uma string não vazia'
            }), 400
        
        # Fazer a predição
        result, confidence, confidence_method = predict_hate_speech(comment, model)
        
        if confidence_method == "error":
            return jsonify({
                'error': 'Erro na predição',
                'message': result
            }), 500
        
        # Preparar resposta
        response = {
            'comment': comment,
            'prediction': result,
            'is_hate_speech': result == "É discurso de ódio",
            'confidence': round(confidence, 2),
            'confidence_method': confidence_method,
            'processed_comment': preprocess_text(comment),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log da predição
        logger.info(f"Predição realizada: {result} (confiança: {confidence:.2f}%)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro no endpoint /predict: {e}")
        return jsonify({
            'error': 'Erro interno do servidor',
            'message': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Endpoint para classificação em lote de múltiplos comentários"""
    try:
        if model is None:
            return jsonify({
                'error': 'Modelo não carregado',
                'message': 'O modelo de ML não foi carregado corretamente'
            }), 500
        
        data = request.get_json()
        
        if not data or 'comments' not in data:
            return jsonify({
                'error': 'Campo obrigatório ausente',
                'message': 'Campo "comments" é obrigatório'
            }), 400
        
        comments = data['comments']
        
        if not isinstance(comments, list):
            return jsonify({
                'error': 'Formato inválido',
                'message': 'Campo "comments" deve ser uma lista'
            }), 400
        
        if len(comments) > 100:  # Limite de segurança
            return jsonify({
                'error': 'Muitos comentários',
                'message': 'Máximo de 100 comentários por requisição'
            }), 400
        
        results = []
        
        for i, comment in enumerate(comments):
            if not comment or not isinstance(comment, str):
                results.append({
                    'index': i,
                    'comment': comment,
                    'error': 'Comentário inválido'
                })
                continue
            
            result, confidence, confidence_method = predict_hate_speech(comment, model)
            
            results.append({
                'index': i,
                'comment': comment,
                'prediction': result,
                'is_hate_speech': result == "É discurso de ódio",
                'confidence': round(confidence, 2),
                'confidence_method': confidence_method
            })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro no endpoint /predict/batch: {e}")
        return jsonify({
            'error': 'Erro interno do servidor',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint não encontrado',
        'message': 'Verifique a URL e o método HTTP'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Método não permitido',
        'message': 'Verifique o método HTTP da requisição'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Erro interno do servidor',
        'message': 'Ocorreu um erro inesperado'
    }), 500

if __name__ == '__main__':
    try:
        # Carregar o modelo na inicialização
        load_model()
        
        # Iniciar o servidor
        print("🚀 Iniciando servidor Flask...")
        print("📊 API de Classificação de Discurso de Ódio")
        print("🔗 Endpoints disponíveis:")
        print("   GET  / - Status da API")
        print("   GET  /health - Health check")
        print("   GET  /model-info - Informações do modelo")
        print("   POST /predict - Classificar um comentário")
        print("   POST /predict/batch - Classificar múltiplos comentários")
        
        app.run(host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar o servidor: {e}")
        print("Certifique-se de que os arquivos do modelo estão no diretório correto!")