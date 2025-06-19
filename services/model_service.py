"""
Serviço responsável pelo carregamento e predições do modelo
"""
import joblib
import json
import os
import numpy as np
from datetime import datetime
from config.settings import MODEL_PATH, MODEL_INFO_PATH, RESPONSE_LABELS, logger
from utils.text_preprocessor import preprocess_text


class ModelService:
    """Serviço responsável pelo modelo de classificação de discurso de ódio"""
    
    def __init__(self):
        self.model = None
        self.model_info = None
        
    def load_model(self):
        """
        Carrega o modelo e suas informações do disco.
        
        Raises:
            FileNotFoundError: Se o arquivo do modelo não for encontrado
            Exception: Se houver erro ao carregar o modelo
        """
        try:
            # Carregar modelo
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                logger.info("✅ Modelo carregado com sucesso!")
            else:
                raise FileNotFoundError(f"Arquivo do modelo não encontrado: {MODEL_PATH}")
            
            # Carregar informações do modelo
            if os.path.exists(MODEL_INFO_PATH):
                with open(MODEL_INFO_PATH, 'r') as f:
                    self.model_info = json.load(f)
                logger.info("✅ Informações do modelo carregadas com sucesso!")
            else:
                logger.warning("⚠️ Arquivo de informações do modelo não encontrado!")
                self.model_info = {}
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar o modelo: {e}")
            raise e
    
    def is_loaded(self):
        """Verifica se o modelo está carregado"""
        return self.model is not None
    
    def get_model_info(self):
        """Retorna as informações do modelo"""
        return self.model_info if self.model_info else {}
    
    def predict_single(self, comment):
        """
        Faz predição para um único comentário.
        
        Args:
            comment: Comentário a ser classificado
            
        Returns:
            dict: Resultado da predição com confiança e outros metadados
        """
        try:
            # Preprocessar texto
            processed_comment = preprocess_text(comment)
            
            if not processed_comment.strip():
                return {
                    'error': True,
                    'message': 'Comentário inválido',
                    'result': None
                }
            
            # Fazer predição
            prediction = self.model.predict([processed_comment])[0]
            
            # Calcular confiança
            confidence_data = self._calculate_confidence(processed_comment)
            
            # Determinar resultado
            result = RESPONSE_LABELS['NOT_HATE_SPEECH'] if prediction == 1 else RESPONSE_LABELS['HATE_SPEECH']
            
            return {
                'error': False,
                'comment': comment,
                'prediction': result,
                'is_hate_speech': result == RESPONSE_LABELS['HATE_SPEECH'],
                'confidence': confidence_data['confidence'],
                'confidence_method': confidence_data['method'],
                'processed_comment': processed_comment,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                'error': True,
                'message': str(e),
                'result': None
            }
    
    def _calculate_confidence(self, processed_text):
        """
        Calcula a confiança da predição.
        
        Args:
            processed_text: Texto já processado
            
        Returns:
            dict: Dicionário com confiança e método usado
        """
        confidence = 0.0
        method = "none"
        
        try:
            # Tentar usar probabilidade
            probability = self.model.predict_proba([processed_text])[0]
            confidence = max(probability) * 100
            method = "probability"
        except AttributeError:
            try:
                # Tentar usar decision function
                decision = self.model.decision_function([processed_text])[0]
                confidence = 100 * (1 / (1 + np.exp(-abs(decision))))
                method = "decision_function"
            except:
                # Valor padrão
                confidence = 50.0
                method = "default"
        
        return {
            'confidence': round(confidence, 2),
            'method': method
        }


# Instância singleton do serviço
model_service = ModelService() 