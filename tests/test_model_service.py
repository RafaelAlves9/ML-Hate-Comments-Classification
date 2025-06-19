"""
Testes para o ModelService
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np
from services.model_service import ModelService
from config.settings import RESPONSE_LABELS


class TestModelService(unittest.TestCase):
    """Testes unitários para o ModelService"""
    
    def setUp(self):
        """Configuração inicial para cada teste"""
        self.service = ModelService()
        
    @patch('services.model_service.joblib.load')
    @patch('services.model_service.os.path.exists')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"accuracy": 0.95}')
    def test_load_model_success(self, mock_open, mock_exists, mock_joblib):
        """Testa carregamento bem-sucedido do modelo"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_model = Mock()
        mock_joblib.return_value = mock_model
        
        # Executar
        self.service.load_model()
        
        # Verificar
        self.assertIsNotNone(self.service.model)
        self.assertEqual(self.service.model_info, {"accuracy": 0.95})
        mock_joblib.assert_called_once()
        
    @patch('services.model_service.os.path.exists')
    def test_load_model_file_not_found(self, mock_exists):
        """Testa erro quando arquivo do modelo não existe"""
        # Configurar mock
        mock_exists.return_value = False
        
        # Executar e verificar
        with self.assertRaises(FileNotFoundError):
            self.service.load_model()
            
    def test_is_loaded(self):
        """Testa verificação se modelo está carregado"""
        # Modelo não carregado
        self.assertFalse(self.service.is_loaded())
        
        # Modelo carregado
        self.service.model = Mock()
        self.assertTrue(self.service.is_loaded())
        
    def test_get_model_info(self):
        """Testa obtenção de informações do modelo"""
        # Sem informações
        self.assertEqual(self.service.get_model_info(), {})
        
        # Com informações
        self.service.model_info = {"accuracy": 0.95}
        self.assertEqual(self.service.get_model_info(), {"accuracy": 0.95})
        
    @patch('services.model_service.preprocess_text')
    def test_predict_single_success(self, mock_preprocess):
        """Testa predição bem-sucedida de um comentário"""
        # Configurar mocks
        mock_preprocess.return_value = "texto processado"
        mock_model = Mock()
        mock_model.predict.return_value = [0]  # Discurso de ódio
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        self.service.model = mock_model
        
        # Executar
        result = self.service.predict_single("Comentário de teste")
        
        # Verificar
        self.assertFalse(result['error'])
        self.assertEqual(result['prediction'], RESPONSE_LABELS['HATE_SPEECH'])
        self.assertTrue(result['is_hate_speech'])
        self.assertEqual(result['confidence'], 80.0)
        self.assertEqual(result['confidence_method'], 'probability')
        
    @patch('services.model_service.preprocess_text')
    def test_predict_single_empty_comment(self, mock_preprocess):
        """Testa predição com comentário vazio"""
        # Configurar mock
        mock_preprocess.return_value = ""
        self.service.model = Mock()
        
        # Executar
        result = self.service.predict_single("")
        
        # Verificar
        self.assertTrue(result['error'])
        self.assertEqual(result['message'], 'Comentário inválido')
        
    def test_predict_batch(self):
        """Testa predição em lote"""
        # Configurar mock
        self.service.model = Mock()
        self.service.model.predict.return_value = [1]  # Não é discurso de ódio
        self.service.model.predict_proba.return_value = [[0.9, 0.1]]
        
        # Executar
        comments = ["Comentário 1", "Comentário 2", None, "Comentário 3"]
        results = self.service.predict_batch(comments)
        
        # Verificar
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0]['prediction'], RESPONSE_LABELS['NOT_HATE_SPEECH'])
        self.assertEqual(results[2]['error'], 'Comentário inválido')
        
    def test_calculate_confidence_probability(self):
        """Testa cálculo de confiança usando probabilidade"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        self.service.model = mock_model
        
        # Executar
        result = self.service._calculate_confidence("texto")
        
        # Verificar
        self.assertEqual(result['confidence'], 70.0)
        self.assertEqual(result['method'], 'probability')
        
    def test_calculate_confidence_decision_function(self):
        """Testa cálculo de confiança usando decision function"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict_proba.side_effect = AttributeError()
        mock_model.decision_function.return_value = [2.0]
        self.service.model = mock_model
        
        # Executar
        result = self.service._calculate_confidence("texto")
        
        # Verificar
        self.assertGreater(result['confidence'], 50.0)
        self.assertEqual(result['method'], 'decision_function')
        
    def test_calculate_confidence_default(self):
        """Testa cálculo de confiança com valor padrão"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict_proba.side_effect = AttributeError()
        mock_model.decision_function.side_effect = Exception()
        self.service.model = mock_model
        
        # Executar
        result = self.service._calculate_confidence("texto")
        
        # Verificar
        self.assertEqual(result['confidence'], 50.0)
        self.assertEqual(result['method'], 'default')


if __name__ == '__main__':
    unittest.main() 