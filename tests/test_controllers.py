"""
Testes para os Controllers
"""
import unittest
from unittest.mock import Mock, patch
import json
from flask import Flask, request
from controllers import prediction_controller, health_controller


class TestControllers(unittest.TestCase):
    """Testes de integração para os controllers"""
    
    def setUp(self):
        """Configuração inicial para cada teste"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Registrar rotas
        self.app.add_url_rule('/', 'home', health_controller.home, methods=['GET'])
        self.app.add_url_rule('/health', 'health_check', health_controller.health_check, methods=['GET'])
        self.app.add_url_rule('/model-info', 'get_model_info', health_controller.get_model_info, methods=['GET'])
        self.app.add_url_rule('/predict', 'predict', prediction_controller.predict, methods=['POST'])
        self.app.add_url_rule('/predict/batch', 'predict_batch', prediction_controller.predict_batch, methods=['POST'])
        
    @patch('controllers.health_controller.model_service')
    def test_home_endpoint(self, mock_service):
        """Testa endpoint home"""
        # Configurar mock
        mock_service.is_loaded.return_value = True
        mock_service.get_model_info.return_value = {"accuracy": 0.95}
        
        # Executar
        response = self.client.get('/')
        data = json.loads(response.data)
        
        # Verificar
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'API funcionando!')
        self.assertTrue(data['model_loaded'])
        self.assertIn('endpoints', data)
        
    @patch('controllers.health_controller.model_service')
    def test_health_check_endpoint(self, mock_service):
        """Testa endpoint de health check"""
        # Configurar mock
        mock_service.is_loaded.return_value = True
        
        # Executar
        response = self.client.get('/health')
        data = json.loads(response.data)
        
        # Verificar
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])
        self.assertIn('timestamp', data)
        
    @patch('controllers.health_controller.model_service')
    def test_model_info_endpoint(self, mock_service):
        """Testa endpoint de informações do modelo"""
        # Configurar mock
        mock_service.get_model_info.return_value = {"accuracy": 0.95, "algorithm": "SVM"}
        
        # Executar
        response = self.client.get('/model-info')
        data = json.loads(response.data)
        
        # Verificar
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['accuracy'], 0.95)
        self.assertEqual(data['algorithm'], 'SVM')
        
    @patch('controllers.prediction_controller.model_service')
    def test_predict_endpoint_success(self, mock_service):
        """Testa predição bem-sucedida"""
        # Configurar mock
        mock_service.is_loaded.return_value = True
        mock_service.predict_single.return_value = {
            'error': False,
            'comment': 'Teste',
            'prediction': 'Não é discurso de ódio',
            'is_hate_speech': False,
            'confidence': 85.5,
            'confidence_method': 'probability',
            'processed_comment': 'teste',
            'timestamp': '2024-01-01T00:00:00'
        }
        
        # Executar
        response = self.client.post('/predict',
                                   json={'comment': 'Teste'},
                                   content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['prediction'], 'Não é discurso de ódio')
        self.assertFalse(data['is_hate_speech'])
        self.assertEqual(data['confidence'], 85.5)
        
    @patch('controllers.prediction_controller.model_service')
    def test_predict_endpoint_model_not_loaded(self, mock_service):
        """Testa erro quando modelo não está carregado"""
        # Configurar mock
        mock_service.is_loaded.return_value = False
        
        # Executar
        response = self.client.post('/predict',
                                   json={'comment': 'Teste'},
                                   content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        self.assertEqual(response.status_code, 500)
        self.assertEqual(data['error'], 'Modelo não carregado')
        
    @patch('controllers.prediction_controller.model_service')
    def test_predict_batch_endpoint_success(self, mock_service):
        """Testa predição em lote bem-sucedida"""
        # Configurar mock
        mock_service.is_loaded.return_value = True
        mock_service.predict_batch.return_value = [
            {
                'index': 0,
                'comment': 'Comentário 1',
                'prediction': 'Não é discurso de ódio',
                'is_hate_speech': False,
                'confidence': 90.0,
                'confidence_method': 'probability'
            },
            {
                'index': 1,
                'comment': 'Comentário 2',
                'prediction': 'É discurso de ódio',
                'is_hate_speech': True,
                'confidence': 75.0,
                'confidence_method': 'probability'
            }
        ]
        
        # Executar
        response = self.client.post('/predict/batch',
                                   json={'comments': ['Comentário 1', 'Comentário 2']},
                                   content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data['results']), 2)
        self.assertEqual(data['total_processed'], 2)
        self.assertIn('timestamp', data)
        
    def test_predict_endpoint_missing_comment(self):
        """Testa erro quando comentário está ausente"""
        # Executar
        response = self.client.post('/predict',
                                   json={},
                                   content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'Campo obrigatório ausente')
        
    def test_predict_batch_endpoint_too_many_comments(self):
        """Testa erro quando há muitos comentários"""
        # Criar lista com mais de 100 comentários
        comments = [f'Comentário {i}' for i in range(101)]
        
        # Executar
        response = self.client.post('/predict/batch',
                                   json={'comments': comments},
                                   content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'Muitos comentários')


if __name__ == '__main__':
    unittest.main() 