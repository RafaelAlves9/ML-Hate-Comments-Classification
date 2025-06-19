"""
Testes para os Controllers usando PyTest
"""
import pytest
from unittest.mock import Mock, patch
import json
from flask import Flask
from controllers import prediction_controller, health_controller


class TestControllers:
    """Testes de integração para os controllers"""
    
    @pytest.fixture
    def app(self):
        """Fixture para criar aplicação Flask de teste"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        
        # Registrar rotas
        app.add_url_rule('/', 'home', health_controller.home, methods=['GET'])
        app.add_url_rule('/health', 'health_check', health_controller.health_check, methods=['GET'])
        app.add_url_rule('/model-info', 'get_model_info', health_controller.get_model_info, methods=['GET'])
        app.add_url_rule('/predict', 'predict', prediction_controller.predict, methods=['POST'])
        app.add_url_rule('/predict/batch', 'predict_batch', prediction_controller.predict_batch, methods=['POST'])
        
        return app
    
    @pytest.fixture
    def client(self, app):
        """Fixture para criar cliente de teste"""
        return app.test_client()
    
    @patch('controllers.health_controller.model_service')
    def test_home_endpoint(self, mock_service, client):
        """Testa endpoint home"""
        # Configurar mock
        mock_service.is_loaded.return_value = True
        mock_service.get_model_info.return_value = {"accuracy": 0.95}
        
        # Executar
        response = client.get('/')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 200
        assert data['status'] == 'API funcionando!'
        assert data['model_loaded'] is True
        assert 'endpoints' in data
    
    @patch('controllers.health_controller.model_service')
    def test_health_check_endpoint(self, mock_service, client):
        """Testa endpoint de health check"""
        # Configurar mock
        mock_service.is_loaded.return_value = True
        
        # Executar
        response = client.get('/health')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 200
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True
        assert 'timestamp' in data
    
    @patch('controllers.health_controller.model_service')
    def test_model_info_endpoint(self, mock_service, client):
        """Testa endpoint de informações do modelo"""
        # Configurar mock
        mock_service.get_model_info.return_value = {"accuracy": 0.95, "algorithm": "SVM"}
        
        # Executar
        response = client.get('/model-info')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 200
        assert data['accuracy'] == 0.95
        assert data['algorithm'] == 'SVM'
    
    @patch('controllers.prediction_controller.model_service')
    def test_predict_endpoint_success(self, mock_service, client):
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
        response = client.post('/predict',
                              json={'comment': 'Teste'},
                              content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 200
        assert data['prediction'] == 'Não é discurso de ódio'
        assert data['is_hate_speech'] is False
        assert data['confidence'] == 85.5
    
    @patch('controllers.prediction_controller.model_service')
    def test_predict_endpoint_model_not_loaded(self, mock_service, client):
        """Testa erro quando modelo não está carregado"""
        # Configurar mock
        mock_service.is_loaded.return_value = False
        
        # Executar
        response = client.post('/predict',
                              json={'comment': 'Teste'},
                              content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 500
        assert data['error'] == 'Modelo não carregado'
    
    @patch('controllers.prediction_controller.model_service')
    def test_predict_batch_endpoint_success(self, mock_service, client):
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
        response = client.post('/predict/batch',
                              json={'comments': ['Comentário 1', 'Comentário 2']},
                              content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 200
        assert len(data['results']) == 2
        assert data['total_processed'] == 2
        assert 'timestamp' in data
    
    @patch('controllers.prediction_controller.model_service')
    def test_predict_endpoint_missing_comment(self, mock_service, client):
        """Testa erro quando comentário está ausente"""
        # Configurar mock para evitar erro interno
        mock_service.is_loaded.return_value = True
        
        # Executar
        response = client.post('/predict',
                              json={},
                              content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 400
        # A mensagem de erro quando o JSON está vazio é 'Dados inválidos'
        # Isso ocorre antes de verificar campos específicos
        assert data['error'] == 'Dados inválidos' or data['error'] == 'Campo obrigatório ausente'
    
    @patch('controllers.prediction_controller.model_service')
    def test_predict_batch_endpoint_too_many_comments(self, mock_service, client):
        """Testa erro quando há muitos comentários"""
        # Configurar mock para evitar erro interno
        mock_service.is_loaded.return_value = True
        
        # Criar lista com mais de 100 comentários
        comments = [f'Comentário {i}' for i in range(101)]
        
        # Executar
        response = client.post('/predict/batch',
                              json={'comments': comments},
                              content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 400
        assert data['error'] == 'Muitos comentários' 