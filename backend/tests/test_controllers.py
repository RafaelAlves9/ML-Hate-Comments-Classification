"""
Testes para os Controllers usando PyTest
"""
import pytest
from unittest.mock import Mock, patch
import json
from flask import Flask
from backend.controllers import prediction_controller, health_controller


class TestControllers:
    """Testes de integração para os controllers"""
    
    @pytest.fixture
    def app(self):
        """Fixture para criar aplicação Flask de teste"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        
        # Registrar rotas
        app.add_url_rule('/api', 'home', health_controller.home, methods=['GET'])
        app.add_url_rule('/api/health', 'health_check', health_controller.health_check, methods=['GET'])
        app.add_url_rule('/api/predict', 'predict', prediction_controller.predict, methods=['POST'])
        
        return app
    
    @pytest.fixture
    def client(self, app):
        """Fixture para criar cliente de teste"""
        return app.test_client()
    
    @patch('backend.controllers.health_controller.model_service')
    def test_home_endpoint(self, mock_service, client):
        """Testa endpoint home"""
        # Configurar mock
        mock_service.is_loaded.return_value = True
        mock_service.get_model_info.return_value = {"accuracy": 0.95}
        
        # Executar
        response = client.get('/api')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 200
        assert data['status'] == 'API funcionando!'
        assert data['model_loaded'] is True
        assert 'endpoints' in data
    
    @patch('backend.controllers.health_controller.model_service')
    def test_health_check_endpoint(self, mock_service, client):
        """Testa endpoint de health check"""
        # Configurar mock
        mock_service.is_loaded.return_value = True
        
        # Executar
        response = client.get('/api/health')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 200
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True
        assert 'timestamp' in data
    
    @patch('backend.controllers.prediction_controller.model_service')
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
        response = client.post('/api/predict',
                              json={'comment': 'Teste'},
                              content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 200
        assert data['prediction'] == 'Não é discurso de ódio'
        assert data['is_hate_speech'] is False
        assert data['confidence'] == 85.5
    
    @patch('backend.controllers.prediction_controller.model_service')
    def test_predict_endpoint_model_not_loaded(self, mock_service, client):
        """Testa erro quando modelo não está carregado"""
        # Configurar mock
        mock_service.is_loaded.return_value = False
        
        # Executar
        response = client.post('/api/predict',
                              json={'comment': 'Teste'},
                              content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 500
        assert data['error'] == 'Modelo não carregado'
    
    @patch('backend.controllers.prediction_controller.model_service')
    def test_predict_endpoint_missing_comment(self, mock_service, client):
        """Testa erro quando comentário está ausente"""
        # Configurar mock para evitar erro interno
        mock_service.is_loaded.return_value = True
        
        # Executar
        response = client.post('/api/predict',
                              json={},
                              content_type='application/json')
        data = json.loads(response.data)
        
        # Verificar
        assert response.status_code == 400
        # A mensagem de erro quando o JSON está vazio é 'Dados inválidos'
        # Isso ocorre antes de verificar campos específicos
        assert data['error'] == 'Dados inválidos' or data['error'] == 'Campo obrigatório ausente' 