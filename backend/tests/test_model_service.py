"""
Testes para o ModelService usando PyTest
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import numpy as np
from backend.services.model_service import ModelService
from backend.config.settings import RESPONSE_LABELS


class TestModelService:
    """Testes unitários para o ModelService"""
    
    @pytest.fixture
    def service(self):
        """Fixture para criar instância do ModelService"""
        return ModelService()
    
    @patch('backend.services.model_service.joblib.load')
    @patch('backend.services.model_service.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"accuracy": 0.95}')
    def test_load_model_success(self, mock_file_open, mock_exists, mock_joblib, service):
        """Testa carregamento bem-sucedido do modelo"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_model = Mock()
        mock_joblib.return_value = mock_model
        
        # Executar
        service.load_model()
        
        # Verificar
        assert service.model is not None
        assert service.model_info == {"accuracy": 0.95}
        mock_joblib.assert_called_once()
    
    @patch('backend.services.model_service.os.path.exists')
    def test_load_model_file_not_found(self, mock_exists, service):
        """Testa erro quando arquivo do modelo não existe"""
        # Configurar mock
        mock_exists.return_value = False
        
        # Executar e verificar
        with pytest.raises(FileNotFoundError):
            service.load_model()
    
    def test_is_loaded(self, service):
        """Testa verificação se modelo está carregado"""
        # Modelo não carregado
        assert service.is_loaded() is False
        
        # Modelo carregado
        service.model = Mock()
        assert service.is_loaded() is True
    
    def test_get_model_info(self, service):
        """Testa obtenção de informações do modelo"""
        # Sem informações
        assert service.get_model_info() == {}
        
        # Com informações
        service.model_info = {"accuracy": 0.95}
        assert service.get_model_info() == {"accuracy": 0.95}
    
    @patch('backend.services.model_service.preprocess_text')
    def test_predict_single_success(self, mock_preprocess, service):
        """Testa predição bem-sucedida de um comentário"""
        # Configurar mocks
        mock_preprocess.return_value = "texto processado"
        mock_model = Mock()
        mock_model.predict.return_value = [0]  # Discurso de ódio
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        service.model = mock_model
        
        # Executar
        result = service.predict_single("Comentário de teste")
        
        # Verificar
        assert result['error'] is False
        assert result['prediction'] == RESPONSE_LABELS['HATE_SPEECH']
        assert result['is_hate_speech'] is True
        assert result['confidence'] == 80.0
        assert result['confidence_method'] == 'probability'
    
    @patch('backend.services.model_service.preprocess_text')
    def test_predict_single_empty_comment(self, mock_preprocess, service):
        """Testa predição com comentário vazio"""
        # Configurar mock
        mock_preprocess.return_value = ""
        service.model = Mock()
        
        # Executar
        result = service.predict_single("")
        
        # Verificar
        assert result['error'] is True
        assert result['message'] == 'Comentário inválido'
    
    def test_calculate_confidence_probability(self, service):
        """Testa cálculo de confiança usando probabilidade"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        service.model = mock_model
        
        # Executar
        result = service._calculate_confidence("texto")
        
        # Verificar
        assert result['confidence'] == 70.0
        assert result['method'] == 'probability'
    
    def test_calculate_confidence_decision_function(self, service):
        """Testa cálculo de confiança usando decision function"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict_proba.side_effect = AttributeError()
        mock_model.decision_function.return_value = [2.0]
        service.model = mock_model
        
        # Executar
        result = service._calculate_confidence("texto")
        
        # Verificar
        assert result['confidence'] > 50.0
        assert result['method'] == 'decision_function'
    
    def test_calculate_confidence_default(self, service):
        """Testa cálculo de confiança com valor padrão"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict_proba.side_effect = AttributeError()
        mock_model.decision_function.side_effect = Exception()
        service.model = mock_model
        
        # Executar
        result = service._calculate_confidence("texto")
        
        # Verificar
        assert result['confidence'] == 50.0
        assert result['method'] == 'default'
    
    @pytest.mark.parametrize("prediction,expected_label,expected_is_hate", [
        (0, RESPONSE_LABELS['HATE_SPEECH'], True),
        (1, RESPONSE_LABELS['NOT_HATE_SPEECH'], False)
    ])
    def test_predict_with_different_classes(self, service, prediction, expected_label, expected_is_hate):
        """Testa predições para diferentes classes"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict.return_value = [prediction]
        mock_model.predict_proba.return_value = [[0.5, 0.5]]
        service.model = mock_model
        
        # Executar
        result = service.predict_single("Teste")
        
        # Verificar
        assert result['prediction'] == expected_label
        assert result['is_hate_speech'] == expected_is_hate 