"""
Arquivo de configuração compartilhada do PyTest
"""
import pytest
import sys
import os

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def pytest_configure(config):
    """Configuração inicial do pytest"""
    config.addinivalue_line(
        "markers", 
        "performance: marca testes relacionados ao desempenho do modelo"
    )
    config.addinivalue_line(
        "markers", 
        "integration: marca testes de integração da API"
    )
    config.addinivalue_line(
        "markers", 
        "unit: marca testes unitários"
    )
    config.addinivalue_line(
        "markers", 
        "slow: marca testes que podem demorar mais tempo"
    )


def pytest_collection_modifyitems(config, items):
    """Modificar itens coletados para adicionar marcadores automaticamente"""
    for item in items:
        # Adicionar marcador baseado no nome do arquivo
        if "test_model_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "test_controllers" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_model_service" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def test_data_dir():
    """Retorna o diretório de dados de teste"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def model_path(test_data_dir):
    """Retorna o caminho do modelo"""
    return os.path.join(test_data_dir, 'hate_speech_classifier_model.pkl')


@pytest.fixture(scope="session")
def dataset_path(test_data_dir):
    """Retorna o caminho do dataset"""
    return os.path.join(test_data_dir, 'hate.csv')


def pytest_report_header(config):
    """Adicionar informações ao cabeçalho do relatório"""
    return [
        "Testes de Classificação de Discurso de Ódio",
        "Projeto MVP - Machine Learning",
        f"Python: {sys.version.split()[0]}",
    ]


def pytest_runtest_makereport(item, call):
    """Personalizar relatório de testes"""
    if call.when == "call" and call.excinfo is not None:
        if "performance" in [m.name for m in item.iter_markers()]:
            # Para testes de performance, adicionar informação extra
            print(f"\n⚠️  Teste de performance falhou: {item.name}")
            print("   Verifique os thresholds definidos e as métricas do modelo")


# Hook para relatório final
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Adicionar resumo personalizado ao final dos testes"""
    if terminalreporter.stats.get('failed'):
        terminalreporter.section("ATENÇÃO - Testes de Performance")
        terminalreporter.write_line(
            "Se os testes de performance falharam, o modelo não atende aos requisitos mínimos de qualidade."
        )
        terminalreporter.write_line(
            "NÃO implante este modelo em produção até que todos os testes passem."
        ) 