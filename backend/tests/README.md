# Testes do Sistema de Classificação de Discurso de Ódio

## 📋 Visão Geral

Este diretório contém todos os testes automatizados do projeto, implementados com **PyTest**. Os testes garantem que o modelo atenda aos requisitos mínimos de desempenho e que a API funcione corretamente.

## 🎯 Tipos de Testes

### 1. **Testes de Desempenho do Modelo** (`test_model_performance.py`)
Valida se o modelo atende aos requisitos mínimos de qualidade estabelecidos.

**Métricas e Thresholds:**
- **Acurácia mínima**: 65%
- **Precisão mínima**: 60%
- **Recall mínimo**: 60%
- **F1-Score mínimo**: 60%

**Validações incluídas:**
- ✅ Carregamento correto do modelo
- ✅ Tamanho adequado do dataset de teste
- ✅ Métricas dentro dos thresholds estabelecidos
- ✅ Desempenho balanceado entre classes
- ✅ Análise de falsos positivos e falsos negativos

### 2. **Testes de Integração** (`test_controllers.py`)
Testa os endpoints da API Flask.

**Endpoints testados:**
- `GET /` - Página inicial
- `GET /health` - Health check
- `GET /model-info` - Informações do modelo
- `POST /predict` - Predição individual
- `POST /predict/batch` - Predição em lote

### 3. **Testes Unitários** (`test_model_service.py`)
Testa as funcionalidades do serviço de modelo.

**Funcionalidades testadas:**
- Carregamento do modelo
- Predições individuais e em lote
- Cálculo de confiança
- Tratamento de erros

## 🚀 Como Executar os Testes

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Executar todos os testes
```bash
python run_tests.py
```

### Executar apenas testes de desempenho
```bash
python run_tests.py performance
```

### Executar com cobertura de código
```bash
python run_tests.py coverage
```

### Executar teste específico
```bash
python run_tests.py -k test_model_accuracy_threshold
```

### Comandos PyTest diretos
```bash
# Todos os testes
pytest

# Apenas testes de desempenho
pytest tests/test_model_performance.py -v

# Com marcadores
pytest -m performance    # Apenas testes de performance
pytest -m unit          # Apenas testes unitários
pytest -m integration   # Apenas testes de integração

# Com cobertura
pytest --cov=. --cov-report=html
```

## 🔍 Interpretação dos Resultados

### Testes de Desempenho

Se os testes de desempenho **FALHAREM**, significa que:
- ❌ O modelo **NÃO** atende aos requisitos mínimos de qualidade
- ❌ O modelo **NÃO DEVE** ser implantado em produção
- ❌ É necessário retreinar ou ajustar o modelo

### Exemplo de saída:
```
=== RELATÓRIO DE DESEMPENHO DO MODELO ===

Métricas Gerais:
- Acurácia: 67.89% (threshold: 65.00%)
- Precisão: 65.38% (threshold: 60.00%)
- Recall: 64.49% (threshold: 60.00%)
- F1-Score: 64.93% (threshold: 60.00%)

Dataset de Teste:
- Total de amostras: 600
- Distribuição de classes: [400, 200]

Status: ✅ APROVADO
```

## 📊 Arquivos de Configuração

### `pytest.ini`
Configurações globais do PyTest, incluindo:
- Diretórios de teste
- Marcadores customizados
- Opções padrão
- Timeout

### `conftest.py`
Configurações compartilhadas entre testes:
- Fixtures globais
- Hooks personalizados
- Relatórios customizados

## 🛡️ Garantia de Qualidade

Os testes implementados garantem:

1. **Qualidade do Modelo**: Validação contínua das métricas de desempenho
2. **Estabilidade da API**: Todos os endpoints funcionam corretamente
3. **Tratamento de Erros**: Sistema responde adequadamente a entradas inválidas
4. **Barreira de Segurança**: Impede deploy de modelos inadequados

## 🔄 CI/CD

Estes testes devem ser executados:
- Antes de cada commit (pre-commit hook)
- Em cada pull request
- Antes de cada deploy
- Após retreinar o modelo

## 📝 Notas Importantes

1. **Dados de Teste**: Os testes usam 20% do dataset `hate.csv` como dados de teste
2. **Reprodutibilidade**: Random seed fixo (42) garante resultados consistentes
3. **Timeout**: Testes têm timeout de 5 minutos para evitar travamentos
4. **Paralelização**: Testes podem ser executados em paralelo com `pytest -n auto` 