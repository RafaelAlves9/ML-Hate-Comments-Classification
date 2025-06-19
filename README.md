# API de Classificação de Discurso de Ódio

API REST para classificação de comentários como discurso de ódio ou não, utilizando Machine Learning, com um frontend integrado para testes manuais.

## 📋 Estrutura do Projeto

```
ML-Hate-Comments-Classification/
├── frontend/               # Arquivos do frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
├── app.py                  # Arquivo principal da aplicação
├── config/                 # Configurações
├── controllers/            # Controllers (rotas)
├── services/               # Serviços de negócio
├── utils/                  # Utilitários
├── tests/                  # Testes automatizados
├── hate_speech_classifier_model.pkl # Modelo treinado
├── HATE_COMMENTS_CLASSIFICATION.ipynb # Notebook com código de treinamento
└── requirements.txt        # Dependências
```

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd ML-Hate-Comments-Classification
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
```

3. Ative o ambiente virtual:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

4. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🔧 Execução

Execute a aplicação:
```bash
python app.py
```

O comando irá:
1.  Iniciar o servidor backend.
2.  Abrir automaticamente o frontend no seu navegador em `http://localhost:5000`.

## 📌 Endpoints da API

Todas as rotas da API estão disponíveis sob o prefixo `/api`. O frontend já está configurado para usá-las.

### Health Check
- `GET /api` - Status da API
- `GET /api/health` - Health check

### Predição
- `POST /api/predict` - Classificar um comentário

### Exemplo de Requisição

#### Predição única:
```json
POST /api/predict
{
    "comment": "Seu comentário aqui"
}
```

#### Resposta:
```json
{
    "comment": "Seu comentário aqui",
    "prediction": "Não é discurso de ódio",
    "is_hate_speech": false,
    "confidence": 85.5,
    "processed_comment": "seu comentário aqui",
    "timestamp": "2024-01-01T12:00:00"
}
```

## 🧪 Testes

O projeto possui uma suíte de testes robusta utilizando **PyTest** para garantir a qualidade e o desempenho da aplicação.

### Tipos de Testes

1.  **Testes de Desempenho do Modelo** (`tests/test_model_performance.py`):
    -   **O quê?** Avalia o modelo de Machine Learning (`.pkl`) contra um conjunto de dados de teste.
    -   **Para quê?** Garante que o modelo atenda a métricas mínimas de qualidade (acurácia, precisão, recall, F1-score) antes de ir para produção. Isso impede que um modelo com desempenho ruim seja implantado.

2.  **Testes de Integração** (`tests/test_controllers.py`):
    -   **O quê?** Testa os endpoints da API (`/api/*`).
    -   **Para quê?** Simula requisições HTTP e valida as respostas, garantindo que as rotas funcionem como esperado e tratem os erros corretamente.

3.  **Testes Unitários** (`tests/test_model_service.py`):
    -   **O quê?** Testa as funções e classes de forma isolada, como a `ModelService`.
    -   **Para quê?** Verifica a lógica interna de cada componente, como o cálculo de confiança da predição e o tratamento de dados inválidos.

### Como Executar os Testes

Para facilitar a execução, utilize o script `run_tests.py`:

```bash
# Executar todos os testes
python run_tests.py

# Executar apenas os testes de desempenho do modelo
python run_tests.py performance

# Executar testes com relatório de cobertura de código
python run_tests.py coverage
```
Para mais detalhes sobre os testes, consulte o `README.md` dentro da pasta `tests`.

## 🏗️ Arquitetura

O projeto segue uma arquitetura em camadas:

1.  **Frontend**: Interface do usuário (HTML/CSS/JS) para interação com a API.
2.  **Controllers**: Responsáveis por receber requisições HTTP da API e retornar respostas.
3.  **Services**: Contém a lógica de negócio (carregamento e uso do modelo ML).
4.  **Utils**: Funções auxiliares reutilizáveis.
5.  **Config**: Configurações centralizadas da aplicação.

Esta estrutura facilita:
- Manutenção e evolução do código
- Testes unitários e de integração
- Reutilização de componentes
- Separação de responsabilidades (SOLID)

## 🤖 Sobre o Modelo

O modelo de Machine Learning foi treinado no **Google Colab** e todo o código de treinamento, análise exploratória e avaliação está disponível no arquivo `HATE_COMMENTS_CLASSIFICATION.ipynb`. Este notebook contém:

- Análise exploratória dos dados
- Pré-processamento e limpeza de texto
- Treinamento e comparação de diferentes algoritmos
- Avaliação de métricas de desempenho
- Exportação do modelo final

## 📝 Notas

- O modelo (`hate_speech_classifier_model.pkl`) deve estar presente no diretório raiz.