# API de Classificação de Discurso de Ódio

API REST para classificação de comentários como discurso de ódio ou não, utilizando Machine Learning.

## 📋 Estrutura do Projeto

```
ML-Hate-Comments-Classification/
├── app.py                  # Arquivo principal da aplicação
├── config/                 # Configurações
│   └── settings.py        # Configurações gerais e constantes
├── controllers/           # Controllers (rotas)
│   ├── health_controller.py    # Rotas de health e informações
│   └── prediction_controller.py # Rotas de predição
├── services/              # Serviços de negócio
│   └── model_service.py   # Serviço do modelo ML
├── utils/                 # Utilitários
│   └── text_preprocessor.py # Preprocessamento de texto
├── tests/                 # Testes automatizados
│   ├── test_model_service.py  # Testes do serviço
│   └── test_controllers.py    # Testes dos controllers
├── hate_speech_classifier_model.pkl # Modelo treinado
├── model_info.json        # Informações do modelo
└── requirements.txt       # Dependências
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

Execute a API:
```bash
python app.py
```

A API estará disponível em `http://localhost:5000`

## 📌 Endpoints

### Health Check
- `GET /` - Status da API
- `GET /health` - Health check

### Predição
- `POST /predict` - Classificar um comentário

### Exemplo de Requisição

#### Predição única:
```json
POST /predict
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
    "confidence_method": "probability",
    "processed_comment": "seu comentário aqui",
    "timestamp": "2024-01-01T12:00:00"
}
```

## 🧪 Testes

Execute os testes:
```bash
python -m pytest tests/
```

Ou execute individualmente:
```bash
python tests/test_model_service.py
python tests/test_controllers.py
```

## 🏗️ Arquitetura

O projeto segue uma arquitetura em camadas:

1. **Controllers**: Responsáveis por receber requisições HTTP e retornar respostas
2. **Services**: Contém a lógica de negócio (carregamento e uso do modelo ML)
3. **Utils**: Funções auxiliares reutilizáveis
4. **Config**: Configurações centralizadas da aplicação

Esta estrutura facilita:
- Manutenção e evolução do código
- Testes unitários e de integração
- Reutilização de componentes
- Separação de responsabilidades (SOLID)

## 📝 Notas

- O modelo (`hate_speech_classifier_model.pkl`) deve estar presente no diretório raiz
- As informações do modelo (`model_info.json`) são opcionais mas recomendadas