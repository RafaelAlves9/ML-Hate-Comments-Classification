"""
API de Classificação de Discurso de Ódio
Arquivo principal da aplicação Flask
"""
from flask import Flask
from flask_cors import CORS
from services.model_service import model_service
from controllers import prediction_controller, health_controller
from config.settings import HOST, PORT, logger


# Criar aplicação Flask
app = Flask(__name__)
CORS(app)


# Registrar rotas de health e informações
app.add_url_rule('/', 'home', health_controller.home, methods=['GET'])
app.add_url_rule('/health', 'health_check', health_controller.health_check, methods=['GET'])
app.add_url_rule('/model-info', 'get_model_info', health_controller.get_model_info, methods=['GET'])


# Registrar rotas de predição
app.add_url_rule('/predict', 'predict', prediction_controller.predict, methods=['POST'])
app.add_url_rule('/predict/batch', 'predict_batch', prediction_controller.predict_batch, methods=['POST'])


# Registrar handlers de erro
app.register_error_handler(404, health_controller.handle_404)
app.register_error_handler(405, health_controller.handle_405)
app.register_error_handler(500, health_controller.handle_500)


def initialize_app():
    """Inicializa a aplicação carregando o modelo"""
    try:
        logger.info("🚀 Iniciando aplicação...")
        model_service.load_model()
        logger.info("✅ Aplicação inicializada com sucesso!")
        
        print("\n📊 API de Classificação de Discurso de Ódio")
        print("=" * 50)
        print("🔗 Endpoints disponíveis:")
        print("   GET  / - Status da API")
        print("   GET  /health - Health check")
        print("   GET  /model-info - Informações do modelo")
        print("   POST /predict - Classificar um comentário")
        print("   POST /predict/batch - Classificar múltiplos comentários")
        print("=" * 50)
        print(f"\n🌐 Servidor rodando em http://{HOST}:{PORT}\n")
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar aplicação: {e}")
        print("\n⚠️  Certifique-se de que os arquivos do modelo estão no diretório correto!")
        raise e


if __name__ == '__main__':
    try:
        # Inicializar aplicação
        initialize_app()
        
        # Executar servidor
        app.run(host=HOST, port=PORT, debug=False)
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        exit(1)