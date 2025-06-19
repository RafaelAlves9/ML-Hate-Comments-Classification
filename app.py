"""
API de Classificação de Discurso de Ódio
Arquivo principal da aplicação Flask
"""
from flask import Flask
from flask_cors import CORS
from services.model_service import model_service
from controllers import prediction_controller, health_controller
from config.settings import HOST, PORT, logger
import webbrowser
import threading


# Configuração da aplicação Flask com suporte a CORS e arquivos estáticos
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)


@app.route('/')
def serve_index():
    """Serve a página inicial do frontend"""
    return app.send_static_file('index.html')

# Rotas de health check e informações do sistema
app.add_url_rule('/api', 'home', health_controller.home, methods=['GET'])
app.add_url_rule('/api/health', 'health_check', health_controller.health_check, methods=['GET'])

# Rotas de predição do modelo
app.add_url_rule('/api/predict', 'predict', prediction_controller.predict, methods=['POST'])

# Handlers para tratamento de erros HTTP
app.register_error_handler(404, health_controller.handle_404)
app.register_error_handler(405, health_controller.handle_405)
app.register_error_handler(500, health_controller.handle_500)


def initialize_app():
    """Inicializa a aplicação carregando o modelo de ML"""
    try:
        logger.info("Iniciando aplicação...")
        model_service.load_model()
        logger.info("Aplicação inicializada com sucesso!")
        
        print("\nAPI de Classificação de Discurso de Ódio")
        print("=" * 50)
        print("🔗 Endpoints da API disponíveis em /api/*")
        print("   GET  /api - Status da API")
        print("   GET  /api/health - Health check")
        print("   POST /api/predict - Classificar um comentário")
        print(f"\n🏠 Frontend disponível em http://{HOST}:{PORT}")
        print(f"🌐 Servidor rodando em http://{HOST}:{PORT}\n")
        
    except Exception as e:
        logger.error(f"Erro ao inicializar aplicação: {e}")
        print("\nCertifique-se de que os arquivos do modelo estão no diretório correto!")
        raise e


if __name__ == '__main__':
    try:
        # Inicializar aplicação
        initialize_app()
        
        # Abrir navegador após um pequeno delay
        def open_browser():
            """Abre o navegador na URL da aplicação"""
            webbrowser.open_new(f"http://{HOST}:{PORT}")

        threading.Timer(1.5, open_browser).start()
        
        # Executar servidor
        app.run(host=HOST, port=PORT, debug=False)
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        exit(1)