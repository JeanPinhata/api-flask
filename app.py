from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import json
import os # Importar o módulo os para acessar variáveis de ambiente

app = Flask(__name__)

# --- Carregar o pipeline do modelo e os dados de apoio ---
# Certifique-se de que esses arquivos estão na raiz do seu projeto
# ou em um subdiretório que você especificará.
try:
    pipeline = joblib.load('model_pipeline.pkl')
    with open('politicas_base.json', 'r', encoding='utf-8') as f:
        politicas_base = json.load(f)
    with open('all_difficulties.json', 'r', encoding='utf-8') as f:
        all_difficulties = json.load(f)
    with open('categorical_cols_for_ohe.json', 'r', encoding='utf-8') as f:
        categorical_cols_for_ohe = json.load(f)
    print("Modelo, políticas e dados de apoio carregados com sucesso.")
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos: {e}. Certifique-se de que 'model_pipeline.pkl', 'politicas_base.json', 'all_difficulties.json' e 'categorical_cols_for_ohe.json' estão na mesma pasta do 'app.py'.")
    # Para evitar que a aplicação falhe completamente em produção,
    # você pode configurar um comportamento de fallback ou relatar o erro de forma mais robusta.
    # Por enquanto, vamos manter a lógica de None para pipeline.
    pipeline = None
    politicas_base = {}
    all_difficulties = []
    categorical_cols_for_ohe = []
except Exception as e:
    print(f"Ocorreu um erro inesperado ao carregar arquivos: {e}")
    pipeline = None
    politicas_base = {}
    all_difficulties = []
    categorical_cols_for_ohe = []


# --- Dicionário de políticas sugeridas por cluster (Exemplo) ---
# ... (seu dicionário de políticas sugeridas permanece o mesmo) ...
politicas_sugeridas_por_cluster = {
    0: [
        "Criar editais específicos para grupos vulneráveis (PCD, baixa renda, sem formação).",
            "Garantir bolsas de apoio financeiro para participação em oficinas e cursos.",
            "Criar programas de capacitação inicial em arte e cultura voltados à inclusão social.",
            "Oferecer apoio logístico (transporte, alimentação) para participação em atividades culturais.",
            "Incentivar projetos de arte comunitária em bairros periféricos."
    ],
    1: [
        "Fortalecer canais de divulgação e formação técnica para artistas em início de carreira.",
            "Disponibilizar mentorias com artistas experientes e gestores culturais.",
            "Incentivar parcerias entre artistas emergentes e instituições culturais locais.",
            "Criar feiras, mostras e festivais voltados a talentos em ascensão.",
            "Oferecer cursos de capacitação em gestão de carreira e produção cultural."
    ],
    2: [
         "Ampliar incentivos à profissionalização e registro formal de artistas.",
            "Criar linhas de microcrédito específicas para artistas formalizados.",
            "Oferecer capacitação em gestão de carreira, marketing e direitos autorais.",
            "Promover programas de certificação de competências artísticas.",
            "Incentivar a participação em cooperativas e associações de classe."
    ],
    3: [
         "Estimular redes de cooperação entre artistas e produtores culturais.",
            "Promover intercâmbios culturais regionais, nacionais e internacionais.",
            "Fomentar o uso de tecnologias digitais para ampliar o alcance do trabalho artístico.",
            "Criar programas de residência artística em diferentes linguagens.",
            "Apoiar iniciativas de economia criativa e inovação cultural."
    ]
}

# --- Rotas da Aplicação Flask ---

# Rota para a página inicial (formulário HTML)
@app.route('/')
def home():
    # Certifique-se de ter uma pasta 'templates' na raiz do seu projeto
    # e dentro dela o arquivo 'index.html'
    return render_template('index.html')

# Rota para as recomendações (API endpoint)
@app.route('/recommend', methods=['POST'])
def recommend():
    if pipeline is None:
        # Quando o modelo não é carregado, o erro 500 já é retornado.
        # Aqui, estamos aprimorando a mensagem para o usuário final.
        return jsonify({"error": "Serviço indisponível: Modelo não carregado. Contacte o administrador."}), 503 # 503 Service Unavailable

    data = request.json # Recebe os dados JSON do frontend

    # Validar se 'data' é um dicionário e não está vazio
    if not isinstance(data, dict) or not data:
        return jsonify({"error": "Requisição inválida: Dados JSON esperados."}), 400

    # Preparar os dados de entrada para o modelo
    input_data = {}
    for col in categorical_cols_for_ohe:
        input_data[col] = [data.get(col, 'Não informado')]

    # Lidar com as dificuldades digitais (recriar as colunas binárias como no treinamento)
    input_difficulties = data.get('Dificuldades_Divulgacao_Digital', [])
    if isinstance(input_difficulties, str):
        input_difficulties = [d.strip() for d in input_difficulties.split(',') if d.strip()]

    for difficulty in all_difficulties:
        # Melhorar a limpeza do nome para garantir consistência
        clean_difficulty_name = f'Dificuldade_{difficulty.replace(" ", "_").replace("/", "_").replace("-", "_").replace("(", "").replace(")", "").replace(":", "").replace(".", "").replace("__", "_")}'
        input_data[clean_difficulty_name] = [1 if difficulty in input_difficulties else 0]

    # Criar um DataFrame a partir dos dados de entrada
    input_df = pd.DataFrame(input_data)

    try:
        # Prever o cluster do artista
        predicted_cluster = pipeline.predict(input_df)[0]

        # Obter as políticas sugeridas para o cluster
        recommendations = politicas_sugeridas_por_cluster.get(predicted_cluster, ["Nenhuma política sugerida para este perfil no momento."])

        return jsonify({
            "cluster": int(predicted_cluster), # Converter para int para JSON
            "recommendations": recommendations
        })
    except KeyError as e:
        return jsonify({"error": f"Erro de chave: Verifique se todos os campos esperados estão no JSON de entrada. Detalhe: {e}"}), 400
    except Exception as e:
        # Logar o erro completo para depuração no Render
        print(f"Erro inesperado ao fazer a predição: {e}")
        return jsonify({"error": f"Erro interno no servidor ao processar a requisição. Detalhe: {str(e)}"}), 500

# Esta parte só será executada se você rodar 'python app.py' localmente.
# No Render, o Gunicorn é quem irá iniciar a aplicação.
if __name__ == '__main__':
    # Obtém a porta da variável de ambiente PORT fornecida pelo Render
    # Se PORT não estiver definida (ex: rodando localmente), usa 5000 como padrão
    port = int(os.environ.get("PORT", 5000))

    # NUNCA USE debug=True EM PRODUÇÃO!
    # No Render, você não usará app.run() diretamente.
    # Este bloco é apenas para testar localmente.
    print(f"Rodando localmente em http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)