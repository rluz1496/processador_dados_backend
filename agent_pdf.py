import time
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.media import File as AgnoFile
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from agno.utils.pprint import pprint_run_response

load_dotenv()

    #Unificar os emails -- OK
    #Identificar o fixo e movel - OK
    #Identificar o perfil se inquilo/proprietario (Somente o primeiro) - OK
    #Identificar o tipo da unidade - OK
    #Identificador de CPF condosuite --
    #Identificador de telefone --condosuite busca dados--
    #baixar csv para conferencia - OK
    #https://condosuite.condoconta.com.br/#/signin
    #Inserir calculo 90/10 visivel

# ------------------------------
# MODELOS DE RESPOSTA
# ------------------------------

class UnidadeInfo(BaseModel):
    Unidade: str
    Bloco: str
    Tipo: str
    Perfil: str
    Proprietario_Nome: str
    Proprietario_CPF_CNPJ: str
    Proprietario_Celular: str
    Proprietario_Telefone_fixo: str
    Proprietario_Email: str
    Responsavel_Nome: str
    Responsavel_CPF_CNPJ: str
    Responsavel_Celular: str
    Responsavel_Telefone_fixo: str
    Responsavel_Email: str


class DocumentJSON(BaseModel):
    unidades: List[UnidadeInfo]
    total_unidades: int
    condominio: str = ""


# ------------------------------
# AGENTE CONFIGURADO
# ------------------------------

data_processor = Agent(
    name="extractor_data",
    model=OpenAIChat(id="gpt-5"),
    response_model=DocumentJSON,
    use_json_mode=True,
    debug_mode=False,
    instructions=[
        """
        Você é um assistente especializado em extração de informações estruturadas a partir de documentos PDF.

        TAREFA:
        - Leia o documento PDF completo.
        - Extraia todas as unidades habitacionais encontradas, incluindo apartamentos, garagens, salas ou qualquer outro tipo de unidade mencionada.
        - Para cada unidade, extraia exatamente os campos definidos no modelo abaixo.
        - Sempre retorne a saída como um JSON válido, sem comentários, texto extra ou explicações.

        CAMPOS A EXTRAIR:
        - Unidade: Número da unidade. Se vier acompanhado do bloco (ex.: "Bloco 01 Unidade 01-01" ou "01-02"), extraia apenas o número da unidade ("01-01", "01-02"). Se não houver, deixe em branco.
        - Bloco: Número ou nome do bloco, se presente. Se estiver embutido no número da unidade, extraia separadamente. Se não houver, deixe em branco.
        - Tipo: Tipo da unidade (ex.: "Apto", "Garagem", "Sala"). Se não especificado, padronize como "Apto".
        - Perfil: Sempre "Proprietário".
        - Proprietario_Nome
        - Proprietario_CPF_CNPJ
        - Proprietario_Celular
        - Proprietario_Telefone_fixo
        - Proprietario_Email
        - Responsavel_Nome: Nome do responsável, se houver, e for diferente do proprietário (nome e documento diferentes). Caso contrário, deixe em branco.
        - Responsavel_CPF_CNPJ
        - Responsavel_Celular
        - Responsavel_Telefone_fixo
        - Responsavel_Email

        REGRAS:
        1. Se a unidade tiver apenas proprietário, todos os campos de responsável ficam em branco.
        2. Se houver proprietário e responsável, ambos devem estar na mesma entrada (mesmo objeto no JSON).
        3. Dependentes não devem ser incluídos.
        4. Não aplicar nenhuma formatação adicional aos valores extraídos — manter exatamente como no documento.
        5. Ignorar informações irrelevantes (ex.: síndico, administradora) que não estejam associadas a uma unidade.
        6. Contar cada unidade apenas uma vez no campo "total_unidades".

        FORMATO DE SAÍDA:
        Retorne somente um JSON válido com a seguinte estrutura:
        {
        "unidades": [
            {
            "Unidade": "",
            "Bloco": "",
            "Tipo": "",
            "Perfil": "",
            "Proprietario_Nome": "",
            "Proprietario_CPF_CNPJ": "",
            "Proprietario_Celular": "",
            "Proprietario_Telefone_fixo": "",
            "Proprietario_Email": "",
            "Responsavel_Nome": "",
            "Responsavel_CPF_CNPJ": "",
            "Responsavel_Celular": "",
            "Responsavel_Telefone_fixo": "",
            "Responsavel_Email": ""
            }
        ]
        }
        """
    ]
)


def processar_arquivo_pdf(filepath: str):
    print(f"Iniciando processamento do arquivo: {filepath}")
    start_time = time.time()

    file = AgnoFile(filepath=filepath)

    response = data_processor.run(
        message="Extraia os dados.",
        files=[file]
    )

    pprint_run_response(response)

    duration = time.time() - start_time
    print(f"\n✅ Finalizado em {duration:.2f} segundos.")
    return response
