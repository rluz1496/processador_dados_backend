import time
from agno.agent import Agent
from agno.models.google import Gemini
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
    Nome: str
    CPF_CNPJ: str
    Celular: str
    Telefone_fixo: str
    Email: str


class DocumentJSON(BaseModel):
    unidades: List[UnidadeInfo]
    total_unidades: int
    condominio: str = ""


# ------------------------------
# AGENTE CONFIGURADO
# ------------------------------

data_processor = Agent(
    name="extractor_data",
    model=Gemini(id="gemini-2.5-pro", temperature=0),
    response_model=DocumentJSON,
    use_json_mode=True,
    debug_mode=False,
    instructions=[
        """
        Leia TODO o documento PDF e extraia informações de todas as unidades habitacionais encontradas, incluindo apartamentos, garagens, salas ou qualquer outro tipo de unidade mencionado. Para cada unidade, capture exatamente os seguintes campos:

        Unidade: O número da unidade, extraído de forma limpa. Se a unidade for descrita com o bloco incluído (ex.: "Bloco 01 Unidade 01-01" ou "01-02"), extraia apenas o número da unidade (ex.: "01-01", "01-02"). Se não houver unidade explícita, deixe em branco.
        Bloco: O número ou nome do bloco, se presente. Se o bloco estiver embutido no número da unidade, extraia-o separadamente (ex.: para "Bloco 01 Unidade 01-01", o Bloco é "01"). Se não houver bloco, deixe em branco.
        Tipo: O tipo da unidade (ex.: "Apto", "Garagem", "Sala", etc.). Se o tipo não estiver especificado no documento, padronize como "Apto".
        Perfil: O perfil da pessoa associada à unidade. Pode ser "Proprietário", "Inquilino" ou "Dependente". Sempre inclua o proprietário. Se o inquilino estiver destacado no documento, inclua-o como uma entrada separada para a mesma unidade, com o mesmo número de unidade e bloco, mas com o perfil "Inquilino" e seus respectivos dados. Dependentes (pessoas que moram na unidade, mas não são proprietários nem inquilinos) devem ser incluídos apenas se explicitamente mencionados, com o perfil "Dependente". Se o perfil não estiver destacado, considere a pessoa como "Proprietário".
        Nome: O nome completo da pessoa (proprietário, inquilino ou dependente). Se não houver, deixe em branco.
        CPF_CNPJ: O CPF ou CNPJ da pessoa, formatado como aparece no documento. Se não houver, deixe em branco.
        Telefone: O número de telefone, identificado como "Fixo" ou "Móvel". Formate o número com o DDD no padrão "(XX) XXXX-XXXX" para fixo ou "(XX) XXXXX-XXXX" para móvel. Se não for possível determinar o tipo, considere "Móvel". Se não houver telefone, deixe em branco.
        WhatsApp: O número de WhatsApp, formatado com o DDD no padrão "(XX) XXXXX-XXXX". Se o documento não especificar um número de WhatsApp, mas houver um número de telefone móvel, assuma que o WhatsApp é o mesmo número do telefone móvel. Se não houver WhatsApp, deixe em branco.
        Email: O endereço de e-mail da pessoa. Se não houver, deixe em branco.

        Instruções adicionais:

        Processe todas as unidades mencionadas no documento, mesmo que sejam de tipos diferentes (ex.: apartamentos, garagens, salas) e pertençam ao mesmo proprietário.
        Garanta que cada unidade seja representada apenas uma vez por perfil (ex.: uma unidade com proprietário e inquilino terá duas entradas, uma com perfil "Proprietário" e outra com perfil "Inquilino", mas com o mesmo número de unidade e bloco).
        Ignore informações irrelevantes, como dados de outras entidades (ex.: síndico, administradora) que não estejam vinculadas a uma unidade específica.
        Se um campo não estiver presente no documento, deixe-o em branco no JSON.
        Ao concluir, conte o total de unidades processadas (considerando cada unidade por perfil) e inclua esse total no campo "total_unidades" no JSON.

        Formato de saída:
        Retorne somente um JSON válido, sem comentários ou texto extra, no seguinte formato:

        {
          "unidades": [ ... ],
          "total_unidades": 0
        }

        Garantias:
        - Todas as unidades existentes no PDF devem ser incluídas.
        - Formate os números de telefone e WhatsApp corretamente.
        - Separe unidade de bloco.
        - Identifique tipo e perfil corretamente.
        - Não repita entradas.
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
