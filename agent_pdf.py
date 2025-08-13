import time
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.media import File as AgnoFile
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from agno.utils.pprint import pprint_run_response

load_dotenv()

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

class UnidadeNormalizada(BaseModel):
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

class DocumentoNormalizado(BaseModel):
    unidades: List[UnidadeNormalizada]
    total_unidades: int
    condominio: str = ""

# ------------------------------
# AGENTE EXTRATOR
# ------------------------------

data_processor = Agent(
    name="extractor_data",
    model=OpenAIChat(id="gpt-5-mini"),
    response_model=DocumentJSON,
    use_json_mode=True,
    debug_mode=True,
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
        ],
        "total_unidades": 0
        }
        """
    ]
)

# ------------------------------
# AGENTE NORMALIZADOR
# ------------------------------

data_normalizer = Agent(
    name="data_normalizer",
    model=OpenAIChat(id="gpt-5-mini"),
    response_model=DocumentoNormalizado,
    use_json_mode=True,
    debug_mode=True,
    instructions=[
        """
        Você é um especialista em normalização e padronização de dados estruturados.

        TAREFA:
        - Receber um JSON contendo dados extraídos de documentos.
        - Normalizar e padronizar todos os campos de acordo com as regras abaixo.
        - Retornar exatamente o mesmo JSON, mantendo a estrutura original, porém com todos os campos normalizados.

        REGRAS DE NORMALIZAÇÃO:

        1. TELEFONES
        - Remover todos os caracteres não numéricos.
        - Celular:
        - Deve ter 11 dígitos: DDD (2 dígitos) + número (9 dígitos).
        - Se possuir apenas 10 dígitos e o número for celular, inserir o dígito 9 após o DDD.
        - Formatar como: (XX) XXXXX-XXXX.
        - Telefone fixo:
        - Deve ter 10 dígitos: DDD (2 dígitos) + número (8 dígitos).
        - Formatar como: (XX) XXXX-XXXX.
        - Se não for possível determinar o tipo, considerar como celular.

        2. CPF/CNPJ
        - Remover todos os caracteres não numéricos.
        - CPF:
        - Deve ter 11 dígitos.
        - Formatar como: XXX.XXX.XXX-XX.
        - CNPJ:
        - Deve ter 14 dígitos.
        - Formatar como: XX.XXX.XXX/XXXX-XX.

        3. EMAIL
        - Converter para minúsculas.
        - Remover espaços extras.
        - Validar se contém "@" e "." (formato básico).

        4. NOMES
        - Converter para formato título (primeira letra maiúscula de cada palavra).
        - Remover espaços extras no início e no fim.
        - Remover caracteres especiais desnecessários.

        5. UNIDADE
        - Remover espaços extras.
        - Manter consistência na numeração.
        - Se o bloco estiver repetido dentro da unidade (ex.: "Unidade: 01-01 Bloco 01"), remover o bloco e manter apenas a unidade (ex.: "01-01").

        6. BLOCO
        - Remover espaços extras.
        - Padronizar numeração e formato.
        - Se não houver bloco informado, deixar em branco.

        7. TIPO
        - Padronizar para: "Apartamento", "Garagem", "Sala", "Loja", etc.
        - Converter abreviações: "Apto" → "Apartamento".

        SAÍDA:
        - Retornar exclusivamente o JSON normalizado, sem explicações adicionais ou texto extra.

        """
    ]
)

def processar_pdf(filepath: str) -> str:
    """Processa PDF com extração e normalização sequencial"""
    
    # Etapa 1: Extração
    response_extracao = data_processor.run(
        message=f"extraia os dados conforme o prompt: {filepath}",
        files=[{"filepath": filepath}]
    )
    
    dados_extraidos = response_extracao.content
    print(f"Extração concluída. {dados_extraidos.total_unidades} unidades encontradas.")
    
    # Etapa 2: Normalização
    print("🔧 Iniciando normalização de dados...")
    response_normalizacao = data_normalizer.run(
        message=f"Normalize os seguintes dados extraídos: {dados_extraidos.model_dump_json()}"
    )
    
    dados_normalizados = response_normalizacao.content
    print(f" {dados_normalizados}")
    
    return dados_normalizados

if __name__ == "__main__":
    resultado = processar_pdf("unidades_condominio-deville.pdf")
    print("\n" + "="*50)
    print("RESULTADO FINAL NORMALIZADO:")
    print("="*50)
    print(resultado.model_dump_json(indent=2))
