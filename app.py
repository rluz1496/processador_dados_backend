from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import os

from agent_pdf import data_processor  # agente que usa o modelo Gemini + File
from agno.media import File as AgnoFile

app = FastAPI(title="API de Processamento de PDFs", version="2.0.0")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://processador-dados-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/processar-pdf")
async def processar_pdf(arquivo: UploadFile = File(...)):
    if not arquivo.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos")

    try:
        # Cria arquivo temporário com o conteúdo do UploadFile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_path = tmp.name
            content = await arquivo.read()
            tmp.write(content)

        # Cria o objeto File do Agno
        agno_file = AgnoFile(filepath=temp_path)

        # Executa o agente com o arquivo
        resultado = data_processor.run(
            message="Extraia os dados.",
            files=[agno_file]
        )

        document_data = resultado.content  # Deve ser um DocumentJSON

        resposta = {
            "success": True,
            "unidades": [u.model_dump() for u in document_data.unidades],
            "total": document_data.total_unidades,
            "condominio": document_data.condominio
        }

        return resposta

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar PDF: {str(e)}")

    finally:
        # Remove o arquivo temporário (boa prática)
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def root():
    return {"message": "API de Processamento de PDFs funcionando!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
