import re
import os
import torch
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.responses import JSONResponse

# ─── Configuration ────────────────────────────────────────────────────────────

PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")
NOM_MODELE = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = os.getenv("CACHE_DIR", "./models_cache")
CONFIG_GENERATION = {
    "max_new_tokens": 200,
    "do_sample": False,
    "repetition_penalty": 1.1,
    "use_cache": True,  # KV-cache GPU activé
}

INTENTS = {
    "Conditions d'accès": [
        "condition", "conditions", "accès", "admission",
        "inscription", "prérequis", "pré requis", "intégrer"
    ],
    "Objectifs de la formation": [
        "objectif", "objectifs", "but", "compétence"
    ],
    "Poursuites d'études": [
        "master", "poursuite", "études", "après la licence"
    ],
    "Débouchés professionnels": [
        "emploi", "travail", "métier", "débouché"
    ],
}

# Global state
vectorstore = None
tokeniseur = None
modele = None


# ─── Text splitting ────────────────────────────────────────────────────────────

def split_by_sections(text: str, filename: str) -> list[dict]:
    text = re.sub(r'\s+', ' ', text)

    pattern = (
        r"(Informations générale"
        r"|Objectifs de la formation"
        r"|Compétences visées"
        r"|Perspectives professionnelles"
        r"|Perspectives académiques"
        r"|Admission"
        r"|Conditions de passage"
        r"|Contact"
        r"|Responsables?\s+de\s+la\s+formation"
        r"|Organisation\s+et\s+contenu\s+des\s+études"
        r"|Débouchés professionnels"
        r"|DEBOUCHES PROFESSIONNELS"
        r"|Poursuites d'études"
        r"|Conditions d'accès"
        r"|Modalités d'admission"
        r"|ORGANISATION ET CONTENU DES ÉTUDES)"
    )

    parts = re.split(pattern, text, flags=re.IGNORECASE)
    filiere = os.path.splitext(os.path.basename(filename))[0]
    structured_chunks = []

    for i in range(1, len(parts), 2):
        section = parts[i].strip()
        content = parts[i + 1].strip()
        if len(content) > 50:
            structured_chunks.append({
                "page_content": content,
                "metadata": {"filiere": filiere, "section": section}
            })

    return structured_chunks


# ─── Document loading ──────────────────────────────────────────────────────────

def load_documents(folder_path: str) -> list[Document]:
    all_chunks = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return []
        
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            try:
                loader = PyPDFLoader(path)
                pages = loader.load()
                full_text = " ".join([p.page_content for p in pages])
                structured = split_by_sections(full_text, file)
                all_chunks.extend(structured)
            except Exception as e:
                print(f"❌ Erreur lors de la lecture de {file}: {e}")

    documents = [
        Document(
            page_content=(
                f"Filière : {chunk['metadata'].get('filiere', '')}\n"
                f"Section : {chunk['metadata'].get('section', '')}\n\n"
                f"{chunk['page_content']}"
            ),
            metadata=chunk["metadata"],
        )
        for chunk in all_chunks
    ]
    return documents


# ─── Model loading ─────────────────────────────────────────────────────────────

def charger_modele(nom_modele: str):
    tok = AutoTokenizer.from_pretrained(nom_modele)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    mod = AutoModelForCausalLM.from_pretrained(
        nom_modele,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    mod.eval()
    
    # ⚠️ torch.compile peut causer des instabilités avec .generate(). 
    # Si des erreurs surviennent, commentez la ligne ci-dessous.
    try:
        mod = torch.compile(mod, mode="reduce-overhead")
    except Exception as e:
        print(f"⚠️ torch.compile échoué, passage en mode standard : {e}")
        
    return tok, mod


# ─── RAG helpers ───────────────────────────────────────────────────────────────

def construire_prompt(contexte: str, question: str) -> str:
    return (
        "Tu es un assistant universitaire francophone.\n"
        "Réponds uniquement avec les informations du contexte, en français.\n"
        "Si l'information n'est pas dans le contexte, dis \"Information non disponible\".\n"
        "Surtout n'hallucinez, ne créez pas et n'inventez pas.\n"
        "Base tout juste sur le contexte pour répondre aux questions.\n\n"
        f"Contexte:\n{contexte}\n\n"
        f"Question:\n{question}\n\n"
        "Réponse:\n"
    )


def recuperer_contexte(
    base_vectorielle,
    question: str,
    k: int = 5,
    max_caracteres: int = 2000,
) -> tuple[str, list]:
    docs = base_vectorielle.similarity_search(question, k=k)
    morceaux = []
    total = 0
    for i, doc in enumerate(docs):
        texte = f"[Source {i+1}] {doc.page_content}"
        if total + len(texte) > max_caracteres:
            restant = max_caracteres - total
            if restant > 100:
                morceaux.append(texte[:restant] + "…")
            break
        morceaux.append(texte)
        total += len(texte)
    return "\n\n".join(morceaux), docs


@torch.inference_mode()
def poser_question(question: str, k: int = 5, afficher_sources: bool = False):
    contexte, docs = recuperer_contexte(vectorstore, question, k=k)
    prompt = construire_prompt(contexte, question)

    entrees = tokeniseur(
        prompt, return_tensors="pt", truncation=True, max_length=1800
    ).to("cuda")

    input_length = entrees.input_ids.shape[1]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        sorties = modele.generate(
            **entrees,
            **CONFIG_GENERATION,
            pad_token_id=tokeniseur.eos_token_id,
            eos_token_id=tokeniseur.eos_token_id,
        )

    # 💎 FIX: Extraire UNIQUEMENT les nouveaux tokens générés (évite le bug du split)
    nouveaux_tokens = sorties[0][input_length:]
    reponse = tokeniseur.decode(nouveaux_tokens, skip_special_tokens=True).strip()

    # Nettoyages annexes
    reponse = reponse.replace("\\n", "\n")
    reponse = re.sub(r' {2,}', ' ', reponse)

    if afficher_sources:
        sources = [doc.metadata for doc in docs]
        return {"reponse": reponse, "sources": sources}

    return reponse


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, tokeniseur, modele

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("⏳ Chargement des documents PDF…")
    documents = load_documents(PDF_FOLDER)
    print(f"✅ {len(documents)} documents chargés.")

    if not documents:
        print("⚠️ Aucun document trouvé dans le dossier PDF. Initialisation d'une base vide.")
        # Fallback pour éviter que FAISS ne crash s'il n'y a pas de PDFs
        documents = [Document(page_content="Base vide initiale.", metadata={"filiere": "Aucune", "section": "Aucune"})]

    print("⏳ Construction du vectorstore…")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=CACHE_DIR,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"batch_size": 64},
    )
    vectorstore = FAISS.from_documents(documents, embedding_model)
    print("✅ Vectorstore prêt.")

    print("⏳ Chargement du modèle LLM…")
    tokeniseur, modele = charger_modele(NOM_MODELE)
    print("✅ Modèle chargé.")

    yield
    
    print("👋 Arrêt du serveur.")


# ─── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Assistant Universitaire RAG",
    description="API de questions-réponses sur les formations universitaires.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ───────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    afficher_sources: Optional[bool] = False


class ReponseSimple(BaseModel):
    reponse: str


class ReponseAvecSources(BaseModel):
    reponse: str
    sources: list[dict]


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/question", response_model=Union[ReponseSimple, ReponseAvecSources])
def question(body: QuestionRequest):
    if vectorstore is None or modele is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé ou base vide.")

    resultat = poser_question(
        question=body.question,
        k=body.k,
        afficher_sources=body.afficher_sources,
    )

    if isinstance(resultat, dict):
        return ReponseAvecSources(**resultat)
    return ReponseSimple(reponse=resultat)


@app.get("/intents")
def get_intents():
    return INTENTS


@app.get("/ping")
def runpod_ping():
    return JSONResponse(status_code=200, content={"status": "healthy"})