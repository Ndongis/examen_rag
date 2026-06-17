import re
import os
import torch
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─── Configuration ────────────────────────────────────────────────────────────

PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")
NOM_MODELE = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
CONFIG_GENERATION = {
    "max_new_tokens": 200,
    "temperature": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.1,
    "do_sample": False,
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
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            loader = PyPDFLoader(path)
            pages = loader.load()
            full_text = " ".join([p.page_content for p in pages])
            structured = split_by_sections(full_text, file)
            all_chunks.extend(structured)

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
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    mod = AutoModelForCausalLM.from_pretrained(
        nom_modele,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    mod.eval()
    return tok, mod


# ─── RAG helpers ───────────────────────────────────────────────────────────────

def construire_prompt(contexte: str, question: str) -> str:
    return (
        "Tu es un assistant universitaire francophone.\n"
        "Réponds uniquement avec les informations du contexte, en français.\n"
        "Si l'information n'est pas dans le contexte, dis \"Information non disponible\".\n"
        "Surtout n'hallucinez, ne créez pas et n'inventez pas.\n"
        "Base tout juste sur le contexte pour répondre aux questions.\n"
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
def poser_question(
    question: str,
    k: int = 5,
    afficher_sources: bool = False,
) -> str | dict:
    contexte, docs = recuperer_contexte(vectorstore, question, k=k)
    prompt = construire_prompt(contexte, question)

    entrees = tokeniseur(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(modele.device)

    sorties = modele.generate(
        **entrees,
        **CONFIG_GENERATION,
        pad_token_id=tokeniseur.eos_token_id,
        eos_token_id=tokeniseur.eos_token_id,
    )

    reponse = tokeniseur.decode(sorties[0], skip_special_tokens=True).strip()

    if "Réponse:" in reponse:
        reponse = reponse.split("Réponse:")[-1].strip()

    reponse = reponse.replace(": ", ".\n")

    if afficher_sources:
        sources = [doc.metadata for doc in docs]
        return {"reponse": reponse, "sources": sources}

    return reponse


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, tokeniseur, modele

    print("⏳ Chargement des documents PDF…")
    documents = load_documents(PDF_FOLDER)
    print(f"✅ {len(documents)} documents chargés.")

    print("⏳ Construction du vectorstore…")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
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


@app.post("/question", response_model=ReponseSimple | ReponseAvecSources)
def question(body: QuestionRequest):
    if vectorstore is None or modele is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé.")

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
    """Retourne la liste des intentions reconnues et leurs mots-clés."""
    return INTENTS