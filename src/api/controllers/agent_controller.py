"""
Agent Controller — Routeur FastAPI pour l'agent LangChain + Gemini.

POST /api/v1/agent/chat   → envoie une question, retourne la réponse de l'agent
DELETE /api/v1/agent/session/{session_id} → efface la mémoire d'une session
"""

import logging
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from src.api.core.security import verify_api_key
from src.api.services.agent_service import run_agent, clear_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["Agent IA"])


class ChatRequest(BaseModel):
    question: str = Field(..., description="Question en langage naturel pour l'agent")
    session_id: str = Field(default="default", description="ID de session pour la mémoire de conversation")


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat avec l'agent IA SmartRecovery (Gemini + LangChain)",
    description=(
        "Envoie une question en langage naturel. L'agent utilise Gemini 1.5 Flash "
        "et peut appeler automatiquement les modèles ML, interroger PostgreSQL "
        "et calculer le scoring selon le besoin."
    ),
)
async def chat(request: ChatRequest, _=Depends(verify_api_key)):
    logger.info(f"Agent chat — session={request.session_id} | question={request.question[:80]}")
    answer = run_agent(question=request.question, session_id=request.session_id)
    return ChatResponse(
        session_id=request.session_id,
        question=request.question,
        answer=answer,
    )


@router.delete(
    "/session/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Efface la mémoire d'une session",
)
async def delete_session(session_id: str, _=Depends(verify_api_key)):
    clear_session(session_id)
    return {"message": f"Session '{session_id}' effacée."}
