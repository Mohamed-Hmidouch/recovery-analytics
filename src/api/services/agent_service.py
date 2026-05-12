"""
Agent Service — Initialise et exécute l'agent LangChain avec Gemini.

LLM    : Gemini 1.5 Flash (via GEMINI_API_KEY dans .env)
Tools  : predict_dossier, query_history, get_segment_scoring
Memory : ConversationBufferMemory par session_id (dict en mémoire)
"""

import logging
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import hub

from src.api.services.agent_tools import predict_dossier, query_history, get_segment_scoring

logger = logging.getLogger(__name__)

# Mémoire par session — dict[session_id → ConversationBufferMemory]
_session_memories: dict[str, ConversationBufferMemory] = {}

TOOLS = [predict_dossier, query_history, get_segment_scoring]

SYSTEM_PROMPT = """Tu es SmartRecovery Assistant, un expert en recouvrement de créances.
Tu aides les gestionnaires à analyser des dossiers, consulter l'historique et obtenir des recommandations IA.
Tu réponds toujours en français, de façon claire et professionnelle.
Tu as accès à 3 outils :
- predict_dossier : simule la prédiction ML complète pour un dossier
- query_history : interroge l'historique des dossiers en base de données
- get_segment_scoring : calcule le scoring par segment client

Utilise les outils chaque fois que la question nécessite des données réelles.
"""


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY manquante dans les variables d'environnement.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )


def _get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in _session_memories:
        _session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    return _session_memories[session_id]


def run_agent(question: str, session_id: str = "default") -> str:
    """
    Execute l'agent Gemini avec la question de l'utilisateur.
    Maintient la mémoire de conversation par session_id.
    """
    try:
        llm = _get_llm()
        memory = _get_memory(session_id)

        # Prompt ReAct depuis LangChain Hub
        prompt = hub.pull("hwchase17/react-chat")

        agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        response = executor.invoke({
            "input": f"{SYSTEM_PROMPT}\n\nQuestion: {question}",
            "chat_history": memory.chat_memory.messages,
        })
        return response["output"]

    except Exception as e:
        logger.error(f"Agent error (session={session_id}): {e}", exc_info=True)
        return f"Erreur de l'agent : {str(e)}"


def clear_session(session_id: str) -> None:
    """Efface la mémoire d'une session."""
    if session_id in _session_memories:
        del _session_memories[session_id]
