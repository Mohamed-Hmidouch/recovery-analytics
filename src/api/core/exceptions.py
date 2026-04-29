"""
Core — Gestion centralisée des exceptions (Global Exception Handler).
Intercepte les erreurs (Pydantic, HTTP) et renvoie des JSON standardisés et user-friendly.
Équivalent strict de @ControllerAdvice / @ExceptionHandler dans Spring Boot.
"""

from fastapi import Request, HTTPException, FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Intercepte les erreurs 422 de Pydantic et les rend lisibles par un humain."""
    erreurs_lisibles = []
    for error in exc.errors():
        # Extrait le nom du champ en erreur (ex: 'revenu_estime')
        champ = " -> ".join(str(loc) for loc in error["loc"] if loc != "body")
        message_technique = error.get("msg", "")
        
        # Traduction simple de quelques messages techniques courants
        if "greater than" in message_technique:
            msg_humain = "La valeur doit être strictement positive (supérieure à 0)."
        elif "Input should be" in message_technique:
            msg_humain = f"Valeur non reconnue. {message_technique.replace('Input should be', 'Les choix possibles sont :')}"
        else:
            msg_humain = message_technique
            
        erreurs_lisibles.append(f"Erreur sur le champ '{champ}' : {msg_humain}")

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Les données du dossier envoyées sont invalides ou incomplètes. Veuillez vérifier les détails ci-dessous.",
            "details": erreurs_lisibles
        }
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """Intercepte les autres erreurs HTTP pour harmoniser la structure JSON."""
    error_code = "HTTP_ERROR"
    if exc.status_code == 401:
        error_code = "UNAUTHORIZED_ACCESS"
    elif exc.status_code == 403:
        error_code = "ACCESS_DENIED"
    elif exc.status_code == 413:
        error_code = "PAYLOAD_TOO_LARGE"
    elif exc.status_code == 429:
        error_code = "RATE_LIMIT_EXCEEDED"
    elif exc.status_code == 503:
        error_code = "SERVICE_UNAVAILABLE"
    elif exc.status_code == 500:
        error_code = "INTERNAL_SERVER_ERROR"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": error_code,
            "message": exc.detail,
            "details": []
        }
    )

def register_exception_handlers(app: FastAPI):
    """Enregistre tous les gestionnaires d'exceptions personnalisés sur l'application."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
