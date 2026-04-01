"""Bearer token verification for protected routes."""

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from utils.settings import get_settings

security = HTTPBearer(description="Shared secret for POST /run (set API_TOKEN in .env)")


def verify_run_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> None:
    expected = get_settings().api_token.strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not configured: set API_TOKEN in the environment",
        )
    if credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
