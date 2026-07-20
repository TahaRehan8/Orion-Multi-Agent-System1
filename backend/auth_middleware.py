"""
Authentication Middleware for Orion Multi-Agent RAG System
Role-based access control decorators and utilities
"""

from functools import wraps
from typing import Callable, Optional
from fastapi import HTTPException, Header, Depends
from backend.database import get_user_role, is_super_user, ROLE_SUPER_USER, ROLE_LIMITED


def check_permission(user_role: str, required_role: str) -> bool:
    """
    Check if a user role satisfies the required role.
    super_user has access to everything.
    """
    if user_role == ROLE_SUPER_USER:
        return True
    if required_role == ROLE_LIMITED and user_role == ROLE_LIMITED:
        return True
    return False


def require_super_user(username: str) -> bool:
    """Check if username has super_user role, raise HTTPException if not"""
    if not is_super_user(username):
        raise HTTPException(
            status_code=403,
            detail="This action requires super_user privileges"
        )
    return True


def get_current_user_role(x_username: Optional[str] = Header(None)) -> tuple:
    """
    Dependency to get current user and role from request header.
    Returns (username, role) tuple.
    """
    if not x_username:
        return (None, None)
    
    role = get_user_role(x_username)
    return (x_username, role)


class RoleChecker:
    """
    Dependency class for checking user roles in FastAPI routes.
    
    Usage:
        @app.get("/admin-only")
        async def admin_route(user: tuple = Depends(RoleChecker(required_role="super_user"))):
            username, role = user
            ...
    """
    
    def __init__(self, required_role: str = ROLE_LIMITED):
        self.required_role = required_role
    
    def __call__(self, x_username: Optional[str] = Header(None)) -> tuple:
        if not x_username:
            raise HTTPException(
                status_code=401,
                detail="X-Username header is required"
            )
        
        user_role = get_user_role(x_username)
        
        if not user_role:
            raise HTTPException(
                status_code=401,
                detail="User not found or inactive"
            )
        
        if not check_permission(user_role, self.required_role):
            raise HTTPException(
                status_code=403,
                detail=f"This action requires {self.required_role} privileges"
            )
        
        return (x_username, user_role)


# Pre-configured dependency instances
require_authenticated = RoleChecker(required_role=ROLE_LIMITED)
require_admin = RoleChecker(required_role=ROLE_SUPER_USER)


def can_deploy_agents(username: str) -> bool:
    """Check if user can deploy agents"""
    return is_super_user(username)


def can_simulate_agents(username: str) -> bool:
    """Check if user can run simulations"""
    return is_super_user(username)


def can_modify_agents(username: str) -> bool:
    """Check if user can modify agent configurations"""
    return is_super_user(username)


def can_view_registry(username: str) -> bool:
    """Check if user can view the full agent registry"""
    return is_super_user(username)
