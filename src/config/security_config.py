from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.config.settings import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """Security utilities for authentication and authorization."""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain password against a hashed password.
        
        Args:
            plain_password: Plain text password
            hashed_password: BCrypt hashed password
        
        Returns:
            True if password matches
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """
        Hash a password using BCrypt.
        
        Args:
            password: Plain text password
        
        Returns:
            BCrypt hashed password
        """
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Payload data
            expires_delta: Optional custom expiry time
        
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(
            to_encode,
            settings.secret_key,
            algorithm=settings.algorithm
        )
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT refresh token.
        
        Args:
            data: Payload data
            expires_delta: Optional custom expiry time
        
        Returns:
            Encoded JWT refresh token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=settings.refresh_token_expire_days
            )
        
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(
            to_encode,
            settings.secret_key,
            algorithm=settings.algorithm
        )
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.algorithm]
            )
            return payload
        except JWTError:
            return None
    
    @staticmethod
    def get_token_type(token: str) -> Optional[str]:
        """
        Get the type of token (access or refresh).
        
        Args:
            token: JWT token string
        
        Returns:
            Token type or None if invalid
        """
        payload = SecurityManager.verify_token(token)
        if payload:
            return payload.get("type")
        return None


# Global security manager instance
security = SecurityManager()