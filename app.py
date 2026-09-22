"""Compatibility entrypoint for the containerized Deepcam API.

The actual service implementation lives in backend.app so the existing
repository layout and mobile API remain intact.
"""
from backend.app import app

__all__ = ["app"]
