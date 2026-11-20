"""`ego_pod.service` —— `Ego` 快慢双系统三合一本地认知服务与 Python API。"""

from __future__ import annotations

from ego_pod.service.engine import (
    PROFILE_INTUITION_BACKEND,
    SERVICE_SCHEMA_VERSION,
    EgoCognitiveEngine,
)
from ego_pod.service.profile import (
    COGNITIVE_PROFILE_FILENAME,
    COGNITIVE_PROFILE_SCHEMA_VERSION,
    CognitiveProfile,
    IntuitionExemplar,
    RedlineRule,
    build_default_cognitive_profile,
    compile_cognitive_profile,
    read_cognitive_profile,
    scrub_pii_and_secrets,
    write_cognitive_profile,
)
from ego_pod.service.server import (
    DEFAULT_SERVE_HOST,
    DEFAULT_SERVE_PORT,
    EgoServiceServer,
)

__all__ = [
    "SERVICE_SCHEMA_VERSION",
    "PROFILE_INTUITION_BACKEND",
    "COGNITIVE_PROFILE_FILENAME",
    "COGNITIVE_PROFILE_SCHEMA_VERSION",
    "DEFAULT_SERVE_HOST",
    "DEFAULT_SERVE_PORT",
    "EgoCognitiveEngine",
    "EgoServiceServer",
    "CognitiveProfile",
    "IntuitionExemplar",
    "RedlineRule",
    "build_default_cognitive_profile",
    "compile_cognitive_profile",
    "read_cognitive_profile",
    "write_cognitive_profile",
    "scrub_pii_and_secrets",
]
