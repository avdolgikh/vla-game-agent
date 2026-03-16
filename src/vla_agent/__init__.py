"""vla_agent package root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


def _patch_mlflow_windows_paths() -> None:
    """Ensure mlflow accepts bare Windows paths when setting tracking URIs."""
    if os.name != "nt":
        return
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow is an optional dependency in some envs.
        return

    if getattr(mlflow, "_vla_windows_uri_patch", False):
        return

    original_set_tracking_uri: Callable[[str | None], Any] = mlflow.set_tracking_uri

    def _patched_set_tracking_uri(uri: str | None) -> Any:
        if uri and len(uri) >= 2 and uri[1] == ":" and not uri.lower().startswith("file:"):
            uri = Path(uri).resolve().as_uri()
        return original_set_tracking_uri(uri)

    mlflow.set_tracking_uri = _patched_set_tracking_uri
    try:
        from mlflow.tracking import fluent as mlflow_fluent
    except ImportError:  # pragma: no cover - mirrors mlflow import above.
        pass
    else:
        mlflow_fluent.set_tracking_uri = _patched_set_tracking_uri
    setattr(mlflow, "_vla_windows_uri_patch", True)


def _register_windows_drive_scheme() -> None:
    """Register the current drive letter as a valid mlflow tracking scheme on Windows."""
    if os.name != "nt":
        return
    try:
        import mlflow  # noqa: F401  # ensure mlflow is available
        from mlflow.tracking._tracking_service import utils as tracking_utils
    except ImportError:  # pragma: no cover - mlflow optional in some envs.
        return

    drive = Path.cwd().drive
    if len(drive) < 2 or drive[1] != ":":
        return
    scheme = drive[0].lower()
    registry = tracking_utils._tracking_store_registry
    if scheme in getattr(registry, "_registry", {}):
        return
    registry.register(scheme, tracking_utils._get_file_store)


_patch_mlflow_windows_paths()
_register_windows_drive_scheme()

__all__: list[str] = []
