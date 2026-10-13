"""One place that turns an open GGUF into its dossier key.

The key is GGUFFile.model_hash(): since 0.3.1 the header plus fixed samples of
the tensor data. Dossiers made before 0.3.1 are keyed by the header-only hash,
which two models of one geometry and size share. When such a model is opened
again, its dossier, runs, inbox items and harvest folder are carried over to
the new key here, once, and the mapping is recorded in hash_migrations.
"""
from __future__ import annotations

from ..core.source.gguf import GGUFFile
from .shared_lib import SharedLib


def model_identity(lib: SharedLib, g: GGUFFile) -> str:
    new = g.model_hash()
    lib.rekey_model(g.model_hash_legacy(), new)
    return new
