from __future__ import annotations


class PrepError(RuntimeError):
    """Raised when a preparation step fails unrecoverably.

    Covers: prepare_receptor4.py non-zero exit, autogrid4 HD map missing.
    The message is human-readable and always contains the underlying cause.
    """
