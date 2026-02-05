"""Speaker-aware local transcription using WhisperX and pyannote."""

__version__ = "0.1.0"

# PyTorch 2.6+ compatibility fix for pyannote model loading
# torch.load changed default to weights_only=True, which breaks pyannote
def _patch_torch_load():
    """Patch torch.load to force weights_only=False for compatibility.

    PyTorch 2.6 changed the default, breaking pyannote and many other
    ML libraries that use pickle-based checkpoints. Lightning explicitly
    passes weights_only=True, so we force it to False.
    """
    try:
        import torch
        _original_load = torch.load

        def _patched_load(*args, **kwargs):
            # Force weights_only=False for backward compatibility
            # Lightning and other libs explicitly pass True, so we override
            kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        torch.load = _patched_load
    except Exception:
        pass


_patch_torch_load()
