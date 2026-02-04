import os
from abc import ABC, abstractmethod

class BaseTool(ABC):
    allowed_root = None  # Set this at runtime to a base directory

    @classmethod
    def validate_path(cls, path: str):
        if cls.allowed_root is None:
            raise ValueError("allowed_root not set in BaseTool")

        # Normalize both the allowed root and the requested path.
        root = os.path.abspath(cls.allowed_root) # it should be set to the workspace root
        abs_path = os.path.abspath(os.path.join(root, path))

        try:
            common = os.path.commonpath([root, abs_path])
        except ValueError:
            # Raised when drives differ on Windows; treat as outside the sandbox.
            raise ValueError(f"Disallowed path: {abs_path}")

        if common != root:
            raise ValueError(f"Disallowed path: {abs_path}")

        return abs_path

    @abstractmethod
    def run(self, **kwargs):
        pass