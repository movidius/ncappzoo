"""Shared pytest fixtures and configuration for the test suite."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a sample image file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to the sample image file
    """
    import numpy as np
    image_path = temp_dir / "test_image.npy"
    sample_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    np.save(str(image_path), sample_data)
    return image_path


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Provide a mock neural network model configuration.
    
    Returns:
        Dict containing sample model configuration
    """
    return {
        "name": "test_model",
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 1000],
        "framework": "openvino",
        "precision": "FP16",
        "device": "MYRIAD",
    }


@pytest.fixture
def mock_ncs_device(mocker):
    """Mock Intel Neural Compute Stick device.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock object representing NCS device
    """
    device = mocker.Mock()
    device.name = "Intel Movidius Neural Compute Stick"
    device.is_open = True
    device.allocate_graph = mocker.Mock(return_value=mocker.Mock())
    device.close = mocker.Mock()
    return device


@pytest.fixture
def sample_graph_file(temp_dir: Path) -> Path:
    """Create a sample compiled graph file.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to the sample graph file
    """
    graph_path = temp_dir / "test_model.blob"
    graph_path.write_bytes(b"Mock compiled graph data")
    return graph_path


@pytest.fixture
def env_backup() -> Generator[Dict[str, str], None, None]:
    """Backup and restore environment variables.
    
    Yields:
        Dict: Copy of original environment variables
    """
    original_env = os.environ.copy()
    yield original_env
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_openvino_env(env_backup, mocker):
    """Mock OpenVINO environment setup.
    
    Args:
        env_backup: Environment backup fixture
        mocker: pytest-mock fixture
    """
    os.environ["INTEL_OPENVINO_DIR"] = "/opt/intel/openvino"
    os.environ["PYTHONPATH"] = "/opt/intel/openvino/python/python3"
    return env_backup


@pytest.fixture
def sample_inference_result():
    """Provide sample inference results for testing.
    
    Returns:
        Dict with sample inference outputs
    """
    import numpy as np
    return {
        "predictions": np.random.rand(1, 1000).astype(np.float32),
        "inference_time": 25.5,  # milliseconds
        "preprocessing_time": 5.2,
        "postprocessing_time": 2.1,
    }


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset imported modules between tests to avoid state pollution."""
    import sys
    modules_to_reset = [
        module for module in sys.modules.keys() 
        if module.startswith(("apps.", "networks.", "shared."))
    ]
    for module in modules_to_reset:
        sys.modules.pop(module, None)


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements.
    
    Returns:
        List that collects printed output
    """
    captured = []
    
    def mock_print(*args, **kwargs):
        captured.append(" ".join(str(arg) for arg in args))
    
    monkeypatch.setattr("builtins.print", mock_print)
    return captured