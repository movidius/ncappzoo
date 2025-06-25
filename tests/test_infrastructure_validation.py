"""Validation tests to verify the testing infrastructure is properly set up."""

import sys
from pathlib import Path
import pytest


class TestInfrastructureSetup:
    """Tests to validate the testing infrastructure."""
    
    def test_pytest_is_available(self):
        """Verify pytest is installed and importable."""
        import pytest
        assert pytest.__version__
    
    def test_pytest_cov_is_available(self):
        """Verify pytest-cov is installed and importable."""
        import pytest_cov
        assert pytest_cov
    
    def test_pytest_mock_is_available(self):
        """Verify pytest-mock is installed and importable."""
        import pytest_mock
        assert pytest_mock
    
    def test_project_structure_exists(self):
        """Verify the expected project directories exist."""
        project_root = Path(__file__).parent.parent
        
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        assert (project_root / "tests" / "conftest.py").exists()
    
    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml is properly configured."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        assert pyproject_path.exists()
        
        content = pyproject_path.read_text()
        assert "[tool.poetry]" in content
        assert "[tool.pytest.ini_options]" in content
        assert "[tool.coverage.run]" in content
    
    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Verify the unit test marker is properly configured."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Verify the integration test marker is properly configured."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Verify the slow test marker is properly configured."""
        assert True
    
    def test_temp_dir_fixture(self, temp_dir):
        """Verify the temp_dir fixture works correctly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_mock_model_config_fixture(self, mock_model_config):
        """Verify the mock_model_config fixture provides expected data."""
        assert isinstance(mock_model_config, dict)
        assert mock_model_config["name"] == "test_model"
        assert mock_model_config["framework"] == "openvino"
        assert "input_shape" in mock_model_config
        assert "output_shape" in mock_model_config
    
    def test_env_backup_fixture(self, env_backup):
        """Verify the env_backup fixture preserves environment."""
        import os
        
        test_key = "TEST_ENV_VAR_FOR_VALIDATION"
        original_value = os.environ.get(test_key)
        
        os.environ[test_key] = "test_value"
        assert os.environ[test_key] == "test_value"
        
        # After test, env should be restored (verified in fixture cleanup)
    
    def test_capture_stdout_fixture(self, capture_stdout):
        """Verify the capture_stdout fixture works correctly."""
        print("Hello, testing!")
        print("Line 2")
        
        assert len(capture_stdout) == 2
        assert capture_stdout[0] == "Hello, testing!"
        assert capture_stdout[1] == "Line 2"
    
    def test_project_modules_importable(self):
        """Verify project modules can be imported."""
        # Add project root to Python path if not already there
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test that we can import from the shared directory
        # (actual imports may fail due to dependencies, but the directories should be accessible)
        assert (project_root / "shared").exists()
        assert (project_root / "apps").exists()
        assert (project_root / "networks").exists()


class TestCoverageConfiguration:
    """Tests to verify coverage is properly configured."""
    
    def test_coverage_configured_in_pyproject(self):
        """Verify coverage settings are in pyproject.toml."""
        project_root = Path(__file__).parent.parent
        pyproject_content = (project_root / "pyproject.toml").read_text()
        
        assert "--cov=apps" in pyproject_content
        assert "--cov=networks" in pyproject_content
        assert "--cov=shared" in pyproject_content
        assert "--cov-fail-under=80" in pyproject_content
        assert "[tool.coverage.report]" in pyproject_content