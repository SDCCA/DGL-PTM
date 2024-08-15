"""A test file for config.py."""
import pytest
import yaml

from dgl_ptm.config import Config


@pytest.fixture
def config_parameters():
    return {
        "_model_identifier": "test_config",
        "number_agents": 100,
        "steering_parameters": {
            "deletion_prob": 0.05,
            "edata": ['all']
            }
    }
@pytest.fixture
def config_file(tmp_path, config_parameters):
    """Return a config file."""
    # change a value
    config_parameters["number_agents"] = 150
    filename = tmp_path / "config.yaml"
    with open(filename , "w") as f:
        yaml.dump(config_parameters, f, sort_keys=False)
    return filename

def test_from_yaml(config_file):
    """Test Config.from_yaml."""
    cfg = Config.from_yaml(config_file)
    assert cfg.model_identifier == "test_config"
    assert cfg.number_agents == 150
    assert cfg.step_target == 5

def test_from_dict(config_parameters):
    """Test Config.from_dict."""
    cfg = Config.from_dict(config_parameters)
    assert cfg.model_identifier == "test_config"
    assert cfg.number_agents == 100
    assert cfg.step_target == 5

def test_to_yaml(tmp_path):
    """Test Config.to_yaml."""
    cfg = Config()
    cfg.to_yaml(tmp_path / "config.yaml")

    with open(tmp_path / "config.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    assert cfg_dict["_model_identifier"] == "test"
    assert cfg_dict["number_agents"] == 100
    assert cfg_dict["steering_parameters"]["del_method"] == "probability"
    assert cfg_dict["steering_parameters"]["del_threshold"] == 0.05
    assert cfg_dict["steering_parameters"]["adapt_m"][1] == 0.5

def test_invalid_fields(config_parameters):
    """Test that invalid fields are not accepted."""
    config_parameters["invalid_field"] = 100
    with pytest.raises(ValueError):
        _ = Config.from_dict(config_parameters)

def test_invalid_values(config_parameters):
    """Test that invalid values are not accepted."""
    config_parameters["number_agents"] = -100
    with pytest.raises(ValueError):
        _ = Config.from_dict(config_parameters)
