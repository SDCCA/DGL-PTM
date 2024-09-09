"""Configuration parameters for DGL_PTM model.

The configuration parameters are stored in a pydantic object. The model is
initialized with default values. The default values can be overwritten by
providing a yaml file or a dictionary. The keys and values are validated by
pydantic which is a data validation library.
"""

import logging
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, typing

logger = logging.getLogger(__name__)

class MThetaDist(BaseModel):
    """Base class for model_theta distribution."""
    type: str = "multinomial"
    parameters: list[int | float | list[int | float]] = [[0.02, 0.03, 0.05, 0.9], [0.7, 0.8, 0.9, 1]]
    round: bool = False
    decimals: int | None = None

    @field_validator("parameters")
    def _convert_parameters(cls, v, values):
        if values.data["type"] == "multinomial":
            for i in v:
                if not isinstance(i, list):
                    raise TypeError("multinomial parameters must be a list of lists")
            return [torch.tensor(i) for i in v]
        else:
            return torch.tensor(v)

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)


class SteeringParams(BaseModel):
    """Base class for steering parameters.
    These are the parameters used within each step of the model.
    """
    edata: list[str] | None = ["all"]
    epath: str = "./edge_data"
    format: str = "xarray"
    mode: str = "w"
    ndata: list[str | list[str | list[str]]] | None = ["all_except", ["a_table"]]
    npath: str = "./agent_data.zarr"
    capital_method: str = "present_shock"
    wealth_method: str = "singular_transfer"
    income_method: str = "income_generation"
    consume_method: str = "fitted_consumption"
    nn_path: str | None = "default"
    capital_update_method: str = "default"
    characteristic_distance: int | float = 35
    homophily_parameter: int | float = 0.69
    adapt_m: list[float] = [0.0, 0.5, 0.9]
    adapt_cost: list[float] = [0.0, 0.25, 0.45]
    depreciation: float = 0.6
    discount: float = 0.95
    model_theta: list[float] | None = None 
    model_theta_dist: MThetaDist | None = MThetaDist()
    tech_gamma: list[float] = [0.3, 0.35, 0.45]
    tech_cost: list[float] = [0.0, 0.15, 0.65]
    del_method: str = "probability"
    del_threshold: int | float | None | typing.Literal["balance"] = 0.05
    noise_ratio: float = 0.05
    local_ratio: float = 0.25
    truncation_weight: float = 1.0e-10
    step_type: str = "default"
    data_collection_period: int = 1
    data_collection_list: list[int] | None = None

    @field_validator("adapt_m")
    def _convert_adapt_m(cls, v):
        return torch.tensor(v)

    @field_validator("adapt_cost")
    def _convert_adapt_cost(cls, v):
        return torch.tensor(v)

    @field_validator("tech_gamma")
    def _convert_tech_gamma(cls, v):
        return torch.tensor(v)

    @field_validator("tech_cost")
    def _convert_tech_cost(cls, v):
        return torch.tensor(v)

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)

class InitialGraphArgs(BaseModel):
    """Base class for initial graph arguments."""
    seed: int = 100
    new_node_edges: int = 1

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)

class AlphaDist(BaseModel):
    """Base class for alpha distribution."""
    type: str = "normal"
    parameters: list[float] = [1.08, 0.074]
    round: bool = False
    decimals: int | None = None

    @field_validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)


class CapitalDist(BaseModel):
    """Base class for capital distribution."""
    type: str = "uniform"
    parameters: list[float] = [0., 1.0]
    round: bool = False
    decimals: int | None = None

    @field_validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)


class LambdaDist(BaseModel):
    """Base class for lambda distribution."""
    type: str = "uniform"
    parameters: list[float] = [0.1, 0.9]
    round: bool = True
    decimals: int | None = 1

    @field_validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)


class SigmaDist(BaseModel):
    """Base class for sigma distribution."""
    type: str = "uniform"
    parameters: list[float] = [0.1, 1.9]
    round: bool = True
    decimals: int | None = 1

    @field_validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)


class TechnologyDist(BaseModel):
    """Base class for technology distribution."""
    type: str = "bernoulli"
    parameters: list[float | None] = [0.5, None]
    round: bool = False
    decimals: int | None = None

    @field_validator("parameters")
    def _convert_parameters(cls, v):
        return v if None in v else torch.tensor(v)

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)


class AThetaDist(BaseModel):
    """Base class for a_theta distribution."""
    type: str = "uniform"
    parameters: list[float] = [0.1, 1.0]
    round: bool = False
    decimals: int | None = None

    @field_validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)


class SensitivityDist(BaseModel):
    """Base class for sensitivity distribution."""
    type: str = "uniform"
    parameters: list[float] = [0.0, 1.0]
    round: bool = False
    decimals: int | None = None

    @field_validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    # Make sure pydantic validates the default values
    model_config = ConfigDict(validate_default = True)


class Config(BaseModel):
    """Base class for configuration parameters.
    These are the parameters used by the overarching process.
    """
    model_identifier: str = Field("test", alias='_model_identifier') # because pydantic does not like underscores
    description: str = "" # Never used to influence processing. This value is meant purely to add a description to identify a parameter setting.
    device: str = "cpu"
    seed: int = 42
    number_agents: PositiveInt = 100
    initial_graph_type: str = "barabasi-albert"
    initial_graph_args: InitialGraphArgs = InitialGraphArgs()
    step_target: PositiveInt = 5
    checkpoint_period: int = 10
    milestones: list[PositiveInt] | None = None
    steering_parameters: SteeringParams = SteeringParams()
    alpha_dist: AlphaDist = AlphaDist()
    capital_dist: CapitalDist = CapitalDist()
    cost_vals: list = [0.0, 0.45]
    gamma_vals: list = [0.3, 0.45]
    lambda_dist: LambdaDist = LambdaDist()
    sigma_dist: SigmaDist = SigmaDist()
    technology_dist: TechnologyDist = TechnologyDist()
    technology_levels: list = [0, 1]
    a_theta_dist: AThetaDist = AThetaDist()
    sensitivity_dist: SensitivityDist = SensitivityDist()

    # Make sure pydantic validates the default values
    model_config = ConfigDict(
        validate_default = True,
        protected_namespaces = (), # because _model is a protected namespace
        populate_by_name = True,
        validate_assignment = True,
        extra = "forbid",
        )

    @classmethod
    def from_yaml(cls, config_file):
        """Read configs from a config.yaml file.

        If key is not found in config.yaml, the default value is used.
        """
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file {config_file} not found.")

        with open(config_file) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise SyntaxError(f"Error parsing config file {config_file}.") from exc
        return cls(**cfg)

    @classmethod
    def from_dict(cls, cfg):
        """Read configs from a dict."""
        if not isinstance(cfg, dict):
            raise TypeError("Input must be a dictionary.")
        return cls(**cfg)

    def to_yaml(self, config_file):
        """Write configs to a yaml config_file."""
        if Path(config_file).exists():
            logger.warning(f"Overwriting config file {config_file}.")

        cfg = self.model_dump(by_alias=True, warnings=False)

        # if there are tensors, convert them to lists before saving
        def _convert_value(nested_dict):
            for key, value in nested_dict.items():
                if isinstance(value, torch.Tensor):
                    nested_dict[key] = value.tolist()
                elif isinstance(value, list):
                    nested_dict[key] = [
                        i.tolist() if isinstance(i, torch.Tensor) else i for i in value
                        ]
                elif isinstance(value, dict):
                    nested_dict[key] = _convert_value(value)
            return nested_dict

        cfg = _convert_value(cfg)
        with open(config_file, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

CONFIG = Config()
