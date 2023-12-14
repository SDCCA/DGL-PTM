from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
import torch
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

class MThetaDist(BaseModel):
    """Base class for m_theta distribution."""
    type: str = "multinomial"
    parameters: List[List[float]] = [[0.02, 0.03, 0.05, 0.9], [0.7, 0.8, 0.9, 1]]
    round: bool = False
    decimals: Optional[int] = None

    @validator("parameters")
    def _convert_parameters(cls, v):
        for i in v:
            if not isinstance(i, list):
                raise TypeError("parameters must be a list of lists")
        return [torch.tensor(i) for i in v]

    class Config:
        validate_default = True

class SteeringParams(BaseModel):
    """Base class for steering parameters."""
    deletion_prob: float = 0.05
    edata: List[str] = ["all"]
    epath: str = "./edge_data"
    format: str = "xarray"
    mode: str = "w"
    ndata: List[Union[str, List[str]]] = ["all_except", ["a_table"]]
    npath: str = "./agent_data.zarr"
    ratio: float = 0.1
    wealth_method: str = "singular_transfer"
    income_method: str = "default"
    consume_method: str = "default"
    capital_update_method: str = "default"
    characteristic_distance: int = 35
    homophily_parameter: float = 0.69
    perception_method: str = "default"
    adapt_m: List[float] = [0.0, 0.5, 0.9]
    adapt_cost: List[float] = [0.0, 0.25, 0.45]
    depreciation: float = 0.6
    discount: float = 0.95
    m_theta_dist: MThetaDist = MThetaDist()
    weight_a: float = 0.69
    weight_b: int = 35
    tech_gamma: List[float] = [0.3, 0.35, 0.45]
    tech_cost: List[float] = [0.0, 0.15, 0.65]
    truncation_weight: float = 1.0e-10
    step_type: str = "custom"

    @validator("adapt_m")
    def _convert_adapt_m(cls, v):
        return torch.tensor(v)

    @validator("adapt_cost")
    def _convert_adapt_cost(cls, v):
        return torch.tensor(v)

    @validator("tech_gamma")
    def _convert_tech_gamma(cls, v):
        return torch.tensor(v)

    @validator("tech_cost")
    def _convert_tech_cost(cls, v):
        return torch.tensor(v)

    class Config:
        validate_default = True


class AlphaDist(BaseModel):
    """Base class for alpha distribution."""
    type: str = "normal"
    parameters: List[float] = [1.08, 0.074]
    round: bool = False
    decimals: Optional[int] = None

    @validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    class Config:
        validate_default = True


class CapitalDist(BaseModel):
    """Base class for capital distribution."""
    type: str = "uniform"
    parameters: List[float] = [0., 1.0]
    round: bool = False
    decimals: Optional[int] = None

    @validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    class Config:
        validate_default = True


class LambdaDist(BaseModel):
    """Base class for lambda distribution."""
    type: str = "uniform"
    parameters: List[float] = [0.1, 0.9]
    round: bool = True
    decimals: Optional[int] = 1

    @validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    class Config:
        validate_default = True


class SigmaDist(BaseModel):
    """Base class for sigma distribution."""
    type: str = "uniform"
    parameters: List[float] = [0.1, 1.9]
    round: bool = True
    decimals: Optional[int] = 1

    @validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    class Config:
        validate_default = True


class TechnologyDist(BaseModel):
    """Base class for technology distribution."""
    type: str = "bernoulli"
    parameters: List[float | None] = [0.5, None]
    round: bool = False
    decimals: Optional[int] = None

    @validator("parameters")
    def _convert_parameters(cls, v):
        return v if None in v else torch.tensor(v)

    class Config:
        validate_default = True


class AThetaDist(BaseModel):
    """Base class for a_theta distribution."""
    type: str = "uniform"
    parameters: List[float] = [0.1, 1.0]
    round: bool = False
    decimals: Optional[int] = None

    @validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    class Config:
        validate_default = True


class SensitivityDist(BaseModel):
    """Base class for sensitivity distribution."""
    type: str = "uniform"
    parameters: List[float] = [0.0, 1.0]
    round: bool = False
    decimals: Optional[int] = None

    @validator("parameters")
    def _convert_parameters(cls, v):
        return torch.tensor(v)

    class Config:
        validate_default = True


class Config(BaseModel):
    """Base class for configuration parameters."""
    model_identifier: str = Field("test", alias='_model_identifier') # because pydantic does not like underscores
    number_agents: int = 100
    initial_graph_type: str = "barabasi-albert"
    model_data: dict = {}
    model_graph: object = None # TODO confirm that model_graph is a class obj
    step_count: int = 0
    step_target: int = 5
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
    adapt_m: list = [0.0, 0.5, 0.9]
    adapt_cost: list = [0.0, 0.25, 0.45]
    depreciation: float = 0.6
    discount: float = 0.95

    @validator("adapt_m")
    def _convert_adapt_m(cls, v):
        return torch.tensor(v)

    @validator("adapt_cost")
    def _convert_adapt_cost(cls, v):
        return torch.tensor(v)

    class Config:
        validate_default = True
        protected_namespaces = () # because model_ is used internally
        populate_by_name = True

    @classmethod
    def from_yaml(cls, config_file):
        """Read configs from a config.yaml file.

        If key is not found in config.yaml, the default value is used.
        """
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file {config_file} not found.")

        with open(config_file, "r") as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise SyntaxError(f"Error parsing config file {config_file}.") from exc
        return cls(**cfg)

    # add a class method to read configs from a dict
    @classmethod
    def from_dict(cls, cfg):
        """Read configs from a dict."""
        if not isinstance(cfg, dict):
            raise TypeError("Input must be a dict.")
        return cls(**cfg)

    def to_yaml(self, config_file):
        """Write configs to a config.yaml file."""
        if Path(config_file).exists():
            logger.warning(f"Overwriting config file {config_file}.")

        cfg = self.model_dump(by_alias=True, warnings=False)
        # if there are tensors, convert them to lists before saving
        def _convert_value(nested_dict):
            for key, value in nested_dict.items():
                if isinstance(value, torch.Tensor):
                    nested_dict[key] = value.tolist()
                elif isinstance(value, List):
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
