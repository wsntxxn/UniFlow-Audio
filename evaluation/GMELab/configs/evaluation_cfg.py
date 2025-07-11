import dataclasses
from dataclasses import dataclass, Field, fields
from pathlib import Path, PosixPath
import typing as tp
from warnings import warn

from omegaconf import OmegaConf

from eval_utils.utils import dataclass_from_dict
from eval_utils.exceptions import ConfigurationError, ConfigurationWarning


@dataclass
class FADCfg:
    model_name: str = "vggish"
    sample_rate: int = 16_000
    use_pca: bool = False
    use_activation: bool = False
    dtype: str = "float32"
    embeddings_fn: tp.Optional[str] = None

    def __post_init__(self):
        # TODO: checking
        if self.embeddings_fn is None:
            if self.model_name == "vggish":
                self.embeddings_fn = f"{self.model_name}"
                if self.use_pca:
                    self.embeddings_fn += "_pca"
                if self.use_activation:
                    self.embeddings_fn += "_activation"
                self.embeddings_fn += f"_embeddings.npy"
            else:
                self.embeddings_fn = (
                    f"{self.model_name}_{self.sample_rate}_embeddings.npy"
                )


@dataclass
class KLDCfg:
    pretrained_length: int = 10
    batch_size: int = 10
    num_workers: int = 10
    duration: float = 2.56
    apply_metadata_to_samples: bool = False

    def __post_init__(self):
        # TODO: checking
        pass


@dataclass
class InSyncCfg:
    exp_name: str = "24-01-04T16-39-21"
    device: str = "cuda:0"
    vfps: int = 25
    afps: int = 16_000
    input_size: int = 256
    ckpt_parent_path: str = "./checkpoints/sync_models"

    def __post_init__(self):
        # TODO: checking
        pass


# this is basically the same as InSyncCfg
@dataclass
class AVClipScoreCfg:
    exp_name: str = "24-01-04T16-39-21"
    device: str = "cuda:0"
    vfps: int = 25
    afps: int = 16_000
    input_size: int = 256
    ckpt_parent_path: str = "./checkpoints/sync_models"

    def __post_init__(self):
        # TODO: checking
        pass


@dataclass
class ZCRCfg:
    afps: int = 24_000
    duration: float = 2.56
    apply_metadata_to_samples: bool = False


@dataclass
class RhythmSimilarityCfg:
    afps: int = 24_000
    duration: float = 2.56
    apply_metadata_to_samples: bool = False


@dataclass
class SpectralContrastSimilarityCfg:
    afps: int = 24_000
    duration: float = 2.56
    apply_metadata_to_samples: bool = False


@dataclass
class ImageBindScore:
    device: str = "cuda:0"
    afps: int = 16_000
    get_diagonal_scores: bool = False


@dataclass
class FIDCfg:
    afps: int = 22_050
    ckpt_path: str = "./checkpoints/melception_models/melception-21-05-10T09-28-40.pt"
    device: str = "cuda:0"
    batch_size: int = 10
    num_workers: int = 10
    duration: float = 2.56
    apply_metadata_to_gt: bool = False


@dataclass
class PipelineCfg:
    fad: FADCfg = None
    kld: KLDCfg = None
    insync: InSyncCfg = None
    avclip_score: AVClipScoreCfg = None
    zcr: ZCRCfg = None
    rhythm_similarity: RhythmSimilarityCfg = None
    spectral_contrast_similarity: SpectralContrastSimilarityCfg = None
    imagebind_score: ImageBindScore = None
    fid: FIDCfg = None


@dataclass
class EvaluationCfg:
    # unique identifier for the evaluation
    id: str
    # directory containing evaluation data (.mp4 and .wav)
    sample_directory: Path
    # defined evaluation pipeline
    pipeline: PipelineCfg
    # ground truth data (.wav)
    gt_directory: Path
    # sample directory specific metadata
    sample_vfps: int = 25
    sample_afps: int = 24_000
    sample_min_side: int = 256
    sample_acodec: str = "aac"
    sample_vcodec: str = "h264"
    # ground truth directory specific metadata
    gt_vfps: int = 25
    gt_afps: int = 24_000
    gt_min_side: int = 256
    gt_acodec: str = "aac"
    gt_vcodec: str = "h264"
    # metadata
    metadata: tp.Optional[Path] = None
    # directories to save evaluation results
    result_directory: tp.Optional[Path] = None
    verbose: bool = False

    def __post_init__(self):
        self.sample_directory = Path(self.sample_directory)
        self.gt_directory = Path(self.gt_directory)
        self.metadata = Path(self.metadata) if self.metadata else None

        if not self.sample_directory.is_dir():
            raise ConfigurationError("sample_directory not existing directory")

        if (
            self.pipeline.fad is not None
            or self.pipeline.kld is not None
            or self.pipeline.zcr is not None
            or self.pipeline.rhythm_similarity is not None
            or self.pipeline.spectral_contrast_similarity is not None
        ):
            if not self.gt_directory.is_dir():
                raise ConfigurationError(
                    "gt_directory must be a Path object to an existing directory"
                )

        if self.result_directory is None:
            self.result_directory = self.sample_directory
        if not self.result_directory.is_dir():
            raise ConfigurationError(
                "result_directory must be a Path object to an existing directory"
            )

        self._print_pipeline()

    def _print_pipeline(self):
        print("Evaluation pipeline:")
        print(f"id: {self.id}")
        print(f"sample_directory: {self.sample_directory}")
        print(f"gt_directory: {self.gt_directory}")
        print(f"metadata: {self.metadata}")
        print(f"result_directory: {self.result_directory}")
        print(self.pipeline)
        print(f"verbose: {self.verbose}")
        print()


def get_evaluation_config(eval_cfg_dict: tp.Union[dict, Path, str]) -> EvaluationCfg:
    """Method to get evaluation config from a dictionary.

    Args:
        eval_cfg_dict (dict, Path, str): Dictionary defining the evaluation configuration
        or path to the configuration file.

    Returns:
        EvaluationCfg: Evaluation configuration object.
    """
    if type(eval_cfg_dict) in [str, Path, PosixPath]:
        eval_cfg_dict = OmegaConf.to_container(
            OmegaConf.load(eval_cfg_dict), resolve=True
        )
    assert type(eval_cfg_dict) == dict
    _check_evaluation_cfg_dict(eval_cfg_dict)
    evaluation_cfg = dataclass_from_dict(EvaluationCfg, eval_cfg_dict)
    return evaluation_cfg


def _check_evaluation_cfg_dict(eval_cfg_dict: dict):
    """Method to check evaluation config dictionary. Highlights the errors in the
    user defined configuration.

    Note, this function does not check that directories exists etc. That is done in
    the dataclass init.

    Args:
        eval_cfg_dict (dict): User defined evaluation configuration dictionary.

    Raises:
        AssertionError: If the evaluation configuration dictionary is not valid.
    """

    # check the main configuration
    req_fields, all_fields = _get_dataclass_fields(EvaluationCfg)
    dict_keys = set(eval_cfg_dict.keys())

    missing_keys = req_fields - dict_keys
    extra_keys = dict_keys - all_fields
    if missing_keys:
        raise ConfigurationError(f"EvaluationCfg missing keys: {missing_keys}")
    if extra_keys:
        raise ConfigurationError(f"EvaluationCfg has extra keys: {extra_keys}")

    # check the pipeline configuration
    _, all_fields = _get_dataclass_fields(PipelineCfg)
    if not eval_cfg_dict["pipeline"]:
        raise ConfigurationError(f"PipelineCfg is empty. No metrics configured.")

    extra_keys = eval_cfg_dict["pipeline"].keys() - all_fields
    if extra_keys:
        raise ConfigurationError(f"PipelineCfg has extra keys: {extra_keys}")

    # check individual metric configurations
    for metric_name, metric_cfg in eval_cfg_dict["pipeline"].items():
        if metric_cfg is None:
            raise ConfigurationError(f"{metric_name} configuration is empty.")
        metric_dataclass = _match_dataclass_field_name_to_type(metric_name, PipelineCfg)
        req_fields, all_fields = _get_dataclass_fields(metric_dataclass)
        dict_keys = set(metric_cfg.keys())

        missing_keys = req_fields - dict_keys
        extra_keys = dict_keys - all_fields
        if missing_keys:
            raise ConfigurationError(f"{metric_name} missing keys: {missing_keys}")
        if extra_keys:
            raise ConfigurationError(f"{metric_name} has extra keys: {extra_keys}")


def _get_dataclass_fields(dataclass: tp.Type) -> tp.Tuple[tp.Set[str], tp.Set[str]]:
    req_dataclass_fields = {f.name for f in fields(dataclass) if _is_required_field(f)}
    all_dataclass_fields = {f.name for f in fields(dataclass)}
    return req_dataclass_fields, all_dataclass_fields


def _is_required_field(field: Field) -> bool:
    return (
        field.default == dataclasses.MISSING
        and field.default_factory == dataclasses.MISSING
    )


def _match_dataclass_field_name_to_type(field_name: str, dataclass: tp.Type) -> tp.Type:
    return next(f.type for f in fields(dataclass) if f.name == field_name)
