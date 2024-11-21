import timm, yaml, os
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

from transformers import TrainingArguments

@dataclass
class SNNArguments:
    """
    Arguments pertaining to the SNN model and training details.
    """

    model_name: str = field(
        default=None,
        metadata={"help": ("The name of the vision model to use from the timm library.")},
    )
    
    train: bool = field(
        default=False,
        metadata={"help": (
                "Determines whether the model will be in training or evaluation mode."
            )},
    )
    
    similarity_fn: str = field(
        default="cosine_similarity",
        metadata={
            "help": (
                "Defines the similarity function used in the SNN. "
                "Must be 'cosine_similarity' or 'euclidean_distance'."
            )},
    )
    
    add_layer: bool = field(
        default=False,
        metadata={"help": ("If True, adds a trainable linear transformation layer "
                           "between the output layer and the similarity computation.")},
    )
    
    use_logits: str = field(
        default=False,
        metadata={"help": ("if True, return the logits (before softmax). "
                           "If set to False, return the penultimate layer features.")},
    )
    
    train_added_layer_only: str = field(
        default=False,
        metadata={"help": ("if True, return the logits (before softmax). "
                           "If set to False, return the penultimate layer features.")},
    )
    
    def __post_init__(self):
        available_models = timm.list_models()

        if self.model_name not in (None, *available_models):
            raise ValueError(f"The provided model_name '{self.model_name}' is not a valid model from timm.")
        
        if self.model_name == None:
            raise ValueError("A valid 'model_name' must be provided. Please select a model from the timm library.")
        
@dataclass
class SNNTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to the SNN model and training details.
    """

    model_name: str = field(
        default=None,
        metadata={"help": ("The name of the vision model to use from the timm library.")},
    )
    
    train: bool = field(
        default=False,
        metadata={"help": (
                "Determines whether the model will be in training or evaluation mode."
            )},
    )
    
    similarity_fn: str = field(
        default="cosine_similarity",
        metadata={
            "help": (
                "Defines the similarity function used in the SNN. "
                "Must be 'cosine_similarity' or 'euclidean_distance'."
            )},
    )
    
    add_layer: bool = field(
        default=False,
        metadata={"help": ("If True, adds a trainable linear transformation layer "
                           "between the output layer and the similarity computation.")},
    )
    
    use_logits: str = field(
        default=False,
        metadata={"help": ("if True, return the logits (before softmax). "
                           "If set to False, return the penultimate layer features.")},
    )
    
    train_added_layer_only: str = field(
        default=False,
        metadata={"help": ("if True, return the logits (before softmax). "
                           "If set to False, return the penultimate layer features.")},
    )
    
    train_dataset_path: str = field(
        default = None,
        metadata={"help": ("Dataset path to be trained on.")}
    )
    
    eval_dataset_path: str = field(
        default = None,
        metadata={"help": ("Dataset path to be evaluated on.")}
    )
    
    # def __post_init__(self):
    #     available_models = timm.list_models()

    #     if self.model_name not in (None, *available_models):
    #         raise ValueError(f"The provided model_name '{self.model_name}' is not a valid model from timm.")
        
    #     if self.model_name == None:
    #         raise ValueError("A valid 'model_name' must be provided. Please select a model from the timm library.")

@dataclass
class ONNXExportArguments:
    """
    Arguments pertaining to exporting the model to ONNX format.
    """

    output_path: str = field(
        default="mobilenetv3_large_snn.onnx",
        metadata={"help": ("The path where the ONNX model will be saved.")}
    )
    
    export_params: bool = field(
        default=True,
        metadata={"help": ("Store the trained parameter weights inside the model file.")}
    )
    
    opset_version: int = field(
        default=11,
        metadata={"help": ("The ONNX opset version to export the model to.")}
    )
    
    do_constant_folding: bool = field(
        default=True,
        metadata={"help": ("Whether to perform constant folding optimization during export.")}
    )
    
    input_names: List[str] = field(
        default_factory=lambda: ['input'],
        metadata={"help": ("List of input names for the ONNX model.")}
    )
    
    output_names: List[str] = field(
        default_factory=lambda: ['output'],
        metadata={"help": ("List of output names for the ONNX model.")}
    )

    def __post_init__(self):
        if not isinstance(self.input_names, list):
            raise ValueError("'input_names' must be a list of input names.")
        if not isinstance(self.output_names, list):
            raise ValueError("'output_names' must be a list of output names.")
    
    
