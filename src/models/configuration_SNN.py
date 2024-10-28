import torch, timm

import torch.onnx
import torch.nn as nn
import torch.nn.functional as F

from timm.data import resolve_data_config

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from configs.dataclasses import SNNArguments, ONNXExportArguments


class SNN(nn.Module):
    
    """
    Siamese Neural Network (SNN) for comparing images using a pretrained vision model from the timm library.
    
    This model architecture converts any vision model into an SNN with an optional trainable additional layer
    to transform the output of the penultimate layer before comparing the two image representations.

    Args:
        snn_args.model_name (str): 
            The name of the vision model to use from the timm library.
        snn_args.similarity_fn (str):
            Defines the similarity function used in the SNN. 
            Must be 'cosine_similarity' or 'euclidean_distance'.
        snn_args.add_layer (bool, optional, defaults to False): 
            If True, adds a trainable linear transformation layer between the output layer and the similarity computation.
        snn_args.use_logits (bool, optional, defaults to False): 
            if True, return the logits (before softmax). If set to False, return the penultimate layer features.
        snn_args.train_added_layer_only (bool, optional, defaults to False): 
            if True, only the added layer will be trained in training mode.

    Attributes:
        model (torch.nn.Module): 
            The pretrained vision model from the timm library.
        added_layer (torch.nn.Module or None): 
            Optional linear layer to transform the penultimate layer's output if `snn_args.add_layer` is True.
    """
        
    def __init__(self, snn_args: SNNArguments):
        
        super().__init__()
        
        self.snn_args = snn_args
        
        self.base_model = (timm.create_model(snn_args.model_name, pretrained=True) if snn_args.use_logits
                      else timm.create_model(snn_args.model_name, pretrained=True, num_classes=0))
        
        self.base_model.eval()
        
        if snn_args.similarity_fn == "cosine_similarity": 
            self.snn_args.similarity_fn = "cosine_similarity"
        elif snn_args.similarity_fn == "euclidean_distance": 
            self.snn_args.similarity_fn = "euclidean_distance"
        else:  
            raise ValueError(f"Invalid similarity function: {snn_args.similarity_fn}. Must be 'cosine_similarity' or 'euclidean_distance'")
        
        base_output_shape = self.get_base_output_shape()
        self.added_layer = nn.Linear(base_output_shape[-1], base_output_shape[-1]) if snn_args.add_layer else None
        
        # If train_added_layer_only is selected, freeze everything except the added layer
        if snn_args.train_added_layer_only:
            for param in self.base_model.parameters():
                param.requires_grad = False

            for param in self.added_layer.parameters():
                param.requires_grad = True
    
    def get_base_input_shape(self):
        data_config = resolve_data_config(self.base_model.pretrained_cfg, model=self.base_model)
        input_shape = (1, *data_config['input_size'])
        
        return input_shape
    
    def get_base_output_shape(self):
        input_shape = self.get_base_input_shape()
        output_shape = self.base_model(torch.randn(input_shape)).shape
        
        return output_shape
    
    def get_final_layers(self, *images: torch.Tensor):
        if (self.added_layer is not None):
            transformed_layers = [self.added_layer(image) for image in images]
            return transformed_layers
        
        return images
    
    def similarity(self, a: torch.Tensor, b: torch.Tensor, dim=1):
        if self.snn_args.similarity_fn == "cosine_similarity":
            a_norm = a / a.norm(dim=dim, keepdim=True)
            b_norm = b / b.norm(dim=dim, keepdim=True)
            
            return a_norm @ b_norm.t()
        
        if self.snn_args.similarity_fn == "euclidean_distance":
            return torch.sqrt(torch.sum(torch.square(a - b), dim=dim))
        
    def forward(
                 self, 
                 input_images: torch.Tensor, 
                 target_images: torch.Tensor,
                 negative_images: torch.Tensor = None
                 ) -> torch.Tensor:
        
        """
        Forward pass for comparing an image to one or more other images.

        Args:
            input_images (torch.Tensor): 
                The tensor representing the primary image to be compared with the target and negative images.
            target_images (torch.Tensor): 
                During evaluation, this is the tensor representing the second image for comparison,
                and it a value error is raised if it is more than 1 image.
                During training, this can be 1 or more images,
                and these will be the positive images.
            negative_images (torch.Tensor):
                One or more tensors representing images that are dissimilar to the input_images.
                Only used when training is set to True.
            
        Returns:
            torch.Tensor: 
                If training is set to False, returns the cosine_similarity between the input_images and target_image.
                If training is set to True, returns the contrastive loss value for the input_images, target_image, and negative_images.
        """
        
        if input_images.ndim != 4:
            raise ValueError(f"Expected input image with 4 dimensions but got input image with {input_images.ndim} dimenions instead.")
        
        if target_images.ndim != 4:
            raise ValueError(f"Expected target images with 4 dimensions but got target images with {target_images.ndim} dimenions instead.")
        
        if self.training:
            if negative_images == None:
                raise ValueError("At least one negative image must be provided in training mode.")
            if negative_images.ndim != 4:
                raise ValueError(f"Expected negative images with 4 dimensions but got negative images with {negative_images.ndim} dimenions instead.")
        
        if not self.training and negative_images != None:
            raise ValueError("Cannot pass negative_images in eval mode.")
        
        if self.training:
            # Forward logic for training mode
            input_images_features    = self.base_model(input_images)
            target_images_features   = self.base_model(target_images)
            negative_images_features = self.base_model(negative_images)
            
            (input_images_final,
            target_images_final,
            negative_images_final) = self.get_final_layers(
                input_images_features,
                target_images_features,
                negative_images_features
            )
                
            target_images_similarity   = self.similarity(input_images_final, target_images_final, dim=1)
            negative_images_similarity = self.similarity(input_images_final, negative_images_final, dim=1)
            
            loss = -torch.log(torch.exp(target_images_similarity) / torch.sum(torch.exp(negative_images_similarity))).mean()
            return loss
                
        # Forward logic for eval mode
        input_images_features   = self.base_model(input_images)
        target_images_features = self.base_model(target_images)
        
        (input_images_final,
        target_images_final)  = self.get_final_layers(
            input_images_features,
            target_images_features)
            
        similarity = self.similarity(input_images_final, target_images_final)
        
        return similarity
    
    def export_to_onnx(self, export_args: ONNXExportArguments):
        """
        Export the model to ONNX format.

        Args:
            output_path (str): Name of the output ONNX model file.
        """
        
        self.eval()
        dummy_input = (torch.rand(self.get_base_input_shape()), 
                       torch.rand(self.get_base_input_shape())) 

        torch.onnx.export(self,              
                          dummy_input,
                          export_args.output_path,
                          export_params=export_args.export_params,
                          opset_version=export_args.opset_version,
                          do_constant_folding=export_args.do_constant_folding,
                          input_names=export_args.input_names,
                          output_names=export_args.output_names)