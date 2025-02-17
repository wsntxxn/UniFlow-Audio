from pathlib import Path
import torch
import torch.nn as nn
from utils.torch_utilities import load_pretrained_model, merge_matched_keys


class LoadPretrainedBase(nn.Module):
    def process_state_dict(
        self, model_dict: dict[str, torch.Tensor],
        state_dict: dict[str, torch.Tensor]
    ):
        """
        Custom processing functions of each model that transforms `state_dict` loaded from 
        checkpoints to the state that can be used in `load_state_dict`.
        Use `merge_mathced_keys` to update parameters with matched names and shapes by 
        default.  

        Args
            model_dict:
                The state dict of the current model, which is going to load pretrained parameters
            state_dict:
                A dictionary of parameters from a pre-trained model.

            Returns:
                dict[str, torch.Tensor]:
                    The updated state dict, where parameters with matched keys and shape are 
                    updated with values in `state_dict`.      
        """
        state_dict = merge_matched_keys(model_dict, state_dict)
        return state_dict

    def load_pretrained(self, ckpt_path: str | Path):
        load_pretrained_model(
            self, ckpt_path, state_dict_process_fn=self.process_state_dict
        )


class CountParamsBase(nn.Module):
    def count_params(self):
        num_params = 0
        trainable_params = 0
        for param in self.parameters():
            num_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return num_params, trainable_params


class SaveTrainableParamsBase(nn.Module):
    @property
    def param_names_to_save(self):
        names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                names.append(name)
        for name, _ in self.named_buffers():
            names.append(name)
        return names

    def load_state_dict(self, state_dict, strict=True):
        for key in self.param_names_to_save:
            if key not in state_dict:
                raise Exception(
                    f"{key} not found in either pre-trained models (e.g. BERT)"
                    " or resumed checkpoints (e.g. epoch_40/model.pt)"
                )
        return super().load_state_dict(state_dict, strict)
