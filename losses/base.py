import torch
import torch.nn as nn


class IndentityWrapper(nn.Module):
    def forward(self, loss: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": loss}


class LossSumWrapper(nn.Module):
    def __init__(self, weights: dict[str, float]):
        super().__init__()
        self.weights = weights

    def forward(self,
                loss_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        total_loss = 0
        for loss_name, loss_val in loss_dict.items():
            total_loss += loss_val * self.weights[loss_name]
        output = {"loss": total_loss}
        output.update(loss_dict)
        return output
