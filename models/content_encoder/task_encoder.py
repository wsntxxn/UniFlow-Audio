import torch.nn as nn
import torch

class TaskEncoder(nn.Module):
    def __init__(
        self,
        d_model: int=256,
        n_tasks:int=4,
    ):
        super().__init__()
        self.task_embed=nn.Embedding(
            num_embeddings=n_tasks+1,
            embedding_dim=d_model,
            padding_idx=0,
        )
        self.task_list=[]
    def forward(self, task:str):
        #将task 映射到唯一的id
        if task not in self.task_list:
            self.task_list.append(task)
        task_id=self.task_list.index(task)+1
        return self.task_embed(torch.tensor(task_id, device=self.task_embed.weight.device))

