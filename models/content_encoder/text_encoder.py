import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput


class TransformersTextEncoderBase(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(
        self,
        text: list[str],
    ):
        device = self.model.device
        batch = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        output: BaseModelOutput = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output = output.last_hidden_state
        mask = (attention_mask == 1).to(device)

        return {"output": output, "mask": mask}


class T5TextEncoder(TransformersTextEncoderBase):
    def __init__(self, model_name: str = "google/flan-t5-large",d_out=None):
        nn.Module.__init__(self)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval()
        self.d_out=d_out
        if d_out!=None:
            self.out_proj=nn.Linear(self.model.config.d_model,d_out)

    def forward(
        self,
        text: list[str],
    ):
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", enabled=False
        ):
            # res.output[bs,seq_len,d_model]
            res=super().forward(text)
            output=res["output"]
            if self.d_out!=None:
                output=self.out_proj(output)
            mask=res["mask"]
            return {
                "output": output,
                "mask": mask
            }


if __name__ == '__main__':
    text_encoder = T5TextEncoder()
    text = ["a man is speaking", "a woman is singing while a dog is barking"]
    text_encoder.eval()
    with torch.no_grad():
        output = text_encoder(text)
        print(output)
        print(output.shape)
