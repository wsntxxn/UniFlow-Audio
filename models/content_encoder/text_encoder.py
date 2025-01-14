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
        max_length: int | None = None,
        padding: bool | str = True,
    ):
        device = self.model.device
        batch = self.tokenizer(
            text,
            max_length=max_length or self.tokenizer.model_max_length,
            padding=padding,
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
    def __init__(self, model_name: str = "google/flan-t5-large"):
        nn.Module.__init__(self)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.eval()

    def forward(
        self,
        text: list[str],
        max_length: int | None = None,
        padding: bool | str = True
    ):
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", enabled=False
        ):
            return super().forward(text, max_length, padding)


if __name__ == '__main__':
    text_encoder = T5TextEncoder()
    text = ["a man is speaking", "a woman is singing while a dog is barking"]
    text_encoder.eval()
    with torch.no_grad():
        output = text_encoder(text)
