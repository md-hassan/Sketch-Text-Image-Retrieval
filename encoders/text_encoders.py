from transformers import CLIPTokenizer, CLIPModel

class text_encoder_clip16():
    def __init__(self):
        # vision_config = CLIPVisionConfig(patch_size = 16)
        # clip_conf = CLIPConfig.from_text_vision_config(vision_conf=vision_config)
        # self.clip_model = CLIPModel(clip_conf)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    
    def forward(self,x):
        inputs = self.text_tokenizer(x, padding=True, return_tensors="pt")
        return self.clip_model.get_text_features(**inputs) 
