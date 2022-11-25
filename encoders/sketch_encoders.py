from transformers import CLIPProcessor, CLIPModel

class sketch_encoder_clip16():
    def __init__(self):
        # vision_config = CLIPVisionConfig(patch_size = 16)
        # clip_conf = CLIPConfig.from_text_vision_config(vision_conf=vision_config)
        # self.clip_model = CLIPModel(clip_conf)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        # self.image_loader = CLIPFeatureExtractor()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def forward(self,x):
        # pixels = self.image_loader(x)
        # return self.clip_model.get_image_features(pixel_values = pixels)

        inputs = self.processor(images=x, padding=True, return_tensors="pt")
        return self.clip_model.get_image_features(**inputs) 



        
        




        
