!pip install -q transformers torch torchvision pillow matplotlib gradio
!pip install -q accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("\nLoading BLIP for image description...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
class VL_JEPA_System:

    def __init__(self, clip_model, clip_processor, blip_model, blip_processor, device):
        self.device = device
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.blip_model = blip_model
        self.blip_processor = blip_processor
        self.current_image = None
        self.image_features = None
        self.description = None
        self.patch_features = None

    def upload_and_describe(self, image):
        """Step 1: Analyze image"""
        self.current_image = image
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(**inputs)
            self.patch_features = vision_outputs.last_hidden_state
            self.image_features = vision_outputs.pooler_output
        self.description = self._generate_detailed_description(image)
        return self.description

    def _generate_detailed_description(self, image):
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=100)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        prompts = [ "this image shows"]
        descriptions = [caption]

        for prompt in prompts:
            inputs = self.blip_processor(image, prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=100)
            desc = self.blip_processor.decode(out[0], skip_special_tokens=True)
            if desc not in descriptions:
                descriptions.append(desc)

        return ". ".join(descriptions[:3]) + "."

    def answer_question(self, question):
        if self.current_image is None or self.patch_features is None:
            return "⚠️ Please upload an image first!", None, 0.0
        answer = self._generate_answer(question)
        with torch.no_grad():
            text_inputs = self.clip_processor(text=[question], return_tensors="pt", padding=True).to(self.device)
            text_outputs = self.clip_model.text_model(**text_inputs)
            text_hidden = text_outputs.last_hidden_state
            text_features = text_hidden[:, 0, :]
            if text_features.shape[1] != self.patch_features.shape[2]:
                pad_size = self.patch_features.shape[2] - text_features.shape[1]
                if pad_size > 0:
                    text_features = torch.nn.functional.pad(text_features, (0, pad_size))
                else:
                    text_features = text_features[:, :self.patch_features.shape[2]]
            text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            patches_norm = self.patch_features / self.patch_features.norm(dim=-1, keepdim=True)

            attention_scores = torch.matmul(text_norm, patches_norm.transpose(1, 2))
            attention_weights = F.softmax(attention_scores * 100, dim=-1)

        confidence = attention_weights.max().item()
        return answer, attention_weights, confidence

    def _generate_answer(self, question):

        question_lower = question.lower().strip()
        if "indoor" in question_lower or "outdoor" in question_lower:
            inputs = self.blip_processor(
                self.current_image,
                text="a photo taken",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=20)

            caption = self.blip_processor.decode(out[0], skip_special_tokens=True).lower()
            print(f"DEBUG Indoor/Outdoor - Caption: '{caption}'")
            indoor_words = ["indoor", "inside", "room", "kitchen", "office", "building", "house", "wall", "ceiling"]
            outdoor_words = ["outdoor", "outside", "park", "street", "field", "sky", "tree", "nature"]

            if any(word in caption for word in indoor_words):
                return "indoors"
            elif any(word in caption for word in outdoor_words):
                return "outdoors"
            else:
                return "indoors"
        elif "color" in question_lower or "colour" in question_lower:
            inputs = self.blip_processor(
                self.current_image,
                text="the main color is",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=15)

            answer = self.blip_processor.decode(out[0], skip_special_tokens=True)
            print(f"DEBUG Color - Raw: '{answer}'")
            colors = ["red", "blue", "green", "yellow", "orange", "purple",
                     "pink", "brown", "black", "white", "gray", "grey", "tan", "beige"]

            for color in colors:
                if color in answer.lower():
                    return color

            cleaned = answer.replace("the main color is", "").replace("the color is", "").strip()
            return cleaned if cleaned else "mixed colors"

        elif any(phrase in question_lower for phrase in ["what is this", "what type", "what kind", "what's this"]):

            inputs = self.blip_processor(
                self.current_image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=30)

            answer = self.blip_processor.decode(out[0], skip_special_tokens=True)
            print(f"DEBUG What Is - Caption: '{answer}'")

            prefixes = ["this is a ", "this is an ", "there is a ", "there is an ",
                       "a picture of ", "an image of "]

            answer_lower = answer.lower()
            for prefix in prefixes:
                if answer_lower.startswith(prefix):
                    answer = answer[len(prefix):]
                    break

            return answer.strip()

        elif "doing" in question_lower or "holding" in question_lower or "wearing" in question_lower:
            inputs = self.blip_processor(
                self.current_image,
                text="a person",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=40)

            answer = self.blip_processor.decode(out[0], skip_special_tokens=True)
            print(f"DEBUG Person Action - Caption: '{answer}'")
            answer = answer.replace("a person", "").strip()
            if answer and not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]

            return answer if answer else "a person in the image"


        elif "how many" in question_lower:

            inputs = self.blip_processor(
                self.current_image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=30)

            description = self.blip_processor.decode(out[0], skip_special_tokens=True).lower()
            import re
            numbers = re.findall(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', description)

            if numbers:
                return numbers[0]
            elif "group" in description or "several" in description:
                return "several"
            else:
                return "one"
        elif question_lower.startswith(("is ", "are ", "does ", "do ", "can ", "will ")):
            inputs = self.blip_processor(
                self.current_image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=40)

            description = self.blip_processor.decode(out[0], skip_special_tokens=True).lower()
            print(f"DEBUG Yes/No - Description: '{description}'")
            question_words = question_lower.replace("is ", "").replace("are ", "").replace("?", "").split()
            matches = sum(1 for word in question_words if word in description)

            if matches >= len(question_words) * 0.5:
                return "yes"
            else:
                return "no"
        else:
            inputs = self.blip_processor(
                self.current_image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=30)

            answer = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return answer

    def visualize_grounding(self, attention_weights, question, answer, confidence):
        attn = attention_weights.squeeze().cpu().numpy()

        if len(attn) == 50:
            attn = attn[1:]

        attn_map = attn.reshape(7, 7)
        attn_resized = np.array(Image.fromarray(attn_map).resize(
            self.current_image.size, Image.BILINEAR
        ))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(self.current_image)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        im = axes[1].imshow(attn_map, cmap='jet', interpolation='nearest')
        axes[1].set_title("Attention Map", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(self.current_image)
        axes[2].imshow(attn_resized, alpha=0.5, cmap='jet')
        axes[2].set_title(f"Q: {question}\nA: {answer}\nConf: {confidence:.1%}",
                         fontsize=10, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        return fig

vl_jepa = VL_JEPA_System(
    clip_model, clip_processor,
    blip_model, blip_processor,
    device
)
import nest_asyncio
nest_asyncio.apply()

def process_image_and_question(image, question):
    if image is None:
        return "Please upload an image", "Please upload an image first", None

    try:
        description = vl_jepa.upload_and_describe(image)
        if question and len(question.strip()) > 0:
            answer, attention, confidence = vl_jepa.answer_question(question)
            fig = vl_jepa.visualize_grounding(attention, question, answer, confidence)
            answer_text = f"**Answer:** {answer}\n**Confidence:** {confidence:.1%}"
            return description, answer_text, fig
        else:
            return description, "Enter a question above", None

    except Exception as e:
        import traceback
        error = f"Error: {traceback.format_exc()}"
        return error, error, None
demo = gr.Interface(
    fn=process_image_and_question,
    inputs=[
        gr.Image(type="pil", label="📸 Upload Image"),
        gr.Textbox(label="❓ Your Question", placeholder="What is the dog doing?")
    ],
    outputs=[
        gr.Textbox(label="🔍 Image Understanding", lines=5),
        gr.Markdown(label="💡 Answer"),
        gr.Plot(label="🎨 Grounded Evidence")
    ],
    title="🧠 IMAGE ANALYZER VL-JEPA INSPIRED",
    description="Upload an image and ask questions about it!",
    examples=[
        [None, "What is in this image?"],
        [None, "What color is dominant?"],
        [None, "Is this indoors or outdoors?"]
    ],
    cache_examples=False
)
demo.launch(share=True, inline=False, debug=False)
