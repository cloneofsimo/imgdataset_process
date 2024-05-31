import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import os
import huggingface_hub
from typing import List, Tuple
from hpsv2.utils import root_path, hps_version_map

warnings.filterwarnings("ignore", category=UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_model(hps_version: str = "v2.0") -> Tuple[torch.nn.Module, callable]:
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )

    # Check if the checkpoint exists and download if necessary
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    return model, preprocess_val

@torch.no_grad()
def score_pair(model: torch.nn.Module, preprocess_val: callable, prompts_image_pairs: List[Tuple[Image.Image, str]]) -> List[float]:
    tokenizer = get_tokenizer('ViT-H-14')
    results = []
    image_tensors = []
    texts = []

    for image, prompt in prompts_image_pairs:
        
            # Process the image
        image_tensor = preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
        # Process the prompt
        text = tokenizer([prompt], context_length=77).to(device=device, non_blocking=True)
            # Calculate the HPS
        image_tensors.append(image_tensor)
        texts.append(text)

    image_tensor = torch.cat(image_tensors, dim=0)
    text = torch.cat(texts, dim=0)

    with torch.cuda.amp.autocast():
        outputs = model(image_tensor, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        inner_product = torch.einsum("bf,bf->b", image_features, text_features)
        hps_score = inner_product.cpu().numpy().tolist()

    return hps_score

if __name__ == '__main__':
    import argparse
    
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', nargs='+', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--hps-version', type=str, default="v2.0", help='HPS version to use for the model checkpoint')

    args = parser.parse_args()

    # Initialize the model
    model, preprocess_val = initialize_model(args.hps_version)

    # Prepare image-prompt pairs
    image_prompt_pairs = [(Image.open(img_path).convert('RGB'), args.prompt) for img_path in args.image_path] * 2

    # Calculate the scores
    hps_scores = score_pair(model, preprocess_val, image_prompt_pairs)
    print('HPSv2 scores:', hps_scores)