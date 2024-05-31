import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from diffusers.models import AutoencoderKL
from streaming import MDSWriter

import logging
import time
import numpy as np
from typing import Any
import json
from streaming.base.format.mds.encodings import Encoding, _encodings
from tqdm import tqdm
from torch.utils.data import DataLoader
import webdataset as wds
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import hpsv2
from torchvision import transforms
from hpsv2r import initialize_model, score_pair

# Initialize logging
logging.basicConfig(level=logging.INFO)


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.uint8)


class np16(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float16)


_encodings["np16"] = np16
_encodings["uint8"] = uint8


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose(
    [
        transforms.Resize(288),
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        normalize,
    ]
)


def crop_to_center(image, new_size=768):
    width, height = image.size
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def prepare_image(pil_image):

    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image


def wds_preprocess(x):
    key, pil_image, _json = x
    pil_image = pil_image.convert("RGB")
    # resize one side to 256
    w, h = pil_image.size
    if w > h:
        pil_image = pil_image.resize((256, int(h * 256 / w)))
    else:
        pil_image = pil_image.resize((int(w * 256 / h), 256))

    pil_image = crop_to_center(pil_image, new_size=256)

    image_for_vae = prepare_image(pil_image)
    image_for_sscd = small_288(pil_image)

    caption = _json["caption"]
    # print(_json) # {'uid': '95ff62922b3536189768bcc883598109', 'clip_b32_similarity_score': 0.299560546875, 'clip_l14_similarity_score': 0.3017578125, 'caption': 'Picture of Car seat 0+ cover Little Goose', 'url': 'https://dealers.little-dutch.com/content/images/thumbs/002/0023598_1000.jpeg', 'key': '000010028', 'status': 'success', 'error_message': None, 'width': None, 'height': None, 'original_width': None, 'original_height': None, 'exif': '{}', 'sha256': 'b1e6f78d70b10645f54682c5cb01a8ba9584f6e34b4f292b431350fa93e94060'}

    # return {"image_for_vae": image_for_vae, "caption": caption, "image_for_sscd": image_for_sscd, 'uid' : _json['uid'], 'clip_simscore': _json['clip_l14_similarity_score']}
    return (image_for_vae, caption, image_for_sscd, key)

def parse_captions(captions_file: str) -> dict[str, str]:
    with open(captions_file, "r") as f:
        lines = f.readlines()
    return {kv[0]: kv[1] for line in lines if len(kv := line.split("\t", 1)) == 2}



COLUMNS = {
    "key": "str",
    "caption": "str",
    "image": "jpeg",
    "vae_256x256_latents": "np16",
    "t5_xl_embeddings": "uint8",
    "sscd_embeddings": "np16",
    "hps_score": "str",
}


@torch.no_grad()
def convert_to_mds(dataset_paths, out_roots, device, is_test=False):
    logging.info(f"Processing on {device}")

    vae_model = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    vae_model = vae_model.to(device).eval()
    vae_model.to(memory_format=torch.channels_last)

    t5tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pile-t5-xl", use_fast=False)
    t5tokenizer.pad_token = t5tokenizer.bos_token
    t5model = AutoModelForSeq2SeqLM.from_pretrained("EleutherAI/pile-t5-xl", torch_dtype=torch.bfloat16)
    t5model = t5model.to(device).eval()

    hps_model, hps_preprocess_val = initialize_model("v2.0")

    #t5model.encoder = torch.compile(t5model.encoder, mode='reduce-overhead')

    sscd_model = torch.jit.load("sscd_disc_mixup.torchscript.pt").to("cuda")

    parsed_captions_kv = parse_captions("/home/ubuntu/here/captions.tsv")
    #parsed_captions_kv = {}

    # randomly sample 10 items
    for i, (k, v) in enumerate(parsed_captions_kv.items()):
        print(k, v)
        if i > 10:
            break

    dataset_bulks = [
        dataset_paths[i : i + 8] for i in range(0, len(dataset_paths), 8)
    ]
    out_roots_bulks = [out_roots[i : i + 8] for i in range(0, len(out_roots), 8)]

    for dataset_paths, out_roots in zip(dataset_bulks, out_roots_bulks):

        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                logging.info(f"Dataset not found: {dataset_path}")
                return
            
        out_root = out_roots[0]

        dataset = wds.DataPipeline(
            wds.SimpleShardList(dataset_paths),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.decode("pil", handler=wds.warn_and_continue),
            wds.to_tuple("__key__", "jpg", "json", handler=wds.warn_and_continue),
            wds.map(wds_preprocess),
            wds.batched(512),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=24,
            prefetch_factor=2,
            shuffle=False,
            drop_last=False,
            timeout=120,
            collate_fn=lambda x: x,
        )

        t0 = time.time()
        sub_data_root = os.path.join(out_root, "data")

        if os.path.exists(sub_data_root):
            for file in os.listdir(sub_data_root):
                os.remove(os.path.join(sub_data_root, file))

        os.makedirs(sub_data_root, exist_ok=True)
        inference_latencies = []
        keys = []

        with MDSWriter(out=sub_data_root, columns=COLUMNS) as out:

            for idx, batch in tqdm(enumerate(dataloader)):

                if is_test and idx > 0:
                    break

                start_time = time.time()

                # image_for_vae, captions = batch["image_for_vae"], batch["caption"]
                # image_for_sscd = batch["image_for_sscd"]
                image_for_vae, captions, image_for_sscd, uids = batch

                # Replace Captions:
                new_captions = [parsed_captions_kv.get(uid, old_caption) for uid, old_caption in zip(uids, captions)]

                for nc, oc in zip(new_captions, captions):
                    print(f"New Caption: {nc}")
                    print(f"Old Caption: {oc}")
                    break
                    
                captions = new_captions

                image_back_to_pil = [
                    Image.fromarray(
                        ((x.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(
                            np.uint8
                        )
                    )
                    for x in image_for_vae
                ]

                ### SSCD
                image_for_sscd = image_for_sscd.to(
                    "cuda", memory_format=torch.channels_last
                )
                sscd_embeddings = sscd_model(image_for_sscd)
                sscd_embeddings = sscd_embeddings.cpu().numpy().astype(np.float16)

                ### VAE
                image_for_vae = image_for_vae.to(device).half()
                vae_latents = vae_model.encode(image_for_vae).latent_dist.sample()
                vae_outputs = vae_latents.cpu().numpy().astype(np.float16)

                ### T5
                t5_inputs = t5tokenizer(
                    captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                t5_inputs = {k: v.to(device) for k, v in t5_inputs.items()}
                t5_outputs = t5model.encoder(**t5_inputs)[0]
                mask = (
                    t5_inputs["attention_mask"].unsqueeze(-1).expand(t5_outputs.shape)
                )
                t5_outputs = t5_outputs * mask
                t5_outputs = (
                    ((t5_outputs.clip(-0.25, 0.25) / 0.5 + 0.5) * 255.0)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                ### HPSv2
                image_prompt_pairs = [(img, prompt) for img, prompt in zip(image_back_to_pil, captions)]
                hps_scores = score_pair(hps_model, hps_preprocess_val, image_prompt_pairs)


                ### Write
                for i in range(len(captions)):
                    # COLUMNS = {
                    #     "key": "str",
                    #     "caption": "str",
                    #     "image": "jpeg",
                    #     "vae_256x256_latents": "np16",
                    #     "t5_xl_embeddings": "uint8",
                    #     "sscd_embeddings": "np16",
                    # }
                    sample = {
                        "vae_256x256_latents": vae_outputs[i],
                        "caption": str(captions[i]),
                        "t5_xl_embeddings": t5_outputs[i],
                        "sscd_embeddings": sscd_embeddings[i],
                        "key": uids[i],
                        "image": image_back_to_pil[i],
                        "hps_score": str(hps_scores[i]),
                    }
                    out.write(sample)

                inference_latencies.append(time.time() - start_time)
                keys.extend(uids)

            logging.info(
                f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds"
            )
            logging.info(
                f"Total Inference Time on {device}: {time.time() - t0} seconds"
            )

        save_to_json(keys, os.path.join(out_root, "keys.json"))


def main(datasetinfos, out_roots, is_test=False, device_name="cuda"):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Processing on {device}")
    convert_to_mds(datasetinfos, out_roots, device, is_test=is_test)
    logging.info("Finished processing images.")


def detect_small_or_nonexistent_dirs(current_dir, start=0, end=14000, max_size=1024):
    small_or_nonexistent_dirs = []

    for i in range(start, end + 1):
        dir_name = f"{i:05d}"
        dir_path = os.path.join(current_dir, dir_name)

        if not os.path.exists(dir_path):
            small_or_nonexistent_dirs.append(i)
        elif os.path.isdir(dir_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)

            if total_size < max_size:
                small_or_nonexistent_dirs.append(i)

    return small_or_nonexistent_dirs


def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to MDS format.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for processing (cuda or cpu).",
    )
    parser.add_argument(
        "--file_index", type=int, default=1, help="File index to process."
    )
    parser.add_argument(
        "--is_test", action="store_true", help="Run in test mode with reduced dataset."
    )

    parser.add_argument(
        "--outdir_basepath",
        type=str,
        default="/jfs/mds_original",
        help="Output directory path.",
    )
    parser.add_argument(
        "--tar_indir_basepath",
        type=str,
        default="/home/ubuntu/data",
        help="Input directory path.",
    )

    args = parser.parse_args()

    # reqsids = json.load(open("{outdir_basepath}/small_or_nonexistent_dirs.json"))
    reqsids = range(18552)

    out_roots, datasetinfos = [], []
    for i, reqid in enumerate(reqsids):
        if i % 8 == args.file_index:
            out_root = f"{args.outdir_basepath}/{str(int(reqid)).zfill(5)}"
            dataset_path = f"{args.tar_indir_basepath}/{str(int(reqid)).zfill(5)}.tar"
            out_roots.append(out_root)
            datasetinfos.append(dataset_path)

    main(datasetinfos, out_roots, is_test=args.is_test, device_name=args.device)
