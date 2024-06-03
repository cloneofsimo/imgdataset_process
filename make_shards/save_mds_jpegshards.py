import os
import json
from PIL import Image
import logging
from torch.utils.data import DataLoader
import webdataset as wds
import argparse
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import time
import torch
# Initialize logging
logging.basicConfig(level=logging.INFO)

def crop_to_center(image, new_size=768):
    width, height = image.size
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def wds_preprocess(x):
    key, pil_image, _json = x
    pil_image = pil_image.convert("RGB")
    
    # Resize and crop the image
    w, h = pil_image.size
    if w > h:
        pil_image = pil_image.resize((256, int(h * 256 / w)))
    else:
        pil_image = pil_image.resize((int(w * 256 / h), 256))

    pil_image = crop_to_center(pil_image, new_size=256)
    
    return key, pil_image

def main(dataset_paths, out_root, is_test=False):
    logging.info(f"Processing dataset")

    dataset = wds.DataPipeline(
        wds.SimpleShardList(dataset_paths),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.decode("pil", handler=wds.warn_and_continue),
        wds.to_tuple("__key__", "jpg", "json", handler=wds.warn_and_continue),
        wds.map(wds_preprocess),
        wds.batched(2048),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=32,
        prefetch_factor=8,
        shuffle=False,
        drop_last=False,
    )

    t0 = time.time()
    os.makedirs(out_root, exist_ok=True)

    keys = []

    for idx, batch in tqdm(enumerate(dataloader), smoothing=0.0):
        if is_test and idx > 0:
            break
        
        print(len(batch))
        keys_batch, images_batch = batch

        # for key, image in zip(keys_batch, images_batch):
        #     image_save_path = os.path.join(out_root, f"{key}.png")
        #     image.save(image_save_path, format="PNG")
        #     keys.append(key)

    logging.info(
        f"Total Processing Time: {time.time() - t0} seconds"
    )
    save_to_json(keys, os.path.join(out_root, "keys.json"))
    logging.info("Finished processing images.")

def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from webdataset.")
    
    parser.add_argument(
        "--is_test", action="store_true", help="Run in test mode with reduced dataset."
    )
    parser.add_argument(
        "--outdir_basepath",
        type=str,
        default="/jfs/imgshards",
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
    reqsids = range(64)

    out_roots, datasetinfos = [], []
    for i, reqid in enumerate(reqsids):
      
        out_root = f"{args.outdir_basepath}/{str(int(reqid)).zfill(5)}"
        dataset_path = f"{args.tar_indir_basepath}/{str(int(reqid)).zfill(5)}.tar"
        out_roots.append(out_root)
        datasetinfos.append(dataset_path)

    main(datasetinfos, out_root, is_test=args.is_test)
