import argparse
import os
import sys
import types
import torch
import transformers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from segearth_r2.datasets.dataset import RRSISDDataset, DataCollatorForCOCODatasetV2, get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from transformers import SiglipImageProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-data-path', required=True)
    p.add_argument('--model-name-or-path', required=True)
    p.add_argument('--vision-tower', required=True)
    p.add_argument('--mask-config', default='segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml')
    p.add_argument('--split', default='val')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-samples', type=int, default=3)
    args = p.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=2048, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    mask_cfg = get_mask_config(args.mask_config)
    model = SegEarthR2.from_pretrained(args.model_name_or_path, mask_decoder_cfg=mask_cfg, add_cross_attn=True)
    if not model.is_train_mask_decode:
        model.initial_mask_module(None, None)
    model.runtime_tokenizer = tokenizer
    tokenizer.add_tokens('[SEG]')
    model.resize_token_embeddings(len(tokenizer))
    model.get_special_token(SEG=tokenizer('[SEG]', return_tensors='pt', add_special_tokens=False)['input_ids'], EOS=tokenizer.eos_token_id)
    model.eval().to(args.device)

    data_args = types.SimpleNamespace(base_data_path=args.base_data_path)
    ds = RRSISDDataset(base_data_path=args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=args.split)
    proc = SiglipImageProcessor.from_pretrained(args.vision_tower)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=proc)

    samples = [ds[i] for i in range(min(args.num_samples, len(ds)))]
    batch = collator(samples)
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(args.device)
    if 'token_refer_id' in batch:
        batch['token_refer_id'] = [x.to(args.device) for x in batch['token_refer_id']]

    with torch.no_grad():
        _, _, _, inputs_embeds, _, seg_idx, img_idx = model.prepare_inputs_labels_for_multimodal(
            batch['input_ids'], batch['attention_mask'], None, batch['labels'], batch['images_clip'],
            token_refer_id=batch['token_refer_id'], SEG_token_embedding_indices=batch['SEG_token_embedding_indices']
        )

    print('=== SEG input alignment check ===')
    for i in range(min(args.num_samples, inputs_embeds.shape[0])):
        seg_pos = torch.where(seg_idx[i].bool())[0].tolist()
        img_pos = torch.where(img_idx[i].bool())[0].tolist()
        print(f'sample={i} seq_len={inputs_embeds.shape[1]} seg_positions={seg_pos} seg_count={len(seg_pos)} image_token_count={len(img_pos)}')
        if len(seg_pos) == 0:
            raise RuntimeError(f'sample {i} has no [SEG] position in inputs_embeds')
        for s in seg_pos:
            if s < 0 or s >= inputs_embeds.shape[1]:
                raise RuntimeError(f'sample {i} invalid seg position {s}')
    print('PASS: SEG input embedding positions map correctly to inputs_embeds.')


if __name__ == '__main__':
    main()
