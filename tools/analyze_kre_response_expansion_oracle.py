#!/usr/bin/env python3
import os, json
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from pycocotools import mask as mask_utils
from detectron2.structures import ImageList
from transformers import SiglipImageProcessor

REPO_ROOT = '/home/wangchengjun/huangziyi/reseg/resegearth+source'
import sys
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, RRSISDDataset
from segearth_r2.eval.eval import DataArguments as EvalDataArguments
from segearth_r2.utils.builder import load_pretrained_model
from segearth_r2.utils import conversation as conversation_lib

IN_SEL = os.path.join(REPO_ROOT, 'outputs/source/empty_mask_query_anatomy/selected_vehicle_samples.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'outputs/source/kre_response_expansion_oracle')
DOC = os.path.join(REPO_ROOT, 'docs/KRE_RESPONSE_EXPANSION_ORACLE_TEST.md')
os.makedirs(OUT_DIR, exist_ok=True)

sel = pd.read_csv(IN_SEL)
sel = sel[sel['group'].isin(['empty_vehicle','good_vehicle'])].copy()

model_candidates = [
    '/home/wangchengjun/huangziyi/reseg/output/source/rrsisd_baseline_raw_7w/merged_model',
    '/home/wangchengjun/huangziyi/reseg/output/base/standard-base-siglip1-28w/merged_model',
]
MODEL_PATH = next((p for p in model_candidates if os.path.isdir(p) and os.path.isfile(os.path.join(p,'config.json'))), None)
if MODEL_PATH is None:
    raise RuntimeError('baseline merged model not found')

BASE_DATA_PATH = '/home/wangchengjun/huangziyi/data/RRSISD'

# setup model/dataset
data_args = replace(
    EvalDataArguments(),
    base_data_path=BASE_DATA_PATH,
    dataset_name='rrsisd',
    split='test',
    vision_tower='/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384',
    vision_tower_mask='/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl',
    mask_config=os.path.join(REPO_ROOT, 'segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml'),
)
setattr(data_args, 'concept_public_semantic_library', None)
setattr(data_args, 'concept_refaware_prior', False)
setattr(data_args, 'concept_match_strict', False)
setattr(data_args, 'debug_concept_match_strict', False)
setattr(data_args, 'debug_concept_refaware_prior', False)
conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

tokenizer, model, _, _ = load_pretrained_model(MODEL_PATH, model_args=data_args, mask_config=data_args.mask_config, device='cpu')
model = model.cpu().float().eval()
clip_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_processor)
ds = RRSISDDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=data_args.split)

refid_to_idx = {int(r.get('ref_id', -1)): i for i, r in enumerate(ds.reason_file)}
ann_dict = ds.ann_dict


def decode_rle_mask(mask_item):
    rle = {'size': mask_item['size'], 'counts': mask_item['counts']}
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)

def merge_masks(mask_list):
    decoded = [decode_rle_mask(m) for m in mask_list]
    merged = np.zeros_like(decoded[0], dtype=np.uint8)
    for m in decoded:
        merged = np.logical_or(merged, m).astype(np.uint8)
    return merged

def mask_metrics(pred_bool, gt_bool):
    inter = int(np.logical_and(pred_bool, gt_bool).sum())
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    iou = inter / (union + 1e-7)
    parea = int(pred_bool.sum())
    garea = int(gt_bool.sum())
    prec = inter / (parea + 1e-7)
    rec = inter / (garea + 1e-7)
    return inter, union, iou, prec, rec, parea, garea

def gt_bbox(gt):
    ys, xs = np.where(gt)
    if len(xs)==0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

@torch.no_grad()
def forward_logits(batch):
    device = torch.device('cpu')
    mdtype = torch.float32
    b = {}
    for k,v in batch.items():
        if torch.is_tensor(v):
            b[k]=v.to(device)
        else:
            b[k]=v

    image_features = model.get_vision_tower_feature(b['images'].to(device=device, dtype=mdtype))
    input_ids2, attention_mask2, pkv, inputs_embeds, labels2, seg_indices2, image_features_indices = model.prepare_inputs_labels_for_multimodal(
        b['input_ids'], b['attention_mask'], None, b['labels'], b['images_clip'].to(device=device, dtype=mdtype),
        token_refer_id=[x.to(device) for x in b['token_refer_id']],
        SEG_token_embedding_indices=b['SEG_token_embedding_indices'],
    )
    outputs = model.model(
        input_ids=input_ids2,
        attention_mask=attention_mask2,
        past_key_values=pkv,
        inputs_embeds=inputs_embeds,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    seg_query = model.SEG_token_projector(model.get_SEG_embedding(hidden_states, seg_indices2))
    mask_features, _, multi_scale_features = model.pixel_decoder.forward_features(image_features)

    images_rep = [image.repeat((num,1,1,1)) for image,num in zip(b['images'], b['mask_num'])]
    images_rep = [s[0] for image_repeat in images_rep for s in torch.split(image_repeat,1,dim=0)]
    mask_num_t = torch.tensor(b['mask_num'], device=mask_features.device)
    mask_features = torch.repeat_interleave(mask_features, repeats=mask_num_t, dim=0)
    multi_scale_features = [torch.repeat_interleave(feat, repeats=mask_num_t, dim=0) for feat in multi_scale_features]

    mask_outputs = model.predictor(multi_scale_features, mask_features, None, None, seg_query)
    mask_logits = mask_outputs['pred_masks']
    images_il = ImageList.from_tensors(images_rep, model.size_divisibility)
    mask_logits = F.interpolate(mask_logits, size=(images_il.tensor.shape[-2], images_il.tensor.shape[-1]), mode='bilinear', align_corners=False)
    logit = mask_logits[0,0].detach().float().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logit))
    return logit, prob

all_rows = []
best_rows = []

kernels = [3,5,7]
iters = [1,2,3]
topks = [100,300,500,1000]

for _, srow in sel.iterrows():
    ref_id = int(srow['ref_id'])
    idx = refid_to_idx.get(ref_id)
    if idx is None:
        continue
    item = ds[idx]
    batch = collator([item])
    logit, prob = forward_logits(batch)

    ann = ann_dict[int(srow['ann_id'])]
    gt = merge_masks(ann['segmentation']).astype(bool)
    h,w = gt.shape
    if logit.shape != gt.shape:
        logit = cv2.resize(logit, (w,h), interpolation=cv2.INTER_LINEAR)
        prob = 1.0 / (1.0 + np.exp(-logit))

    raw_mask = prob >= 0.5
    raw_inter, raw_union, raw_iou, raw_prec, raw_rec, raw_area, gt_area = mask_metrics(raw_mask, gt)

    candidates = []
    # threshold baselines
    for thr in [0.5,0.3,0.1]:
        cmask = prob >= thr
        candidates.append((f'threshold_{thr}', None, None, None, cmask))

    # top-k and dilations
    flat = logit.reshape(-1)
    order = np.argsort(flat)[::-1]
    gb = gt_bbox(gt)

    for k in topks:
        kk = min(k, flat.size)
        seed = np.zeros(flat.size, dtype=np.uint8)
        seed[order[:kk]] = 1
        seed = seed.reshape(gt.shape).astype(bool)
        candidates.append(('topk_seed', k, 0, 0, seed))

        for ker in kernels:
            kernel = np.ones((ker,ker), np.uint8)
            for it in iters:
                d = cv2.dilate(seed.astype(np.uint8), kernel, iterations=it).astype(bool)
                candidates.append(('topk_dilation', k, ker, it, d))

        # optional oracle variant: components intersect gt bbox
        if gb is not None:
            n, lbl = cv2.connectedComponents(seed.astype(np.uint8))
            keep = np.zeros_like(seed, dtype=bool)
            x0,y0,x1,y1 = gb
            for cid in range(1,n):
                comp = lbl==cid
                if comp[y0:y1+1, x0:x1+1].any():
                    keep |= comp
            candidates.append(('topk_seed_oracle_bbox_intersect', k, 0, 0, keep))

    sample_rows = []
    for ctype, k, ker, it, cmask in candidates:
        inter, union, iou, prec, rec, carea, garea = mask_metrics(cmask, gt)
        overflow = max(carea - inter, 0) / (carea + 1e-7)
        undercov = max(garea - inter, 0) / (garea + 1e-7)
        row = {
            'ref_id': ref_id,
            'idx': int(srow['idx']),
            'category_name': srow['category_name'],
            'group': srow['group'],
            'gt_area': int(garea),
            'raw_pred_area': int(raw_area),
            'raw_iou': float(raw_iou),
            'candidate_type': ctype,
            'top_k': -1 if k is None else int(k),
            'kernel': -1 if ker is None else int(ker),
            'iterations': -1 if it is None else int(it),
            'candidate_area': int(carea),
            'candidate_iou': float(iou),
            'delta_iou_vs_raw': float(iou - raw_iou),
            'precision': float(prec),
            'recall': float(rec),
            'candidate_intersection': int(inter),
            'candidate_union': int(union),
            'candidate_overflow_ratio': float(overflow),
            'candidate_undercoverage_ratio': float(undercov),
            'is_oracle_variant': bool('oracle' in ctype),
        }
        sample_rows.append(row)
        all_rows.append(row)

    best = max(sample_rows, key=lambda r: r['candidate_iou'])
    best_rows.append(best)

cand_df = pd.DataFrame(all_rows)
best_df = pd.DataFrame(best_rows)

cand_df.to_csv(os.path.join(OUT_DIR, 'kre_oracle_per_candidate.csv'), index=False, encoding='utf-8')
best_df.to_csv(os.path.join(OUT_DIR, 'kre_oracle_per_sample_best.csv'), index=False, encoding='utf-8')

# group summary
summary_rows = []
for g, gdf in cand_df.groupby('group'):
    raw = gdf[['ref_id','raw_iou']].drop_duplicates()['raw_iou']
    best = best_df[best_df['group']==g]
    summary_rows.append({
        'group': g,
        'sample_count': int(best.shape[0]),
        'raw_mean_iou': float(raw.mean()),
        'best_mean_iou': float(best['candidate_iou'].mean()),
        'mean_delta_iou': float((best['candidate_iou'] - best['raw_iou']).mean()),
        'improved_count': int((best['delta_iou_vs_raw'] > 1e-6).sum()),
        'degraded_count': int((best['delta_iou_vs_raw'] < -1e-6).sum()),
        'best_mean_precision': float(best['precision'].mean()),
        'best_mean_recall': float(best['recall'].mean()),
    })

gsum = pd.DataFrame(summary_rows)
gsum.to_csv(os.path.join(OUT_DIR, 'kre_oracle_group_summary.csv'), index=False, encoding='utf-8')

empty_best = best_df[best_df['group']=='empty_vehicle']
good_best = best_df[best_df['group']=='good_vehicle']
empty_delta = float((empty_best['candidate_iou'] - empty_best['raw_iou']).mean()) if len(empty_best) else float('nan')
improved = int((empty_best['delta_iou_vs_raw'] > 1e-6).sum()) if len(empty_best) else 0
degraded = int((empty_best['delta_iou_vs_raw'] < -1e-6).sum()) if len(empty_best) else 0
oracle_share = float((empty_best['is_oracle_variant']).mean()) if len(empty_best) else 0.0

good_raw_mean = float(good_best['raw_iou'].mean()) if len(good_best) else float('nan')
good_best_mean = float(good_best['candidate_iou'].mean()) if len(good_best) else float('nan')

good_hurt = (good_best_mean + 1e-9) < (good_raw_mean - 0.03)

# verdict
if len(empty_best)==0 or len(good_best)==0:
    verdict = 'D'; label='inconclusive'
elif empty_delta > 0.05 and improved >= int(0.6*len(empty_best)) and (not good_hurt) and oracle_share < 0.5:
    verdict = 'A'; label='response_expansion_promising'
elif empty_delta > 0.02 and oracle_share >= 0.5:
    verdict = 'B'; label='response_expansion_oracle_only'
elif empty_delta <= 0.01:
    verdict = 'C'; label='response_expansion_not_promising'
else:
    verdict = 'D'; label='inconclusive'

best_pattern = best_df.groupby(['candidate_type','top_k','kernel','iterations']).size().sort_values(ascending=False).head(10)

summary = {
    'success': True,
    'baseline_model_path': MODEL_PATH,
    'empty_vehicle_count': int(len(empty_best)),
    'good_vehicle_count': int(len(good_best)),
    'empty_raw_mean_iou': float(empty_best['raw_iou'].mean()) if len(empty_best) else None,
    'empty_best_mean_iou': float(empty_best['candidate_iou'].mean()) if len(empty_best) else None,
    'empty_mean_delta_iou': empty_delta,
    'empty_improved_count': improved,
    'empty_degraded_count': degraded,
    'good_raw_mean_iou': good_raw_mean,
    'good_best_mean_iou': good_best_mean,
    'good_hurt': bool(good_hurt),
    'oracle_best_share_empty': oracle_share,
    'best_pattern_top10': [
        {'candidate_type':k[0], 'top_k':int(k[1]), 'kernel':int(k[2]), 'iterations':int(k[3]), 'count':int(v)}
        for k,v in best_pattern.items()
    ],
    'verdict': verdict,
    'verdict_label': label,
}
with open(os.path.join(OUT_DIR, 'kre_oracle_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

md = []
md.append('# KRE Response Expansion Oracle Test')
md.append('')
md.append(f"- verdict: {verdict} ({label})")
md.append(f"- empty raw mean IoU: {summary['empty_raw_mean_iou']:.4f}")
md.append(f"- empty best mean IoU: {summary['empty_best_mean_iou']:.4f}")
md.append(f"- empty mean delta IoU: {summary['empty_mean_delta_iou']:.4f}")
md.append(f"- empty improved/degraded: {improved}/{degraded}")
md.append(f"- good raw/best mean IoU: {good_raw_mean:.4f}/{good_best_mean:.4f}")
md.append(f"- oracle best share (empty): {oracle_share:.2%}")
md.append('')
md.append('## Outputs')
md.append(f"- `{os.path.join(OUT_DIR, 'kre_oracle_per_candidate.csv')}`")
md.append(f"- `{os.path.join(OUT_DIR, 'kre_oracle_per_sample_best.csv')}`")
md.append(f"- `{os.path.join(OUT_DIR, 'kre_oracle_group_summary.csv')}`")
md.append(f"- `{os.path.join(OUT_DIR, 'kre_oracle_summary.json')}`")

with open(DOC, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')

print(json.dumps(summary, ensure_ascii=False, indent=2))
