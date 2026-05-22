#!/usr/bin/env python3
import os, json, math, pickle
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

BASE_DATA_PATH = '/home/wangchengjun/huangziyi/data/RRSISD'
SUMMARY_JSON = os.path.join(REPO_ROOT, 'outputs/source/baseline_error_anatomy/baseline_error_anatomy_summary.json')
PER_SAMPLE_CSV = os.path.join(REPO_ROOT, 'outputs/source/baseline_error_anatomy/baseline_error_per_sample.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'outputs/source/empty_mask_query_anatomy')
DOC_PATH = os.path.join(REPO_ROOT, 'docs/EMPTY_MASK_QUERY_ANATOMY.md')

os.makedirs(OUT_DIR, exist_ok=True)

with open(SUMMARY_JSON, 'r', encoding='utf-8') as f:
    base_summary = json.load(f)
PRED_DIR = base_summary.get('pred_dir', '/home/wangchengjun/huangziyi/reseg/output/base/standard-base-siglip1-28w/test_results')

candidate_models = [
    '/home/wangchengjun/huangziyi/reseg/output/source/rrsisd_baseline_raw_7w/merged_model',
    '/home/wangchengjun/huangziyi/reseg/output/base/standard-base-siglip1-28w/merged_model',
]
MODEL_PATH = None
for p in candidate_models:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'config.json')):
        MODEL_PATH = p
        break
if MODEL_PATH is None:
    raise RuntimeError('No baseline merged model found')

# sample selection from prior anatomy
df = pd.read_csv(PER_SAMPLE_CSV)
veh = df[df['category_name'].astype(str).str.lower() == 'vehicle'].copy()
empty_pool = veh[(veh['error_type'] == 'empty_or_near_empty_pred') | (veh['pred_area_ratio'] <= 1e-3)].copy()
good_pool = veh[(veh['iou'] >= 0.7) & (veh['pred_area'] > 0)].copy()

if len(empty_pool) < 10 or len(good_pool) < 10:
    raise RuntimeError(f'insufficient vehicle samples: empty={len(empty_pool)}, good={len(good_pool)}')

empty_sel = empty_pool.sort_values(['gt_area_ratio', 'iou'], ascending=[True, True]).head(10).copy()
remaining_good = good_pool.copy()
matched_good_rows = []
for _, er in empty_sel.iterrows():
    if len(remaining_good) == 0:
        break
    d = (remaining_good['gt_area_ratio'] - er['gt_area_ratio']).abs()
    j = d.idxmin()
    matched_good_rows.append(remaining_good.loc[j])
    remaining_good = remaining_good.drop(index=j)

good_sel = pd.DataFrame(matched_good_rows)
if len(good_sel) < 10:
    add = good_pool.drop(index=good_sel.index, errors='ignore').head(10 - len(good_sel))
    good_sel = pd.concat([good_sel, add], ignore_index=False)

good_sel = good_sel.head(10).copy()

empty_sel['group'] = 'empty_vehicle'
good_sel['group'] = 'good_vehicle'
selected = pd.concat([empty_sel, good_sel], ignore_index=True)

selected_cols = [
    'group','idx','ref_id','image_id','ann_id','category_name','expression','gt_area','pred_area',
    'gt_area_ratio','pred_area_ratio','iou','precision','recall','error_type'
]
selected[selected_cols].to_csv(os.path.join(OUT_DIR, 'selected_vehicle_samples.csv'), index=False, encoding='utf-8')

# model / dataset setup
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
model = model.cpu().float()
clip_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_processor)
ds = RRSISDDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=data_args.split)

# maps
refid_to_idx = {}
for i, r in enumerate(ds.reason_file):
    refid_to_idx[int(r.get('ref_id', -1))] = i
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


def bbox_center(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def bbox_xyxy(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


@torch.no_grad()
def forward_with_logits(batch):
    device = torch.device('cpu')
    model.eval()
    model.to(device)
    mdtype = next(model.parameters()).dtype

    b = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            b[k] = v.to(device)
        else:
            b[k] = v

    image_features = model.get_vision_tower_feature(b['images'].to(device=device, dtype=mdtype))
    input_ids2, attention_mask2, pkv, inputs_embeds, labels2, seg_indices2, image_features_indices = model.prepare_inputs_labels_for_multimodal(
        b['input_ids'], b['attention_mask'], None, b['labels'], b['images_clip'].to(device=device, dtype=mdtype),
        token_refer_id=[x.to(device) for x in b['token_refer_id']],
        SEG_token_embedding_indices=b['SEG_token_embedding_indices'],
    )
    if device.type == 'cuda':
        inputs_embeds = inputs_embeds.to(dtype=mdtype)

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
    seg_hidden = model.get_SEG_embedding(hidden_states, seg_indices2)
    seg_query = model.SEG_token_projector(seg_hidden)

    mask_features, _, multi_scale_features = model.pixel_decoder.forward_features(image_features)
    images_rep = [image.repeat((num, 1, 1, 1)) for image, num in zip(b['images'], b['mask_num'])]
    images_rep = [s[0] for image_repeat in images_rep for s in torch.split(image_repeat, 1, dim=0)]
    mask_num_t = torch.tensor(b['mask_num'], device=mask_features.device)
    mask_features = torch.repeat_interleave(mask_features, repeats=mask_num_t, dim=0)
    multi_scale_features = [torch.repeat_interleave(feat, repeats=mask_num_t, dim=0) for feat in multi_scale_features]
    mask_outputs = model.predictor(multi_scale_features, mask_features, None, None, seg_query)
    mask_logits = mask_outputs['pred_masks']
    images_il = ImageList.from_tensors(images_rep, model.size_divisibility)
    mask_logits = F.interpolate(mask_logits, size=(images_il.tensor.shape[-2], images_il.tensor.shape[-1]), mode='bilinear', align_corners=False)
    logit = mask_logits[0, 0].detach().float().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logit))

    seg_hidden_np = seg_hidden.detach().float().cpu().numpy().reshape(-1)
    seg_query_np = seg_query.detach().float().cpu().numpy().reshape(-1)
    model.cpu()
    return seg_hidden_np, seg_query_np, logit, prob

rows = []
for _, r in selected.iterrows():
    ref_id = int(r['ref_id'])
    ds_idx = refid_to_idx.get(ref_id)
    if ds_idx is None:
        continue
    item = ds[ds_idx]
    batch = collator([item])
    seg_hidden, seg_query, logit_map, prob_map = forward_with_logits(batch)

    ann = ann_dict[int(r['ann_id'])]
    gt_mask = merge_masks(ann['segmentation']).astype(bool)
    h, w = gt_mask.shape
    if logit_map.shape != gt_mask.shape:
        logit_map = cv2.resize(logit_map, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_map = 1.0 / (1.0 + np.exp(-logit_map))

    inside = gt_mask
    outside = ~gt_mask

    top1_idx = int(np.argmax(logit_map))
    y1, x1 = np.unravel_index(top1_idx, logit_map.shape)
    top1_inside = bool(gt_mask[y1, x1])

    flat_idx_desc = np.argsort(logit_map.reshape(-1))[::-1]
    def topk_rate(k):
        k = min(k, flat_idx_desc.size)
        ids = flat_idx_desc[:k]
        return float(gt_mask.reshape(-1)[ids].mean()) if k > 0 else float('nan')

    gt_center = bbox_center(gt_mask.astype(np.uint8))
    if gt_center is None:
        dist = float('nan')
    else:
        dist = float(np.linalg.norm(np.array([x1, y1], dtype=np.float32) - gt_center))

    gt_bbox = bbox_xyxy(gt_mask.astype(np.uint8))
    p99 = np.percentile(logit_map, 99)
    cluster = logit_map >= p99
    cluster_overlap_bbox = False
    if gt_bbox is not None:
        x0,y0,x2,y2 = gt_bbox
        cluster_overlap_bbox = bool(cluster[y0:y2+1, x0:x2+1].any())

    rows.append({
        'group': r['group'], 'idx': int(r['idx']), 'ref_id': ref_id, 'image_id': int(r['image_id']), 'ann_id': int(r['ann_id']),
        'category_name': r['category_name'], 'expression': str(r['expression']),
        'gt_area': int(r['gt_area']), 'pred_area': int(r['pred_area']), 'gt_area_ratio': float(r['gt_area_ratio']), 'pred_area_ratio': float(r['pred_area_ratio']),
        'iou': float(r['iou']), 'precision': float(r['precision']), 'recall': float(r['recall']), 'error_type': str(r['error_type']),
        'llm_seg_hidden_shape': str(tuple(seg_hidden.shape)), 'llm_seg_hidden_norm': float(np.linalg.norm(seg_hidden)), 'llm_seg_hidden_mean': float(seg_hidden.mean()), 'llm_seg_hidden_std': float(seg_hidden.std()),
        'projected_query_shape': str(tuple(seg_query.shape)), 'projected_query_norm': float(np.linalg.norm(seg_query)), 'projected_query_mean': float(seg_query.mean()), 'projected_query_std': float(seg_query.std()),
        'logit_min': float(logit_map.min()), 'logit_max': float(logit_map.max()), 'logit_mean': float(logit_map.mean()), 'logit_std': float(logit_map.std()),
        'logit_p95': float(np.percentile(logit_map,95)), 'logit_p99': float(np.percentile(logit_map,99)),
        'prob_max': float(prob_map.max()), 'prob_mean': float(prob_map.mean()), 'prob_p95': float(np.percentile(prob_map,95)), 'prob_p99': float(np.percentile(prob_map,99)),
        'predicted_area_at_threshold_0.5': int((prob_map >= 0.5).sum()),
        'predicted_area_at_threshold_0.3': int((prob_map >= 0.3).sum()),
        'predicted_area_at_threshold_0.1': int((prob_map >= 0.1).sum()),
        'mean_logit_inside_gt': float(logit_map[inside].mean()), 'mean_logit_outside_gt': float(logit_map[outside].mean()),
        'max_logit_inside_gt': float(logit_map[inside].max()), 'max_logit_outside_gt': float(logit_map[outside].max()),
        'mean_prob_inside_gt': float(prob_map[inside].mean()), 'mean_prob_outside_gt': float(prob_map[outside].mean()),
        'inside_outside_logit_margin': float(logit_map[inside].mean() - logit_map[outside].mean()),
        'top1_logit_coord': f'({int(x1)},{int(y1)})', 'top1_inside_gt': top1_inside,
        'top100_logit_inside_gt_rate': topk_rate(100), 'top500_logit_inside_gt_rate': topk_rate(500), 'top1000_logit_inside_gt_rate': topk_rate(1000),
        'top1_to_gt_center_distance': dist, 'top_logit_cluster_overlaps_gt_bbox': cluster_overlap_bbox,
        'attention_unavailable': True,
    })

rdf = pd.DataFrame(rows)
rdf.to_csv(os.path.join(OUT_DIR, 'empty_mask_query_per_sample.csv'), index=False, encoding='utf-8')

num_cols = [
    'gt_area_ratio','iou','projected_query_norm','logit_max','prob_max','mean_logit_inside_gt','mean_logit_outside_gt','inside_outside_logit_margin',
    'predicted_area_at_threshold_0.5','predicted_area_at_threshold_0.3','predicted_area_at_threshold_0.1',
    'top100_logit_inside_gt_rate','top500_logit_inside_gt_rate','top1000_logit_inside_gt_rate'
]
group_summary = rdf.groupby('group')[num_cols].mean().reset_index()
group_summary['count'] = rdf.groupby('group').size().values
group_summary.to_csv(os.path.join(OUT_DIR, 'empty_mask_query_group_summary.csv'), index=False, encoding='utf-8')

# verdict rules
es = group_summary[group_summary['group']=='empty_vehicle'].iloc[0]
gs = group_summary[group_summary['group']=='good_vehicle'].iloc[0]
if (es['projected_query_norm'] < 0.8 * gs['projected_query_norm']) and (es['logit_max'] < 0.8 * gs['logit_max']):
    verdict = 'A'
    diagnosis = 'query_activation_weak'
elif (es['predicted_area_at_threshold_0.5'] < 1.0) and (es['predicted_area_at_threshold_0.3'] > es['predicted_area_at_threshold_0.5'] * 3.0):
    verdict = 'B'
    diagnosis = 'threshold_calibration_issue'
elif (es['logit_max'] >= 0.8 * gs['logit_max']) and (es['top100_logit_inside_gt_rate'] < 0.6 * gs['top100_logit_inside_gt_rate']):
    verdict = 'C'
    diagnosis = 'query_mislocalized'
elif (es['projected_query_norm'] >= 0.8 * gs['projected_query_norm']) and (es['mean_logit_inside_gt'] < gs['mean_logit_inside_gt'] * 0.7):
    verdict = 'D'
    diagnosis = 'visual_feature_or_decoder_failure'
else:
    verdict = 'E'
    diagnosis = 'inconclusive'

summary = {
    'success': True,
    'baseline_model_path': MODEL_PATH,
    'pred_dir': PRED_DIR,
    'selected_empty_vehicle_count': int((rdf['group']=='empty_vehicle').sum()),
    'selected_good_vehicle_count': int((rdf['group']=='good_vehicle').sum()),
    'gt_area_ratio_comparable': {
        'empty_mean': float(rdf[rdf.group=='empty_vehicle']['gt_area_ratio'].mean()),
        'good_mean': float(rdf[rdf.group=='good_vehicle']['gt_area_ratio'].mean()),
        'empty_median': float(rdf[rdf.group=='empty_vehicle']['gt_area_ratio'].median()),
        'good_median': float(rdf[rdf.group=='good_vehicle']['gt_area_ratio'].median()),
    },
    'group_means': group_summary.to_dict(orient='records'),
    'diagnosis_verdict': verdict,
    'diagnosis_label': diagnosis,
    'attention_unavailable': True,
}
with open(os.path.join(OUT_DIR, 'empty_mask_query_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

md = []
md.append('# Empty-Mask Query Anatomy')
md.append('')
md.append('## 0. 一页结论')
md.append(f"- verdict: {verdict} ({diagnosis})")
md.append(f"- empty/good 样本数: {summary['selected_empty_vehicle_count']} / {summary['selected_good_vehicle_count']}")
md.append(f"- projected_query_norm: empty={es['projected_query_norm']:.4f}, good={gs['projected_query_norm']:.4f}")
md.append(f"- logit_max: empty={es['logit_max']:.4f}, good={gs['logit_max']:.4f}")
md.append(f"- prob_max: empty={es['prob_max']:.4f}, good={gs['prob_max']:.4f}")
md.append(f"- inside_outside_logit_margin: empty={es['inside_outside_logit_margin']:.4f}, good={gs['inside_outside_logit_margin']:.4f}")
md.append(f"- top100 inside GT rate: empty={es['top100_logit_inside_gt_rate']:.4f}, good={gs['top100_logit_inside_gt_rate']:.4f}")
md.append('')
md.append('## 1. 数据与样本')
md.append(f"- baseline model: `{MODEL_PATH}`")
md.append(f"- pred_dir: `{PRED_DIR}`")
md.append('- selected samples 见 selected_vehicle_samples.csv')
md.append('')
md.append('## 2. 组间对比')
md.append('- 详见 empty_mask_query_group_summary.csv 与 per-sample.csv')
md.append('')
md.append('## 3. 注意')
md.append('- attention_unavailable=True（未改 decoder，使用 logits heatmap proxy）')
md.append('')
md.append('## 4. 下一步建议')
if verdict == 'A':
    md.append('- 病因偏 query activation 弱。')
elif verdict == 'B':
    md.append('- 病因偏阈值校准问题。')
elif verdict == 'C':
    md.append('- 病因偏 query 定位偏移。')
elif verdict == 'D':
    md.append('- 病因偏视觉特征/decoder。')
else:
    md.append('- 结论混杂，需更多受控样本。')

with open(DOC_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')

print(json.dumps(summary, ensure_ascii=False, indent=2))
