import os, json, pickle
import numpy as np
import pandas as pd
import tifffile as tiff
import cv2
from pycocotools import mask as mask_utils

repo = '/home/wangchengjun/huangziyi/reseg/resegearth+source'
pred_dir = '/home/wangchengjun/huangziyi/reseg/output/base/standard-base-siglip1-28w/test_results'
base_data = '/home/wangchengjun/huangziyi/data/RRSISD'
refs_path = os.path.join(base_data, 'rrsisd', 'refs(unc).p')
inst_path = os.path.join(base_data, 'rrsisd', 'instances.json')
out_dir = os.path.join(repo, 'outputs/source/baseline_error_anatomy')
doc_path = os.path.join(repo, 'docs/BASELINE_ERROR_ANATOMY.md')
os.makedirs(out_dir, exist_ok=True)

with open(refs_path, 'rb') as f:
    refs = pickle.load(f)
with open(inst_path, 'r', encoding='utf-8') as f:
    inst = json.load(f)

ann_dict = {a['id']: a for a in inst['annotations']}
cat_dict = {c['id']: c['name'] for c in inst.get('categories', [])}
test_refs = [r for r in refs if r.get('split') == 'test']


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


def load_pred_mask(pred_path):
    arr = tiff.imread(pred_path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def resize_pred_to_gt(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_mask = (pred_mask > 0).astype(np.uint8)
    return pred_mask


def get_pred_path(image_name, sample_id, idx):
    image_stem = os.path.splitext(image_name)[0]
    cands = [
        os.path.join(pred_dir, f'{image_name}_{sample_id}_test_data_0.tif'),
        os.path.join(pred_dir, f'{image_name}_{sample_id}_test_0.tif'),
        os.path.join(pred_dir, f'{image_stem}_{sample_id}_test_data_0.tif'),
        os.path.join(pred_dir, f'{image_stem}_{sample_id}_test_0.tif'),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    tif_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.tif')])
    p1 = f'{image_name}_{sample_id}_'
    p2 = f'{image_stem}_{sample_id}_'
    for f in tif_files:
        if f.startswith(p1) or f.startswith(p2):
            return os.path.join(pred_dir, f)
    if idx < len(tif_files):
        return os.path.join(pred_dir, tif_files[idx])
    return None


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1 + 1) * max(0.0, y2 - y1 + 1)


rows = []
skipped_empty_gt = 0
missing_pred = 0

for idx, ref in enumerate(test_refs):
    ann = ann_dict[ref['ann_id']]
    gt = merge_masks(ann['segmentation'])
    if gt.sum() == 0:
        skipped_empty_gt += 1
        continue
    pred_path = get_pred_path(ref['file_name'], ref['ref_id'], idx)
    if pred_path is None or not os.path.exists(pred_path):
        missing_pred += 1
        continue

    pred = resize_pred_to_gt(load_pred_mask(pred_path), gt)

    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    gt_area = int(gt.sum())
    pred_area = int(pred.sum())

    iou = inter / (union + 1e-7)
    dice = (2.0 * inter) / (gt_area + pred_area + 1e-7)
    precision = inter / (pred_area + 1e-7)
    recall = inter / (gt_area + 1e-7)
    fp_area = pred_area - inter
    fn_area = gt_area - inter

    h, w = gt.shape
    img_area = float(h * w)
    gt_area_ratio = gt_area / img_area
    pred_area_ratio = pred_area / img_area
    area_ratio_pred_over_gt = pred_area / (gt_area + 1e-7)

    gt_box = mask_to_bbox(gt)
    pred_box = mask_to_bbox(pred)
    bbox_gt_area_ratio = (bbox_area(gt_box) / img_area) if gt_box is not None else 0.0
    bbox_pred_area_ratio = (bbox_area(pred_box) / img_area) if pred_box is not None else 0.0

    if pred_area <= max(1, int(0.001 * img_area)):
        error_type = 'empty_or_near_empty_pred'
    elif (iou < 0.5) and (area_ratio_pred_over_gt > 3.0) and (precision < 0.3):
        error_type = 'huge_overflow'
    elif iou >= 0.7:
        error_type = 'good'
    elif iou >= 0.5:
        error_type = 'mild_error'
    elif (recall >= 0.6) and (precision < 0.5):
        error_type = 'over_segmentation'
    elif (precision >= 0.6) and (recall < 0.5):
        error_type = 'under_segmentation'
    elif (precision < 0.5) and (recall < 0.5):
        error_type = 'localization_or_confusion'
    elif fp_area >= fn_area:
        error_type = 'over_segmentation'
    else:
        error_type = 'under_segmentation'

    if iou >= 0.7:
        quality_bucket = 'high_iou'
    elif iou >= 0.5:
        quality_bucket = 'mid_iou'
    else:
        quality_bucket = 'low_iou'

    cat_id = ann.get('categories_id', ann.get('category_id'))
    cat_name = cat_dict.get(cat_id, str(cat_id))
    sent = ''
    if ref.get('sentences'):
        s0 = ref['sentences'][0]
        sent = s0.get('sent', s0.get('raw', ''))

    rows.append({
        'idx': idx,
        'ref_id': ref['ref_id'],
        'image_id': ref.get('image_id', ann.get('image_id')),
        'ann_id': ref['ann_id'],
        'category_id': cat_id,
        'category_name': cat_name,
        'expression': sent,
        'gt_area': gt_area,
        'pred_area': pred_area,
        'intersection': inter,
        'union': union,
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'fp_area': fp_area,
        'fn_area': fn_area,
        'pred_area_ratio': pred_area_ratio,
        'gt_area_ratio': gt_area_ratio,
        'area_ratio_pred_over_gt': area_ratio_pred_over_gt,
        'bbox_gt_area_ratio': bbox_gt_area_ratio,
        'bbox_pred_area_ratio': bbox_pred_area_ratio,
        'error_type': error_type,
        'scale_bucket': '',
        'quality_bucket': quality_bucket,
    })

if len(rows) == 0:
    raise RuntimeError('no valid samples')

df = pd.DataFrame(rows)

def fixed_bucket(x):
    if x < 0.001:
        return 'tiny'
    if x < 0.005:
        return 'small'
    if x < 0.02:
        return 'medium'
    if x < 0.10:
        return 'large'
    return 'scene'

q20, q40, q60, q80 = df['gt_area_ratio'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()

def q_bucket(x):
    if x <= q20:
        return 'tiny'
    if x <= q40:
        return 'small'
    if x <= q60:
        return 'medium'
    if x <= q80:
        return 'large'
    return 'scene'

df['scale_bucket_fixed'] = df['gt_area_ratio'].map(fixed_bucket)
df['scale_bucket_quantile'] = df['gt_area_ratio'].map(q_bucket)
df['scale_bucket'] = df['scale_bucket_fixed']

sample_count = int(len(df))
mean_iou = float(df['iou'].mean())
median_iou = float(df['iou'].median())
mean_precision = float(df['precision'].mean())
mean_recall = float(df['recall'].mean())
mean_pred_area = float(df['pred_area'].mean())
mean_gt_area = float(df['gt_area'].mean())
mean_area_ratio_pred_over_gt = float(df['area_ratio_pred_over_gt'].mean())

err_count = df['error_type'].value_counts()
err_pct = df['error_type'].value_counts(normalize=True) * 100.0

low = df[df['iou'] < 0.5]
low_ratios = {
    'over_segmentation': float((low['error_type'] == 'over_segmentation').mean() * 100.0) if len(low) else 0.0,
    'under_segmentation': float((low['error_type'] == 'under_segmentation').mean() * 100.0) if len(low) else 0.0,
    'localization_or_confusion': float((low['error_type'] == 'localization_or_confusion').mean() * 100.0) if len(low) else 0.0,
}

cat_summary = df.groupby('category_name', as_index=False).agg(
    sample_count=('iou', 'size'),
    mean_iou=('iou', 'mean'),
    median_iou=('iou', 'median'),
    mean_precision=('precision', 'mean'),
    mean_recall=('recall', 'mean'),
    mean_pred_area=('pred_area', 'mean'),
    mean_gt_area=('gt_area', 'mean'),
    mean_area_ratio_pred_over_gt=('area_ratio_pred_over_gt', 'mean'),
    over_segmentation_rate=('error_type', lambda s: np.mean(s == 'over_segmentation') * 100.0),
    under_segmentation_rate=('error_type', lambda s: np.mean(s == 'under_segmentation') * 100.0),
    localization_or_confusion_rate=('error_type', lambda s: np.mean(s == 'localization_or_confusion') * 100.0),
    empty_pred_rate=('error_type', lambda s: np.mean(s == 'empty_or_near_empty_pred') * 100.0),
    huge_overflow_rate=('error_type', lambda s: np.mean(s == 'huge_overflow') * 100.0),
    tiny_sample_rate=('scale_bucket_fixed', lambda s: np.mean(s == 'tiny') * 100.0),
    small_sample_rate=('scale_bucket_fixed', lambda s: np.mean(s == 'small') * 100.0),
    large_sample_rate=('scale_bucket_fixed', lambda s: np.mean(s == 'large') * 100.0),
    scene_sample_rate=('scale_bucket_fixed', lambda s: np.mean(s == 'scene') * 100.0),
)

scale_summary = df.groupby('scale_bucket_fixed', as_index=False).agg(
    sample_count=('iou', 'size'),
    mean_iou=('iou', 'mean'),
    mean_precision=('precision', 'mean'),
    mean_recall=('recall', 'mean'),
    over_segmentation_rate=('error_type', lambda s: np.mean(s == 'over_segmentation') * 100.0),
    under_segmentation_rate=('error_type', lambda s: np.mean(s == 'under_segmentation') * 100.0),
    localization_or_confusion_rate=('error_type', lambda s: np.mean(s == 'localization_or_confusion') * 100.0),
    mean_area_ratio_pred_over_gt=('area_ratio_pred_over_gt', 'mean'),
).rename(columns={'scale_bucket_fixed': 'scale_bucket'})

corrs = {
    'gt_area_ratio_vs_iou_pearson': float(df['gt_area_ratio'].corr(df['iou'], method='pearson')),
    'pred_area_ratio_vs_iou_pearson': float(df['pred_area_ratio'].corr(df['iou'], method='pearson')),
    'area_ratio_pred_over_gt_vs_iou_pearson': float(df['area_ratio_pred_over_gt'].corr(df['iou'], method='pearson')),
    'gt_area_ratio_vs_iou_spearman': float(df['gt_area_ratio'].corr(df['iou'], method='spearman')),
    'pred_area_ratio_vs_iou_spearman': float(df['pred_area_ratio'].corr(df['iou'], method='spearman')),
    'area_ratio_pred_over_gt_vs_iou_spearman': float(df['area_ratio_pred_over_gt'].corr(df['iou'], method='spearman')),
}

low_loc = low_ratios['localization_or_confusion']
low_over = low_ratios['over_segmentation']
low_under = low_ratios['under_segmentation']
abs_corr_area = abs(corrs['area_ratio_pred_over_gt_vs_iou_spearman'])

if (low_over + low_under >= 55.0) and (abs_corr_area >= 0.25) and (low_loc < 45.0):
    verdict = 'A'
    verdict_label = 'scope_prior_promising'
elif (low_loc >= 50.0) and (low_over + low_under < 50.0):
    verdict = 'B'
    verdict_label = 'scope_prior_not_primary'
elif (low_loc >= 40.0) and (low_over + low_under >= 35.0):
    verdict = 'C'
    verdict_label = 'scope_prior_category_limited'
else:
    verdict = 'D'
    verdict_label = 'inconclusive'

focus_categories = ['airport','overpass','vehicle','ship','harbor','chimney','golffield','basketballcourt','bridge','trainstation','airplane','storagetank','windmill']
focus_df = cat_summary[cat_summary['category_name'].isin(focus_categories)].sort_values('mean_iou')
worst10 = cat_summary.sort_values('mean_iou').head(10)
over10 = cat_summary.sort_values('over_segmentation_rate', ascending=False).head(10)
under10 = cat_summary.sort_values('under_segmentation_rate', ascending=False).head(10)
loc10 = cat_summary.sort_values('localization_or_confusion_rate', ascending=False).head(10)
ratio_hi10 = cat_summary.sort_values('mean_area_ratio_pred_over_gt', ascending=False).head(10)
ratio_lo10 = cat_summary.sort_values('mean_area_ratio_pred_over_gt', ascending=True).head(10)

summary = {
    'pred_dir': pred_dir,
    'sample_count': sample_count,
    'skipped_empty_gt': int(skipped_empty_gt),
    'missing_pred': int(missing_pred),
    'mean_iou': mean_iou,
    'median_iou': median_iou,
    'mean_precision': mean_precision,
    'mean_recall': mean_recall,
    'mean_pred_area': mean_pred_area,
    'mean_gt_area': mean_gt_area,
    'mean_area_ratio_pred_over_gt': mean_area_ratio_pred_over_gt,
    'error_type_count': {k: int(v) for k, v in err_count.to_dict().items()},
    'error_type_percent': {k: float(v) for k, v in err_pct.to_dict().items()},
    'low_iou_sample_count': int(len(low)),
    'low_iou_error_ratio_percent': low_ratios,
    'correlations': corrs,
    'scale_bucket_fixed_thresholds': {'tiny': '<0.001', 'small': '[0.001,0.005)', 'medium': '[0.005,0.02)', 'large': '[0.02,0.10)', 'scene': '>=0.10'},
    'scale_bucket_quantile_edges': {'q20': float(q20), 'q40': float(q40), 'q60': float(q60), 'q80': float(q80)},
    'rankings': {
        'worst_mean_iou_top10': worst10.to_dict(orient='records'),
        'over_segmentation_rate_top10': over10.to_dict(orient='records'),
        'under_segmentation_rate_top10': under10.to_dict(orient='records'),
        'localization_or_confusion_rate_top10': loc10.to_dict(orient='records'),
        'area_ratio_pred_over_gt_high_top10': ratio_hi10.to_dict(orient='records'),
        'area_ratio_pred_over_gt_low_top10': ratio_lo10.to_dict(orient='records'),
        'focus_categories': focus_df.to_dict(orient='records'),
    },
    'verdict': verdict,
    'verdict_label': verdict_label,
}

os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, 'baseline_error_per_sample.csv'), index=False, encoding='utf-8')
cat_summary.sort_values('mean_iou', ascending=True).to_csv(os.path.join(out_dir, 'baseline_error_category_summary.csv'), index=False, encoding='utf-8')
pd.DataFrame([{'error_type': k, 'count': int(v), 'percent': float(err_pct.get(k, 0.0))} for k, v in err_count.to_dict().items()]).sort_values('count', ascending=False).to_csv(os.path.join(out_dir, 'baseline_error_type_summary.csv'), index=False, encoding='utf-8')
scale_summary.to_csv(os.path.join(out_dir, 'scale_error_summary.csv'), index=False, encoding='utf-8')
with open(os.path.join(out_dir, 'baseline_error_anatomy_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

md = []
md.append('# Baseline Error Anatomy')
md.append('')
md.append('## 0. 一页结论')
md.append(f"- baseline 主要错误：低 IoU 样本以 {max(low_ratios, key=low_ratios.get)} 为主。")
md.append(f"- 低 IoU 三类占比：over={low_over:.2f}%, under={low_under:.2f}%, localization/confusion={low_loc:.2f}%。")
md.append(f"- 面积/尺度相关：Spearman corr(gt_area_ratio, IoU)={corrs['gt_area_ratio_vs_iou_spearman']:.4f}; corr(area_ratio_pred_over_gt, IoU)={corrs['area_ratio_pred_over_gt_vs_iou_spearman']:.4f}。")
md.append(f"- Mask Scope Prior 判断：{verdict} ({verdict_label})。")
md.append('')
md.append('## 1. 数据来源与样本数')
md.append(f"- Pred: `{pred_dir}`")
md.append(f"- GT: `{inst_path}` + `{refs_path}`")
md.append(f"- sample_count={sample_count}, skipped_empty_gt={skipped_empty_gt}, missing_pred={missing_pred}")
md.append('')
md.append('## 2. 整体错误类型分布')
for k, v in err_count.to_dict().items():
    md.append(f"- {k}: {int(v)} ({float(err_pct.get(k, 0.0)):.2f}%)")
md.append('')
md.append('## 3. Per-category 错误排名')
md.append('- 详见 outputs/source/baseline_error_anatomy/baseline_error_category_summary.csv 与 summary.json 排名字段。')
md.append('')
md.append('## 4. Precision vs Recall 分析')
md.append(f"- mean_precision={mean_precision:.4f}, mean_recall={mean_recall:.4f}")
md.append(f"- low-IoU: over={low_over:.2f}% under={low_under:.2f}% loc/conf={low_loc:.2f}%")
md.append('')
md.append('## 5. Scale / Area 与 IoU 关系')
for k, v in corrs.items():
    md.append(f"- {k}: {v:.6f}")
md.append('- 见 scale_error_summary.csv。')
md.append('')
md.append('## 6. 重点类别分析')
for _, r in focus_df.iterrows():
    md.append(f"- {r['category_name']}: n={int(r['sample_count'])}, mIoU={r['mean_iou']:.4f}, over={r['over_segmentation_rate']:.2f}%, under={r['under_segmentation_rate']:.2f}%, loc/conf={r['localization_or_confusion_rate']:.2f}%")
md.append('')
md.append('## 7. 对 Mask Scope Prior 的判断')
md.append(f"- verdict: {verdict} ({verdict_label})")
md.append('- evidence_for / evidence_against 见 summary.json 与本文结论段。')
md.append('')
md.append('## 8. 下一步建议')
md.append('- 本轮仅完成方向诊断，不引入新方法，不改模型/训练/语义库/eval。')
with open(doc_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')

print(json.dumps({
    'sample_count': sample_count,
    'skipped_empty_gt': int(skipped_empty_gt),
    'missing_pred': int(missing_pred),
    'mean_iou': mean_iou,
    'mean_precision': mean_precision,
    'mean_recall': mean_recall,
    'low_iou_count': int(len(low)),
    'low_iou_ratios': low_ratios,
    'verdict': verdict,
    'verdict_label': verdict_label,
    'doc_path': doc_path,
    'out_dir': out_dir,
}, ensure_ascii=False, indent=2))
