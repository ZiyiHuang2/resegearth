# Baseline Error Anatomy

## 0. 一页结论
- baseline 主要错误：低 IoU 样本以 localization_or_confusion 为主。
- 低 IoU 三类占比：over=14.72%, under=14.58%, localization/confusion=28.61%。
- 面积/尺度相关：Spearman corr(gt_area_ratio, IoU)=0.6197; corr(area_ratio_pred_over_gt, IoU)=0.0702。
- Mask Scope Prior 判断：D (inconclusive)。

## 1. 数据来源与样本数
- Pred: `/home/wangchengjun/huangziyi/reseg/output/base/standard-base-siglip1-28w/test_results`
- GT: `/home/wangchengjun/huangziyi/data/RRSISD/rrsisd/instances.json` + `/home/wangchengjun/huangziyi/data/RRSISD/rrsisd/refs(unc).p`
- sample_count=3480, skipped_empty_gt=1, missing_pred=0

## 2. 整体错误类型分布
- good: 2105 (60.49%)
- mild_error: 453 (13.02%)
- empty_or_near_empty_pred: 428 (12.30%)
- localization_or_confusion: 208 (5.98%)
- over_segmentation: 107 (3.07%)
- under_segmentation: 106 (3.05%)
- huge_overflow: 73 (2.10%)

## 3. Per-category 错误排名
- 详见 outputs/source/baseline_error_anatomy/baseline_error_category_summary.csv 与 summary.json 排名字段。

## 4. Precision vs Recall 分析
- mean_precision=0.7723, mean_recall=0.7582
- low-IoU: over=14.72% under=14.58% loc/conf=28.61%

## 5. Scale / Area 与 IoU 关系
- gt_area_ratio_vs_iou_pearson: 0.318054
- pred_area_ratio_vs_iou_pearson: 0.323037
- area_ratio_pred_over_gt_vs_iou_pearson: -0.217839
- gt_area_ratio_vs_iou_spearman: 0.619661
- pred_area_ratio_vs_iou_spearman: 0.612201
- area_ratio_pred_over_gt_vs_iou_spearman: 0.070222
- 见 scale_error_summary.csv。

## 6. 重点类别分析
- harbor: n=27, mIoU=0.3826, over=3.70%, under=11.11%, loc/conf=18.52%
- vehicle: n=558, mIoU=0.5320, over=2.15%, under=1.08%, loc/conf=6.81%
- bridge: n=269, mIoU=0.5489, over=2.23%, under=4.46%, loc/conf=11.15%
- airport: n=196, mIoU=0.6098, over=10.20%, under=11.22%, loc/conf=1.02%
- windmill: n=260, mIoU=0.6117, over=1.54%, under=3.46%, loc/conf=1.54%
- trainstation: n=128, mIoU=0.6412, over=4.69%, under=14.84%, loc/conf=8.59%
- overpass: n=226, mIoU=0.6416, over=5.31%, under=2.21%, loc/conf=7.96%
- basketballcourt: n=121, mIoU=0.7029, over=5.79%, under=2.48%, loc/conf=9.92%
- ship: n=182, mIoU=0.7224, over=2.20%, under=0.55%, loc/conf=7.14%
- airplane: n=99, mIoU=0.7339, over=3.03%, under=0.00%, loc/conf=3.03%
- storagetank: n=110, mIoU=0.8015, over=1.82%, under=0.00%, loc/conf=5.45%
- golffield: n=112, mIoU=0.8021, over=2.68%, under=1.79%, loc/conf=0.89%
- chimney: n=107, mIoU=0.8107, over=0.00%, under=0.00%, loc/conf=4.67%

## 7. 对 Mask Scope Prior 的判断
- verdict: D (inconclusive)
- evidence_for / evidence_against 见 summary.json 与本文结论段。

## 8. 下一步建议
- 本轮仅完成方向诊断，不引入新方法，不改模型/训练/语义库/eval。
