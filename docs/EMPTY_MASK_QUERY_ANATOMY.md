# Empty-Mask Query Anatomy

## 0. 一页结论
- verdict: D (visual_feature_or_decoder_failure)
- empty/good 样本数: 10 / 10
- projected_query_norm: empty=174.0314, good=166.9830
- logit_max: empty=9.4585, good=11.2884
- prob_max: empty=0.9006, good=0.9002
- inside_outside_logit_margin: empty=32.1310, good=39.9304
- top100 inside GT rate: empty=0.5410, good=0.8000

## 1. 数据与样本
- baseline model: `/home/wangchengjun/huangziyi/reseg/output/source/rrsisd_baseline_raw_7w/merged_model`
- pred_dir: `/home/wangchengjun/huangziyi/reseg/output/base/standard-base-siglip1-28w/test_results`
- selected samples 见 selected_vehicle_samples.csv

## 2. 组间对比
- 详见 empty_mask_query_group_summary.csv 与 per-sample.csv

## 3. 注意
- attention_unavailable=True（未改 decoder，使用 logits heatmap proxy）

## 4. 下一步建议
- 病因偏视觉特征/decoder。
