import importlib.util
from pathlib import Path
import torch

module_path = Path(__file__).resolve().parents[1] / 'segearth_r2' / 'model' / 'tgmsa' / 'tgmsa.py'
spec = importlib.util.spec_from_file_location('tgmsa_module', module_path)
tgmsa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tgmsa_module)
SwinOutputTargetFilter = tgmsa_module.SwinOutputTargetFilter
PixelTargetBackgroundCalibrator = tgmsa_module.PixelTargetBackgroundCalibrator
DynamicQueryBinding = tgmsa_module.DynamicQueryBinding


def test_swin_output_target_filter_is_default_noop_and_preserves_shapes():
    torch.manual_seed(0)
    features = {
        'res2': torch.randn(2, 128, 16, 16),
        'res3': torch.randn(2, 256, 8, 8),
        'res4': torch.randn(2, 512, 4, 4),
        'res5': torch.randn(2, 1024, 2, 2),
    }
    seg = torch.randn(3, 1, 256)
    module = SwinOutputTargetFilter(seg_dim=256, feature_dims={'res2': 128, 'res3': 256, 'res4': 512, 'res5': 1024})

    out = module(features, seg, mask_num=[1, 2])

    assert set(out) == set(features)
    for name in features:
        assert out[name].shape == features[name].shape
        assert torch.allclose(out[name], features[name])


def test_pixel_target_background_calibrator_is_default_noop_and_preserves_shapes():
    torch.manual_seed(1)
    mask_features = torch.randn(3, 256, 8, 8)
    multi_scale = [torch.randn(3, 256, s, s) for s in (4, 8, 16)]
    seg = torch.randn(3, 1, 256)
    module = PixelTargetBackgroundCalibrator(hidden_dim=256)

    out_mask, out_multi = module(mask_features, multi_scale, seg)

    assert out_mask.shape == mask_features.shape
    assert torch.allclose(out_mask, mask_features)
    assert len(out_multi) == len(multi_scale)
    for got, expected in zip(out_multi, multi_scale):
        assert got.shape == expected.shape
        assert torch.allclose(got, expected)


def test_dynamic_query_binding_refines_shape_and_reports_diversity_loss():
    torch.manual_seed(2)
    seg = torch.randn(4, 1, 256)
    module = DynamicQueryBinding(hidden_dim=256, query_bank_size=4, alpha_init=0.1, diversity_margin=0.2, num_heads=4)

    refined, info = module(seg, mask_num=[2, 2])

    assert refined.shape == seg.shape
    assert info['query_pos'].shape == seg.shape
    assert info['loss_tgmsa_query_diversity'].ndim == 0
    assert info['loss_tgmsa_segment_separation'].ndim == 0
    assert info['loss_tgmsa_binding_entropy'].ndim == 0
    assert info['loss_tgmsa_peer_contrast'].ndim == 0
    assert info['tgmsa_binding_logits'].shape == (4, 4)
    assert torch.isfinite(info['loss_tgmsa_query_diversity'])
    assert torch.isfinite(info['loss_tgmsa_segment_separation'])
    assert torch.isfinite(info['loss_tgmsa_binding_entropy'])
    assert torch.isfinite(info['loss_tgmsa_peer_contrast'])

if __name__ == '__main__':
    test_swin_output_target_filter_is_default_noop_and_preserves_shapes()
    test_pixel_target_background_calibrator_is_default_noop_and_preserves_shapes()
    test_dynamic_query_binding_refines_shape_and_reports_diversity_loss()
    print('TGMSA module tests passed')
