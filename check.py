import torch

ckpt = torch.load(
    '/media/Hulu/面部反应生成/baseline_react2026-main-v2/pretrained_models/diffusion_model/TransformerDenoiser/checkpoint.pth',
    map_location='cpu',
    weights_only=False
)

CHECK_KEYWORDS = ['stitch', 'future']

if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print(f'顶层 keys 数量: {len(keys)}')
    print(f'顶层 keys: {keys}')

    for top_k in keys:
        if top_k == 'optimizer':
            continue
        if isinstance(ckpt[top_k], dict):
            inner = ckpt[top_k]
            for kw in CHECK_KEYWORDS:
                kw_inner = [k for k in inner.keys() if kw in k.lower()]
                print(f'\n{top_k} 内 {kw} 相关 keys 数量: {len(kw_inner)}')
                for k in kw_inner[:50]:
                    v = inner[k]
                    print(f'  {k}: shape={v.shape if hasattr(v, "shape") else type(v).__name__}')
                if len(kw_inner) > 50:
                    print(f'  ... 还有 {len(kw_inner) - 50} 个')
                if len(kw_inner) == 0:
                    print('  (无)')
else:
    print(f'类型: {type(ckpt)}')