import os

import torch

from archive.models.dit.attention import Attention

ATTENTION_SOURCE = "archive/models/dit/attention.py"


def require_runtime():
    if not torch.cuda.is_available():
        raise SystemExit("This script needs a CUDA GPU.")

    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise SystemExit("This script needs flash-attn installed.") from exc


def make_pair(dim, context_dim=None, dtype=torch.float16):
    sdpa = Attention(
        dim=dim,
        context_dim=context_dim,
        num_heads=4,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        rope_mode="none",
        attn_mode="sdpa",
    ).cuda().to(dtype).eval()

    flash = Attention(
        dim=dim,
        context_dim=context_dim,
        num_heads=4,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        rope_mode="none",
        attn_mode="flash_attn",
    ).cuda().to(dtype).eval()

    flash.load_state_dict(sdpa.state_dict())
    return sdpa, flash


def compare(
    name, sdpa, flash, x, x_mask=None, context=None, context_mask=None
):
    with torch.no_grad():
        sdpa_out = sdpa(
            x=x,
            context=context,
            context_mask=context_mask,
        )
        flash_out = flash(
            x=x,
            context=context,
            context_mask=context_mask,
        )

    if sdpa_out.shape != flash_out.shape:
        print(
            f"FAIL {name}: shape sdpa={tuple(sdpa_out.shape)} flash={tuple(flash_out.shape)}"
        )
        return False

    if context is None:
        lhs = sdpa_out[context_mask]
        rhs = flash_out[context_mask]
    else:
        lhs = sdpa_out.reshape(-1, sdpa_out.shape[-1])
        rhs = flash_out.reshape(-1, flash_out.shape[-1])

    diff = (lhs.float() - rhs.float()).abs()
    max_abs = diff.max().item()
    ok = torch.allclose(lhs.float(), rhs.float(), atol=2e-2, rtol=2e-2)
    print(f"{'PASS' if ok else 'FAIL'} {name}: max_abs={max_abs:.6f}")
    return ok


def add_noise_to_masked_tokens(x, mask, scale=4.0):
    x = x.clone()
    x[~mask] = x[~mask] + scale
    return x


def main():
    require_runtime()

    # archive Attention also accepts the ATTN_MODE env override. Remove it so
    # the two modules below really run different branches.
    os.environ.pop("ATTN_MODE", None)

    torch.manual_seed(1234)
    dtype = torch.bfloat16
    device = "cuda"

    print(f"Attention source: {ATTENTION_SOURCE}")
    print(f"dtype: {dtype}")

    dim = 64
    context_dim = 40
    batch = 2
    self_len = 12
    query_len = 9
    context_len = 13

    self_x = torch.randn(batch, self_len, dim, device=device, dtype=dtype)
    cross_x = torch.randn(batch, query_len, dim, device=device, dtype=dtype)
    context = torch.randn(
        batch, context_len, context_dim, device=device, dtype=dtype
    )

    self_mask = torch.ones(batch, self_len, device=device, dtype=torch.bool)
    self_mask[0, -3:] = False
    self_mask[1, -5:] = False

    query_mask = torch.ones(batch, query_len, device=device, dtype=torch.bool)
    query_mask[0, -2:] = False
    query_mask[1, -4:] = False

    context_mask = torch.ones(
        batch, context_len, device=device, dtype=torch.bool
    )
    context_mask[0, -4:] = False
    context_mask[1, -6:] = False

    # Make ignored padding obvious if a branch accidentally attends to it.
    self_x_with_noisy_pad = add_noise_to_masked_tokens(self_x, self_mask)
    context_with_noisy_pad = add_noise_to_masked_tokens(context, context_mask)

    self_sdpa, self_flash = make_pair(dim=dim, dtype=dtype)
    cross_sdpa, cross_flash = make_pair(
        dim=dim, context_dim=context_dim, dtype=dtype
    )

    results = [
        compare(
            "self/context_mask_only",
            self_sdpa,
            self_flash,
            x=self_x_with_noisy_pad,
            context=None,
            context_mask=self_mask,
        ),
        compare(
            "cross/context_mask_only",
            cross_sdpa,
            cross_flash,
            x=cross_x,
            x_mask=None,
            context=context_with_noisy_pad,
            context_mask=context_mask,
        ),
    ]

    print("")
    if all(results):
        print("All checks passed.")
    else:
        print("Some checks failed.")


if __name__ == "__main__":
    main()
