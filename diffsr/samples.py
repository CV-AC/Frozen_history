from __future__ import annotations
import math
from typing import Sequence
import torch
import torch.nn.functional as F

__all__ = [
    "CONVENTIONS",
    "as_jitter",
    "sample_offset",
    "registered_sample_positions",
    "splat_table",
    "splat_gather",
    "registered_bicubic",
    "gather_nearest",
    "warp_bilinear",
    "warp_separable",
    "warp_catmull_rom",
    "cubic_weights",
    "lanczos2_weights",
    "velocity_cap_T",
    "velocity_cap",
    "variance_clip",
    "neighbourhood_moments",
    "sample_moments",
    "depth_test",
    "parse_depth_spec",
    "dilate_motion",
    "motion_to_hr",
]
CONVENTIONS = ("image", "legacy")


def as_jitter(
    jitter, batch: int | None = None, device=None, dtype=torch.float32
) -> torch.Tensor:
    j = torch.as_tensor(jitter, device=device, dtype=dtype)
    if j.dim() == 1:
        j = j[None]
    if j.shape[-1] != 2:
        raise ValueError(f"jitter must end in (jy, jx), got {tuple(j.shape)}")
    if batch is not None and j.shape[0] == 1 and (batch > 1):
        j = j.expand(batch, 2)
    return j


def sample_offset(scale: int, jitter, convention: str = "image") -> torch.Tensor:
    j = as_jitter(
        jitter,
        dtype=jitter.dtype
        if torch.is_tensor(jitter) and jitter.is_floating_point()
        else torch.float32,
        device=jitter.device if torch.is_tensor(jitter) else None,
    )
    if convention == "image":
        return torch.stack((-j[:, 0] * scale, j[:, 1] * scale), dim=1)
    if convention == "legacy":
        return torch.stack((j[:, 0] * scale, j[:, 1] * scale), dim=1)
    raise ValueError(f"convention must be one of {CONVENTIONS}")


def registered_sample_positions(
    scale: int, jitter_yx, h: int, w: int, convention: str = "image"
) -> torch.Tensor:
    o = sample_offset(scale, jitter_yx, convention)
    c = 0.5 * (scale - 1)
    rows = torch.arange(h, device=o.device, dtype=o.dtype) * scale + c
    cols = torch.arange(w, device=o.device, dtype=o.dtype) * scale + c
    y = rows[None, :, None] + o[:, 0, None, None]
    x = cols[None, None, :] + o[:, 1, None, None]
    return torch.stack((y.expand(-1, h, w), x.expand(-1, h, w)), dim=1)


def _tap_offsets(scale: int, jitter, taps: int, convention: str):
    o = sample_offset(scale, jitter, convention)
    r = taps // 2
    c = 0.5 * (scale - 1)
    u = torch.arange(scale, device=o.device, dtype=o.dtype)
    a = torch.arange(-r, r + 1, device=o.device, dtype=o.dtype)
    base = scale * a[None, :] + c - u[:, None]
    dy = base[None] + o[:, 0, None, None]
    dx = base[None] + o[:, 1, None, None]
    return (dy, dx)


def _raw_table(scale, jitter, sigma, taps, convention, kernel):
    (dy, dx) = _tap_offsets(scale, jitter, taps, convention)
    batch = dy.shape[0]
    d2 = dy[:, :, None, :, None] ** 2 + dx[:, None, :, None, :] ** 2
    if kernel == "gauss":
        sig = torch.as_tensor(sigma, device=d2.device, dtype=d2.dtype)
        k = torch.exp(-d2 / (2.0 * sig * sig))
    elif kernel == "nearest":
        flat = d2.reshape(batch, scale, scale, taps * taps)
        k = F.one_hot(flat.argmin(-1), taps * taps).to(d2.dtype)
        k = k.reshape_as(d2)
    elif kernel == "box":
        k = torch.ones_like(d2)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    return k.reshape(batch, scale * scale, taps * taps)


_EXACT_GRID = 32


def splat_table(
    scale: int,
    jitter,
    sigma=0.7,
    taps: int = 3,
    convention: str = "image",
    kernel: str = "gauss",
    normalize: str = "analytic",
) -> torch.Tensor:
    k = _raw_table(scale, jitter, sigma, taps, convention, kernel)
    if kernel != "gauss" or normalize == "none":
        return k
    if normalize == "analytic":
        sig = torch.as_tensor(sigma, device=k.device, dtype=k.dtype)
        return k * (scale * scale / (2.0 * math.pi * sig * sig))
    if normalize == "exact":
        g = (
            torch.arange(_EXACT_GRID, device=k.device, dtype=k.dtype) + 0.5
        ) / _EXACT_GRID - 0.5
        grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        z = _raw_table(scale, grid, sigma, taps, convention, "gauss")
        return k / z.sum(-1).mean()
    raise ValueError(f"unknown normalize {normalize!r}")


def splat_gather(
    lr: torch.Tensor,
    jitter,
    scale: int,
    sigma=0.7,
    taps: int = 3,
    convention: str = "image",
    normalize: str = "analytic",
    border: str = "zero",
    kernel: str = "gauss",
    table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    (batch, chans, h, w) = lr.shape
    if table is None:
        table = splat_table(
            scale,
            as_jitter(jitter, batch, lr.device, lr.dtype),
            sigma,
            taps,
            convention,
            kernel,
            normalize,
        )
    table = table.to(device=lr.device, dtype=lr.dtype)
    if table.shape[0] == 1 and batch > 1:
        table = table.expand(batch, -1, -1)
    r = taps // 2
    ones = torch.ones(batch, 1, h, w, device=lr.device, dtype=lr.dtype)
    x = torch.cat((lr, ones), dim=1)
    x = F.pad(x, (r, r, r, r), mode="constant" if border == "zero" else "replicate")
    cols = F.unfold(x, taps).view(batch, chans + 1, taps * taps, h, w)
    out = torch.einsum("bcthw,but->bcuhw", cols, table)
    out = F.pixel_shuffle(out.reshape(batch, (chans + 1) * scale * scale, h, w), scale)
    return (out[:, :chans], out[:, chans:])


def gather_nearest(
    lr_stat: torch.Tensor, jitter, scale: int, convention: str = "image"
) -> torch.Tensor:
    (s_f, _) = splat_gather(
        lr_stat,
        jitter,
        scale,
        taps=3,
        convention=convention,
        kernel="nearest",
        border="replicate",
    )
    return s_f


def registered_bicubic(
    lr: torch.Tensor, jitter, scale: int, convention: str = "image", clamp: bool = True
) -> torch.Tensor:
    (batch, _, h, w) = lr.shape
    o = sample_offset(
        scale, as_jitter(jitter, batch, lr.device, torch.float32), convention
    ).float()
    c = 0.5 * (scale - 1)
    ys = torch.arange(h * scale, device=lr.device, dtype=torch.float32)
    xs = torch.arange(w * scale, device=lr.device, dtype=torch.float32)
    ly = (ys[None] - c - o[:, 0, None]) / scale
    lx = (xs[None] - c - o[:, 1, None]) / scale
    ny = 2.0 * ly / max(h - 1, 1) - 1.0
    nx = 2.0 * lx / max(w - 1, 1) - 1.0
    grid = torch.stack(
        (
            nx[:, None, :].expand(-1, h * scale, -1),
            ny[:, :, None].expand(-1, -1, w * scale),
        ),
        dim=-1,
    )
    out = F.grid_sample(
        lr.float(), grid, mode="bicubic", padding_mode="border", align_corners=True
    ).to(lr.dtype)
    return out.clamp(0, 1) if clamp else out


def _grid_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.promote_types(torch.float32, x.dtype)


def warp_bilinear(
    x: torch.Tensor, motion: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    (batch, _, height, width) = x.shape
    gd = _grid_dtype(x)
    rows = torch.arange(height, device=x.device, dtype=gd)
    cols = torch.arange(width, device=x.device, dtype=gd)
    (grid_y, grid_x) = torch.meshgrid(rows, cols, indexing="ij")
    src_y = grid_y[None] + motion[:, 0].to(gd)
    src_x = grid_x[None] + motion[:, 1].to(gd)
    norm_y = 2.0 * src_y / max(height - 1, 1) - 1.0
    norm_x = 2.0 * src_x / max(width - 1, 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    out = F.grid_sample(
        x.to(gd), grid, mode="bilinear", padding_mode="border", align_corners=True
    ).to(x.dtype)
    inside = (norm_x.abs() <= 1.0) & (norm_y.abs() <= 1.0)
    return (out, inside[:, None].to(x.dtype))


def cubic_weights(t: torch.Tensor, a: float = -0.5):

    def near(x):
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0

    def far(x):
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a

    return (far(1.0 + t), near(t), near(1.0 - t), far(2.0 - t))


def lanczos2_weights(t: torch.Tensor):

    def lz(x):
        x = x.abs()
        px = math.pi * x
        val = torch.where(
            x < 1e-06,
            torch.ones_like(x),
            2.0 * torch.sin(px) * torch.sin(px / 2.0) / (px * px).clamp_min(1e-12),
        )
        return torch.where(x < 2.0, val, torch.zeros_like(x))

    w = [lz(1.0 + t), lz(t), lz(1.0 - t), lz(2.0 - t)]
    total = w[0] + w[1] + w[2] + w[3]
    return tuple((v / total for v in w))


_KERNELS = ("catmull_rom", "lanczos2", "bicubic075")


def _weights(kernel: str, t: torch.Tensor):
    if kernel == "catmull_rom":
        return cubic_weights(t, -0.5)
    if kernel == "bicubic075":
        return cubic_weights(t, -0.75)
    if kernel == "lanczos2":
        return lanczos2_weights(t)
    raise ValueError(f"kernel must be one of {_KERNELS}")


def _fetch16(x: torch.Tensor, motion: torch.Tensor, kernel: str):
    (batch, chans, height, width) = x.shape
    gd = _grid_dtype(x)
    rows = torch.arange(height, device=x.device, dtype=gd)
    cols = torch.arange(width, device=x.device, dtype=gd)
    py = rows[None, :, None] + motion[:, 0].to(gd)
    px = cols[None, None, :] + motion[:, 1].to(gd)
    (fy, fx) = (torch.floor(py), torch.floor(px))
    wy = _weights(kernel, py - fy)
    wx = _weights(kernel, px - fx)
    iy = [(fy + (k - 1)).clamp(0, height - 1).long() for k in range(4)]
    ix = [(fx + (k - 1)).clamp(0, width - 1).long() for k in range(4)]
    flat = x.reshape(batch, chans, height * width)
    out = None
    for a in range(4):
        for b in range(4):
            idx = (iy[a] * width + ix[b]).view(batch, 1, -1).expand(-1, chans, -1)
            term = (wy[a] * wx[b]).to(x.dtype)[:, None] * flat.gather(2, idx).view(
                batch, chans, height, width
            )
            out = term if out is None else out + term
    inside = ((py >= 0) & (py <= height - 1) & (px >= 0) & (px <= width - 1))[
        :, None
    ].to(x.dtype)
    footprint = [iy[a] * width + ix[b] for a in (1, 2) for b in (1, 2)]
    return (out, inside, footprint)


def _gather_flat(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    (batch, chans, height, width) = x.shape
    return (
        x.reshape(batch, chans, -1)
        .gather(2, idx.view(batch, 1, -1).expand(-1, chans, -1))
        .view(batch, chans, height, width)
    )


def warp_separable(
    x,
    motion: torch.Tensor,
    kernel: str = "catmull_rom",
    premultiplied: bool = False,
    clamp_to_neighbourhood: bool = True,
    eps: float = 1e-06,
):
    if premultiplied:
        (s_prev, w_prev) = x
        chans = s_prev.shape[1]
        (both, inside, fp) = _fetch16(torch.cat((s_prev, w_prev), 1), motion, kernel)
        (s_hat, w_hat) = (both[:, :chans], both[:, chans:].clamp_min(0.0))
        if not clamp_to_neighbourhood:
            return (s_hat, w_hat, inside)
        mu_prev = s_prev / w_prev.clamp_min(eps)
        has_prev = (w_prev > eps).to(s_prev.dtype)
        mus = torch.stack([_gather_flat(mu_prev, i) for i in fp])
        ws = torch.stack([_gather_flat(w_prev, i) for i in fp])
        hs = torch.stack([_gather_flat(has_prev, i) for i in fp]) > 0.5
        big = torch.finfo(s_prev.dtype).max
        lo = torch.where(hs, mus, torch.full_like(mus, big)).amin(0)
        hi = torch.where(hs, mus, torch.full_like(mus, -big)).amax(0)
        any_w = hs.any(0)
        mu_hat = s_hat / w_hat.clamp_min(eps)
        mu_c = torch.where(any_w, torch.minimum(torch.maximum(mu_hat, lo), hi), mu_hat)
        w_c = torch.minimum(torch.maximum(w_hat, ws.amin(0)), ws.amax(0))
        s_c = torch.where(w_c > eps, mu_c * w_c, torch.zeros_like(s_hat))
        return (s_c, w_c, inside)
    (out, inside, fp) = _fetch16(x, motion, kernel)
    if clamp_to_neighbourhood:
        vals = torch.stack([_gather_flat(x, i) for i in fp])
        out = torch.minimum(torch.maximum(out, vals.amin(0)), vals.amax(0))
    return (out, inside)


def warp_catmull_rom(
    x,
    motion: torch.Tensor,
    premultiplied: bool = False,
    clamp_to_neighbourhood: bool = True,
    eps: float = 1e-06,
):
    return warp_separable(
        x, motion, "catmull_rom", premultiplied, clamp_to_neighbourhood, eps
    )


def _speed(v_hr: torch.Tensor) -> torch.Tensor:
    if v_hr.shape[1] == 2:
        return v_hr.float().pow(2).sum(1, keepdim=True).sqrt().to(v_hr.dtype)
    return v_hr.abs()


def velocity_cap_T(
    v_hr: torch.Tensor, vmax: float = 64.0, vmin: float = 2.0, v0: float = 2.0
) -> torch.Tensor:
    return (vmax / (1.0 + _speed(v_hr) / v0)).clamp_min(vmin)


def velocity_cap(
    S: torch.Tensor,
    W: torch.Tensor,
    v_hr: torch.Tensor,
    vmax: float = 64.0,
    vmin: float = 2.0,
    v0: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    excess = (W / velocity_cap_T(v_hr, vmax, vmin, v0)).clamp_min(1.0)
    return (S / excess, W / excess)


def neighbourhood_moments(
    x: torch.Tensor, k: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    r = k // 2
    xp = F.pad(x, (r, r, r, r), mode="replicate")
    mu = F.avg_pool2d(xp, k, stride=1)
    m2 = F.avg_pool2d(xp * xp, k, stride=1)
    return (mu, (m2 - mu * mu).clamp_min(0.0).sqrt())


def sample_moments(
    lr: torch.Tensor, jitter, scale: int, k: int = 3, convention: str = "image"
) -> tuple[torch.Tensor, torch.Tensor]:
    chans = lr.shape[1]
    r = k // 2
    xp = F.pad(lr, (r, r, r, r), mode="replicate")
    mu = F.avg_pool2d(xp, k, stride=1)
    m2 = F.avg_pool2d(xp * xp, k, stride=1)
    both = gather_nearest(torch.cat((mu, m2), 1), jitter, scale, convention)
    (mu_h, m2_h) = (both[:, :chans], both[:, chans:])
    return (mu_h, (m2_h - mu_h * mu_h).clamp_min(0.0).sqrt())


def variance_clip(
    hist: torch.Tensor,
    ref_nbhd,
    gamma: float = 1.0,
    mode: str = "clamp",
    eps: float = 1e-06,
) -> torch.Tensor:
    if torch.is_tensor(ref_nbhd):
        mu = ref_nbhd.mean(1)
        sigma = ref_nbhd.std(1, unbiased=False)
    else:
        (mu, sigma) = ref_nbhd
    ext = gamma * sigma
    if mode == "clamp":
        return torch.minimum(torch.maximum(hist, mu - ext), mu + ext)
    if mode == "clip":
        d = hist - mu
        t = (d.abs() / (ext + eps)).amax(1, keepdim=True)
        return mu + d / t.clamp_min(1.0)
    raise ValueError(f"unknown mode {mode!r}")


def motion_to_hr(motion: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(motion, scale_factor=scale, mode="nearest") * scale


def dilate_motion(
    motion: torch.Tensor, depth: torch.Tensor, near_is_small: bool = True
) -> torch.Tensor:
    key = -depth if near_is_small else depth
    (_, index) = F.max_pool2d(key, 3, stride=1, padding=1, return_indices=True)
    flat = motion.flatten(2)
    picked = flat.gather(2, index.flatten(2).expand(-1, motion.shape[1], -1))
    return picked.view_as(motion)


def depth_test(
    depth: torch.Tensor,
    depth_prev: torch.Tensor,
    motion: torch.Tensor,
    rel: float = 0.1,
    mode: str = "far",
) -> tuple[torch.Tensor, torch.Tensor]:
    (warped, inside) = warp_bilinear(depth_prev, motion)
    err = (depth - warped) / depth.abs().clamp_min(1e-06)
    if mode == "far":
        bad = err > rel
    elif mode == "sym":
        bad = err.abs() > rel
    else:
        raise ValueError(f"unknown depth-test mode {mode!r}")
    return ((~bad).to(depth.dtype) * inside, err)


def parse_depth_spec(spec: str | None) -> tuple[str, float] | None:
    if spec is None or str(spec).lower() in ("none", "off", ""):
        return None
    spec = str(spec)
    for mode in ("far", "sym"):
        if spec.startswith(mode):
            return (mode, float(spec[len(mode) :]))
    raise ValueError(f"depth spec must look like far0.1 / sym0.05 / none, got {spec!r}")


def _brute_splat(lr, jitter, scale, sigma, taps, convention, normalize):
    (b, c, h, w) = lr.shape
    o = sample_offset(scale, jitter, convention).double()
    cc = 0.5 * (scale - 1)
    r = taps // 2
    S = torch.zeros(b, c, h * scale, w * scale, dtype=torch.float64)
    W = torch.zeros(b, 1, h * scale, w * scale, dtype=torch.float64)
    norm = (
        scale * scale / (2 * math.pi * sigma * sigma)
        if normalize == "analytic"
        else 1.0
    )
    for bi in range(b):
        for Y in range(h * scale):
            for X in range(w * scale):
                (i0, j0) = (Y // scale, X // scale)
                for a in range(-r, r + 1):
                    for bb in range(-r, r + 1):
                        (i, j) = (i0 + a, j0 + bb)
                        if not (0 <= i < h and 0 <= j < w):
                            continue
                        qy = i * scale + cc + o[bi, 0]
                        qx = j * scale + cc + o[bi, 1]
                        k = norm * math.exp(
                            -((qy - Y) ** 2 + (qx - X) ** 2) / (2 * sigma * sigma)
                        )
                        S[bi, :, Y, X] += k * lr[bi, :, i, j].double()
                        W[bi, 0, Y, X] += k
    return (S, W)


def _selftest() -> None:
    torch.manual_seed(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{('PASS' if cond else 'FAIL')}] {name} {detail}")

    p = registered_sample_positions(4, [[0.25, -0.125]], 2, 3)
    check(
        "positions (image)",
        torch.allclose(p[0, :, 1, 2], torch.tensor([4 + 1.5 - 1.0, 8 + 1.5 - 0.5])),
        f"{p[0, :, 1, 2].tolist()}",
    )
    p = registered_sample_positions(4, [[0.25, -0.125]], 2, 3, "legacy")
    check(
        "positions (legacy)",
        torch.allclose(p[0, :, 1, 2], torch.tensor([4 + 1.5 + 1.0, 8 + 1.5 - 0.5])),
    )
    for scale, sigma, conv in (
        (4, 0.7, "image"),
        (4, 1.3, "legacy"),
        (3, 0.6, "image"),
        (2, 0.9, "image"),
    ):
        lr = torch.rand(2, 3, 5, 6, dtype=torch.float64)
        jit = torch.rand(2, 2, dtype=torch.float64) - 0.5
        (S, W) = splat_gather(lr, jit, scale, sigma, convention=conv)
        (Sb, Wb) = _brute_splat(lr, jit, scale, sigma, 3, conv, "analytic")
        err = max((S - Sb).abs().max().item(), (W - Wb).abs().max().item())
        check(
            f"splat == brute x{scale} sigma {sigma} {conv}",
            err < 1e-10,
            f"max err {err:.2e}",
        )
    g = (torch.arange(16, dtype=torch.float64) + 0.5) / 16 - 0.5
    grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    for scale in (4, 3, 2):
        for sigma in (0.5, 0.7, 1.0):
            K = splat_table(scale, grid, sigma)
            e = K.sum(-1).mean().item()
            Ke = splat_table(scale, grid, sigma, normalize="exact")
            ee = Ke.sum(-1).mean().item()
            tol = 0.001 if scale == 4 else 0.03
            check(
                f"E[W_f] x{scale} sigma {sigma}",
                abs(e - 1) < tol and abs(ee - 1) < 0.01,
                f"analytic {e:.5f} exact {ee:.5f}",
            )
    lr = torch.rand(1, 2, 3, 3, dtype=torch.float64, requires_grad=True)
    sig = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    jit = torch.tensor([[0.1, -0.3]], dtype=torch.float64)

    def fn(x, s_):
        (S, W) = splat_gather(x, jit, 4, s_)
        return S / (W + 0.1)

    check(
        "gradcheck splat (colour, sigma)",
        torch.autograd.gradcheck(fn, (lr, sig), eps=1e-06, atol=1e-05),
    )
    lr = torch.rand(1, 3, 12, 16)
    a = registered_bicubic(lr, [[0.0, 0.0]], 4, clamp=False)
    b = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
    err = (a - b).abs().max().item()
    check("registered_bicubic(j=0) == interpolate", err < 1e-05, f"max err {err:.2e}")
    jit2 = torch.tensor([[0.375, -0.125]])
    up2 = registered_bicubic(lr, jit2, 4, clamp=False)
    err2 = (up2[:, :, 4:44:4, 5:61:4] - lr[:, :, 1:11, 1:15]).abs().max().item()
    check(
        "registered_bicubic interpolates samples", err2 < 1e-05, f"max err {err2:.2e}"
    )
    try:
        from .rrta import warp as _rwarp

        x = torch.rand(2, 4, 20, 30)
        m = torch.randn(2, 2, 20, 30) * 3
        (o1, i1) = warp_bilinear(x, m)
        (o2, i2) = _rwarp(x, m)
        check(
            "warp_bilinear == rrta.warp (fp32)",
            torch.equal(o1, o2) and torch.equal(i1, i2),
        )
    except ImportError:
        print("  [skip] rrta not importable")
    (H, W) = (24, 32)
    (yy, xx) = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )

    def quad(y, x):
        return 0.3 + 0.01 * y - 0.02 * x + 0.003 * y * y + 0.002 * x * y - 0.001 * x * x

    img = quad(yy, xx)[None, None]
    for dy, dx in ((0.3, -0.7), (-1.45, 2.2), (0.5, 0.5)):
        m = (
            torch.tensor([dy, dx], dtype=torch.float64)
            .view(1, 2, 1, 1)
            .expand(1, 2, H, W)
        )
        (out, inside) = warp_catmull_rom(img, m, clamp_to_neighbourhood=False)
        ref = quad(yy + dy, xx + dx)
        core = (slice(4, -4), slice(4, -4))
        err = (out[0, 0][core] - ref[core]).abs().max().item()
        (ob, _) = warp_bilinear(img, m)
        errb = (ob[0, 0][core] - ref[core]).abs().max().item()
        (oc, _) = warp_separable(img, m, "bicubic075", clamp_to_neighbourhood=False)
        errc = (oc[0, 0][core] - ref[core]).abs().max().item()
        check(
            f"CR exact on quadratic shift ({dy},{dx})",
            err < 1e-12,
            f"CR {err:.1e}  bilinear {errb:.1e}  keys-0.75 {errc:.1e}",
        )
    xs = torch.arange(256, dtype=torch.float64)
    for f in (0.25, 0.5):
        errs = {}
        for phase in torch.linspace(0, 2 * math.pi, 9)[:-1]:
            sig = (
                torch.sin(math.pi / 2 * xs + phase)
                .view(1, 1, 1, -1)
                .expand(1, 1, 8, 256)
            )
            m = torch.zeros(1, 2, 8, 256, dtype=torch.float64)
            m[:, 1] = f
            ref = torch.sin(math.pi / 2 * (xs + f) + phase)[8:-8]
            for name, fn_ in (
                ("bilinear", lambda: warp_bilinear(sig, m)[0]),
                (
                    "cr",
                    lambda: warp_catmull_rom(sig, m, clamp_to_neighbourhood=False)[0],
                ),
                ("cr_clamp", lambda: warp_catmull_rom(sig, m)[0]),
                (
                    "lanczos2",
                    lambda: warp_separable(
                        sig, m, "lanczos2", clamp_to_neighbourhood=False
                    )[0],
                ),
            ):
                e = (fn_()[0, 0, 4, 8:-8] - ref).pow(2).mean().sqrt().item()
                errs.setdefault(name, []).append(e)
        rms = {k: sum(v) / len(v) for (k, v) in errs.items()}
        check(
            f"sinusoid 0.5 Nyquist shift {f}: CR < bilinear",
            rms["cr"] < rms["bilinear"],
            " ".join((f"{k} {v:.4f}" for (k, v) in rms.items())),
        )
    S = torch.rand(1, 3, 16, 20, dtype=torch.float64)
    Wt = torch.rand(1, 1, 16, 20, dtype=torch.float64) + 0.1
    z = torch.zeros(1, 2, 16, 20, dtype=torch.float64)
    (s2, w2, _) = warp_catmull_rom((S, Wt), z, premultiplied=True)
    check(
        "premultiplied identity at zero motion",
        torch.allclose(s2, S) and torch.allclose(w2, Wt),
    )
    m = torch.randn(1, 2, 16, 20, dtype=torch.float64)
    mu = torch.rand(1, 3, 16, 20, dtype=torch.float64)
    Wc = torch.full((1, 1, 16, 20), 2.0, dtype=torch.float64)
    (s3, w3, _) = warp_catmull_rom((mu * Wc, Wc), m, premultiplied=True)
    (p3, _) = warp_catmull_rom(mu, m)
    check(
        "premultiplied == plain for constant W",
        torch.allclose(s3 / w3, p3) and torch.allclose(w3, Wc),
    )
    zeros = torch.zeros(1, 1, 16, 20, dtype=torch.float64)
    (s4, w4, _) = warp_catmull_rom((S * 0, zeros), m, premultiplied=True)
    check(
        "premultiplied empty history stays empty",
        bool((w4 == 0).all()) and bool((s4 == 0).all()),
    )
    S = torch.rand(1, 3, 4, 4) * 50
    Wt = torch.rand(1, 1, 4, 4) * 50 + 1
    v = torch.randn(1, 2, 4, 4) * 5
    (S2, W2) = velocity_cap(S, Wt, v, 64, 2, 2)
    T = velocity_cap_T(v, 64, 2, 2)
    check(
        "velocity cap",
        torch.allclose(S2 / W2, S / Wt)
        and bool((W2 <= T + 1e-05).all())
        and bool((T >= 2).all()),
    )
    check(
        "velocity cap T(0) = vmax",
        torch.allclose(
            velocity_cap_T(torch.zeros(1, 2, 1, 1), 64, 2, 2), torch.tensor(64.0)
        ),
    )
    mu = torch.zeros(1, 3, 2, 2)
    sd = torch.ones(1, 3, 2, 2)
    h = torch.tensor([3.0, 0.5, -2.0]).view(1, 3, 1, 1).expand(1, 3, 2, 2)
    c1 = variance_clip(h, (mu, sd), 1.0, "clamp")
    c2 = variance_clip(h, (mu, sd), 1.0, "clip")
    check(
        "variance clamp", torch.allclose(c1[0, :, 0, 0], torch.tensor([1.0, 0.5, -1.0]))
    )
    check(
        "variance clip-to-centre",
        torch.allclose(
            c2[0, :, 0, 0], torch.tensor([1.0, 0.5 / 3, -2.0 / 3]), atol=1e-05
        ),
    )
    lr = torch.arange(20.0).view(1, 1, 4, 5)
    jit = torch.tensor([[0.3, -0.4]])
    near = gather_nearest(lr, jit, 4)
    pos = registered_sample_positions(4, jit, 4, 5)[0]
    (Y, X) = (7, 9)
    d = (pos[0] - Y) ** 2 + (pos[1] - X) ** 2
    idx = int(d.flatten().argmin())
    check("gather_nearest", float(near[0, 0, Y, X]) == float(lr.flatten()[idx]))
    print("ALL PASS" if ok else "SOME TESTS FAILED")
    if not ok:
        raise SystemExit(1)


def _static_segment_test(sigma: float = 0.8) -> dict:
    import json
    import sys
    from pathlib import Path
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from diffsr.data_qrisp import _Lmdb, QRISP_LMDB
    from diffsr.data_sisr import to_tensor
    from diffsr.metrics import psnr

    fr = json.loads((QRISP_LMDB / "QRISP_test_index.json").read_text())["frames"]
    L = _Lmdb(QRISP_LMDB / "QRISP_test_LR270MipBiasJit.lmdb")
    G = _Lmdb(QRISP_LMDB / "QRISP_test_GT1080Enhanced.lmdb")
    seg = [
        f
        for f in fr
        if f["scene"] == "AbandonedSchool"
        and f["sequence"] == "0001"
        and (275 <= int(f["frame"]) <= 290)
    ]
    ref = G.image(seg[-1]["key"])[..., ::-1].astype(np.float64)
    res = {}
    for conv in ("image", "legacy"):
        S = W = None
        for f in seg:
            lr = to_tensor(L.image(f["key"]))[None].double()
            (s_, w_) = splat_gather(
                lr,
                torch.tensor([f["jitter_yx"]], dtype=torch.float64),
                4,
                sigma,
                convention=conv,
            )
            S = s_ if S is None else S + s_
            W = w_ if W is None else W + w_
        out = (S / W.clamp_min(1e-12))[0].permute(1, 2, 0).numpy() * 255.0
        res[conv] = psnr(out, ref, 4, crop_border=8)
    import cv2

    bic = cv2.resize(
        L.image(seg[-1]["key"]), (1920, 1080), interpolation=cv2.INTER_CUBIC
    )[..., ::-1].astype(np.float64)
    res["bicubic"] = psnr(bic, ref, 4, crop_border=8)
    print(
        f"static segment, 16 frames, sigma {sigma}: image {res['image']:.3f} (expect 34.185)  legacy {res['legacy']:.3f} (expect 29.784)  bicubic {res['bicubic']:.3f} (expect 28.379)"
    )
    return res


if __name__ == "__main__":
    import sys

    if "--static" in sys.argv:
        _static_segment_test()
    else:
        _selftest()
