from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rrta import jitter_image_offset, warp

__all__ = ["QRISPNet", "QRISP_SIZES", "block_dilate"]
QRISP_SIZES: dict[str, tuple[int, int]] = {"S": (16, 1), "M": (32, 3), "L": (64, 5)}


def block_dilate(
    motion: torch.Tensor, depth: torch.Tensor, scale: int, block: int = 8
) -> torch.Tensor:
    (batch, chans, height, width) = motion.shape
    (H, W) = (height * scale, width * scale)
    if block % scale == 0:
        b = block // scale
        (m, key) = (motion, -depth)
    else:
        b = block
        m = F.interpolate(motion, scale_factor=scale, mode="nearest")
        key = -F.interpolate(depth, scale_factor=scale, mode="nearest")
    (_, index) = F.max_pool2d(key, b, stride=b, ceil_mode=True, return_indices=True)
    picked = m.flatten(2).gather(2, index.flatten(2).expand(-1, chans, -1))
    picked = picked.view(batch, chans, *index.shape[-2:])
    hr = picked.repeat_interleave(block, dim=-2).repeat_interleave(block, dim=-1)
    return hr[..., :H, :W] * scale


def _dyn_conv3(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    (batch, chans, height, width) = x.shape
    out = weight.shape[1]
    y = F.conv2d(
        x.reshape(1, batch * chans, height, width),
        weight.reshape(batch * out, chans, 3, 3),
        bias.repeat(batch),
        padding=1,
        groups=batch,
    )
    return y.view(batch, out, height, width)


class JitterHyperNet(nn.Module):
    def __init__(
        self, out_features: int, base: torch.Tensor, hidden: int = 2048, layers: int = 7
    ) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("the hypernetwork needs >= 2 layers")
        dims = [2] + [hidden] * (layers - 1)
        self.hidden = nn.ModuleList(
            (nn.Linear(a, b) for (a, b) in zip(dims[:-1], dims[1:]))
        )
        for lin in self.hidden:
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
        self.out = nn.Linear(hidden, out_features)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(base)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for lin in self.hidden:
            z = F.relu(lin(z))
        return self.out(z)

    def wide_weights(self) -> list[nn.Parameter]:
        return [lin.weight for lin in self.hidden[1:]] + [self.out.weight]


class QRISPNet(nn.Module):
    arch = "qrisp"
    state_guard = True
    STATE_GUARD_COLOUR = (-2.0, 3.0)
    STATE_GUARD_FEATURE = 1000.0
    STATE_GUARD_DILATE = 1

    def __init__(
        self,
        scale: int = 4,
        f: int = 64,
        m: int = 5,
        hyper_hidden: int = 2048,
        hyper_layers: int = 7,
        hyper_param: str = "mup",
        motion_dilation: bool = True,
        dilation_block: int = 8,
        dejitter: bool = False,
    ) -> None:
        super().__init__()
        if hyper_param not in ("mup", "plain"):
            raise ValueError("hyper_param must be 'mup' or 'plain'")
        s2 = scale * scale
        (self.scale, self.f, self.m) = (int(scale), int(f), int(m))
        (self.dim, self.blocks) = (self.f, self.m)
        (self.hyper_hidden, self.hyper_layers) = (int(hyper_hidden), int(hyper_layers))
        self.hyper_param = hyper_param
        self.motion_dilation = bool(motion_dilation)
        self.dilation_block = int(dilation_block)
        self.dejitter = bool(dejitter)
        self.hist = 4
        self.cin = 3 + 1 + 2 + self.hist * s2
        self.cout = 5 * s2
        self.n_first = self.f * self.cin * 9
        self.n_last = self.cout * self.f * 9
        first = nn.Conv2d(self.cin, self.f, 3, padding=1)
        last = nn.Conv2d(self.f, self.cout, 3, padding=1)
        self.first_bias = nn.Parameter(first.bias.detach().clone())
        self.last_bias = nn.Parameter(last.bias.detach().clone())
        base = torch.cat(
            (first.weight.detach().flatten(), last.weight.detach().flatten())
        )
        self.hyper = JitterHyperNet(
            self.n_first + self.n_last, base, self.hyper_hidden, self.hyper_layers
        )
        self.mid = nn.ModuleList(
            (nn.Conv2d(self.f, self.f, 3, padding=1) for _ in range(self.m))
        )

    def optimizer_groups(self, lr: float) -> list[dict]:
        if self.hyper_param == "plain":
            return [{"params": list(self.parameters()), "lr": lr}]
        wide = self.hyper.wide_weights()
        ids = {id(p) for p in wide}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [
            {"params": rest, "lr": lr},
            {"params": wide, "lr": lr / self.hyper_hidden},
        ]

    def conditioning(self, jitter: torch.Tensor) -> torch.Tensor:
        return 2.0 * jitter_image_offset(jitter.float())

    def kernels(self, jitter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.conditioning(jitter)
        with torch.autocast(device_type=z.device.type, enabled=False):
            k = self.hyper(z)
        n = k.shape[0]
        return (
            k[:, : self.n_first].view(n, self.f, self.cin, 3, 3),
            k[:, self.n_first :].view(n, self.cout, self.f, 3, 3),
        )

    def _hr_motion(self, motion: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if self.motion_dilation:
            return block_dilate(motion, depth, self.scale, self.dilation_block)
        return (
            F.interpolate(motion, scale_factor=self.scale, mode="nearest") * self.scale
        )

    def step(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        motion: torch.Tensor,
        jitter: torch.Tensor,
        state: tuple | None,
        init_hr: torch.Tensor | None = None,
        init_mask: torch.Tensor | None = None,
        kernels: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple, dict]:
        (batch, _, height, width) = rgb.shape
        s = self.scale
        (H, W) = (height * s, width * s)
        jit = jitter.float()
        if state is None:
            warped = torch.zeros(
                batch, self.hist, H, W, device=rgb.device, dtype=torch.float32
            )
            if init_hr is not None:
                gate = (
                    torch.ones(batch, device=rgb.device)
                    if init_mask is None
                    else init_mask.to(device=rgb.device, dtype=torch.float32)
                ).view(-1, 1, 1, 1)
                warped = torch.cat((gate * init_hr.float(), warped[:, 3:]), dim=1)
        else:
            (hist, previous_jitter) = state
            mv = motion.float()
            if self.dejitter:
                d = jit - previous_jitter.float()
                mv = mv + torch.stack((d[:, 0], -d[:, 1]), dim=1)[:, :, None, None]
            (warped, _) = warp(hist.float(), self._hr_motion(mv, depth.float()))
            warped = warped.float()
        (w_first, w_last) = kernels if kernels is not None else self.kernels(jitter)
        z = self.conditioning(jitter).to(rgb.dtype)
        x = torch.cat(
            (
                rgb,
                depth.to(rgb.dtype),
                z[:, :, None, None].expand(-1, -1, height, width),
                F.pixel_unshuffle(warped, s).to(rgb.dtype),
            ),
            dim=1,
        )
        x = F.relu(_dyn_conv3(x, w_first, self.first_bias))
        for conv in self.mid:
            x = F.relu(conv(x))
        o = F.pixel_shuffle(_dyn_conv3(x, w_last, self.last_bias).float(), s)
        alpha = torch.sigmoid(o[:, 0:1])
        candidate = F.relu(o[:, 1:4])
        out = alpha * candidate + (1.0 - alpha) * warped[:, :3]
        new_hist = torch.cat((out, o[:, 4:5]), dim=1)
        aux = {"alpha": alpha, "candidate": candidate}
        if self.state_guard:
            (new_hist, aux["guard_px"]) = self._guard_state(new_hist)
        return (out, (new_hist, jit), aux)

    def _guard_state(self, hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (lo, hi) = self.STATE_GUARD_COLOUR
        (colour, feature) = (hist[:, :3], hist[:, 3:])
        ok = ((colour >= lo) & (colour <= hi)).all(1, keepdim=True) & (
            feature.abs() <= self.STATE_GUARD_FEATURE
        ).all(1, keepdim=True)
        bad_lr = F.max_pool2d((~ok).to(hist.dtype), self.scale) > 0
        d = int(self.STATE_GUARD_DILATE)
        if d > 0:
            bad_lr = (
                F.max_pool2d(bad_lr.to(hist.dtype), 2 * d + 1, stride=1, padding=d) > 0
            )
        bad = (
            F.interpolate(
                bad_lr.to(hist.dtype), scale_factor=self.scale, mode="nearest"
            )
            > 0
        )
        return (torch.where(bad, 0.0, hist), bad_lr.sum().detach())

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        motion: torch.Tensor,
        jitter: torch.Tensor,
        state: tuple | None = None,
        return_aux: bool = False,
        init_hr: torch.Tensor | None = None,
        init_mask: torch.Tensor | None = None,
    ):
        (batch, steps) = rgb.shape[:2]
        (w_first, w_last) = self.kernels(jitter.reshape(batch * steps, 2))
        w_first = w_first.view(batch, steps, *w_first.shape[1:])
        w_last = w_last.view(batch, steps, *w_last.shape[1:])
        (outputs, auxes) = ([], [])
        for t in range(steps):
            warm = t == 0 and state is None and (init_hr is not None)
            (out, state, aux) = self.step(
                rgb[:, t],
                depth[:, t],
                motion[:, t],
                jitter[:, t],
                state,
                init_hr=init_hr if warm else None,
                init_mask=init_mask if warm else None,
                kernels=(w_first[:, t], w_last[:, t]),
            )
            outputs.append(out)
            if return_aux:
                auxes.append(aux)
        stacked = torch.stack(outputs, dim=1)
        return (stacked, state, auxes) if return_aux else (stacked, state)

    def param_counts(self) -> dict:
        hyper = sum((p.numel() for p in self.hyper.parameters()))
        total = sum((p.numel() for p in self.parameters()))
        convs = total - hyper + self.n_first + self.n_last
        return {
            "total": total,
            "hypernet": hyper,
            "trunk_static": total - hyper,
            "deployed_convs": convs,
            "kernels_per_jitter": self.n_first + self.n_last,
        }

    def extra_repr(self) -> str:
        return f"scale={self.scale}, f={self.f}, m={self.m}, cin={self.cin}, cout={self.cout}, hyper={self.hyper_layers}x{self.hyper_hidden} ({self.hyper_param}), dilation={self.motion_dilation}/{self.dilation_block}, dejitter={self.dejitter}"


def create_qrisp(scale: int = 4, **kwargs) -> QRISPNet:
    kwargs = {k: v for (k, v) in kwargs.items() if k not in ("size", "arch")}
    return QRISPNet(scale=scale, **kwargs)


def conv_gflop(model: QRISPNet, height: int, width: int) -> float:
    macs = 9 * (
        model.cin * model.f + model.m * model.f * model.f + model.f * model.cout
    )
    return 2.0 * macs * height * width / 1000000000.0


if __name__ == "__main__":
    net = QRISPNet()
    print(net)
    print(net.param_counts(), f"{conv_gflop(net, 270, 480):.2f} GFLOP")
