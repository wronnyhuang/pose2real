import torch, torch.nn as nn, torch.nn.functional as F
# ---------------- discriminator: multi-scale spectral-norm PatchGAN, conditional -------------
def SN(m): return nn.utils.spectral_norm(m)


class Patch(nn.Module):
    def __init__(self, cin=6, ch=64, n=4):
        super().__init__()
        L = [SN(nn.Conv2d(cin, ch, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        c = ch
        for i in range(1, n):
            c2 = min(ch * 2 ** i, 512)
            L += [SN(nn.Conv2d(c, c2, 4, 2 if i < n - 1 else 1, 1)), nn.LeakyReLU(0.2, True)]
            c = c2
        L += [SN(nn.Conv2d(c, 1, 4, 1, 1))]
        self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x)


class MultiD(nn.Module):
    def __init__(self, cin=6):
        super().__init__()
        self.ds = nn.ModuleList([Patch(cin), Patch(cin), Patch(cin, n=3)])

    def forward(self, x):
        outs = []
        for i, d in enumerate(self.ds):
            outs.append(d(x if i == 0 else F.avg_pool2d(x, 2 ** i)))
        return outs


class SmallD(nn.Module):
    """Single-scale patch discriminator for 128 px crops (hand GAN): 128 -> 16x16 logits."""

    def __init__(self, cin=3):
        super().__init__()
        self.d = Patch(cin, ch=64, n=4)

    def forward(self, x): return [self.d(x)]


def d_hinge(D, real, fake):
    l = 0
    for a, b in zip(D(real), D(fake)):
        l = l + F.relu(1 - a).mean() + F.relu(1 + b).mean()
    return l


def g_hinge(D, fake):
    return sum(-o.mean() for o in D(fake))


