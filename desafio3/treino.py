"""Desafio 3 (GBC073) - treino configuravel e instrumentado.

Uso (um experimento):
    python treino.py --dados /caminho/dataset --otimizador adamw --lr 1e-3 \
                     --agenda warmcos --semente 0

O dataset pode ter train/<classe>/*.png e val/<classe>/*.png, ou
images/train/<classe>/*.png e images/val/<classe>/*.png.

Protocolo de dados (declarado, vale para TODAS as execucoes):
  - treino   = images/train (12000)
  - validacao = metade estratificada de images/val (1500) -> selecao de lr/otimizador/agenda
  - teste    = outra metade de images/val (1500) -> usado UMA vez, em run_all.py (etapa final)
  A divisao val/teste usa semente fixa (12345), independente da semente do treino.
"""
import argparse, json, math, os, time, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["normal", "scratch", "crack", "rust", "hole"]
# primeira, uma do meio e ultima camada (exigencia do enunciado)
CAMADAS = ["conv1.weight", "fc1.weight", "head.weight"]
OTIMIZADORES = ["sgdm", "sgdm_nesterov", "adam", "adamw", "muon", "rmsprop"]
AGENDAS = ["const", "cosine", "warmcos", "wsd"]
# weight decay padrao por otimizador (declarado; igual em todas as execucoes do mesmo otimizador)
WD_PADRAO = dict(sgdm=5e-4, sgdm_nesterov=5e-4, adam=0.0, adamw=0.01, muon=0.01, rmsprop=0.0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- dados
def carregar_dados(raiz, tam=64):
    """Le PNGs, reduz para tam x tam, normaliza para [-1,1] e guarda cache .npz."""
    cache = os.path.join(raiz, f"cache_{tam}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["xtr"], z["ytr"], z["xva"], z["yva"], z["xte"], z["yte"]
    from PIL import Image

    if all(os.path.isdir(os.path.join(raiz, split)) for split in ("train", "val")):
        pasta_splits = raiz
    elif all(os.path.isdir(os.path.join(raiz, "images", split)) for split in ("train", "val")):
        pasta_splits = os.path.join(raiz, "images")
    else:
        raise FileNotFoundError(
            f"Dataset invalido em '{raiz}': esperado train/<classe> e val/<classe> "
            "ou images/train/<classe> e images/val/<classe>."
        )

    def ler(split):
        xs, ys = [], []
        for ci, c in enumerate(CLASSES):
            arquivos = sorted(glob.glob(os.path.join(pasta_splits, split, c, "*.png")))
            if not arquivos:
                raise FileNotFoundError(
                    f"Nenhuma imagem PNG encontrada para a classe '{c}' em "
                    f"'{os.path.join(pasta_splits, split, c)}'."
                )
            for f in arquivos:
                im = Image.open(f).convert("L").resize((tam, tam), Image.BILINEAR)
                xs.append(np.asarray(im, dtype=np.uint8)); ys.append(ci)
        return np.stack(xs), np.array(ys, dtype=np.int64)

    xtr, ytr = ler("train")
    xv, yv = ler("val")
    rng = np.random.RandomState(12345)  # divisao val/teste fixa e estratificada
    iva, ite = [], []
    for ci in range(len(CLASSES)):
        idx = rng.permutation(np.where(yv == ci)[0])
        h = len(idx) // 2
        iva += list(idx[:h]); ite += list(idx[h:])
    iva, ite = np.array(sorted(iva)), np.array(sorted(ite))
    np.savez_compressed(cache, xtr=xtr, ytr=ytr, xva=xv[iva], yva=yv[iva], xte=xv[ite], yte=yv[ite])
    return xtr, ytr, xv[iva], yv[iva], xv[ite], yv[ite]


def para_tensor(x, y):
    x = (torch.from_numpy(x).float() / 255.0 - 0.5) / 0.5
    return x.unsqueeze(1).to(DEV), torch.from_numpy(y).to(DEV)


# ----------------------------------------------------------------- modelo
class CNN(nn.Module):
    """3 blocos conv+BN+ReLU+pool -> fc1 (oculta) -> head. Nomes batem com CAMADAS."""
    def __init__(self, w=32, n_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, w, 3, padding=1);      self.bn1 = nn.BatchNorm2d(w)
        self.conv2 = nn.Conv2d(w, 2 * w, 3, padding=1);  self.bn2 = nn.BatchNorm2d(2 * w)
        self.conv3 = nn.Conv2d(2 * w, 4 * w, 3, padding=1); self.bn3 = nn.BatchNorm2d(4 * w)
        self.fc1 = nn.Linear(4 * w * 4 * 4, 128)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        for c, b in ((self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)):
            x = F.max_pool2d(F.relu(b(c(x))), 2)
        x = F.adaptive_avg_pool2d(x, 4).flatten(1)
        return self.head(F.relu(self.fc1(x)))


# ----------------------------------------------------------------- Muon (implementacao propria)
def ortogonaliza(G, passos=5):
    """Newton-Schulz: aproxima U V^T de G = U S V^T (do enunciado)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    T = X.size(0) > X.size(1)
    if T:
        X = X.T
    for _ in range(passos):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if T else X


class Muon(torch.optim.Optimizer):
    """Muon: momento (Nesterov opcional) -> ortogonaliza -> passo. Vale para tensores >=2-D;
    kernels de convolucao entram como matriz achatada (C_out, C_in*kh*kw)."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                M = upd.reshape(upd.size(0), -1)
                O = ortogonaliza(M, g["ns_steps"]).reshape_as(p)
                escala = max(1.0, M.size(0) / M.size(1)) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(O, alpha=-g["lr"] * escala)


def montar_otimizadores(nome, model, lr, wd, lr_adamw=None):
    P = torch.optim
    if nome == "muon":
        muon_p = [p for n, p in model.named_parameters() if p.ndim >= 2 and not n.startswith("head")]
        ids = {id(p) for p in muon_p}
        adamw_p = [p for p in model.parameters() if id(p) not in ids]
        return [Muon(muon_p, lr=lr, weight_decay=wd),
                P.AdamW(adamw_p, lr=lr_adamw or 1e-3, weight_decay=wd)]
    ps = model.parameters()
    if nome == "sgdm":          return [P.SGD(ps, lr=lr, momentum=0.9, weight_decay=wd)]
    if nome == "sgdm_nesterov": return [P.SGD(ps, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)]
    if nome == "adam":          return [P.Adam(ps, lr=lr, weight_decay=wd)]
    if nome == "adamw":         return [P.AdamW(ps, lr=lr, weight_decay=wd)]
    if nome == "rmsprop":       return [P.RMSprop(ps, lr=lr, weight_decay=wd)]
    raise ValueError(nome)


def multiplicador(agenda, T):
    """Fator multiplicativo do lr no passo t (0..T-1). Mesma funcao para todos os otimizadores."""
    w = max(1, int(0.05 * T))
    def cos(t, ini, fim):
        return 0.5 * (1 + math.cos(math.pi * (t - ini) / max(1, fim - ini)))
    if agenda == "const":    return lambda t: 1.0
    if agenda == "cosine":   return lambda t: cos(t, 0, T)
    if agenda == "warmcos":  return lambda t: (t + 1) / w if t < w else cos(t, w, T)
    if agenda == "wsd":  # warmup 5%, estavel ate 70%, decaimento linear ate 0
        d0 = int(0.7 * T)
        return lambda t: (t + 1) / w if t < w else (1.0 if t < d0 else max(0.0, (T - t) / max(1, T - d0)))
    raise ValueError(agenda)


# ----------------------------------------------------------------- avaliacao
@torch.no_grad()
def avaliar(model, x, y, bs=500):
    model.eval()
    L, A = 0.0, 0
    for i in range(0, len(x), bs):
        o = model(x[i:i + bs])
        L += F.cross_entropy(o, y[i:i + bs], reduction="sum").item()
        A += (o.argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return L / len(x), A / len(x)


# ----------------------------------------------------------------- treino
def run_id(c):
    lra = f"_la{c['lr_adamw']:g}" if c["otimizador"] == "muon" else ""
    return f"{c['otimizador']}_lr{c['lr']:g}{lra}_{c['agenda']}_s{c['semente']}"


def treinar(c, dados=None, pasta="."):
    """c: dict com dados, otimizador, lr, lr_adamw, agenda, semente, epocas, bs, k, clip, wd."""
    c = dict(dict(epocas=6, bs=64, k=10, clip=1.0, wd=None, lr_adamw=None, tam=64), **c)
    if c["wd"] is None:
        c["wd"] = WD_PADRAO[c["otimizador"]]
    if dados is None:
        dados = carregar_dados(c["dados"], c["tam"])
    xtr, ytr = para_tensor(dados[0], dados[1]); xva, yva = para_tensor(dados[2], dados[3])
    torch.manual_seed(c["semente"]); np.random.seed(c["semente"])
    treino_loader = DataLoader(
        TensorDataset(xtr, ytr), batch_size=c["bs"], shuffle=True, drop_last=True,
        generator=torch.Generator(device="cpu").manual_seed(c["semente"]),
    )
    model = CNN().to(DEV)
    opts = montar_otimizadores(c["otimizador"], model, c["lr"], c["wd"], c["lr_adamw"])
    T = c["epocas"] * (len(xtr) // c["bs"])           # mesmo orcamento (passos) para todos
    f = multiplicador(c["agenda"], T)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, f) for o in opts]
    params = list(model.parameters()); named = dict(model.named_parameters())
    clipmax = c["clip"] if c["clip"] > 0 else float("inf")

    log, t, divergiu, t0 = [], 0, False, time.time()
    passos_ep = len(xtr) // c["bs"]
    while t < T and not divergiu:
        for xb, yb in treino_loader:
            if t >= T:
                break
            loss = F.cross_entropy(model(xb), yb)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            g = clip_grad = torch.nn.utils.clip_grad_norm_(params, clipmax)  # norma ANTES do corte
            registrar = (t % c["k"] == 0)
            if registrar:
                with torch.no_grad():
                    th = torch.sqrt(sum((p ** 2).sum() for p in params))
                    gl = {n: named[n].grad.norm().item() for n in CAMADAS if named[n].grad is not None}
                    antes = {n: named[n].detach().clone() for n in CAMADAS}
            for o in opts:
                o.step()
            for s in scheds:
                s.step()
            if registrar:
                with torch.no_grad():
                    rho = {n: ((named[n] - antes[n]).norm() / (antes[n].norm() + 1e-12)).item() for n in CAMADAS}
                reg = dict(passo=t, perda=loss.item(), grad=float(g), theta=th.item(),
                           lr=opts[0].param_groups[0]["lr"])
                reg.update({f"grad/{n}": v for n, v in gl.items()})
                reg.update({f"rho/{n}": v for n, v in rho.items()})
                log.append(reg)
            if not (math.isfinite(loss.item()) and math.isfinite(float(g))):
                divergiu = True
                log.append(dict(passo=t, perda=loss.item(), grad=float(g), divergiu=1))
                break
            t += 1
            if t % passos_ep == 0:  # validacao periodica (fim de cada epoca)
                vl, va = avaliar(model, xva, yva)
                log.append(dict(passo=t, val_loss=vl, val_acc=va))
    if divergiu:
        vl, va = float("nan"), float("nan")
    else:
        vl, va = avaliar(model, xva, yva)  # metrica final = fim do treino (sem "melhor epoca")
    rid = run_id(c)
    os.makedirs(os.path.join(pasta, "logs"), exist_ok=True)
    pd.DataFrame(log).to_csv(os.path.join(pasta, "logs", rid + ".csv"), index=False)
    torch.save(model.state_dict(), os.path.join(pasta, "logs", rid + ".pt"))
    return dict(run_id=rid, otimizador=c["otimizador"], lr=c["lr"],
                lr_adamw=c["lr_adamw"] if c["otimizador"] == "muon" else "", agenda=c["agenda"],
                semente=c["semente"], passos=T, bs=c["bs"], wd=c["wd"], clip=c["clip"],
                val_acc=va, val_loss=vl, divergiu=int(divergiu), tempo_s=round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--otimizador", choices=OTIMIZADORES, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--lr_adamw", type=float, default=None, help="so para muon")
    ap.add_argument("--agenda", choices=AGENDAS, default="warmcos")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--clip", type=float, default=1.0)
    a = vars(ap.parse_args())
    print(json.dumps(treinar(a), indent=1))