"""Entrega do Desafio 2: ativacao quase-identidade calibrada."""
import math

import torch


def ativacao(x: torch.Tensor) -> torch.Tensor:
    """Ativacao elemento a elemento, sem depender do lote."""
    return 0.15 * (x + torch.tanh(x))


_gerador = torch.Generator().manual_seed(0)
_e_f2 = ativacao(torch.randn(1_000_000, generator=_gerador)).square().mean().item()


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    s2 = 1.0 / (fan_in * _e_f2)
    fator = 2.0 if camada == n_camadas else 0.8
    W.normal_(0.0, fator * math.sqrt(s2))
    b.zero_()
