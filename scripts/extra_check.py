import sys, torch, numpy as np
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from ouro_model import load_ouro
from routed_ouro import RoutedOuro, fineweb_chunks
from structured_router import round_states, option_ce
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("ByteDance/Ouro-1.4B")
ouro = load_ouro("ByteDance/Ouro-1.4B", torch.bfloat16).cuda().eval()
sd = RoutedOuro(ouro, use_router=False).cuda()
chunks = fineweb_chunks(tok, 1024, 32)
D = []
for i in range(0, 32, 8):
    ids = torch.tensor(chunks[i:i+8], device="cuda")
    H, X = round_states(ouro, ids, 12, sd)
    ce = option_ce(ouro, H, X, ids)          # B,T,R,2
    D.append((ce[..., 1] - ce[..., 0]).reshape(-1, 4).float().cpu())
d = torch.cat(D).numpy()                     # extra minus plain, per token per round (negative = extra helps)
base = None
print("mean change from re-running half a round, per round:", np.round(d.mean(0), 4))
print("share of tokens it helps (< 0):           ", np.round((d < 0).mean(0), 3))
for lam in (0.0, 0.0005, 0.001):
    gain = np.minimum(d + lam * 12, 0)       # hindsight: use extra only where it pays its compute
    print(f"hindsight gain if used only where it pays (lam={lam}):", np.round(-gain.mean(0), 4), " used on", np.round((gain < 0).mean(0), 3))
