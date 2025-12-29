import math, torch, torch.nn as nn
from torch.nn import functional as F

# ---------- helper layers ----------
class DWConvBlock(nn.Module):
    """ConvNeXt-style depth-wise conv block"""
    def __init__(self, c, expansion=4, drop=0.):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, 1, 3, groups=c)
        self.pw1 = nn.Linear(c, c*expansion)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(c*expansion, c)
        self.drop = nn.Dropout(drop)
        self.gamma = nn.Parameter(1e-6*torch.ones(c))

    def forward(self, x):
        u = self.dw(x)
        u = u.flatten(2).transpose(1,2)          # (B,HW,C)
        u = self.pw2(self.drop(self.act(self.pw1(u))))
        u = self.gamma * u
        u = u.transpose(1,2).view_as(x)
        return x + u                              # residual

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim*mlp_ratio))
        self.fc2 = nn.Linear(int(dim*mlp_ratio), dim)
        self.act = nn.GELU(); self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

# ---------- transformer encoder ----------
class Block(nn.Module):
    def __init__(self, dim, heads=8, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ---------- CBAM ----------
class CBAM(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim//r), nn.ReLU(), nn.Linear(dim//r, dim))
        self.spatial = nn.Conv2d(2,1,7,1,3)
    def forward(self, x_c, fmap_spatial):
        # Channel attention
        avg = x_c.mean(1, keepdim=True); max_ = x_c.max(1, keepdim=True)[0]
        ch = torch.sigmoid(self.mlp(avg.squeeze()) + self.mlp(max_.squeeze())).unsqueeze(2).unsqueeze(3)
        fmap_spatial = fmap_spatial * ch
        # Spatial
        avg = fmap_spatial.mean(1,keepdim=True); max_ = fmap_spatial.max(1,keepdim=True)[0]
        sp = torch.sigmoid(self.spatial(torch.cat([avg,max_],1)))
        fmap_spatial = fmap_spatial * sp
        return fmap_spatial.flatten(2).mean(-1)   # GAP

# ---------- main model ----------
class HAMFusionNet(nn.Module):
    def __init__(self, num_classes=5, vit_dim=384, vit_heads=6):
        super().__init__()
        # ---- stem ----
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2), nn.BatchNorm2d(64), nn.SiLU(),
            nn.MaxPool2d(3,2,1))                      # -> 64×32×32

        # ---- branch-A : CNN ----
        self.cnn_stage = nn.Sequential(
            DWConvBlock(64),                          # 64×32×32
            DWConvBlock(64),                          # 64×32×32
            DWConvBlock(64))                          # 64×32×32

        # ---- branch-B : ViT ----
        patch = 16; self.patch_dim = 64*patch*patch   # 64×16×16 tokens
        self.proj  = nn.Conv2d(64, vit_dim, patch, patch)  #  vit tokens  + 1 class
        n_tokens = (128//patch)*(128//patch)
        self.class_tok = nn.Parameter(torch.zeros(1,1,vit_dim))
        self.pos_emb   = nn.Parameter(torch.zeros(1,1+n_tokens,vit_dim))
        self.transformer = nn.Sequential(*[Block(vit_dim, vit_heads) for _ in range(4)])

        # ---- cross-fusion (ViT queries, CNN keys/values) ----
        self.cross_attn = nn.MultiheadAttention(vit_dim, vit_heads, batch_first=True)

        # ---- attention pooling & head ----
        self.cbam = CBAM(vit_dim+64)
        self.fc   = nn.Linear(vit_dim+64, num_classes)
        nn.init.trunc_normal_(self.pos_emb, std=.02)

    def forward(self, x):
        x = self.stem(x)                      # (B,64,32,32)
        cnn_feat = self.cnn_stage(x)          # (B,64,32,32)

        # --- ViT branch ---
        v = self.proj(x)                      # (B,dim,8,8) if patch=16 for 128 in
        v = v.flatten(2).transpose(1,2)       # (B,N,dim)
        cls = self.class_tok.expand(x.size(0),-1,-1)
        v = torch.cat([cls, v], 1) + self.pos_emb
        v = self.transformer(v)               # (B,N+1,dim)

        # --- Cross-fusion ---
        cnn_tokens = cnn_feat.flatten(2).transpose(1,2)  # (B,Hc*Wc,Cc)
        fused_cls,_ = self.cross_attn(v[:,0:1,:], cnn_tokens, cnn_tokens)  # query=cls
        fused = torch.cat([fused_cls.squeeze(1), cnn_feat.flatten(2).mean(-1)], 1) # (B,dim+64)

        # --- CBAM & head ---
        pooled = self.cbam(fused.unsqueeze(2), cnn_feat) # returns (B,dim+64)
        return self.fc(pooled)