# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae

import torch

from segmodel.unext.model import UNext_S
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 num_classes=1):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE UNeXt specifics
        self.seg_model = UNext_S(num_classes=num_classes)
        for p in self.seg_model.parameters():
            p.requires_grad = False
        state_dict = torch.load(r"/path/to/pretrained/unext_s.pth", 
                                map_location='cpu')
        self.seg_model.load_state_dict(state_dict)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        if self.norm_pix_loss:
            print("Warning: norm_pix_loss=True is enabled. This will distort recon_imgs "
                  "passed to seg_model and likely break the cycle loss logic.")

        self.initialize_weights()

    def train(self, mode=True):
        super().train(mode)
        self.seg_model.eval()
        return self

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches ** 0.5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** 0.5),
                                                    cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify_imgs(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p * p * 3))
        return x

    def unpatchify_images(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def patchify_filter(self, filters):
        p = self.patch_embed.patch_size[0]
        assert filters.shape[2] == filters.shape[3] and filters.shape[2] % p == 0

        h = w = filters.shape[2] // p
        x = filters.reshape(shape=(filters.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(filters.shape[0], h * w, p * p * 1))
        return x

    def guided_masking(self, x, guidance_map, mask_ratio, epoch_ratio, alpha_0, alpha_T):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        num_mask = L - len_keep

        alpha_t = alpha_0 + epoch_ratio * (alpha_T - alpha_0)
        target_guided_num = int(num_mask * alpha_t)

        pf = self.patchify_filter(guidance_map)  # (N, L, p^2)
        patch_vessel_pixels = torch.sum(pf, dim=2)  # (N, L)
        vessel_sum = patch_vessel_pixels.sum(dim=1)  # (N,)

        eps = 1e-6
        denom = vessel_sum.unsqueeze(-1) + eps  # (N, 1)
        patch_probs = patch_vessel_pixels / denom  # (N, L)

        ids_shuffle = torch.zeros((N, L), dtype=torch.long, device=x.device)

        zero_mask = (vessel_sum == 0)
        nonzero_mask = ~zero_mask

        all_idx = torch.arange(L, device=x.device).unsqueeze(0).repeat(N, 1)  # (N, L)

        if nonzero_mask.any():
            rows_nz = nonzero_mask.nonzero(as_tuple=True)[0]
            for i in rows_nz:
                valid_candidates = (patch_probs[i] > 0).sum().item()
                actual_guided_num = min(target_guided_num, valid_candidates)

                if actual_guided_num > 0:
                    guided_idx_i = torch.multinomial(
                        patch_probs[i],
                        num_samples=actual_guided_num,
                        replacement=False
                    )
                else:
                    guided_idx_i = torch.zeros(0, dtype=torch.long, device=x.device)

                mask_guided_bool_i = torch.zeros(L, dtype=torch.bool, device=x.device)
                if guided_idx_i.numel() > 0:
                    mask_guided_bool_i[guided_idx_i] = True
                
                rem_idx_all_i = all_idx[i][~mask_guided_bool_i]

                num_rand_i = num_mask - guided_idx_i.numel()
                
                if num_rand_i > 0:
                    rand_weight = torch.rand(rem_idx_all_i.shape[0], device=x.device)
                    perm = rem_idx_all_i[torch.argsort(rand_weight)]
                    random_idx_i = perm[:num_rand_i]
                else:
                    random_idx_i = torch.zeros(0, dtype=torch.long, device=x.device)

                mask_idx_i = torch.cat([guided_idx_i, random_idx_i], dim=0)  # (num_mask,)

                keep_bool_i = torch.ones(L, dtype=torch.bool, device=x.device)
                keep_bool_i[mask_idx_i] = False
                keep_idx_i = all_idx[i][keep_bool_i]  # (len_keep,)

                ids_shuffle[i, :] = torch.cat([keep_idx_i, mask_idx_i], dim=0)

        if zero_mask.any():
            rows_z = zero_mask.nonzero(as_tuple=True)[0]
            for j in rows_z:
                perm = torch.randperm(L, device=x.device)
                ids_shuffle[j, :] = perm

        ids_restore = torch.argsort(ids_shuffle, dim=1)  # (N, L)

        ids_keep = ids_shuffle[:, :len_keep].unsqueeze(-1).repeat(1, 1, D)  # (N, len_keep, D)
        x_masked = torch.gather(x, dim=1, index=ids_keep)  # (N, len_keep, D)

        mask = torch.ones((N, L), device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, guidance_map, mask_ratio, epoch_ratio, alpha_0, alpha_T):
        x = self.patch_embed(x)  # (N, L, D)

        x = x + self.pos_embed[:, 1:, :]  # (N, L, D)

        x, mask, ids_restore = self.guided_masking(x, guidance_map, mask_ratio, epoch_ratio, alpha_0, alpha_T)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # (1, 1, D)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)      # (N, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)                  # (N, L+1, D)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)  # (N, L+1, decoder_embed_dim)

        B, N_tokens, D_dec = x.shape  # N_tokens = L_keep+1
        x_tokens = x[:, 1:, :]  # (N, L_keep, D_dec)
        N_orig = ids_restore.shape[1]
        num_mask_tokens = N_orig + 1 - N_tokens
        mask_tokens = self.mask_token.repeat(B, num_mask_tokens, 1)  # (N, num_mask_tokens, D_dec)
        x_ = torch.cat([x_tokens, mask_tokens], dim=1)  # (N, N_orig, D_dec)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))  # (N, L, D_dec)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # (N, L+1, D_dec)

        x = x + self.decoder_pos_embed  # (N, L+1, D_dec)

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)  # (N, L+1, p*p*3)
        x = x[:, 1:, :]  # (N, L, p*p*3)
        return x

    def forward_loss_rec(self, imgs, pred, mask):
        target = self.patchify_imgs(imgs)  # (N, L, p*p*3)
        mean = None
        var = None
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        loss = (pred - target) ** 2  # (N, L, p*p*3)
        loss = loss.mean(dim=-1)    # (N, L)

        loss = (loss * mask).sum() / mask.sum()
        return loss, mean, var

    def forward(self, imgs, filters,
                mask_ratio=0.75, epoch_ratio=0, alpha_0=0, alpha_T=0.5):
                
        self.seg_model.eval()
        with torch.no_grad():
            imgs_seg = self.seg_model(imgs)
            imgs_probs = torch.sigmoid(imgs_seg)
        
        guidance_map = 0.5 * filters + 0.5 * imgs_probs

        latent, mask, ids_restore = self.forward_encoder(
            imgs, guidance_map, mask_ratio, epoch_ratio, alpha_0, alpha_T
        )  # latent: (N, L+1, D)

        pred = self.forward_decoder(latent, ids_restore)  # (N, L, p*p*3)
        rec_loss, mean, var = self.forward_loss_rec(imgs, pred, mask)

        if self.norm_pix_loss and mean is not None and var is not None:
            pred_restored = pred * (var + 1.e-6) ** 0.5 + mean
        else:
            pred_restored = pred

        recon_imgs = self.unpatchify_images(pred_restored)
        
        pred_seg = self.seg_model(recon_imgs)

        cycle_loss = F.binary_cross_entropy_with_logits(pred_seg, imgs_probs.detach())

        loss = rec_loss + cycle_loss

        return loss, rec_loss.item(), cycle_loss.item(), pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks