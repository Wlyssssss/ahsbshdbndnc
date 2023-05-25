import torch.nn as nn
from ldm.modules.encoders.modules import OpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder
from ldm.util import instantiate_from_config
import torch
from taming.models.vqgan import VQModelInterfaceEncoder, VQModel
from ldm.modules.attention import SpatialTransformer
from ldm.modules.attention import  Normalize, BasicTransformerBlock#, exists
from ldm.modules.diffusionmodules.util import zero_module, identity_init_fc, conv_nd
from einops import rearrange
# from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self



def make_zero_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return zero_module(conv_nd(2, in_channels, out_channels, kernel_size, stride=stride, padding=padding))


class SpatialTransformer_v2(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        # change:
        # if exists(context_dim) and not isinstance(context_dim, list):
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels)) # change: switch
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class trans_glyph_emb(nn.Module):
    def __init__(self, 
        type = "fc", # "conv", "attn"
        input_dim = 256,
        out_dim = 1024,
        # fc
        fc_init = "zero",
        # conv/attn
        conv_ks = 3,
        conv_pad = 1,
        conv_stride = 1,
        # attn
        ch = 512, # 1024 
        num_heads = 8, # 16
        dim_head = 64,
        use_linear_in_transformer = True,
        use_checkpoint = False, #True,
    ):
        super().__init__()
        
        if type == "fc":
            self.model = torch.nn.Linear(input_dim, out_dim)
            if fc_init == "zero":
                self.model = zero_module(self.model)
            elif fc_init == "identity":
                self.model = identity_init_fc(self.model)
        elif type == "conv":
            self.model = make_zero_conv(input_dim, out_dim, conv_ks, stride = conv_stride, padding = conv_pad)
        elif type == "attn":
            model = [
                # nn.Conv2d(input_dim, ch, 3, stride = 1, padding = 1),
                nn.Conv2d(input_dim, ch, conv_ks, stride = conv_stride, padding = conv_pad),
                SpatialTransformer_v2( #SpatialTransformer(
                                    ch, num_heads, dim_head, depth=1, context_dim=None, #ch,
                                    disable_self_attn=False, use_linear=use_linear_in_transformer,
                                    use_checkpoint=use_checkpoint, # False if the context is None
                                ),
                make_zero_conv(ch, out_dim, 1, stride = 1, padding = 0)
                # make_zero_conv(ch, out_dim, conv_ks, stride = conv_stride, padding = conv_pad)
            ]
            self.model = nn.Sequential(*model)
        self.model_type = type
    
    def forward(self, x):
        if self.model_type == "fc":
            # b, c, h, w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
            x = self.model(x)
            # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            # return x
        else:
            x = self.model(x)
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        return x
    


class glyph_control(nn.Module):
    def __init__(self,
    image_encoder = "CLIP", # "VQGAN"
    image_encoder_config = None,
    fuse_way = "concat",
    load_text_encoder = False,
    text_encoder_config = None,
    freeze_image_encoder = True,
    trans_emb = False,
    trans_emb_config = None,
    # use_fp16 = False,
    ):
        super().__init__()
        if image_encoder_config is not None:
            image_encoder_config.params.freeze = freeze_image_encoder
            self.image_encoder = instantiate_from_config(image_encoder_config)
        else:
            if image_encoder == "CLIP":
                self.image_encoder =  OpenCLIPImageEmbedder(freeze=freeze_image_encoder)
            elif image_encoder == "VQGAN":
                print("VQGAN glyph image encoder is missing config")
                raise ValueError
            else:
                print("Other types of glyph image encoder are not supported")
                raise ValueError

        if freeze_image_encoder:
            self.freeze_imenc()
        self.freeze_image_encoder = freeze_image_encoder
        self.image_encoder_type = image_encoder


        if load_text_encoder:
            if text_encoder_config is None:
                self.text_encoder = FrozenOpenCLIPEmbedder()
            else:
                self.text_encoder = instantiate_from_config(text_encoder_config)
        self.fuse_way = fuse_way
        # self.dtype = torch.float16 if use_fp16 else torch.float32
        if trans_emb:
            if trans_emb_config is not None:
                self.trans_glyph_emb_model = instantiate_from_config(trans_emb_config)
            else:
                self.trans_glyph_emb_model = trans_glyph_emb()
        else:
            self.trans_glyph_emb_model = None
        
    def freeze_imenc(self):
        self.image_encoder = self.image_encoder.eval()
        self.image_encoder.train = disabled_train
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, glyph_image, text = None, text_embed = None):
        clgim_num_list = [img.shape[0] for img in glyph_image]
        # image_embeds = self.image_encoder(torch.concat(glyph_image, dim=0))
        gim_concat = torch.concat(glyph_image, dim=0)
        image_embeds = self.image_encoder(gim_concat)
        if self.trans_glyph_emb_model is not None:
            image_embeds = self.trans_glyph_emb_model(image_embeds)
        image_embeds = torch.split(image_embeds, clgim_num_list)
        max_image_tokens = max(clgim_num_list)
        pad_image_embeds = []
        for image_embed in image_embeds:
            if image_embed.shape[0] < max_image_tokens:
                image_embed = torch.concat([
                    image_embed,
                    torch.zeros(
                    (max_image_tokens - image_embed.shape[0], *image_embed.shape[1:]), device=image_embed.device, dtype=image_embed.dtype, # add dtype
                    )], dim=0
                )
            pad_image_embeds.append(image_embed)
        pad_image_embeds = torch.stack(pad_image_embeds, dim = 0)
        if text_embed is None:
            assert self.text_encoder, text is not None
            text_embed = self.text_encoder(text)
        if self.fuse_way == "concat":
            assert pad_image_embeds.shape[-1] == text_embed.shape[-1]
            if len(pad_image_embeds.shape) == 4:
                b, _, _ , embdim = pad_image_embeds.shape 
                pad_image_embeds = pad_image_embeds.view(b, -1, embdim)
            out_embed = torch.concat([text_embed, pad_image_embeds], dim= 1)
            print("concat glyph_embed with text_embed:", out_embed.shape)
            return out_embed
        else:
            raise ValueError("Not support other fuse ways for now!")
