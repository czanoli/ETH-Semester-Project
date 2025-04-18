import os
import torch
import torch.nn as nn
import torchvision.transforms as T

from typing import List, Optional, Tuple, Dict
from torch.utils.hooks import RemovableHandle
from external.crocov2.models.croco import CroCoNet
from external.crocov2.models.dpt_block import DPTOutputAdapter

class CrocoFeatureExtractor(nn.Module):
    """
    Feature extractor that loads CROCOv2 weights from a .pth checkpoint.
    """
    def __init__(self, model_name: str, use_dpt: bool = False) -> None:
        super().__init__()

        if use_dpt:
            self.is_dpt = True
        else:
            self.is_dpt = False

        # Construct checkpoint name
        name_items = model_name.split("_")
        assert name_items[0] == "crocov2"
        self.checkpoint_name = "_".join(name_items[1:-1]) + ".pth"
        
        # Configure model type and paths
        if self.is_dpt:
            self.dpt_checkpoint_name = "crocostereo.pth"
        else:
            self.dpt_checkpoint_name = None
        
        # Get intermediate layer number
        key, val = name_items[-1].split("=")
        if key == "layer":
            self.layer_index = int(val)
            self.decoder_layer_index = None
            self.dpt_index = None
        elif key == "decoder":
            self.decoder_layer_index = int(val)
            self.layer_index = None
            if self.is_dpt:
                self.dpt_index = 3  # DPT index
            else:
                self.dpt_index = None
        else:
            self.layer_index = None
            self.decoder_layer_index = None
            self.dpt_index = None
        
        # Set up normalization
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        # Load the CROCOv2 base model
        self.croco_model = self._load_croco_base_model()
        self.croco_model.eval()
        
        # Set up DPT module if needed
        if self.is_dpt:
            self.dpt = self._load_dpt_module()
            self.dpt.eval()
        else:
            self.dpt = None

    def _load_croco_base_model(self) -> nn.Module:
        """
        Load CROCOv2 base model (encoder-decoder architecture).
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count()>0 else 'cpu')
        ckpt_path = f'croco_pretrained_models/{self.checkpoint_name}'
        
        print(f"Loading CROCOv2 base model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model = CroCoNet(**ckpt.get('croco_kwargs', {})).to(device)
        msg = model.load_state_dict(ckpt['model'], strict=True)
        print(msg)
        
        return model

    def _load_dpt_module(self) -> nn.Module:
        """
        Load DPT module from crocostereo checkpoint.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count()>0 else 'cpu')
        ckpt_path = f'external/crocov2/stereoflow_models/{self.dpt_checkpoint_name}'
        
        print(f"Loading DPT module from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Create DPT module
        # Set up hooks based on the CroCo model structure
        if hasattr(self.croco_model, 'dec_blocks'):
            # For encoder+decoder
            enc_depth = self.croco_model.enc_depth
            dec_depth = self.croco_model.dec_depth
            step = {8: 3, 12: 4, 24: 8}.get(dec_depth, dec_depth // 3)
            hooks_idx = [enc_depth + dec_depth - 1 - i*step for i in range(3, -1, -1)]
        else:
            # For encoder only
            enc_depth = self.croco_model.enc_depth
            step = enc_depth // 4
            hooks_idx = [enc_depth - 1 - i*step for i in range(3, -1, -1)]
        
        print(f"DPT hooks_idx: {hooks_idx}")
        
        # Get token dimensions for each hook
        dim_tokens = [
            self.croco_model.enc_embed_dim if hook < self.croco_model.enc_depth 
            else self.croco_model.dec_embed_dim 
            for hook in hooks_idx
        ]
        
        # Initialize DPT adapter
        dpt = DPTOutputAdapter(
            hooks=hooks_idx,
            output_width_ratio=1,
            num_channels=1
        ).to(device)
        
        # Initialize with appropriate token dimensions
        dpt.init(dim_tokens_enc=dim_tokens)
        
        # Load weights if available in checkpoint
        if 'dpt' in ckpt['model']:
            dpt_state_dict = {k.replace('dpt.', ''): v for k, v in ckpt['model'].items() if k.startswith('dpt.')}
            msg = dpt.load_state_dict(dpt_state_dict, strict=False)
            print(f"DPT module loaded: {msg}")
        
        return dpt

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process images through CroCo model and get features.
        Return:
          {
            "cls_tokens":   BxD
            "feature_maps": BxDxHxW
          }
        """
        # 1) Normalize
        images = self.normalize(images)
        
        # 2) Process through the model
        with torch.no_grad():
            # Get image size for proper token reshaping
            B, C, H, W = images.shape
            
            # First encoding
            enc_list_1, pos1, mask1 = self.croco_model._encode_image(
                images,
                do_mask=False,
                return_all_blocks=True
            )

            if self.layer_index is not None:
                # --- use encoder layer ---
                if self.layer_index >= len(enc_list_1):
                    print(f"Encoder layer index {self.layer_index} out of range, using last layer")
                    x = enc_list_1[-1]
                else:
                    print(f"Using encoder layer #{self.layer_index}")
                    x = enc_list_1[self.layer_index - 1]
                    x = self.croco_model.enc_norm(x)
                
                # Reshape from (B, N, C) => (B, C, H', W')
                B, N, C = x.shape
                h_p = w_p = int(N ** 0.5)
                feature_maps = x.reshape(B, h_p, w_p, C).permute(0, 3, 1, 2)


            elif self.decoder_layer_index is not None:
                enc_list_2, pos2, mask2 = self.croco_model._encode_image(
                    images,
                    do_mask=False,
                    return_all_blocks=True
                )
                
                dec_out_list = self.croco_model._decoder(
                    feat1=enc_list_1[-1],
                    pos1=pos1,
                    masks1=mask1,
                    feat2=enc_list_2[-1],
                    pos2=pos2,
                    return_all_blocks=True
                )
                
                if self.is_dpt:
                    # --- For DPT pipeline ---
                    # Combine encoder and decoder outputs for DPT
                    all_features = enc_list_1 + dec_out_list
                    
                    # Feed features to DPT module
                    dpt_outputs = self.dpt(all_features, (H, W))
                    
                    # Select the appropriate path based on dpt_index
                    paths = [dpt_outputs['path_4'], dpt_outputs['path_3'], 
                            dpt_outputs['path_2'], dpt_outputs['path_1']]
                    out = dpt_outputs['out']

                    #import torch
                    import cv2
                    import numpy as np

                    # Your depth tensor: shape [1, 1, 480, 480]
                    depth_tensor = out.squeeze().detach().cpu().numpy()  # shape becomes [480, 480]

                    # Normalize to 0–255
                    depth_normalized = cv2.normalize(depth_tensor, None, 0, 255, cv2.NORM_MINMAX)
                    depth_normalized = depth_normalized.astype(np.uint8)

                    # Apply colormap
                    #depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                    # Save the image
                    cv2.imwrite('debug/dpt_out_norm.png', depth_normalized)
                    
                    if self.dpt_index == "out":
                        print("Using DPT out")
                        feature_maps = out
                    elif self.dpt_index < 0 or self.dpt_index >= len(paths):
                        print(f"DPT index {self.dpt_index} out of range, using path_1 (highest resolution)")
                        feature_maps = paths[-1]  # Default to path_1 (highest resolution)
                    else:
                        feature_maps = paths[self.dpt_index]
                        print(f"Using DPT path_{4-self.dpt_index} features")
                    
                    # feature_maps already in (B, C, H', W') format
                
                else:
                    # --- use decoder ---
                    if self.decoder_layer_index >= len(dec_out_list):
                        print(f"Decoder layer index {self.decoder_layer_index} out of range, using last layer")
                        x = dec_out_list[-1]
                    else:
                        print(f"Using decoder layer #{self.decoder_layer_index}")
                        x = dec_out_list[self.decoder_layer_index]
                    
                    # Reshape from (B, N, C) => (B, C, H', W')
                    B, N, C = x.shape
                    h_p = w_p = int(N ** 0.5)
                    feature_maps = x.reshape(B, h_p, w_p, C).permute(0, 3, 1, 2)

        # Create dummy CLS tokens
        B, C, _, _ = feature_maps.shape
        cls_tokens = torch.zeros((B, C), device=feature_maps.device)
        
        # Return in the format FoundPose expects
        return {
            "cls_tokens": cls_tokens,
            "feature_maps": feature_maps
        }