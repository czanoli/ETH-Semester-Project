import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from typing import List, Optional, Tuple, Dict
from torch.utils.hooks import RemovableHandle
from external.crocov2.models.croco import CroCoNet
from external.crocov2.models.dpt_block import DPTOutputAdapter
from external.crocov2.stereoflow.test import _load_model_and_criterion
from external.crocov2.stereoflow.engine import tiled_pred

'''
# class used for combining the DPT module with decoder
class FeatureFusion(nn.Module):
    """
    Various methods to fuse decoder output with DPT features for CroCov2 architecture
    """
    def __init__(self, fusion_method='concat', hidden_dim=768): # 768 for the projector of the pose part to not raise error
        """
        Args:
            fusion_method: Method for feature fusion ('concat', 'cosine', 'attention', 'autoencoder', 'path_fusion')
            hidden_dim: Hidden dimension for certain fusion methods
        """
        super().__init__()
        self.fusion_method = fusion_method
        self.hidden_dim = hidden_dim
        
        if fusion_method == 'concat':
            # For concat we need to project both features to same dimension
            hidden_dim = 384 # because then the fused will be with C=768 so no error raised by the projector of the pose part
            self.decoder_proj = nn.Conv2d(768, hidden_dim, kernel_size=1)
            self.dpt_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        
        elif fusion_method == 'cosine':
            # Cosine similarity fusion
            self.decoder_proj = nn.Conv2d(768, hidden_dim, kernel_size=1)
            self.dpt_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
            self.weights = nn.Parameter(torch.ones(2))
        
        elif fusion_method == 'attention':
            # Cross-attention between features
            self.decoder_proj = nn.Conv2d(768, hidden_dim, kernel_size=1)
            self.dpt_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
            self.query_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            self.key_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            self.value_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            self.gamma = nn.Parameter(torch.zeros(1))
        
        elif fusion_method == 'autoencoder':
            # Encoder takes concatenated features
            self.decoder_proj = nn.Conv2d(768, hidden_dim, kernel_size=1)
            self.dpt_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
            
            # Autoencoder fusion
            self.encoder = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            # Decoder to reconstruct original features
            self.decoder = nn.Sequential(
                nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
            )
        
        elif fusion_method == 'path_fusion':
            # Use DPT path features for hierarchical fusion
            self.decoder_proj = nn.Conv2d(768, hidden_dim, kernel_size=1)
            
            # Project each path to hidden_dim
            self.path_projs = nn.ModuleList([
                nn.Conv2d(256, hidden_dim, kernel_size=1) for _ in range(4)
            ])
            # Fusion module for combining all features
            self.fusion_module = nn.Sequential(
                nn.Conv2d(5 * hidden_dim, 2 * hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
    
    def resize_features(self, x, target_size):
        """Resize features to target size"""
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, decoder_output, dpt_output, dpt_paths=None):
        """
        Fuse decoder output and DPT output based on the chosen method
        
        Args:
            decoder_output: Output from decoder block 6 [1, 768, 30, 30]
            dpt_output: Output from DPT module [1, 2, 352, 704]
            dpt_paths: List of intermediate DPT path features (optional)
        
        Returns:
            Fused feature representation
        """
        # Define target size for feature maps
        target_size = (decoder_output.shape[2], decoder_output.shape[3])
        
        # Resize DPT output to match decoder output size
        dpt_resized = self.resize_features(dpt_output, target_size)
        
        if self.fusion_method == 'concat':
            # Project to same dimension and concatenate
            decoder_feat = self.decoder_proj(decoder_output)
            dpt_feat = self.dpt_proj(dpt_resized)
            
            # Concatenate along channel dimension
            fused = torch.cat([decoder_feat, dpt_feat], dim=1)
            
            return fused
        
        elif self.fusion_method == 'cosine':
            # Project to same dimension
            decoder_feat = self.decoder_proj(decoder_output)
            dpt_feat = self.dpt_proj(dpt_resized)
            
            # Compute cosine similarity
            # Normalize features
            decoder_norm = F.normalize(decoder_feat, p=2, dim=1)
            dpt_norm = F.normalize(dpt_feat, p=2, dim=1)
            
            # Weighted cosine similarity fusion
            similarity = self.weights[0] * decoder_norm + self.weights[1] * dpt_norm
            return similarity
        
        elif self.fusion_method == 'attention':
            # Project to same dimension
            decoder_feat = self.decoder_proj(decoder_output)
            dpt_feat = self.dpt_proj(dpt_resized)
            
            # Cross-attention mechanism
            query = self.query_proj(decoder_feat)
            key = self.key_proj(dpt_feat)
            value = self.value_proj(dpt_feat)
            
            # Reshape for attention
            b, c, h, w = query.size()
            query = query.view(b, c, -1).permute(0, 2, 1)  # B, HW, C
            key = key.view(b, c, -1)  # B, C, HW
            value = value.view(b, c, -1).permute(0, 2, 1)  # B, HW, C
            
            # Compute attention
            attention = torch.bmm(query, key)  # B, HW, HW
            attention = F.softmax(attention, dim=-1)
            
            # Apply attention to values
            out = torch.bmm(attention, value)  # B, HW, C
            out = out.permute(0, 2, 1).view(b, c, h, w)  # B, C, H, W
            
            # Residual connection
            out = self.gamma * out + decoder_feat
            return self.output_proj(out)
        
        elif self.fusion_method == 'autoencoder':
            # Project to same dimension
            decoder_feat = self.decoder_proj(decoder_output)
            dpt_feat = self.dpt_proj(dpt_resized)
            
            # Concatenate features
            concat_feat = torch.cat([decoder_feat, dpt_feat], dim=1)
            
            # Encode to compressed representation
            encoded = self.encoder(concat_feat)
            
            # Compute reconstruction loss:
            # TODO
            # decoded = self.decoder(encoded)
            # recon_loss = F.mse_loss(decoded, concat_feat)
            
            return encoded
        
        elif self.fusion_method == 'path_fusion' and dpt_paths is not None:
            # Hierarchical fusion using DPT path features
            decoder_feat = self.decoder_proj(decoder_output)
            
            # Process and resize each path
            path_features = []
            for i, path in enumerate(dpt_paths):
                path_feat = self.path_projs[i](path)
                path_feat_resized = self.resize_features(path_feat, target_size)
                path_features.append(path_feat_resized)
            
            # Combine decoder feature with all path features
            all_features = [decoder_feat] + path_features
            concat_features = torch.cat(all_features, dim=1)
            
            # Final fusion
            fused = self.fusion_module(concat_features)
            return fused
        
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
'''

def fusion_module(decoder_output, dpt_pred, confidence=None, fusion_type='adaptive', scale_factor=None):
    """
    Fuses decoder block output with DPT prediction in a training-free approach.
    
    Args:
        decoder_output: Tensor of shape [B, 768, H_d, W_d] (e.g., [1, 768, 30, 30])
        dpt_pred: Tensor of shape [B, C_dpt, H_dpt, W_dpt] from DPT prediction 
        confidence: Tensor of shape [B, H_dpt, W_dpt] with confidence values (optional)
        fusion_type: Strategy for fusion ('adaptive', 'normalization', 'attention', 'direct')
        scale_factor: If provided, rescales output to this factor of decoder size
    
    Returns:
        fused_features: Feature map combining decoder and DPT information
    """
    # Get dimensions
    batch_size, decoder_channels, dec_h, dec_w = decoder_output.shape
    _, dpt_channels, dpt_h, dpt_w = dpt_pred.shape
    
    # Determine target size (default to decoder size)
    target_h, target_w = dec_h, dec_w
    if scale_factor is not None:
        target_h = int(dec_h * scale_factor)
        target_w = int(dec_w * scale_factor)
    
    # Resize DPT prediction to match decoder output size for processing
    dpt_resized = F.interpolate(dpt_pred, size=(dec_h, dec_w), mode='bilinear', align_corners=False)
    
    # If confidence is None but dpt_pred has a channel that is confidence
    if confidence is None and dpt_channels > 1:
        confidence = dpt_pred[:, -1:, :, :]  # Take last channel
        dpt_resized = dpt_resized[:, :-1, :, :]  # Remove confidence channel from prediction
    
    # If we have explicit confidence, resize it too
    if confidence is not None and confidence.dim() > 3:  # If confidence is [B, 1, H, W]
        conf_resized = F.interpolate(confidence, size=(dec_h, dec_w), mode='bilinear', align_corners=False)
        conf_values = conf_resized.squeeze(1)  # Remove channel dim if present
    elif confidence is not None:  # If confidence is [B, H, W]
        conf_resized = F.interpolate(confidence.unsqueeze(1), size=(dec_h, dec_w), mode='bilinear', align_corners=False)
        conf_values = conf_resized.squeeze(1)
    else:
        # Create uniform confidence if none provided
        conf_values = torch.ones(batch_size, dec_h, dec_w, device=decoder_output.device)
    
    # Projection for DPT features to match decoder dimensionality
    # Using 1x1 convolution equivalent with torch.einsum for training-free approach
    dpt_projected = torch.zeros_like(decoder_output)
    if dpt_resized.shape[1] > 1:  # If DPT has multiple channels
        # Repeat DPT channels to match decoder channels
        repeat_factor = decoder_channels // dpt_resized.shape[1]
        if repeat_factor > 0:
            dpt_projected = dpt_resized.repeat(1, repeat_factor, 1, 1)
            # If not exact division, pad the remaining channels
            if dpt_projected.shape[1] < decoder_channels:
                padding = decoder_channels - dpt_projected.shape[1]
                dpt_projected = torch.cat([
                    dpt_projected, 
                    dpt_resized[:, :padding, :, :]
                ], dim=1)
        else:
            # If DPT has more channels than needed, take the first decoder_channels
            dpt_projected = dpt_resized[:, :decoder_channels, :, :]
    else:
        # If DPT has single channel, broadcast it across all decoder channels
        dpt_projected = dpt_resized.expand(-1, decoder_channels, -1, -1)
    
    # Different fusion strategies
    if fusion_type == 'adaptive':
        print("\n!! ADAPTIVE !!\n")
        # Confidence-weighted fusion
        conf_expanded = conf_values.unsqueeze(1).expand_as(decoder_output)
        fused_features = decoder_output * (1 - conf_expanded) + dpt_projected * conf_expanded
    
    elif fusion_type == 'normalization':
        print("\n!! NORMALIZATION !!\n")
        # Channel-wise normalization and weighted combination
        decoder_norm = F.normalize(decoder_output, p=2, dim=1)
        dpt_norm = F.normalize(dpt_projected, p=2, dim=1)
        
        # Use confidence as weight between normalized features
        conf_expanded = conf_values.unsqueeze(1).expand_as(decoder_output)
        fused_features = decoder_norm * (1 - conf_expanded) + dpt_norm * conf_expanded
    
    elif fusion_type == 'attention':
        print("\n!! ATTENTION !!\n")
        # Compute a similarity map between decoder and DPT features
        decoder_flat = decoder_output.view(batch_size, decoder_channels, -1)
        dpt_flat = dpt_projected.view(batch_size, decoder_channels, -1)
        
        # Normalize features for dot product similarity
        decoder_norm = F.normalize(decoder_flat, p=2, dim=1)
        dpt_norm = F.normalize(dpt_flat, p=2, dim=1)
        
        # Compute attention weights (similarity between features)
        attention = torch.bmm(decoder_norm.transpose(1, 2), dpt_norm)  # [B, H*W, H*W]
        attention = F.softmax(attention, dim=2)
        
        # Apply attention
        attended_features = torch.bmm(dpt_flat, attention.transpose(1, 2))
        attended_features = attended_features.view(batch_size, decoder_channels, dec_h, dec_w)
        
        # Combine with original features
        conf_expanded = conf_values.unsqueeze(1).expand_as(decoder_output)
        fused_features = decoder_output * (1 - conf_expanded) + attended_features * conf_expanded
    
    elif fusion_type == 'direct':
        print("\n!! DIRECT !!\n")
        # Direct addition with confidence weighting
        fused_features = decoder_output + dpt_projected * conf_values.unsqueeze(1)
    
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    # Resize to target size if needed
    if (target_h, target_w) != (dec_h, dec_w):
        fused_features = F.interpolate(fused_features, size=(target_h, target_w), 
                                      mode='bilinear', align_corners=False)
    
    return fused_features


def tiled_fusion(decoder_output, dpt_pred, confidence=None, tile_size=704, overlap=0.9, fusion_type='adaptive'):
    """
    Performs tiled fusion for large feature maps to avoid memory issues.
    
    Args:
        decoder_output: Tensor of shape [B, 768, H_d, W_d]
        dpt_pred: Tensor of shape [B, C_dpt, H_dpt, W_dpt]
        confidence: Optional confidence map
        tile_size: Maximum tile size for processing
        overlap: Overlap between tiles (0-1)
        fusion_type: Type of fusion to apply
        
    Returns:
        Fused feature map
    """
    # Get dimensions
    batch_size, decoder_channels, dec_h, dec_w = decoder_output.shape
    
    # If small enough, process directly
    if dec_h <= tile_size and dec_w <= tile_size:
        return fusion_module(decoder_output, dpt_pred, confidence, fusion_type)
    
    # Calculate tile dimensions with overlap
    stride_h = int(tile_size * (1 - overlap))
    stride_w = int(tile_size * (1 - overlap))
    
    # Initialize output tensor and weight accumulator
    fused_output = torch.zeros_like(decoder_output)
    weight_accumulator = torch.zeros((batch_size, 1, dec_h, dec_w), device=decoder_output.device)
    
    # Process each tile
    for y in range(0, dec_h, stride_h):
        for x in range(0, dec_w, stride_w):
            # Calculate tile boundaries
            end_y = min(y + tile_size, dec_h)
            end_x = min(x + tile_size, dec_w)
            
            # Extract tile from decoder output
            decoder_tile = decoder_output[:, :, y:end_y, x:end_x]
            
            # Calculate corresponding region in DPT prediction
            y_ratio = dpt_pred.shape[2] / dec_h
            x_ratio = dpt_pred.shape[3] / dec_w
            
            dpt_y = int(y * y_ratio)
            dpt_x = int(x * x_ratio)
            dpt_end_y = int(end_y * y_ratio)
            dpt_end_x = int(end_x * x_ratio)
            
            dpt_tile = dpt_pred[:, :, dpt_y:dpt_end_y, dpt_x:dpt_end_x]
            
            # Extract confidence tile if provided
            conf_tile = None
            if confidence is not None:
                if confidence.dim() == 3:  # [B, H, W]
                    conf_tile = confidence[:, dpt_y:dpt_end_y, dpt_x:dpt_end_x]
                else:  # [B, 1, H, W]
                    conf_tile = confidence[:, :, dpt_y:dpt_end_y, dpt_x:dpt_end_x]
            
            # Process tile
            fused_tile = fusion_module(decoder_tile, dpt_tile, conf_tile, fusion_type)
            
            # Create weight mask for smooth blending (higher in center, lower at edges)
            h, w = end_y - y, end_x - x
            y_weights = torch.linspace(0, 1, h//2, device=decoder_output.device)
            y_weights = torch.cat([y_weights, torch.flip(y_weights, [0])]) if h % 2 == 0 else torch.cat([y_weights, torch.flip(y_weights[:-1], [0])])
            
            x_weights = torch.linspace(0, 1, w//2, device=decoder_output.device)
            x_weights = torch.cat([x_weights, torch.flip(x_weights, [0])]) if w % 2 == 0 else torch.cat([x_weights, torch.flip(x_weights[:-1], [0])])
            
            weight_mask = y_weights.view(-1, 1) * x_weights.view(1, -1)
            weight_mask = weight_mask.view(1, 1, h, w)
            
            # Apply weights and add to output
            fused_output[:, :, y:end_y, x:end_x] += fused_tile * weight_mask
            weight_accumulator[:, :, y:end_y, x:end_x] += weight_mask
    
    # Normalize by weights
    fused_output = fused_output / (weight_accumulator + 1e-8)
    
    return fused_output


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
                self.dpt_index = 'out'  # DPT index
                #self.feature_fusion = FeatureFusion(fusion_method="attention", hidden_dim=768)
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
        if not self.is_dpt:
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
                    tile_overlap = 0.97
                    use_gpu = torch.cuda.is_available() and torch.cuda.device_count()>0
                    device = torch.device('cuda:0' if use_gpu else 'cpu')
                    model, _, cropsize, with_conf, task, tile_conf_mode = _load_model_and_criterion('external/crocov2/stereoflow_models/crocostereo.pth', None, device)
                    im1 = images.to(device)
                    im2 = images.to(device)

                    # Resize image
                    #im1 = F.interpolate(im1, size=(2112, 2112), mode='bilinear', align_corners=False)
                    #im2 = F.interpolate(im2, size=(2112, 2112), mode='bilinear', align_corners=False)

                    with torch.inference_mode():
                        pred, _, c, preds = tiled_pred(model, None, im1, im2, None, conf_mode=tile_conf_mode, overlap=tile_overlap, crop=cropsize, with_conf=with_conf, return_time=False)

                    # Select the appropriate path based on dpt_index
                    #dpt_paths = [dpt_outputs['path_4'], dpt_outputs['path_3'], 
                            #dpt_outputs['path_2'], dpt_outputs['path_1']]
                    
                    #dpt_out = dpt_outputs['out']

                    # We also need the decoder layer output for later combination with the dpt output
                    if self.decoder_layer_index >= len(dec_out_list):
                        print(f"[DPT version] -- Decoder layer index {self.decoder_layer_index} out of range, using last layer")
                        x = dec_out_list[-1]
                    else:
                        print(f"[DPT version] -- Using decoder layer #{self.decoder_layer_index}")
                        x = dec_out_list[self.decoder_layer_index]
                    
                    # Reshape from (B, N, C) => (B, C, H', W')
                    B, N, C = x.shape
                    h_p = w_p = int(N ** 0.5)
                    dec_featmap = x.reshape(B, h_p, w_p, C).permute(0, 3, 1, 2)

                    # Combine decoder featmap with dpt module output
                    #feature_maps = self.feature_fusion(dec_featmap, pred, None)
                    
                    # Fuse decoder features with DPT prediction
                    feature_maps = fusion_module(
                        decoder_output=dec_featmap,
                        dpt_pred=pred,
                        confidence=c,
                        fusion_type='adaptive'
                    )

                    # --- [OLD] Start of dpt tests
                    #all_features = enc_list_1 + dec_out_list
                    
                    # Feed features to DPT module
                    #dpt_outputs = self.dpt(all_features, (H, W))
                    
                    # Select the appropriate path based on dpt_index
                    #paths = [dpt_outputs['path_4'], dpt_outputs['path_3'], 
                            #dpt_outputs['path_2'], dpt_outputs['path_1']]
                    #out = dpt_outputs['out']
                    
                    #import cv2
                    #import numpy as np

                    # Your depth tensor: shape [1, 1, 480, 480]
                    #depth_tensor = out.squeeze().detach().cpu().numpy()  # shape becomes [480, 480]

                    # Normalize to 0–255
                    #depth_normalized = cv2.normalize(depth_tensor, None, 0, 255, cv2.NORM_MINMAX)
                    #depth_normalized = depth_normalized.astype(np.uint8)

                    # Apply colormap
                    #depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                    # Save the image
                    #cv2.imwrite('debug/dpt_out_norm.png', depth_normalized)
                    #--- [OLD] End of dpt test
                    
                    # --- [OLD] Start of dpt test
                    #if self.dpt_index == "out":
                        #print("Using DPT out")
                        #feature_maps = out
                    #elif self.dpt_index < 0 or self.dpt_index >= len(paths):
                        #print(f"DPT index {self.dpt_index} out of range, using path_1 (highest resolution)")
                        #feature_maps = paths[-1]  # Default to path_1 (highest resolution)
                    #else:
                        #feature_maps = paths[self.dpt_index]
                        #print(f"Using DPT path_{4-self.dpt_index} features")
                    # --- [OLD] end of dpt test
                    
                    # feature_maps already in (B, C, H', W') format
                
                else:
                    # --- use only decoder ---
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