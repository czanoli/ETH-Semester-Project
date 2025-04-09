import os
import torch
import torch.nn as nn
import torchvision.transforms as T

from typing import List, Optional, Tuple, Dict
from torch.utils.hooks import RemovableHandle
from external.crocov2.models.croco import CroCoNet

class CrocoFeatureExtractor(nn.Module):
    """
    Feature extractor that loads CROCOv2 weights from a .pth checkpoint.
    """
    def __init__(self, model_name: str) -> None:
        super().__init__()

        # Construct checkpoint name
        name_items = model_name.split("_")
        assert name_items[0] == "crocov2"
        self.checkpoint_name = "_".join(name_items[1:-1]) + ".pth"
        
        # Get intermediate layer number
        key, val = name_items[-1].split("=")
        if key == "layer":
            self.layer_index = int(val)
            self.decoder_layer_index = None
            self.dpt_index = None
        elif key == "decoder":
            self.decoder_layer_index = int(val)
            self.layer_index = None
            self.dpt_index = None
        elif key == "dpt":
            self.dpt_index = int(val)
            self.layer_index = None
            self.decoder_layer_index = None
        else:
            self.layer_index = None
            self.decoder_layer_index = None
            self.dpt_index = None

        # 1) normalization (as with DINOv2) 
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        # 2) Load the actual CROCOv2 model
        self.model = self._load_crocov2_model_from_pth()
        self.model.eval()

    def _load_crocov2_model_from_pth(self) -> nn.Module:
        """
        Load CROCOv2 checkpoint.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count()>0 else 'cpu')
        ckpt = torch.load(f'croco_pretrained_models/{self.checkpoint_name}', 'cpu')
        model = CroCoNet( **ckpt.get('croco_kwargs',{})).to(device)
        msg = model.load_state_dict(ckpt['model'], strict=True)
        print("Loading CROCOv2 pretrained model...")
        print(msg)
        return model

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        In FoundPose, we only pass a single image (Bx3xHxW) at a time.
        Return:
          {
            "cls_tokens":   BxD
            "feature_maps": BxDxHxW
          }
        """
        # 1) Normalize
        images = self.normalize(images)

        # 2) Get patch embeddings
        with torch.no_grad():
            # First encoding
            enc_list_1, pos1, mask1 = self.model._encode_image(
                images,
                do_mask=False,
                return_all_blocks=True
            )
            # Use encoder
            if (self.decoder_layer_index is None) and (self.dpt_index is None) and (self.layer_index is not None):
                if self.layer_index is None or self.layer_index >= len(enc_list_1):
                    print(" !!!! Getting from CroCov2, output of last encoder layer")
                    x = enc_list_1[-1]
                else:
                    print(f" !!!! Getting from CroCov2, output of encoder layer #{self.layer_index}")
                    x = enc_list_1[self.layer_index - 1]
                    x = self.model.enc_norm(x)
            
            # Use decoder
            elif (self.decoder_layer_index is not None) and (self.layer_index is None) and (self.dpt_index is None):
                # Second encoding for decoder input (same image)
                enc_list_2, pos2, mask2 = self.model._encode_image(
                    images,
                    do_mask=False,
                    return_all_blocks=True
                )

                dec_out_list = self.model._decoder(
                    feat1=enc_list_1[-1],  # image1 features
                    pos1=pos1,
                    masks1=mask1,
                    feat2=enc_list_2[-1],  # image2 features
                    pos2=pos2,
                    return_all_blocks=True
                )

                if self.decoder_layer_index >= len(dec_out_list):
                    print(f" !!!! self.decoder_layer_index is out of range --> len(dec_out_list): {len(dec_out_list)}")
                    print(" !!!! Getting from CroCov2, output of last decoder layer")
                    x = dec_out_list[-1]
                else:
                    print(f" !!!! Getting from CroCov2, output of decoder layer #{self.decoder_layer_index}")
                    x = dec_out_list[self.decoder_layer_index]
            
            # use dpt module
            elif (self.decoder_layer_index is None) and (self.layer_index is None) and (self.dpt_index is not None):
                self.model.dpt_adapter(encoder_tokens, image_size=(images.shape[0], images.shape[1]), return_features=True)
                
            
            else:
                print(f" !!!! Everything is None !!!! ")

            '''
            #debug_shapes = False
            if debug_shapes:
                # 1) Encode the "first" image
                enc_list_1, pos1, mask1 = self.model._encode_image(
                    images, 
                    do_mask=False, 
                    return_all_blocks=True
                )

                # Print shapes for each encoder block
                print("\n\n------------------------")
                for i, x_block in enumerate(enc_list_1):
                    B, N, C = x_block.shape
                    h_p = w_p = int(N ** 0.5)
                    print(f"Encoder block {i} output: (B={B}, N={N}, C={C}) => (B, C, {h_p}, {w_p})")
                
                # 2) Encode the "second" image (same image again), just to feed the decoder
                enc_list_2, pos2, mask2 = self.model._encode_image(
                    images, 
                    do_mask=False, 
                    return_all_blocks=True
                )

                # 3) Call the decoder on the final enc outputs (from block -1)
                dec_out_list = self.model._decoder(
                    feat1=enc_list_1[-1],  # final features from "image1"
                    pos1=pos1,
                    masks1=mask1,         
                    feat2=enc_list_2[-1], # final features from "image2"
                    pos2=pos2,
                    return_all_blocks=True
                )

                # 4) Print decoder block shapes
                print("------------------------")
                for i, dec_block_out in enumerate(dec_out_list):
                    B, N, C = dec_block_out.shape
                    print(f"Decoder block {i} output: (B={B}, N={N}, C={C}) => (B, C, {h_p}, {w_p})")
                print("------------------------\n\n")

            else:
                # out_list is a list of length enc_depth (12)
                # out_list[0] is the output after block 0
                # out_list[11] might be the last block (for enc_depth=12)
                out_list, pos, masks = self.model._encode_image(
                    images, 
                    do_mask=False, 
                    return_all_blocks=True
                )
            '''
            
            # out_list[-1] has the final, normalized features from block 11
            # But if we want a different layer:
            # If self.layer_index is None or out of range, default to final layer
            #if self.layer_index is None or self.layer_index >= len(out_list):
                #print(" !!!! Getting from CroCov2, output of last layer")
                #x = out_list[-1]  # final block, normalized by default by the previous self.model._encode_image(...)
            #else:
                #print(f" !!!! Getting from CroCov2, output of intermediate layer #{self.layer_index}")
                #x = out_list[self.layer_index-1]
                # We need to explicitly normalize this intermediate layer.
                # If we were to get the output of the final layer this would be
                # automatically normalized, but intermediate layers are not by default.
                #x = self.model.enc_norm(x)

        # 3) reshape from (B, N, C) => (B, C, H', W')
        B, N, C = x.shape
        h_p = w_p = int(N ** 0.5)
        patch_embeddings = x.reshape(B, h_p, w_p, C).permute(0, 3, 1, 2) # => (B, C, H', W') 

        # 4) dummy CLS token
        cls_tokens = torch.zeros((B, C), device=patch_embeddings.device)

        # 5) return in the format FoundPose expects
        return {
            "cls_tokens": cls_tokens,          # shape: (B, C)
            "feature_maps": patch_embeddings   # shape: (B, C, H', W')
        }
