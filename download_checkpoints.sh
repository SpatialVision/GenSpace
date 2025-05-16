# GroundedSAM / GroundedDino / RAM
wget -c  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P osdsynth/external/Grounded-Segment-Anything
wget -c  https://huggingface.co/Uminosachi/sam-hq/resolve/main/sam_hq_vit_h.pth -P osdsynth/external/Grounded-Segment-Anything
wget -c  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P osdsynth/external/Grounded-Segment-Anything
wget -c  wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth -P osdsynth/external/Grounded-Segment-Anything/recognize-anything

# PerspectiveFields
wget -c  https://www.dropbox.com/s/z2dja70bgy007su/paramnet_360cities_edina_rpf.pth -P osdsynth/external/PerspectiveFields/models

wget -c https://huggingface.co/Viglong/OriNet/blob/main/croplargeEX2/dino_weight.pt -P 'osdsynth/Orient_Anything/dino_weight.pt'