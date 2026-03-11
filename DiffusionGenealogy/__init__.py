from .ddpm.diffusion import DDPMDiffusion
from .ddim.diffusion import DDIMDiffusion
from .score_sde.diffusion import ScoreSDEDiffusion
from .edm.diffusion import EDMDiffusion
from .rectified_flow.diffusion import RectifiedFlowDiffusion
from .ot_cfm.diffusion import OTCFMDiffusion

VARIANTS = {
    "ddpm": DDPMDiffusion,
    "ddim": DDIMDiffusion,
    "score_sde": ScoreSDEDiffusion,
    "edm": EDMDiffusion,
    "rectified_flow": RectifiedFlowDiffusion,
    "ot_cfm": OTCFMDiffusion,
}
