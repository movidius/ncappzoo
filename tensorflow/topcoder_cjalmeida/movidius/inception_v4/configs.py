from movidius.profile import ProfileConfig, do_profile
from .common import SHAPE, build_fn


def profile(**kwargs):
    cfg = ProfileConfig()
    cfg.shape = SHAPE
    cfg.build_fn = build_fn
    cfg.temp_dir = kwargs.get('outdir')
    do_profile(cfg)
