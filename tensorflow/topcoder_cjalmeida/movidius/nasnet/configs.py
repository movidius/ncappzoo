from movidius.nasnet.common import SHAPE, build_fn
from movidius.profile import ProfileConfig, do_profile


def profile(**kwargs):
    cfg = ProfileConfig()
    cfg.shape = SHAPE
    cfg.build_fn = build_fn
    cfg.temp_dir = kwargs.get('outdir')
    do_profile(cfg)
