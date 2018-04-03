from pathlib import Path

import shutil

from movidius.dataset.utils import BatchDataset, imagenet_coalesce_fn, Feeder
from movidius.preprocessing import preprocess_eval, preprocess_train
from movidius.splits import dataset_from_split, IMAGENET_EXTRA_DIR
from tqdm import tqdm
import random

out = Path('/tmp/samples')
out.mkdir(exist_ok=True)
files = list(IMAGENET_EXTRA_DIR.glob('*.JPEG'))
samples = random.sample(files, int(0.02 * len(files)))
for f in tqdm(samples):  # type: Path
    shutil.copy(str(f), str(out / f.name))