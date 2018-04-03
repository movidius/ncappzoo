from movidius.dataset.utils import BatchDataset, Feeder, imagenet_coalesce_fn
from movidius.splits import dataset_from_split


def dataset_test(split):
    from movidius.preprocessing import preprocess_train
    from pathlib import Path
    import cv2
    from tqdm import tqdm
    ds, num_classes = dataset_from_split(split)
    batch_ds = BatchDataset(ds, 64)
    feeder = Feeder(batch_ds, imagenet_coalesce_fn(num_classes, preprocess_train, ds.test_size[0], ds.test_size[1],
                                                   return_meta=True), workers_count=1)
    feeder.shuffle()
    dump_batches = 100
    bar = tqdm(total=len(ds))
    for imgs, labels, metas in feeder:
        if dump_batches:
            outdir = Path('/tmp/dataset_test')
            outdir.mkdir(exist_ok=True)
            for i, img in enumerate(imgs):
                img += 1
                img /= 2
                img *= 255
                img = cv2.cvtColor(img.astype('u1'), cv2.COLOR_RGB2BGR)
                f = outdir / f'{metas[i].label}_{metas[i].key}'
                cv2.imwrite(str(f), img)
            dump_batches -= 1
        bar.update(len(labels))
    bar.close()
