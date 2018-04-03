import pickle
import shelve
from collections import defaultdict
from concurrent.futures import Future, Executor, ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Callable

import time
from tqdm import tqdm
import xml.etree.ElementTree as ET
import requests

from movidius.dataset.movidius import MovidiusChallengeDataset
from movidius.splits import dataset_from_split
import magic


class Synsets:
    def __init__(self, structure_xml: Path):
        tree = ET.parse(str(structure_xml))
        self.root: ET.Element = tree.getroot()

    def get_children_wnid(self, wnid, collect=None):
        node: ET.Element = self.root.find(f'.//synset[@wnid="{wnid}"]')
        if not node:
            return []
        return [x.attrib['wnid'] for x in node.iter() if x.tag == 'synset' and x != node]


class ImageURLs:
    def __init__(self, urls_file: Path):
        self.urls_file = urls_file
        self.idx_file = urls_file.with_suffix('.index')
        if not self.idx_file.exists():
            self._preprocess()
        else:
            self.idx = pickle.loads(self.idx_file.read_bytes())

    def _preprocess(self):
        from subprocess import check_output
        print('Indexing URLs file. This may take a while')
        total = check_output(['wc', '-l', str(self.urls_file)])
        total = int(total.split()[0])
        self.idx: Dict[str, List[int]] = dict()
        pos = 0
        with self.urls_file.open('rb') as f:
            for line in tqdm(f, total=total):
                try:
                    line = str(line, 'utf-8')
                    code, url = line.strip().split(maxsplit=1)
                    wnid, idx = code.split('_')
                    self.idx.setdefault(wnid, []).append(pos)
                except:
                    import traceback
                    traceback.print_exc()
                    print('Line: ', line)
                pos = f.tell()

        self.idx_file.write_bytes(pickle.dumps(self.idx))

    def get_urls(self, wnids):
        urls: Dict[str, List[Tuple]] = dict()
        positions = []
        for wnid in wnids:
            positions += self.idx.get(wnid, [])

        positions.sort()
        with self.urls_file.open('rb') as f:
            for pos in positions:
                f.seek(pos)
                line = f.readline()
                line = str(line, 'utf-8')
                code, url = line.strip().split(maxsplit=1)
                wnid, idx = code.split('_')
                urls.setdefault(wnid, []).append((url, idx))

        return urls


def _dowload_func(outfile: Path, url: str):
    skip = False
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            mime = r.headers.get('content-type')
            data = r.content
            if not mime:
                with magic.Magic(flags=magic.MAGIC_MIME) as m:
                    mime = m.id_buffer(data)
            _save_data_to_jpeg(r.content, mime, outfile)
            return
        else:
            skip = True
    except requests.ConnectionError:
        skip = True

    except:
        import traceback
        # traceback.print_exc()

    if skip:
        skipfile = outfile.with_suffix('.SKIP')
        skipfile.touch()


class ImageNetDownloader:
    def __init__(self, image_urls: ImageURLs, synsets: Synsets, images_dir: Path):
        self.images_dir = images_dir
        self.image_urls = image_urls
        self.synsets = synsets

    def download_async(self, wnid, max_jobs, pool: Executor, callback: Callable = None) -> List[Future]:
        self.images_dir.mkdir(exist_ok=True, parents=True)
        synsets = [wnid]
        # synsets += self.synsets.get_children_wnid(wnid)

        futures = []
        urls = self.image_urls.get_urls(synsets)
        for wnid in synsets:
            for url, idx in urls[wnid]:
                if len(futures) > max_jobs:
                    break

                outfile = self.images_dir / f'{wnid}_{idx}.JPEG'
                skipfile = outfile.with_suffix('.SKIP')
                if outfile.exists() or skipfile.exists():
                    continue
                f: Future = pool.submit(_dowload_func, outfile, url)
                futures.append(f)
                f.add_done_callback(callback)
        return futures

    def count_downloaded(self, wnid):
        synsets = [wnid]
        # synsets += self.synsets.get_children_wnid(wnid)

        count = 0
        for wnid in synsets:
            files = list(self.images_dir.glob(f'{wnid}_*.JPEG'))
            count += len(files)
        return count

    def purge_files(self, wnid, count):
        files = list(self.images_dir.glob(f'{wnid}_*.JPEG'))
        for file in files:
            if count <= 0:
                break
            file.unlink()
            count -= 1


def download_imagenet_extra(target_count=800, workers=50):
    from .cfg import IMAGENET_EXTRA_DIR, IMAGENET_URL_FILES, IMAGENET_STRUCTURE_XML
    tqdm.write('Loading image URLs')
    urls = ImageURLs(urls_file=IMAGENET_URL_FILES)
    pool = ThreadPoolExecutor(workers)
    tqdm.write('Getting synsets')
    ds, _ = dataset_from_split('train')  # type: MovidiusChallengeDataset, None
    wnids = sorted(list({x.wnid for x in ds.meta}))
    synsets = Synsets(structure_xml=IMAGENET_STRUCTURE_XML)
    IMAGENET_EXTRA_DIR.mkdir(exist_ok=True, parents=True)
    dl = ImageNetDownloader(image_urls=urls, synsets=synsets, images_dir=IMAGENET_EXTRA_DIR)

    wnid_bar = tqdm(total=len(wnids))
    wnid_bar.set_description('WNID processed')
    download_bar = tqdm(total=0)
    download_bar.set_description('Queue state')
    for wnid in wnids:

        tqdm.write('Preparing to download synset %s' % wnid)
        need = target_count - dl.count_downloaded(wnid)

        if need < 0:
            dl.purge_files(wnid, -need)
            tqdm.write(f'Purging {-need} files for WNID {wnid}')
            wnid_bar.update()
            continue

        def callback(f):
            download_bar.update()

        futures = dl.download_async(wnid, need, pool, callback)
        n = len(futures)
        download_bar.total += n
        tqdm.write(f'Need {need} files. Downloading {n} files for WNID {wnid}')

        time.sleep(n // 100)
        wnid_bar.update()

    pool.shutdown(wait=True)
    download_bar.close()
    wnid_bar.close()

# hashes of placeholder images
BAD_IMAGES_MD5 = {'e180ea048bad9d9530576b16ef4c44d4', '2341281598806dac64d9fc39bfdef231',
                  'ab6be20ad7b3841c5d56c268bb12a971', 'a465f6d4349759b69e0f776b8ba8e7bd',
                  '382bc893965abd93b892dcdf1928a1ec', '2c97f927120e2d321f0793e86e14c01a',
                  '137bfd4864f4b4267fcd40e42c9d781e', '7e38a78e5dc8f67ae17b6eb76a25348c',
                  'ba01f16d5fead9f1c198064da12088bd', '9a5cba3864a7e465eff2f8ad99120501',
                  'bc30d724d22b90ff0ef80f38d3feb343', '199e58c5f39372867fbe09f058a6cf40',
                  'bb81ce3bf5d41551419f241dc5203565', '4515f413e67ce8e7b01661b258cf35e5',
                  '8d01c50773354dfaaa6a349bf645d1da', '7144a9d6c21949bcfb5247518077033b',
                  'fec0a4438a6616027236124f2e423b57', 'b7bf3cd2cf08a83ffc2e70d517d87add',
                  'a88d9fa43aeba0379cafe592ad42da63', 'c0715d86e69d87392b2e395b3ae58ca3',
                  '2b798db6fc237c46b5b261335acfc067', '35a4ccd1be5797b536ff391f0a7a6447',
                  '5b77d777c9c5597d9c81c75fb0dbc488', '5d77f90abab18f0e769d367b26057016',
                  'dafc593d39f2652869fdde120069a405', 'd4dc447ac349922bfb08d8f811407fdd',
                  'aa5ec1a52f6fde4404b75f429cfbc007', '74351742df3c8d71b56e95e5e8a24cba',
                  '83a26d8302920cf45518676f235244be', '07f4f4596928aea410bddfed5018004e',
                  '6d209737d337d379b0f4bce912aa2c85', '96b59f7967e38f257c8069c4f84126f7',
                  '708cc298ad0054a0fec729158b06b748', '3ea5ea3d2aca9784fa906bde66239c56',
                  '45ebd598506472fd04e11dd39ebd86c6', '2d6fcd23457df2439e84fa652402860f',
                  'd41d8cd98f00b204e9800998ecf8427e', 'ddddcacd640f494b71ed844f0b67dfbc',
                  '89134461c860c67cb986df29227db65a', 'e1b3fa89dcfa0cd2b11cad470b2c9cb1',
                  '84c9683d38d74f2c77bb46f05cd80b41', '59e73ab791c2dc34bf466cbd97261816',
                  '4f24cc259f5112909f83a0ebc456c76b', '0497a862e81ea783ce0c74cd71c48dd4',
                  '7e6aca1027a71eb55819d7d245f7bb0d'}


def _save_data_to_jpeg(mime, data, outfile):
    import cv2
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from hashlib import md5
    try:
        _md5 = md5(data).hexdigest()
        if _md5 in BAD_IMAGES_MD5:
            raise Exception
        elif 'image/jpeg' in mime:
            return
        elif 'image/png' in mime:
            img = cv2.imdecode(np.frombuffer(data, 'u1'), cv2.IMREAD_COLOR)
            cv2.imwrite(str(outfile), img)
        elif 'image/gif' in mime:
            img = Image.open(BytesIO(data)).convert('RGB')
            img.save(str(outfile))
        else:
            raise Exception
    except:
        if outfile.exists():
            outfile.unlink()
        outfile.with_suffix('.SKIP').touch()


def convert_images_to_jpeg():
    from .cfg import IMAGENET_EXTRA_DIR
    import magic

    files = list(IMAGENET_EXTRA_DIR.iterdir())
    for file in tqdm(files):  # type: Path
        if file.suffix.lower().endswith('skip'):
            continue
        with magic.Magic(flags=magic.MAGIC_MIME_TYPE) as m:
            mime = m.id_filename(str(file))
            data = file.read_bytes()
            _save_data_to_jpeg(mime, data, file)


if __name__ == '__main__':
    s = Synsets(Path('/mnt/ds/cjalmeida/dataset/Imagenet/structure_released.xml'))
    print(s.get_children_wnid('n13035707'))
