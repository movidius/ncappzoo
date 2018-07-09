import sys

sys.path.insert(0, 'ncsdk/api/python')
import click
from pathlib import Path
from movidius.cfg import set_seed
from movidius.imagenet_utils import convert_images_to_jpeg

set_seed()


@click.group()
def cli():
    pass


def _invalid_model(model, options):
    return Exception('Invalid model "%s", must be one of %s' % (model, options))


@cli.command('validate-imagenet')
@click.argument('model')
def validate_imagenet(model):
    if model == 'nasnet_mobile':
        from movidius.nasnet.validate import run as do_validate
    elif model == 'inception_v4':
        from movidius.inception_v4.validate import run as do_validate
    elif model == 'mobilenet':
        from movidius.mobilenet.configs import validate_imagenet as do_validate
    else:
        raise _invalid_model(model, ['nasnet_mobile', 'inception_v4', 'mobilenet'])

    do_validate()


@cli.command('download-imagenet-extra')
def run_download_imagenet_extra():
    from movidius.imagenet_utils import download_imagenet_extra
    download_imagenet_extra()


@cli.command('validate-imagenet-ncs')
@click.argument('model')
def validate_imagenet_ncs(model):
    if model == 'inception_v3':
        from movidius.inception_v3.submission import validate_imagenet
    elif model == 'inception_v4':
        from movidius.inception_v4.submission import validate_imagenet
    else:
        raise _invalid_model(model, ['inception_v3', 'inception_v4'])
    validate_imagenet()


@cli.command()
@click.option('--work-dir', type=Path, required=True)
@click.option('--epochs', type=int, required=True)
@click.option('--train-split', type=str, default='train')
@click.option('--eval-split', type=str, default='eval')
@click.option('--fine-tune', is_flag=True, default=False)
@click.option('--reset-optim', is_flag=True, default=False)
@click.option('--phase', type=int, default=1)
@click.argument('model')
def train(**kwargs):
    model = kwargs['model']
    if model == 'mobilenet':
        from movidius.mobilenet.configs import train as do_train
    elif model == 'mobilenet_v2':
        from movidius.mobilenet.configs import train_v2 as do_train
    elif model == 'inception_v4':
        from movidius.inception_v4.train import run as do_train
    elif model == 'inception_v3' and kwargs['fine_tune']:
        from movidius.inception_v3.configs import train_finetune as do_train
    else:
        raise _invalid_model(model, ['mobilenet', 'mobilenet_v2', 'inception_v4', 'inception_v3 (fine-tune)'])
    do_train(**kwargs)


@cli.command('train-cifar-10')
@click.option('--work-dir', type=Path, required=True)
@click.option('--epochs', type=int, required=True)
@click.option('--reset-optim', is_flag=True, default=False)
@click.option('--phase', type=int, default=1)
@click.argument('model')
def train_cifar10(**kwargs):
    model = kwargs['model']
    if model == 'mobilenet':
        from movidius.mobilenet.configs import train_cifar10 as do_train
    elif model == 'mobilenet2':
        from movidius.mobilenet2.configs import train_cifar10 as do_train
    elif model == 'inception_v3':
        from movidius.inception_v3.configs import train_cifar10 as do_train
    else:
        raise _invalid_model(model, ['mobilenet', 'mobilenet2', 'inception_v3'])
    do_train(**kwargs)


@cli.command()
@click.option('--work-dir', type=Path, required=True)
@click.option('--step', type=int, required=True)
@click.option('--skip-compile', is_flag=True, default=False)
@click.option('-n', '--num', type=int, default=None)
@click.option('--skip-inference', is_flag=True, default=False)
@click.option('--skip-upload', is_flag=True, default=False)
@click.option('--test-split', default='test')
@click.option('--score', is_flag=True, default=False)
@click.option('--limit', type=int, default=None)
@click.argument('model')
def submit(**kwargs):
    model = kwargs['model']
    if model == 'mobilenet':
        from movidius.mobilenet.submission import submit as do_submit
    elif model == 'inception_v4':
        from movidius.inception_v4.submission import submit as do_submit
    elif model == 'inception_v3':
        from movidius.inception_v3.configs import submit as do_submit
    else:
        raise _invalid_model(model, ['mobilenet', 'inception_v4', 'inception_v3'])

    do_submit(**kwargs)


@cli.command('score-gpu')
@click.option('--work-dir', type=Path, required=True)
@click.option('--step', type=int, required=True)
@click.option('--time', type=float, required=True)
@click.option('--split', default='eval')
@click.option('--limit', type=int, default=None)
@click.argument('model')
def score_gpu(**kwargs):
    model = kwargs['model']
    if model == 'inception_v4':
        from movidius.inception_v4.submission import score_gpu as do_score_gpu
    elif model == 'inception_v3':
        from movidius.inception_v3.configs import score_gpu as do_score_gpu
    else:
        raise _invalid_model(model, ['inception_v4'])

    do_score_gpu(**kwargs)


@cli.command('profile')
@click.argument('model')
@click.option('-o', '--outdir', type=Path, required=False)
def profile(**kwargs):
    model = kwargs['model']
    if model == 'mobilenet':
        from movidius.mobilenet.configs import profile as do_profile
    elif model == 'mobilenet_v2':
        from movidius.mobilenet.configs import profile_v2 as do_profile
    elif model == 'nasnet':
        from movidius.nasnet.configs import profile as do_profile
    elif model == 'inception_v4':
        from movidius.inception_v4.configs import profile as do_profile
    else:
        raise _invalid_model(model, ['mobilenet_v2', 'mobilenet', 'nasnet', 'inception_v4'])

    do_profile(**kwargs)


@cli.command('dataset-test')
@click.argument('split')
def dataset_test(split):
    from movidius.dataset.test import dataset_test as run
    run(split)


@cli.command('extra-convert-jpeg')
def extra_convert_jpeg():
    convert_images_to_jpeg()


@cli.command('validate')
@click.argument('model')
@click.option('--work-dir', type=Path, required=True)
@click.option('--step', type=int, required=True)
def validate(**kwargs):
    model = kwargs['model']
    if model == 'mobilenet_v2':
        from movidius.mobilenet.configs import validate_v2 as do_validate
    else:
        raise _invalid_model(model, ['mobilenet_v2'])

    do_validate(**kwargs)


@cli.command('inference')
@click.option('--submission', type=Path, required=True, help='The submission dir')
@click.option('--images', type=Path, required=True, help='The path to the images dir')
@click.option('--out', type=Path, required=True, help='The output CSV file')
def inference(submission, out, images):
    from movidius.submission import run_inference, SubmissionConfig
    from movidius.inception_v3.common import SHAPE
    from movidius.preprocessing import preprocess_eval
    from movidius.dataset.movidius import MovidiusChallengeDataset
    cfg = SubmissionConfig()
    cfg.model_name = 'Inception V3 (fine tuned)'
    cfg.submission_dir = submission
    cfg.preprocess_fn = preprocess_eval
    cfg.shape = SHAPE
    cfg.custom_inferences_file = out
    cfg.custom_test_dataset_fn = lambda: (MovidiusChallengeDataset(images), 200)
    run_inference(cfg)


@cli.command('compile-inference')
@click.option('--weights', type=Path, required=True, help='The weights dir')
@click.option('--images', type=Path, required=True, help='The path to the images dir')
@click.option('--submission', type=Path, required=True, default='/tmp/submission', help='The submission dir')
def compile_infer(submission, images, weights):
    from movidius.submission import run_inference, SubmissionConfig, build_meta, nc_compile
    from movidius.inception_v3.common import SHAPE, create_build_fn
    from movidius.preprocessing import preprocess_eval
    from movidius.dataset.movidius import MovidiusChallengeDataset
    from multiprocessing import Process
    cfg = SubmissionConfig()
    cfg.model_name = 'Inception V3 (fine tuned)'
    cfg.submission_dir = submission
    cfg.preprocess_fn = preprocess_eval
    cfg.shape = SHAPE
    cfg.custom_test_dataset_fn = lambda: (MovidiusChallengeDataset(images), 200)
    cfg.submission_dir = submission
    cfg.build_fn = create_build_fn(create_aux_logits=False)
    cfg.weights_file = weights / 'model'
    p = Process(target=build_meta, args=(cfg,))
    p.start()
    p.join()

    # compile to movidius graph
    nc_compile(cfg)

    run_inference(cfg)

    print('Submission files in %s' % cfg.submission_dir)

if __name__ == '__main__':
    cli()
