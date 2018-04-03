import click
from pathlib import Path
from movidius.cfg import set_seed

set_seed()


@click.group()
def cli():
    pass


@cli.command()
def export():
    from movidius.inception_v4.export import export as do_export
    do_export()


@cli.command()
@click.option('--clean', is_flag=True, default=False)
@click.option('--work-dir', type=Path, required=True)
@click.option('--epochs', type=int, required=True)
@click.option('--train-split', type=str, default='train')
@click.option('--eval-split', type=str, default='eval')
@click.option('--kind', type=str, default='100_128')
def train(clean, work_dir, epochs, train_split, eval_split, kind):
    from movidius.mobilenet.train import run as do_train
    do_train(clean, work_dir, epochs, train_split, eval_split, kind)


@cli.command('validate-imagenet')
def validate_imagenet():
    from movidius.mobilenet.validate import run as do_validate
    do_validate()


@cli.command()
@click.option('--work-dir', type=Path, required=True)
@click.option('--step', type=int, required=True)
@click.option('--skip-compile', is_flag=True, default=False)
@click.option('-n', '--num', type=int, default=None)
@click.option('--skip-inference', is_flag=True, default=False)
@click.option('--test-split', default='test')
@click.option('--score', is_flag=True, default=False)
def submit(**kwargs):
    from movidius.mobilenet.submission import submit as do_submit
    do_submit(**kwargs)


if __name__ == '__main__':
    cli()
