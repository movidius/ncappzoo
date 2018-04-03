from movidius.cfg import set_seed

set_seed()

from pathlib import Path

from movidius.inception_v4.train import run as do_train
from movidius.imagenet import create_cat_1001
import click


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
def train(clean, work_dir, epochs, train_split, eval_split):
    do_train(clean, work_dir, epochs, train_split, eval_split)


if __name__ == '__main__':
    cli()
