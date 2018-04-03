import click
from pathlib import Path
from movidius.cfg import set_seed

set_seed()


@click.group()
def cli():
    pass


@cli.command('validate-imagenet')
def validate_imagenet():
    from movidius.inception_v3.validate import run as do_validate
    do_validate()


@cli.command('validate-imagenet-ncs')
def validate_imagenet_ncs():
    from movidius.inception_v3.submission import validate_imagenet
    validate_imagenet()


@cli.command()
@click.option('--work-dir', type=Path, required=True)
@click.option('--step', type=int, required=True)
@click.option('--skip-compile', is_flag=True, default=False)
@click.option('-n', '--num', type=int, default=None)
@click.option('--skip-inference', is_flag=True, default=False)
@click.option('--test-split', default='test')
@click.option('--score', is_flag=True, default=False)
def submit(**kwargs):
    from movidius.inception_v3.submission import submit as do_submit
    do_submit(**kwargs)


if __name__ == '__main__':
    cli()
