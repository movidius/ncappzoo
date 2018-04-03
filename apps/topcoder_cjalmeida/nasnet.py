from pathlib import Path

from movidius.nasnet.vis import interactive as do_interactive
from movidius.nasnet.train import run as do_train
import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument('kind', type=str)
def export(kind):
    from movidius.nasnet.export import export as do_export
    do_export(kind)


@cli.command()
def interactive():
    do_interactive()


@cli.command()
@click.argument('kind', type=str)
def train(kind):
    do_train(kind)


if __name__ == '__main__':
    cli()
