from ._cli import cli


@cli.command(name='test1')
def sphere(name: str, number: int):
    print('bla')
