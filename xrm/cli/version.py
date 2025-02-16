import click
import vanya


@click.command(name="version")
@click.argument("filename", default=None, required=False, nargs=1)
def command(filename):
    """Load yaml file and tests tests"""
    with open(filename) as tests_file:
        tests = yaml.load(tests_file, Loader=yaml.Loader)
        print(vanya.utils.)
