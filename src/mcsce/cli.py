"""
DOCSTRING for first public console interface.

USAGE:
    $ mcsce -h
"""
import argparse


ap = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def load_args(ap):
    """Load argparse commands."""
    cmd = ap.parse_args()
    return cmd


def cli(ap, main):
    """Command-line interface entry point."""
    cmd = load_args(ap)
    main(**vars(cmd))


def maincli():
    """Independent client entry point."""
    cli(ap, main)


def main():
    """Print helloe sample."""
    print('hello sample')


if __name__ == '__main__':
    maincli()
