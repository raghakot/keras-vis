import shutil

from md_autogen import MarkdownAPIGenerator
from md_autogen import to_md_file

from vis import backend
from vis.utils import utils
from vis import visualization
from vis import backprop_modifiers
from vis import callbacks
from vis import grad_modifiers
from vis import input_modifiers
from vis import losses
from vis import optimizer
from vis import regularizers


def generate_api_docs():
    modules = [
        backend,
        utils,
        visualization,
        backprop_modifiers,
        callbacks,
        grad_modifiers,
        input_modifiers,
        losses,
        optimizer,
        regularizers
    ]

    md_gen = MarkdownAPIGenerator("vis", "https://github.com/raghakot/keras-vis/tree/master")
    for m in modules:
        md_string = md_gen.module2md(m)
        to_md_file(md_string, m.__name__, "sources")


def update_index_md():
    shutil.copyfile('../README.md', 'sources/index.md')


def copy_templates():
    shutil.rmtree('sources', ignore_errors=True)
    shutil.copytree('templates', 'sources')


if __name__ == "__main__":
    copy_templates()
    update_index_md()
    generate_api_docs()
