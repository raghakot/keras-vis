import os
import shutil

from md_autogen import MarkdownAPIGenerator
from md_autogen import to_md_file

from vis import losses
from vis import regularizers
from vis import optimizer
from vis import visualization

from vis.utils import utils
from vis.utils import vggnet


def generate_api_docs():
    modules = [
        losses,
        regularizers,
        optimizer,
        visualization,
        utils,
        vggnet
    ]

    mkgen = MarkdownAPIGenerator("vis", "https://github.com/raghakot/keras-vis/tree/master")
    for module in modules:
        md_string = mkgen.module2md(module)
        to_md_file(md_string, module.__name__, "sources")


def update_index_md():
    shutil.copyfile('../README.md', 'sources/index.md')


def copy_templates():
    shutil.rmtree('sources', ignore_errors=True)
    shutil.copytree('templates', 'sources')


if __name__ == "__main__":
    copy_templates()
    update_index_md()
    generate_api_docs()
