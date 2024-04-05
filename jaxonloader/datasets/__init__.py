from jaxtyping import install_import_hook


with install_import_hook(modules=["jaxonloader"], typechecker="beartype.beartype"):
    from jaxonloader.datasets._datasets import *  # noqa
