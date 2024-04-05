from jaxtyping import install_import_hook


with install_import_hook(modules=["jaxonloader"], typechecker="beartype.beartype"):
    from jaxonloader.dataset import JaxonDataset, SingleArrayDataset, DataTargetDataset  # noqa
    from jaxonloader.dataloader import JaxonDataLoader  # noqa
