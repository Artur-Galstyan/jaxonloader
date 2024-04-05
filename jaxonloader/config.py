import pathlib


JAXONLOADER_PATH = pathlib.Path.home() / ".jaxonloader"


def set_jaxonloader_path(path: str):
    global JAXONLOADER_PATH
    JAXONLOADER_PATH = pathlib.Path(path)
