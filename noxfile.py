import nox


@nox.session
def tests(session):
    session.install("uv")
    session.install("pip")
    session.run("uv", "pip", "install", ".[dev]")
    session.run("uv", "pip", "install", "-e", ".")
    session.run("pytest", "tests")
