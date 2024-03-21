import nox


@nox.session
def tests(session):
    session.run("pip", "install", ".[dev]")
    session.run("pytest", "tests")
