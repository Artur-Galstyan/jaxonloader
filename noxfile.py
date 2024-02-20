import nox


@nox.session
def tests(session):
    session.run("pip", "install", ".")
    session.run("pytest", "tests")
