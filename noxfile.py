import nox


@nox.session
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest", "tests")
