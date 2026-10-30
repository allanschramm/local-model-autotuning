SUPPORTED_PYTHON_VERSIONS = ["3.11", "3.12"]
DEV_DEPENDENCIES = ["pytest>=8.0"]
CI_CHECK_COMMANDS = ["python -m pytest"]


def requires_python_specifier():
    return ">=3.11"


def ci_python_versions():
    return list(SUPPORTED_PYTHON_VERSIONS)


def dev_extra_dependencies():
    return list(DEV_DEPENDENCIES)


def ci_install_command():
    return "python -m pip install -e ."


def ci_check_commands():
    return list(CI_CHECK_COMMANDS)
