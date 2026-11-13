import tomllib
from pathlib import Path

from packaging.requirements import Requirement

PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _load_pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as f:
        return tomllib.load(f)


def test_httpx_is_declared_in_project_dependencies() -> None:
    data = _load_pyproject()
    deps = data.get("project", {}).get("dependencies", [])
    dep_names = [Requirement(dep).name.lower() for dep in deps]
    assert "httpx" in dep_names


def test_httpx_is_not_only_an_optional_dev_dependency() -> None:
    data = _load_pyproject()
    project = data.get("project", {})
    runtime_names = {Requirement(dep).name.lower() for dep in project.get("dependencies", [])}
    dev_names = {Requirement(dep).name.lower() for dep in project.get("optional-dependencies", {}).get("dev", [])}
    assert "httpx" in runtime_names
    assert not ("httpx" in dev_names and "httpx" not in runtime_names)
