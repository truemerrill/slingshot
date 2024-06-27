from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from jinja2 import Environment

from .components import Index, ENV


Page = Callable[[Path, Environment], None]
"""A Page is a callable that takes a path and renders a page to that path"""


@dataclass
class SitePage:
    name: str
    parent: "SiteDirectory"
    page: Page
    index: bool


class SiteDirectory:
    """Data structure for a directory of static site build artifacts"""

    def __init__(self, name: str, parent: "SiteDirectory | None"):
        self.name = name
        self.parent = parent
        self.pages: list[SitePage] = []
        self.directories: list["SiteDirectory"] = []

    def path(self) -> Path:
        if self.parent is None:
            return Path(self.name)
        return self.parent.path() / self.name

    def add_page(self, path: Path, page: Page, index: bool = True) -> None:
        if len(path.parts) == 1:
            name = path.parts[0]
            self.pages.append(SitePage(name, self, page, index))
            return None

        elif len(path.parts) > 1:
            dirname = path.parts[0]
            remaining_path = Path(*path.parts[1:])

            # Add the page to an already existing subdirectory
            for subdir in self.directories:
                if subdir.name == dirname:
                    return subdir.add_page(remaining_path, page, index)

            # Subdirectory does not exist - create and add page
            subdir = SiteDirectory(dirname, parent=self)
            self.directories.append(subdir)
            return subdir.add_page(remaining_path, page, index)

    def build(self, prefix: Path, env: Environment) -> None:
        path = prefix / self.name
        path.mkdir(parents=True, exist_ok=True)

        for page in self.pages:
            page.page(path / page.name, env)
        for subdir in self.directories:
            subdir.build(path, env)

        index = Index(
            parent=self.parent.name if self.parent else None,
            name=self.name,
            files=[p.name for p in self.pages if p.index],
            subdirs=[s.name for s in self.directories]
        )
        with open(path / "index.html", "w") as f:
            f.write(index.render(env))


class Site:
    """Class managing building static website"""

    def __init__(self):
        self.root = SiteDirectory("data", None)

    def publish(self, path: str | Path, page: Page, index: bool = True) -> None:
        pth = Path(path) if isinstance(path, str) else path
        self.root.add_page(pth, page, index)

    def build(self, prefix: Path | str = Path("."), env: Environment = ENV) -> None:
        pth = Path(prefix) if isinstance(prefix, str) else prefix
        self.root.build(pth, env)
