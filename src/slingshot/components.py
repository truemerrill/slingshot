import base64
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Protocol

import pandas as pd
import plotly.graph_objects as go  # type: ignore
from jinja2 import Environment, PackageLoader, select_autoescape
from jinja2.exceptions import TemplateNotFound
from jinja2.loaders import split_template_path


def indent(value: str, level: int = 1) -> str:
    """Jinja2 filter for managing indentation levels

    Args:
        value (str): the HTML to indent
        level (int, optional): the indentation level. Defaults to 1.

    Returns:
        str: the indented HTML
    """
    id = "    " * level
    lines = [id + line if line else line for line in value.splitlines()]
    return "\n".join(lines)


# Global Jinja2 rendering environment
ENV = Environment(
    loader=PackageLoader("slingshot"),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)
ENV.filters["indent"] = indent


class Component(Protocol):
    """Protocol for a component of a web report"""

    def render(self, env: Environment) -> str:
        """Render as HTML

        Args:
            env (Environment): the Jinja2 environment

        Returns:
            str: the rendered HTML
        """
        ...


def render(component: Component | str, env: Environment = ENV) -> str:
    """Render a component of a web report

    Args:
        component (Component | str): the component or a string
        env (Environment, optional): the Jinja2 environment. Defaults to ENV.

    Returns:
        str: the rendered HTML
    """
    if isinstance(component, str):
        return component
    return component.render(env)


class Heading:
    """Component for headings"""

    def __init__(self, content: str, level: int = 2):
        self.content = content
        self.level = level

    def render(self, env: Environment) -> str:
        return f"<h{self.level}>{self.content}</h{self.level}>"


class Paragraph:
    """Component for paragraph blocks"""

    def __init__(self, contents: str):
        self.contents = contents

    def render(self, env: Environment) -> str:
        return f"<p>{self.contents}</p>"


class Div:
    """Component for div blocks"""

    def __init__(
        self,
        children: Iterable[Component],
        id: Optional[str] = None,
        css_class: Optional[str] = None,
    ):
        self.children = tuple(children)
        self.id = id
        self.css_class = css_class

    def render(self, env: Environment) -> str:
        template = env.get_template("partials/div.html")
        return template.render(
            id=self.id,
            css_class=self.css_class,
            children=[c.render(env) for c in self.children],
        )


class Link:
    def __init__(self, contents: Component | str, url: str):
        self.contents = contents
        self.url = url

    def render(self, env: Environment) -> str:
        return f'<a href="{self.url}">{render(self.contents, env)}</a>'


class Table:
    """Component for HTML tables"""

    def __init__(
        self,
        headers: Iterable[str],
        rows: Iterable[Iterable[str]],
        id: Optional[str] = None,
        css_class: Optional[str] = None,
    ):
        self.headers = tuple(headers)
        self.rows = rows
        self.id = id
        self.css_class = css_class

    def render(self, env: Environment) -> str:
        template = env.get_template("partials/table.html")
        return template.render(
            id=self.id,
            css_class=self.css_class,
            headers=self.headers,
            rows=[list(r) for r in self.rows],
        )


class Dataframe:
    """Component for pandas dataframes as an HTML table"""

    def __init__(
        self,
        df: pd.DataFrame,
        id: Optional[str] = None,
        css_class: Optional[str] = None,
    ):
        self.df = df
        self.id = id
        self.css_class = css_class

    def render(self, env: Environment) -> str:
        return self.df.to_html(  # type: ignore
            index=False, classes=self.css_class, table_id=self.id
        )


class Figure:
    """Component for a Plotly figure"""

    def __init__(self, figure: go.Figure, template: str = "plotly_white"):
        figure.update_layout(template=template)  # type: ignore
        self.figure = figure

    def render(self, env: Environment) -> str:
        return self.figure.to_html(full_html=False)  # type: ignore


class StaticImage:
    """Component for a static embedded image"""

    def __init__(self, template_name: str, mimetype: str = "image/png"):
        self.template_name = template_name
        self.mimetype = mimetype

    def render(self, env: Environment) -> str:
        # Get the template root
        loader = env.loader
        if not hasattr(loader, "_template_root"):
            raise TemplateNotFound(self.template_name)
        template_root = getattr(loader, "_template_root")

        # Read the file as a binary blob
        filename = Path(template_root, *split_template_path(self.template_name))
        if not filename.is_file():
            raise TemplateNotFound(self.template_name)
        with open(filename, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f'<img src="data:{self.mimetype};base64, {data}"/>'


class ReportHeading:
    """Component for the heading of a report"""

    def __init__(self, content: str, css_class: Optional[str] = "report-heading"):
        self.content = content
        self.css_class = css_class

    def render(self, env: Environment) -> str:
        template = env.get_template("partials/div.html")
        return template.render(
            css_class=self.css_class,
            children=[Heading(content=self.content).render(env)],
        )


class ReportSection:
    """Component for a section of a report"""

    def __init__(
        self,
        title: str,
        children: Iterable[Component],
        id: Optional[str] = None,
        css_class: Optional[str] = "report-section",
    ):
        self.title = title
        self.children = tuple(children)
        self.id = id
        self.css_class = css_class

    def render(self, env: Environment) -> str:
        template = env.get_template("partials/div.html")
        return template.render(
            id=self.id,
            css_class=self.css_class,
            children=[
                Heading(self.title, level=3).render(env),
                Div(children=self.children, css_class="report-section-content").render(
                    env
                ),
            ],
        )


def today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


class Report:
    """Component for a report"""

    def __init__(
        self,
        title: str,
        sections: Iterable[ReportSection],
    ):
        self.title = title
        self.sections = sections

    def render(self, env: Environment) -> str:
        template = env.get_template("report.html")
        logo = StaticImage("assets/logo.png")
        return template.render(
            title=self.title,
            logo=logo.render(env),
            date=today(),
            heading=ReportHeading(self.title).render(env),
            sections=[s.render(env) for s in self.sections],
        )


class Index:
    """Component for a directory index"""

    def __init__(
        self,
        parent: str | None,
        name: str,
        files: Iterable[str],
        subdirs: Iterable[str],
    ):
        self.parent = parent
        self.name = name
        self.files = tuple(files)
        self.subdirs = tuple(subdirs)

    def render(self, env: Environment) -> str:
        template = env.get_template("index.html")
        logo = StaticImage("assets/logo.png")
        return template.render(
            title=self.name,
            logo=logo.render(env),
            date=today(),
            parent=self.parent,
            dirname=self.name,
            subdirs=self.subdirs,
            files=self.files,
        )
