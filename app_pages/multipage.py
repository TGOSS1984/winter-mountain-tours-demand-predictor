# app_pages/multipage.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import streamlit as st


PageFunc = Callable[[], None]


@dataclass
class Page:
    title: str
    func: PageFunc


@dataclass
class MultiPage:
    """
    Simple multipage pattern for Streamlit.

    Usage:
        app = MultiPage(app_name="My App")
        app.add_page("Home", home_app)
        app.add_page("EDA", eda_app)
        app.run()
    """
    app_name: str
    pages: List[Page] = field(default_factory=list)

    def add_page(self, title: str, func: PageFunc) -> None:
        self.pages.append(Page(title=title, func=func))

    def run(self) -> None:
        st.sidebar.title(self.app_name)
        page_titles = [p.title for p in self.pages]

        selected_title = st.sidebar.radio("Navigation", page_titles)
        page = next(p for p in self.pages if p.title == selected_title)

        # Render selected page
        page.func()
