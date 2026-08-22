# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
Tests for the shared navigation bar.

Verifies that:
1. All HTML pages use the shared <nav id="main-nav"> placeholder
   (no copy-pasted inline nav blocks).
2. app.js contains the buildNavBar() function and exports it.
3. The nav bar includes the Music link on all pages (was previously
   missing from gallery, download, models, admin, and prompting).
"""

import re
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).parent.parent / "web"

# All HTML pages that should have a navigation bar
NAV_PAGES = [
    "index.html",
    "images.html",
    "gallery.html",
    "download.html",
    "audio.html",
    "models.html",
    "admin.html",
    "prompting.html",
    "login.html",
    "debug.html",
    "debug-cookies.html",
]


# ---------------------------------------------------------------------------
# Nav bar placeholder tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("page", NAV_PAGES)
def test_page_has_nav_placeholder(page):
    """Each page must have a <nav id="main-nav"> placeholder."""
    html = (WEB_DIR / page).read_text()
    assert '<nav id="main-nav">' in html, (
        f"{page} is missing the shared nav placeholder <nav id=\"main-nav\">"
    )


@pytest.mark.parametrize("page", NAV_PAGES)
def test_page_has_no_inline_nav_block(page):
    """No page should contain a copy-pasted inline <nav> block."""
    html = (WEB_DIR / page).read_text()
    # The placeholder should be the ONLY nav element
    nav_matches = re.findall(r'<nav\b[^>]*>', html)
    assert len(nav_matches) == 1, (
        f"{page} has {len(nav_matches)} <nav> tags — expected exactly 1 "
        f"(the shared placeholder)"
    )
    assert nav_matches[0] == '<nav id="main-nav">', (
        f"{page} has a non-placeholder nav tag: {nav_matches[0]}"
    )


@pytest.mark.parametrize("page", NAV_PAGES)
def test_page_has_no_duplicate_nav_links(page):
    """No page should have duplicated nav link markup that was copied
    alongside the inline nav (e.g. duplicate nav-toggle or theme-toggle)."""
    html = (WEB_DIR / page).read_text()
    # After replacing inline nav with placeholder, there should be no
    # nav-brand, nav-toggle, or nav-links divs left from the old inline nav
    inline_markers = ["nav-brand", "nav-toggle", "nav-links"]
    for marker in inline_markers:
        count = html.count(marker)
        assert count == 0, (
            f"{page} still has inline nav marker '{marker}' ({count} occurrences)"
        )


# ---------------------------------------------------------------------------
# app.js nav bar builder tests
# ---------------------------------------------------------------------------

def test_app_js_has_build_nav_bar():
    """app.js must define the buildNavBar function."""
    js = (WEB_DIR / "app.js").read_text()
    assert "function buildNavBar()" in js, (
        "app.js is missing the buildNavBar() function definition"
    )


def test_app_js_exports_build_nav_bar():
    """app.js must export buildNavBar in the SDAPI object."""
    js = (WEB_DIR / "app.js").read_text()
    assert "buildNavBar" in js.split("window.SDAPI = ")[1].split("};")[0], (
        "buildNavBar is not exported in window.SDAPI"
    )


def test_app_js_calls_build_nav_bar():
    """app.js must call buildNavBar() in DOMContentLoaded."""
    js = (WEB_DIR / "app.js").read_text()
    # Should call buildNavBar() inside the DOMContentLoaded handler
    domContentLoaded_match = re.search(
        r"document\.addEventListener\('DOMContentLoaded'.*?\{(.*?)\}",
        js, re.DOTALL
    )
    assert domContentLoaded_match is not None, "DOMContentLoaded handler not found"
    handler_body = domContentLoaded_match.group(1)
    assert "buildNavBar()" in handler_body, (
        "buildNavBar() is not called in the DOMContentLoaded handler"
    )


def test_app_js_nav_links_include_music():
    """The nav link definitions must include the Music link."""
    js = (WEB_DIR / "app.js").read_text()
    assert "'/web/audio.html'" in js, "Music link (/web/audio.html) not found in nav builder"
    assert "Music" in js, "Music label not found in nav builder"


def test_app_js_nav_links_include_dashboard():
    """The nav link definitions must include the Dashboard link."""
    js = (WEB_DIR / "app.js").read_text()
    assert "'/web/'" in js, "Dashboard link (/web/) not found in nav builder"
    assert "Dashboard" in js


def test_app_js_nav_links_include_prompting_guide():
    """The nav link definitions must include the Prompting Guide link."""
    js = (WEB_DIR / "app.js").read_text()
    assert "/web/prompting.html" in js, "Prompting Guide link not found in nav builder"
    assert "Prompting Guide" in js


def test_app_js_has_active_link_logic():
    """The buildNavBar function must set an 'active' class based on current path."""
    js = (WEB_DIR / "app.js").read_text()
    assert "active" in js, "No 'active' class logic found in buildNavBar"
    assert "currentPath" in js or "window.location.pathname" in js, (
        "No current path detection found in buildNavBar"
    )


def test_app_js_has_admin_only_logic():
    """The nav builder must mark admin-only links appropriately."""
    js = (WEB_DIR / "app.js").read_text()
    assert "admin-only" in js, "No 'admin-only' class in nav builder"
