"""The site builds with no results, and loads nothing from the network."""

import re

from gaitpinn import site


def test_builds_empty_and_has_no_remote_urls(tmp_path):
    out = tmp_path / "site"
    site.build(tmp_path / "runs", out)
    for name in ["index.html", "people.html", "runs.html", "method.html", "plotly.min.js"]:
        assert (out / name).exists()
    for f in out.glob("*.html"):
        text = f.read_text()
        assert not re.search(r"""<(script|link)[^>]+(src|href)=["']?(https?:)?//""", text), f
    assert "No results yet" in (out / "index.html").read_text()
