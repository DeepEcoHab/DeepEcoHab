"""The citation people copy names the version they installed."""

import re
from pathlib import Path

import deepecohab

REPO = Path(__file__).parent.parent


def test_citations_name_the_package_version():
	"""CITATION.cff, the docs bibliography and the docs' BibTeX block are bumped with pyproject."""
	cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
	versions = [re.search(r"^version: (\S+)$", cff, re.MULTILINE)]
	for name in ("references.bib", "reference.md"):
		bib = (REPO / "docs" / name).read_text(encoding="utf-8")
		versions.append(re.search(r"version\s*=\s*\{([^}]+)\}", bib))

	assert [match and match[1] for match in versions] == [deepecohab.__version__] * 3
