from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from deepecohab.core import (
	# Imported for their side effect: defining the steps of the analysis pipeline.
	antenna_analysis as antenna_analysis,
	recording_pipeline as recording_pipeline,
)
from deepecohab.core.data_model import (
	AnalysisParams as AnalysisParams,
	Project as Project,
	Recording as Recording,
	recording_status as recording_status,
)

if TYPE_CHECKING:
	from deepecohab.plotting import (
		PlotContext as PlotContext,
		PlotRegistry as PlotRegistry,
		available_attributes as available_attributes,
		export_figure as export_figure,
		fit_for_export as fit_for_export,
		plot as plot,
		to_figure as to_figure,
	)

__version__ = version("deepecohab")

_LAZY_EXPORTS = (
	"PlotContext",
	"PlotRegistry",
	"available_attributes",
	"export_figure",
	"fit_for_export",
	"plot",
	"to_figure",
)


def __getattr__(name: str) -> Any:
	"""Resolve the plotting exports on first access."""
	if name not in _LAZY_EXPORTS:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	return getattr(import_module("deepecohab.plotting"), name)


def __dir__() -> list[str]:
	"""List the eager and lazy exports together."""
	return sorted({*globals(), *_LAZY_EXPORTS})
