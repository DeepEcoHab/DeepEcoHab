from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go

from deepecohab.plotting import (
	# Imported for its side effect: registering every plot in the catalog.
	plot_catalog as plot_catalog,
)
from deepecohab.plotting.animals import available_attributes as available_attributes
from deepecohab.plotting.context import PlotContext as PlotContext
from deepecohab.plotting.export import (
	export_figure as export_figure,
	fit_for_export as fit_for_export,
)
from deepecohab.plotting.registry import Figure as Figure, PlotRegistry as PlotRegistry

if TYPE_CHECKING:
	from deepecohab.core.data_model import Recording


def plot(name: str, target: "Recording | PlotContext", **options: Any) -> Figure:
	"""Build one plot from a recording or a context, as the dict plotly.js draws.

	Hold a :class:`PlotContext` when making several plots: it caches the analysis
	tables it has already read, which a recording passed here cannot do. Pass the
	result to :func:`to_figure` to show it or save it with plotly.

	Args:
		name: registry key, as listed by :meth:`PlotRegistry.list_available`.
		**options: overrides for the plot's keyword arguments.
	"""
	context = target if isinstance(target, PlotContext) else PlotContext.from_recording(target)
	return PlotRegistry.build(name, context, **options)


def to_figure(figure: Figure) -> go.Figure:
	"""``figure`` as a ``go.Figure``, for ``show``, ``write_image`` and the rest of plotly."""
	return go.Figure(figure)
