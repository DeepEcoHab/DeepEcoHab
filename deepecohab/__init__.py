from importlib.metadata import version

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
lazy from deepecohab.plotting import (
	PlotContext as PlotContext,
	PlotRegistry as PlotRegistry,
	available_attributes as available_attributes,
	export_figure as export_figure,
	fit_for_export as fit_for_export,
	plot as plot,
	to_figure as to_figure,
)

__version__ = version("deepecohab")
