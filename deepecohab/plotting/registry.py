import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import UnionType
from typing import (
	TYPE_CHECKING,
	Any,
	ClassVar,
	Literal,
	Union,
	get_args,
	get_origin,
	get_type_hints,
)

import polars as pl

if TYPE_CHECKING:
	from deepecohab.plotting.context import PlotContext

ChoiceResolver = Callable[["PlotContext"], Sequence[Any]]


class Figure(dict):
	"""A plot as the ``data`` and ``layout`` dict plotly.js draws.

	Built without plotly's figure objects, whose validation costs many times what drawing
	the data does; a ``dcc.Graph`` takes the dict as it is, and
	:func:`~deepecohab.plotting.to_figure` turns it into a ``go.Figure``. A ``dict`` subclass
	only so it can carry the frames it was drawn from, which stay out of its JSON.
	"""

	_plotted_frames: dict[str, pl.DataFrame]


def plotted(figure: Figure, **frames: pl.DataFrame) -> Figure:
	"""``figure``, carrying the frames it was drawn from for :meth:`PlotRegistry.data`."""
	figure._plotted_frames = frames
	return figure


@dataclass(frozen=True)
class Option:
	"""One user-settable argument of a plot.

	Attributes:
		name: keyword the builder takes.
		label: human-readable control label.
		choices: allowed values, empty when the option is free-form.
		default: value used when the caller passes nothing.
	"""

	name: str
	label: str
	choices: tuple[Any, ...]
	default: Any


@dataclass(frozen=True, eq=False)
class PlotSpec:
	"""A registered plot and everything a caller needs to drive it.

	Attributes:
		name: registry key.
		title: display title.
		summary: first line of the builder's docstring.
		info: what the plot shows and how it is calculated, for the card's info popup.
		requires: analysis tables the plot reads, checked by :meth:`PlotRegistry.build`.
		options: the builder's keyword arguments.
		builder: the plot function itself.
		dynamic_choices: resolvers for options whose choices depend on the cohort.
	"""

	name: str
	title: str
	summary: str
	info: str
	requires: tuple[str, ...]
	options: tuple[Option, ...]
	builder: Callable[..., Figure]
	dynamic_choices: Mapping[str, ChoiceResolver]

	def options_for(self, context: "PlotContext") -> tuple[Option, ...]:
		"""Options with cohort-dependent choices resolved.

		Returns:
			The options. A resolved option whose declared default is not among its
			choices falls back to the first, so the default is always selectable.
		"""
		resolved = []

		for option in self.options:
			resolver = self.dynamic_choices.get(option.name)
			choices = tuple(resolver(context)) if resolver is not None else ()

			# Nothing to narrow with, so the declared default still stands.
			if not choices:
				resolved.append(option)
				continue

			if isinstance(option.default, (list, tuple)):
				# A multi-select option, e.g. phase_type: narrow to what still applies
				# rather than collapsing to one choice. Its default may be a tuple, so
				# that one shared list cannot be mutated by a caller into every plot.
				default = [value for value in option.default if value in choices] or list(choices)
			else:
				default = option.default if option.default in choices else choices[0]
			resolved.append(Option(option.name, option.label, choices, default))

		return tuple(resolved)


def _literal_choices(annotation: Any) -> tuple[Any, ...]:
	"""Every value a ``Literal`` allows, so a type alias built from one still applies.

	Also flattens a union of ``Literal``s, e.g. ``Unit | Literal["auto"]``. A union
	with a non-``Literal`` member (``str | None``, say) has no fixed choice set, so
	it returns empty rather than a partial one.
	"""
	origin = get_origin(annotation)

	if origin is Literal:
		return get_args(annotation)

	if origin is UnionType or origin is Union:
		choices: list[Any] = []
		for member in get_args(annotation):
			member_choices = _literal_choices(member)
			if not member_choices:
				return ()
			choices.extend(value for value in member_choices if value not in choices)
		return tuple(choices)

	return ()


def _options_from_signature(
	name: str,
	func: Callable[..., Figure],
	dynamic: frozenset[str],
) -> tuple[Option, ...]:
	"""Read a builder's keyword-only parameters as options."""
	hints = get_type_hints(func)
	options = []

	for parameter in inspect.signature(func).parameters.values():
		if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
			continue

		if parameter.default is inspect.Parameter.empty:
			raise TypeError(
				f"plot {name!r} option {parameter.name!r} has no default; "
				"every option needs one so a plot can be built without arguments"
			)

		annotation = hints.get(parameter.name)
		choices = _literal_choices(annotation)

		if choices and parameter.name in dynamic:
			raise TypeError(
				f"plot {name!r} option {parameter.name!r} has both a Literal annotation "
				"and a dynamic resolver, so the annotation would be discarded"
			)

		if choices and parameter.default not in choices:
			raise ValueError(
				f"plot {name!r} option {parameter.name!r} defaults to {parameter.default!r}, "
				f"which is not one of {list(choices)}"
			)

		options.append(
			Option(
				name=parameter.name,
				label=parameter.name.replace("_", " ").capitalize(),
				choices=choices,
				default=parameter.default,
			)
		)

	return tuple(options)


class PlotRegistry:
	"""Which plots exist, what each one accepts, and how to build it.

	Plots register at import time, so the catalog belongs to the class. A plot
	declares its options as keyword-only parameters: a ``Literal`` annotation
	supplies the allowed values and the parameter default supplies the default, so
	the signature stays the single source of truth.
	"""

	_specs: ClassVar[dict[str, PlotSpec]] = {}

	@classmethod
	def register(
		cls,
		name: str,
		*,
		title: str,
		info: str,
		requires: tuple[str, ...],
		dynamic_choices: Mapping[str, ChoiceResolver] | None = None,
	) -> Callable[[Callable[..., Figure]], Callable[..., Figure]]:
		"""Register a plot builder under ``name``.

		Args:
			info: what the plot shows and how its data is transformed, in a few sentences.
			requires: analysis tables the builder reads.
			dynamic_choices: resolvers for options whose choices depend on the
				cohort, keyed by option name.

		Raises:
			TypeError: an option lacks a default, collides with a resolver, or a
				resolver names an option the builder does not take.
			ValueError: an option's default is outside its own choices.

		Returns:
			A decorator returning the builder unchanged.
		"""
		resolvers = dict(dynamic_choices or {})

		def wrapper(func: Callable[..., Figure]) -> Callable[..., Figure]:
			options = _options_from_signature(name, func, frozenset(resolvers))
			unknown = set(resolvers) - {option.name for option in options}

			if unknown:
				raise TypeError(
					f"plot {name!r} declares dynamic choices for unknown "
					f"option(s) {sorted(unknown)}"
				)

			docstring = inspect.getdoc(func) or ""
			cls._specs[name] = PlotSpec(
				name=name,
				title=title,
				summary=docstring.split("\n", 1)[0],
				info=info,
				requires=requires,
				options=options,
				builder=func,
				dynamic_choices=resolvers,
			)

			return func

		return wrapper

	@classmethod
	def spec(cls, name: str) -> PlotSpec:
		"""Return one plot's spec.

		Raises:
			KeyError: no plot is registered under ``name``.
		"""
		if name not in cls._specs:
			raise KeyError(f"unknown plot {name!r}; available: {sorted(cls._specs)}")

		return cls._specs[name]

	@classmethod
	def list_available(cls) -> list[str]:
		"""Every registered plot name."""
		return list(cls._specs)

	@classmethod
	def build(cls, name: str, context: "PlotContext", **options: Any) -> Figure:
		"""Build a plot.

		Args:
			context: the data and cohort metadata to plot.
			**options: overrides for the builder's keyword arguments.

		Raises:
			KeyError: no plot is registered under ``name``.
			ValueError: a required table is missing, or an option was given a value
				outside its choices.
			TypeError: an option is not one the plot accepts.
		"""
		plot_spec = cls.spec(name)
		missing = [table for table in plot_spec.requires if table not in context]

		if missing:
			raise ValueError(
				f"plot {name!r} requires {missing}, which the context does not have; "
				"run the analysis for those steps first"
			)

		resolved = plot_spec.options_for(context)
		values = {option.name: option.default for option in resolved}
		values.update(options)

		for option in resolved:
			if not option.choices:
				continue

			value = values[option.name]
			valid = (
				set(value) <= set(option.choices)
				if isinstance(value, (list, tuple))
				else value in option.choices
			)
			if not valid:
				raise ValueError(
					f"plot {name!r} option {option.name!r} must be one of "
					f"{list(option.choices)}, got {value!r}"
				)

		return plot_spec.builder(context, **values)

	@classmethod
	def data(cls, name: str, context: "PlotContext", **options: Any) -> dict[str, pl.DataFrame]:
		"""The frames a plot is drawn from, by name: what its CSV export holds.

		Takes the arguments of :meth:`build`, and raises as it does.
		"""
		return getattr(cls.build(name, context, **options), "_plotted_frames", {})
