import datetime as dt

import polars as pl

from deepecohab.core import grids, transforms
from deepecohab.core.data_model import (
	CALENDAR_COLUMNS,
	AnalysisParams,
	DataFrameRegistry,
	Layout,
	Recording,
)
lazy from deepecohab.core import ranking_algorithm


def _get_activity(padded: pl.LazyFrame, recording: Recording) -> pl.LazyFrame:
	"""Frame of per-animal dwell time and visit counts for every position.

	Drops tunnel directionality so the two ends of a tunnel collapse to one position,
	then sums dwell time and counts visits per animal in each position and hour.
	Interpolated pieces are excluded from the visit count but still contribute their
	time.
	"""
	return (
		transforms.remove_tunnel_directionality(padded, recording)
		.group_by([*CALENDAR_COLUMNS, "position", "animal_id"])
		.agg(
			pl.sum("time_spent").alias("time_in_position"),
			(~pl.col("interpolated")).sum().alias("visits_to_position"),
		)
	)


def _occupancy_intervals(frame: pl.LazyFrame, recording: Recording) -> pl.LazyFrame:
	"""Frame of occupancy intervals for every real position, tunnels undirected.

	The shared front half of solitary time and co-presence, so the two see the same
	positions. Directionality is dropped first, or two animals passing through one
	tunnel in opposite directions would sit in different positions and never meet, and
	``undefined`` is excluded because it is not a place.
	"""
	return (
		transforms.remove_tunnel_directionality(frame, recording)
		.filter(pl.col("position") != Layout.UNDEFINED)
		.pipe(transforms.add_occupancy_bounds)
	)


def _get_time_alone(
	padded: pl.LazyFrame, recording: Recording, params: AnalysisParams
) -> pl.LazyFrame:
	"""Frame of how long each animal occupied a position with no other animal present.

	A span from :func:`transforms.occupancy_spans` holding one animal is time that
	animal spent alone. The sweep also cuts wherever the animal itself reads or a cell
	of the grid begins, so back-to-back spans of one animal in one position make up a
	single continuous bout: a bout shorter than ``params.minimum_time_alone`` is dropped
	whole, and a longer one keeps every span. ``undefined`` positions are excluded.
	"""
	new_bout = (pl.col("time") != pl.col("end").shift(1)).fill_null(True)

	return (
		transforms.occupancy_spans(_occupancy_intervals(padded, recording))
		.filter(pl.col("animals_present") == 1)
		.sort("animal_id", "position", "time")
		.with_columns(new_bout.cum_sum().over("animal_id", "position").alias("bout"))
		.filter(
			(pl.col("end") - pl.col("time")).sum().over("animal_id", "position", "bout")
			> pl.duration(seconds=params.minimum_time_alone)
		)
		.select(
			"animal_id",
			"position",
			pl.col("time").alias("datetime"),
			pl.col("end").sub(pl.col("time")).alias("duration"),
		)
		# The spans are new intervals rather than registrations, so they take their own
		# calendar columns instead of inheriting any.
		.with_columns(
			grids.get_phase(recording), grids.get_day(recording), grids.get_hour(recording)
		)
		.pipe(grids.assign_phase_count, recording)
		.group_by("animal_id", "position", *CALENDAR_COLUMNS)
		.agg(pl.col("duration").sum().alias("time_alone"))
	)


@DataFrameRegistry.register("activity_df", requires=["padded_df"])
def calculate_activity(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of per-animal occupancy, visits and solitary time for every position.

	Combines dwell time and visit counts with solitary occupancy and reindexes both
	onto the dense experiment grid, so every animal, position and hour is present and
	cells with no activity read ``0``.

	Returns:
		``time_in_position``, ``visits_to_position`` and ``time_alone`` per animal,
		position and hour.
	"""
	padded = recording.load_results("padded_df")

	activity = pl.concat(
		[_get_activity(padded, recording), _get_time_alone(padded, recording, params)],
		how="align",
	)

	return grids.reindex_onto_grid(
		activity, recording, "animal_id", positions=recording.layout.positions_non_directional
	)


@DataFrameRegistry.register("match_df", requires=["main_df"])
def calculate_matches(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of chasing events, one row each.

	A chasing event is a loser entering a tunnel from a cage and a winner following
	through the same tunnel within ``params.chasing_time_window`` seconds. This table
	is the shared input for both the per-hour chasing counts and the ranking, so it is
	computed once here.

	Returns:
		One row per chasing event, with winner, loser, tunnel, how far behind the
		winner left the tunnel, and the calendar columns of the winner's registration.
	"""
	registrations = recording.load_results("main_df").sort("datetime")

	cages = recording.layout.cage_names
	tunnels = recording.layout.tunnel_names_directional

	# A row's `position`/`datetime` mark the position the animal left and when, so the winner's
	# `tunnel_entry` is its entry and the loser's `loser_exit` is its exit at the far end.
	# Winner candidates: a tunnel exit whose previous read was a cage, carrying that tunnel's
	# entry time. Each event is dated by the winner, so it also takes the winner's calendar.
	chasing = (
		registrations.with_columns(
			pl.col("datetime").shift(1).over("animal_id").alias("tunnel_entry"),
			pl.col("position").shift(1).over("animal_id").alias("previous_position"),
		)
		.filter(pl.col("position").is_in(tunnels), pl.col("previous_position").is_in(cages))
		.select(
			"position",
			pl.col("animal_id").alias("winner"),
			pl.col("datetime").alias("winner_exit"),
			"tunnel_entry",
			*CALENDAR_COLUMNS,
		)
	)

	chased = registrations.filter(pl.col("position").is_in(tunnels)).select(
		"position",
		pl.col("animal_id").alias("loser"),
		pl.col("datetime").alias("loser_exit"),
	)

	# Pair each loser exit with every winner still inside the same tunnel via a sweep (as in
	# calculate_pairwise_meetings). Each winner pass is an enter(+bit)/leave(-bit) event and each
	# loser exit a zero-delta query, so the per-tunnel cumulative bit-sum is the presence bitmask
	# of winners inside (one bit per animal, capped at ~63, far above any EcoHab study).
	# The window is about the follow-through: how long after the winner entered the tunnel
	# the loser came out of it. chasing_length is the chase itself, the gap between the two
	# animals' exits, so both are one animal's instant minus the other's - but not the same
	# pair of instants.
	follow_through = pl.col("loser_exit") - pl.col("tunnel_entry")
	chasing_length = pl.col("winner_exit") - pl.col("loser_exit")
	shortest, longest = (pl.duration(seconds=bound) for bound in params.chasing_time_window)

	winners, winner_lookup = transforms.with_animal_bits(chasing, "winner")

	# `kind` orders ties at equal timestamps -- enter(0) < query(1) < leave(2) -- so a winner
	# counts as inside iff tunnel_entry <= loser_exit < winner_exit; the strict window below
	# re-asserts this.
	enters = winners.select(
		pl.col("tunnel_entry").alias("time"),
		"position",
		pl.col("bit").alias("bit_delta"),
		pl.lit(0, dtype=pl.Int8).alias("kind"),
	)
	leaves = winners.select(
		pl.col("winner_exit").alias("time"),
		"position",
		pl.col("bit").neg().alias("bit_delta"),
		pl.lit(2, dtype=pl.Int8).alias("kind"),
	)
	queries = chased.select(
		pl.col("loser_exit").alias("time"),
		"position",
		pl.lit(0, dtype=pl.Int64).alias("bit_delta"),
		pl.lit(1, dtype=pl.Int8).alias("kind"),
		"loser",
	)

	# At each loser exit, `mask` is the OR-set of bits of the winners then inside the tunnel.
	mask_at_query = (
		pl.concat([enters, leaves, queries], how="diagonal")
		.sort("position", "time", "kind")
		.with_columns(pl.col("bit_delta").cum_sum().over("position").alias("mask"))
		.filter(pl.col("kind") == 1)
		.select("position", "loser", pl.col("time").alias("loser_exit"), "mask")
	)

	# Decode the mask into candidate winners, then recover each winner's open pass with an
	# as-of join (its latest entry at or before the loser exit); a plain equi-join would
	# match all passes and be quadratic.
	candidates = mask_at_query.join(winner_lookup, how="cross").filter(
		(pl.col("mask") & pl.col("bit")) != 0
	)

	return (
		candidates.sort("loser_exit")
		.join_asof(
			winners.select(
				"position", "winner", "winner_exit", "tunnel_entry", *CALENDAR_COLUMNS
			).sort("tunnel_entry"),
			left_on="loser_exit",
			right_on="tunnel_entry",
			by=["position", "winner"],
			strategy="backward",
			# Both sides are globally sorted on the as-of key above, which polars
			# can't verify per `by` group.
			check_sortedness=False,
		)
		.filter(
			follow_through > shortest,
			follow_through < longest,
			pl.col("loser_exit") < pl.col("winner_exit"),
			pl.col("winner") != pl.col("loser"),
		)
		.select(
			"position",
			"winner",
			"loser",
			pl.col("winner_exit").alias("datetime"),
			chasing_length.alias("chasing_length"),
			*CALENDAR_COLUMNS,
		)
		.sort("datetime")
	)


@DataFrameRegistry.register("chasings_df", requires=["match_df"])
def calculate_chasings(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame counting chasing events per ordered pair of animals, tunnel and hour.

	The event-level ``winner``/``loser`` become ``chaser``/``chased`` here, since in a
	chasing the winner is the one doing the chasing.

	Returns:
		A ``chasings`` count per chaser, chased, tunnel and hour, for the cells with at
		least one chasing; a missing cell is a zero (see ``time_grid``).
	"""
	return (
		recording.load_results("match_df")
		.group_by([*CALENDAR_COLUMNS, "position", "winner", "loser"])
		.len(name="chasings")
		.rename({"winner": "chaser", "loser": "chased"})
	)


@DataFrameRegistry.register("ranking", requires=["match_df"])
def calculate_ranking(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of a dominance ranking, from replaying chasing events as matches.

	Each chasing event is a one-on-one match the winner won. Replaying them in order
	updates every animal's skill rating after each match, so the result is the full
	rating trajectory rather than only the final standings. Ratings stored with
	:meth:`Recording.set_prev_ranking` continue from an earlier recording of the same
	animals instead of from scratch, unless ``params.use_prev_ranking`` is off.

	Returns:
		One row per animal after each match, with ``mu``, ``sigma``, ``ordinal``, the
		``social_rank`` its ordinal puts it in at that point, and the calendar columns
		of the match.
	"""
	model = ranking_algorithm.MODEL
	ranking = dict.fromkeys(recording.cohort.animal_tags, (model.mu, model.sigma))

	previous_ranking = recording.prev_ranking if params.use_prev_ranking else None
	if previous_ranking is not None:
		for name, mu, sigma in previous_ranking.select("animal_id", "mu", "sigma").iter_rows():
			ranking[name] = (float(mu), float(sigma))

	# The replay is order-dependent, so equal timestamps need a tiebreak or the ratings
	# shift between runs. `position` is deliberately left out: it is Categorical, so its
	# sort order is physical and would put the non-determinism straight back. A winner has
	# one exit per instant, so the triple is already unique.
	matches = (
		recording.load_results("match_df")
		.sort("datetime", "loser", "winner")
		.select("loser", "winner", "datetime", *CALENDAR_COLUMNS)
		.collect()
	)

	animals = list(ranking)
	mu_trajectory, sigma_trajectory = ranking_algorithm.replay(
		ranking, zip(matches["winner"], matches["loser"], strict=True)
	)

	return (
		matches.lazy()
		.select(pl.col("datetime", *CALENDAR_COLUMNS).repeat_by(len(animals)).explode())
		.with_columns(
			animal_id=pl.Series(animals * matches.height, dtype=pl.String),
			mu=pl.Series(mu_trajectory, dtype=pl.Float64),
			sigma=pl.Series(sigma_trajectory, dtype=pl.Float64),
		)
		.with_columns(ordinal=(pl.col("mu") - 3 * pl.col("sigma")).round(3))
		.select("animal_id", "mu", "sigma", "ordinal", "datetime", *CALENDAR_COLUMNS)
	).with_columns(
		pl.when(pl.col("ordinal") == pl.col("ordinal").max().over("datetime"))
		.then(pl.lit("dominant"))
		.when(pl.col("ordinal") == pl.col("ordinal").min().over("datetime"))
		.then(pl.lit("subordinate"))
		.otherwise(pl.lit("middle"))
		.cast(pl.Enum(["dominant", "middle", "subordinate"]))
		.alias("social_rank")
	)


def get_prev_ranking(ranking: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
	"""Frame collapsing a ranking trajectory to each animal's latest rating.

	:func:`calculate_ranking` emits one row per animal after every match; this keeps
	only the last ``mu``/``sigma`` per animal, which is what
	:meth:`Recording.set_prev_ranking` stores to continue ranking the same animals from
	where a previous recording left off. ``ranking`` may already be collapsed to this
	shape - sorting only runs when a ``datetime`` column is there to sort by.

	Raises:
		ValueError: ``ranking`` lacks ``animal_id``, ``mu`` or ``sigma``.
	"""
	ranking = ranking.lazy()
	columns = ranking.collect_schema().names()
	if missing := [column for column in ("animal_id", "mu", "sigma") if column not in columns]:
		raise ValueError(f"The ranking is missing {', '.join(missing)}.")
	if "datetime" in columns:
		ranking = ranking.sort("datetime")

	return ranking.group_by("animal_id", maintain_order=True).agg(pl.last("mu"), pl.last("sigma"))


@DataFrameRegistry.register("pairwise_meetings", requires=["main_df"])
def calculate_pairwise_meetings(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of co-occurrences and shared time for every pair of animals, position and hour.

	Each animal's back-to-back registrations in one position merge into a stay, and two
	animals' overlapping stays in one position are a meeting, from the later arrival to
	the earlier departure. Meetings shorter than ``params.minimum_time`` seconds are
	dropped; the rest are cut at the cells of the grid, so their time lands in the hours
	it was spent, and summed per cell. A meeting counts as one encounter, in
	the cell it began in.

	Cages and tunnels are both covered, on the same positions ``_get_time_alone`` reads.
	Tunnel co-presence is physically different - two animals in a two-antenna tunnel are in
	contact, and such a meeting is what the tube test reads as a contest - but it is held to
	the same ``params.minimum_time`` for now; whether that threshold suits tunnels is a
	question for real data.

	Returns:
		``time_together`` and ``pairwise_encounters`` per pair, position and hour, for
		the cells a pair met in; a missing cell is a zero (see ``time_grid``).
	"""
	stays = transforms.merge_contiguous_stays(
		_occupancy_intervals(recording.load_results("main_df"), recording)
	)
	meetings = transforms.overlapping_stays(stays).filter(
		pl.col("end") - pl.col("start") > dt.timedelta(seconds=params.minimum_time)
	)

	return (
		transforms.split_on_cell_boundaries(meetings, recording, "start", "end")
		.pipe(grids.assign_phase_count, recording)
		.group_by([*CALENDAR_COLUMNS, "position", "animal_id", "animal_id_2"])
		.agg(
			(pl.col("end") - pl.col("start")).sum().alias("time_together"),
			pl.col("first_piece").sum().alias("pairwise_encounters"),
		)
	)


def _require_chance(chance: pl.Series) -> pl.Series:
	"""``chance`` unchanged, or a raise if any of it is null.

	A null means the joins behind it found no occupancy or no phase duration for a
	pair-cage-phase that co-presence was recorded in. Summed, that cage would simply
	contribute its observed time with no chance term subtracted, which reads as
	sociability the animals never showed. A lazy frame has no assert and the step is
	sunk streaming, so the check rides along as a batch-wise map.
	"""
	if chance.null_count():
		raise ValueError(
			f"{chance.null_count()} of {len(chance)} chance terms are null: co-presence was "
			"recorded for a pair, cage and phase that activity_df or phase_durations does "
			"not cover, so those cages would be silently skipped."
		)
	return chance


@DataFrameRegistry.register(
	"incohort_sociability", requires=["pairwise_meetings", "activity_df", "phase_durations"]
)
def calculate_incohort_sociability(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of in-cohort sociability: observed togetherness minus chance expectation.

	For each pair and cage, the time the pair actually spent together, as a fraction
	of phase duration, is compared against the time they would be expected to share by
	chance given each animal's independent occupancy of that cage. ``sociability`` is
	the observed-minus-chance difference summed over cages, so a positive value means
	the pair sought each other out. For background see DOI:10.7554/eLife.19532.

	Returns:
		``proportion_together`` and ``sociability`` per phase for every pair.
	"""
	phase_durations = recording.load_results("phase_durations")
	time_together = recording.load_results("pairwise_meetings")
	activity = recording.load_results("activity_df")

	core_columns = ["phase", "day", "phase_count", "animal_id", "animal_id_2"]
	phase_seconds = pl.col("duration").dt.total_seconds(fractional=True)

	time_together = (
		time_together.filter(pl.col("position").is_in(recording.layout.cage_names))
		.group_by([*core_columns, "position"])
		.agg(pl.sum("time_together"))
	)

	activity_per_phase = (
		activity.filter(pl.col("position").is_in(recording.layout.cage_names))
		.group_by(["phase", "day", "phase_count", "position", "animal_id"])
		.agg(pl.sum("time_in_position"))
	)

	expected_together = activity_per_phase.join(
		activity_per_phase, on=["phase", "day", "phase_count", "position"], suffix="_2"
	).filter(pl.col("animal_id") < pl.col("animal_id_2"))

	return (
		# Full, as pairwise_meetings leaves out the cages a pair never met in, and those
		# still subtract their chance term; a meeting with no occupancy behind it leaves a
		# null chance, which _require_chance turns into a raise.
		time_together.join(
			expected_together, on=[*core_columns, "position"], how="full", coalesce=True
		)
		.with_columns(pl.col("time_together").fill_null(dt.timedelta(0)))
		# An inner join, unlike the one above: the dense grid numbers a phase occurrence for
		# the window's final instant whenever that instant opens one, and phase_durations
		# measures no duration for it. Nothing can happen in an instant, so those cells are
		# zero-filled grid rows only, and reporting a sociability over a null duration would
		# be reporting on no time at all.
		.join(phase_durations, on=["phase_count", "phase"])
		.with_columns(
			pl.col("time_together").dt.total_seconds(fractional=True) / phase_seconds,
			(
				pl.col("time_in_position").dt.total_seconds(fractional=True)
				* pl.col("time_in_position_2").dt.total_seconds(fractional=True)
				/ phase_seconds**2
			)
			.map_batches(_require_chance, return_dtype=pl.Float64)
			.alias("chance"),
		)
		.group_by(core_columns)
		.agg(
			pl.sum("time_together").alias("proportion_together"),
			(pl.col("time_together") - pl.col("chance")).sum().alias("sociability"),
		)
		.sort(core_columns)
	)


@DataFrameRegistry.register(
	"feature_df",
	requires=["chasings_df", "pairwise_meetings", "activity_df", "main_df"],
)
def calculate_features(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of per-animal metrics, each paired with the opportunity it arose from.

	Collapses the upstream tables to one value per animal per hour for each metric -
	chasings given and received, activity, time alone, and time together with
	encounters - and pairs every value with the exposure it should be read against. A
	rate is ``sum(value) / sum(exposure)`` over whatever grouping is wanted, so the
	hourly rows sum to a phase, a day or the whole recording without distorting the
	result.

	Activity, time alone, time together and encounters are held per position, each
	exposed against the time spent there, so a rate filtered to some positions reads
	within them and the positions still sum back to the whole. Chasings belong to no
	position, so theirs is null.

	Durations are held in hours, so a duration metric's rate is a fraction of the time
	observed and a count metric's rate is a count per hour. Metrics that need a partner
	are exposed per available partner, which keeps cohorts of different sizes
	comparable. ``n_chasing_per_detection`` is the same chase count as ``n_chasing``
	exposed per antenna detection instead, normalising by how much the animal was seen
	rather than by how many partners it had.

	Returns:
		One row per animal, hour, metric and position, with ``value`` and ``exposure``.
	"""
	solo = ["activity", "time_alone"]
	paired = ["time_together", "pairwise_encounters", "n_chasing", "n_chased"]
	per_detection = ["n_chasing_per_detection"]
	placed = ["activity", "time_alone", "time_together", "pairwise_encounters"]
	keys = [*CALENDAR_COLUMNS, "animal_id"]

	chasings = recording.load_results("chasings_df")
	pairwise_meetings = recording.load_results("pairwise_meetings")

	# padded_df tiles each animal's whole timeline, so its time summed over positions is
	# how long that animal was observed for in the cell - the denominator for the rest.
	by_position = recording.load_results("activity_df").select(
		*keys,
		"position",
		pl.col("visits_to_position").alias("activity"),
		pl.col("time_alone").dt.total_hours(fractional=True),
		pl.col("time_in_position").dt.total_hours(fractional=True).alias("observed_hours"),
	)
	observed = by_position.group_by(keys).agg(pl.sum("observed_hours"))

	n_detections = (
		recording.load_results("main_df").group_by(keys).agg(pl.len().alias("n_detections"))
	)

	n_chasing = (
		chasings.group_by([*CALENDAR_COLUMNS, "chaser"])
		.agg(pl.sum("chasings").alias("n_chasing"))
		.rename({"chaser": "animal_id"})
	)

	n_chased = (
		chasings.group_by([*CALENDAR_COLUMNS, "chased"])
		.agg(pl.sum("chasings").alias("n_chased"))
		.rename({"chased": "animal_id"})
	)

	pairwise_meetings = (
		pairwise_meetings.unpivot(
			on=["animal_id", "animal_id_2"],
			index=[*CALENDAR_COLUMNS, "position", "time_together", "pairwise_encounters"],
			variable_name="_drop",
			value_name="col",
		)
		.drop("_drop")
		.group_by([*CALENDAR_COLUMNS, "position", "col"])
		.agg(
			pl.sum("time_together").dt.total_hours(fractional=True),
			pl.sum("pairwise_encounters"),
		)
		.rename({"col": "animal_id"})
	)

	partners = recording.cohort.n_mice - 1
	# A lone animal has no partner, so a paired metric has neither a value nor an exposure
	# to speak of; leaving it out of `on` drops the columns with it, as unpivot keeps only
	# `index` and `on`. `solo` and `per_detection` remain, so `on` is never empty.
	metrics = [*solo, *(paired if partners else []), *per_detection]

	# observed_hours is the time spent at the row's position, or in all of them when it
	# has none, so the one exposure rule below serves both halves.
	placed_rows = (
		pl.concat([by_position, pairwise_meetings], how="align")
		.fill_null(0)
		.with_columns(pl.col(placed).cast(pl.Float64))
		.unpivot(
			on=[metric for metric in metrics if metric in placed],
			index=[*keys, "position", "observed_hours"],
			variable_name="metric",
			value_name="value",
		)
	)
	unplaced_rows = (
		pl.concat([observed, n_detections, n_chasing, n_chased], how="align")
		.fill_null(0)
		.with_columns(pl.col("n_chasing").alias("n_chasing_per_detection"))
		.with_columns(pl.col("n_chasing", "n_chased", *per_detection).cast(pl.Float64))
		.unpivot(
			on=[metric for metric in metrics if metric not in placed],
			index=[*keys, "observed_hours", "n_detections"],
			variable_name="metric",
			value_name="value",
		)
	)

	return (
		pl.concat([placed_rows, unplaced_rows], how="diagonal")
		.with_columns(
			pl.when(pl.col("metric").is_in(per_detection))
			.then(pl.col("n_detections"))
			.when(pl.col("metric").is_in(paired))
			.then(pl.col("observed_hours") * partners)
			.otherwise(pl.col("observed_hours"))
			.alias("exposure")
		)
		.drop("observed_hours", "n_detections")
		.sort(*keys, "metric", "position")
	)
