import itertools

lazy import networkx as nx


def antennas(antenna_combinations: dict[str, str]) -> set[str]:
	"""The antennas the layout names."""
	return {*pair.split("_") for pair in antenna_combinations}


def unique_routes(antenna_combinations: dict[str, str]) -> dict[tuple[str, str], list[str]]:
	"""The antennas an animal must have crossed for each step the layout forbids.

	A step between two antennas the layout does not join is only possible if the
	antennas in between failed to read. Where more than one route is equally short
	there is no telling which antennas those were, and the step is left out.

	Returns:
		The antennas crossed, in order, keyed by the antennas the step was seen
		between.
	"""
	pairs = (pair.split("_") for pair in antenna_combinations)
	graph = nx.DiGraph([(source, target) for source, target in pairs if source != target])

	routes: dict[tuple[str, str], list[str]] = {}
	for source, target in itertools.permutations(graph, 2):
		if graph.has_edge(source, target) or not nx.has_path(graph, source, target):
			continue
		shortest = itertools.islice(nx.all_shortest_paths(graph, source, target), 2)
		match list(shortest):
			case [path]:
				routes[source, target] = path[1:-1]

	return routes
