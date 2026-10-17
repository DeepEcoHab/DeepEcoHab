import math
from collections.abc import Iterable

from openskill.models import PlackettLuce

MODEL = PlackettLuce(limit_sigma=True, balance=True)
"""The rating model; its ``mu``/``sigma`` are the rating an animal starts from."""


def replay(
	ratings: dict[str, tuple[float, float]], matches: Iterable[tuple[str, str]]
) -> tuple[list[float], list[float]]:
	"""Rating trajectories from replaying one-on-one matches in order.

	``MODEL.rate`` written out for a match of one animal against one, which is ~20x
	faster than calling it and gives the same floats bit for bit (tests/test_ranking.py
	checks this). Every float operation mirrors openskill's own - ``sigma * sigma`` is not
	``sigma**2`` in the last bit - so keep them as they are.

	Args:
		ratings: each animal's starting ``(mu, sigma)``, in the order the output lists them.
		matches: ``(winner, loser)`` pairs, in the order they were played.

	Returns:
		Every animal's ``mu`` and ``sigma`` after each match, match by match.
	"""
	index = {animal: i for i, animal in enumerate(ratings)}
	mus = [mu for mu, _ in ratings.values()]
	sigmas = [sigma for _, sigma in ratings.values()]
	tau_squared, beta_squared, kappa = MODEL.tau**2, MODEL.beta**2, MODEL.kappa

	mu_trajectory: list[float] = []
	sigma_trajectory: list[float] = []
	for winner, loser in matches:
		won, lost = index[winner], index[loser]
		# Each sigma first widens by tau, the model's drift between matches.
		winner_sigma = math.sqrt(sigmas[won] * sigmas[won] + tau_squared)
		loser_sigma = math.sqrt(sigmas[lost] * sigmas[lost] + tau_squared)
		winner_variance, loser_variance = winner_sigma**2, loser_sigma**2

		c = math.sqrt((winner_variance + beta_squared) + (loser_variance + beta_squared))
		winner_strength, loser_strength = math.exp(mus[won] / c), math.exp(mus[lost] / c)
		p_win = winner_strength / (winner_strength + loser_strength)
		p_loss = loser_strength / (winner_strength + loser_strength)

		mus[won] += (1 - p_win) * (winner_variance / c)
		mus[lost] += -p_loss * (loser_variance / c)
		winner_delta = (
			p_win * (1 - p_win) * (winner_variance / c**2) * (math.sqrt(winner_variance) / c)
		)
		loser_delta = (
			p_loss * (1 - p_loss) * (loser_variance / c**2) * (math.sqrt(loser_variance) / c)
		)
		# limit_sigma: a match never leaves a sigma wider than it went in.
		sigmas[won] = min(winner_sigma * math.sqrt(max(1 - winner_delta, kappa)), sigmas[won])
		sigmas[lost] = min(loser_sigma * math.sqrt(max(1 - loser_delta, kappa)), sigmas[lost])

		mu_trajectory.extend(mus)
		sigma_trajectory.extend(sigmas)

	return mu_trajectory, sigma_trajectory
