import argparse
import logging
import webbrowser
from threading import Timer

from deepecohab.app import create_app
from deepecohab.app.services import CACHE_DIR


def main() -> None:
	"""Parse CLI args and run the app."""
	parser = argparse.ArgumentParser(prog="deepecohab-app")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8050)
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--no-browser", action="store_true")
	args = parser.parse_args()

	# A double-clicked GUI has no stderr to read, so warnings go to a file instead.
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		filename=None if args.debug else CACHE_DIR / "app.log",
		format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
	)

	if args.host not in ("127.0.0.1", "localhost", "::1"):
		warning = (
			f"Serving on {args.host}: the app has no login, so anyone who can reach this machine "
			"can read project data, create folders and update the app."
		)
		print(f"WARNING: {warning}", flush=True)
		logging.warning(warning)

	app = create_app()

	if not args.no_browser:
		url = f"http://{args.host}:{args.port}"
		Timer(1, lambda: webbrowser.open(url)).start()

	app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
	main()
