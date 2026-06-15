"""Entry point for `python -m server worker` or `python -m server coordinator`."""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("worker", "coordinator"):
        print("Usage: python -m server <worker|coordinator> [options]")
        print()
        print("Commands:")
        print("  worker       Start the worker server on this machine")
        print("  coordinator  Start the coordinator server")
        print()
        print("Examples:")
        print("  python -m server worker -c server/worker_config.yaml")
        print("  python -m server coordinator -c server/coordinator_config.yaml")
        sys.exit(1)

    mode = sys.argv.pop(1)

    if mode == "worker":
        from .worker import main as worker_main
        worker_main()
    else:
        from .coordinator import main as coordinator_main
        coordinator_main()


if __name__ == "__main__":
    main()
