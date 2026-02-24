"""
Entry point for ``python -m nightcore_analyzer``.

Launches the PyQt6 GUI.  If PyQt6 is not installed (e.g. a headless server)
an informative error is printed and the CLI alternative is shown.
"""

import sys


def main() -> int:
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print(
            "ERROR: PyQt6 is not installed.  GUI mode requires PyQt6.\n"
            "\n"
            "Install it:\n"
            "  pip install PyQt6\n"
            "\n"
            "Or use the CLI instead:\n"
            "  python -m nightcore_analyzer.cli \\\n"
            "      --nightcore /path/to/nightcore.flac \\\n"
            "      --source    /path/to/original.flac \\\n"
            "      --output    results.json",
            file=sys.stderr,
        )
        return 1

    from .gui import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Nightcore Analyzer")
    app.setOrganizationName("Tealdragon204")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
