from os import system
from shutil import which
from sys import exit

from ollama import Client as _Client
from rich import print

__all__ = []


class Client(_Client):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def exists(self, model: str = "") -> None:
        if which("ollama") is None:
            from rich.table import Table
            from rich.panel import Panel

            download_link = "https://ollama.com/download"
            winget_command = "winget install ollama"

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Label", style="bold white")
            table.add_column("Details")

            table.add_row(
                "",
                "Download the latest version from given link (or) directly install using [bold]winget[/] command.",
            )
            table.add_row(
                "Download:",
                f"[link={download_link}][bright_cyan underline]{download_link}[/]",
            )
            table.add_row("Winget:", f"[bright_green]{winget_command}[/]")

            print(
                Panel(
                    table, title="[bright_red]Ollama Not Found![/]", border_style="red"
                )
            )
            exit(1)
        if model:
            models = [m.model for m in self.list().models]
            if model not in models:
                resp = system(f"ollama pull {model}")
                if resp != 0:
                    print(f"Failed to pull the model: {model}")
                    exit(1)


ollama = Client()


if __name__ == "__main__":
    ollama.exists("gemma")
