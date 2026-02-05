"""
Server commands for Destill3D CLI.

Provides REST API server management (stub for future implementation).
"""

import typer
from rich.console import Console

app = typer.Typer(
    name="server",
    help="REST API server commands.",
    no_args_is_help=True,
)

console = Console()


@app.command("start")
def start_server(
    host: str = typer.Option("127.0.0.1", "-h", "--host"),
    port: int = typer.Option(8000, "-p", "--port"),
    workers: int = typer.Option(1, "-w", "--workers"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the REST API server."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn not installed.[/red] "
            "Install with: pip install uvicorn[standard]"
        )
        raise typer.Exit(1)

    console.print(f"[blue]Starting server at http://{host}:{port}[/blue]")

    uvicorn.run(
        "destill3d.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )
