"""
Model management commands for Destill3D CLI.

Handles listing, downloading, and inspecting ML models.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="models",
    help="Manage classification models.",
    no_args_is_help=True,
)

console = Console()


@app.command("list")
def list_models(
    taxonomy: Optional[str] = typer.Option(None, "-t", "--taxonomy"),
    downloaded_only: bool = typer.Option(False, "--downloaded"),
) -> None:
    """List available classification models."""
    from destill3d.classify.registry import MODEL_REGISTRY

    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Name")
    table.add_column("Taxonomy")
    table.add_column("Accuracy", justify="right")
    table.add_column("Downloaded", justify="center", style="green")

    for model_id, info in MODEL_REGISTRY.items():
        if taxonomy and info.taxonomy != taxonomy:
            continue

        downloaded = (
            "✓"
            if info.weights_path and info.weights_path.exists()
            else "✗"
        )

        if downloaded_only and downloaded == "✗":
            continue

        accuracy = (
            f"{info.modelnet40_accuracy:.1%}"
            if info.modelnet40_accuracy
            else "N/A"
        )

        table.add_row(
            model_id,
            info.name,
            info.taxonomy,
            accuracy,
            downloaded,
        )

    console.print(table)


@app.command("download")
def download_model(
    model_id: str = typer.Argument(..., help="Model ID to download"),
    force: bool = typer.Option(
        False, "-f", "--force", help="Re-download if exists"
    ),
) -> None:
    """Download model weights."""
    from destill3d.classify.registry import MODEL_REGISTRY, download_model_weights

    if model_id not in MODEL_REGISTRY:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        console.print("[yellow]Run 'destill3d models list' to see available models[/yellow]")
        raise typer.Exit(1)

    info = MODEL_REGISTRY[model_id]

    if info.weights_path and info.weights_path.exists() and not force:
        console.print(
            f"[yellow]Model already downloaded: {info.weights_path}[/yellow]"
        )
        console.print("[dim]Use --force to re-download[/dim]")
        return

    console.print(f"[blue]Downloading {info.name}...[/blue]")

    try:
        with console.status("Downloading..."):
            path = download_model_weights(model_id)
        console.print(f"[green]Downloaded to: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        raise typer.Exit(1)


@app.command("info")
def model_info(
    model_id: str = typer.Argument(..., help="Model ID"),
) -> None:
    """Show detailed model information."""
    from destill3d.classify.registry import MODEL_REGISTRY

    if model_id not in MODEL_REGISTRY:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        raise typer.Exit(1)

    info = MODEL_REGISTRY[model_id]

    table = Table(title=f"Model: {model_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Name", info.name)
    table.add_row("Architecture", info.architecture)
    table.add_row("Taxonomy", info.taxonomy)
    table.add_row("Input Points", str(info.input_points))
    table.add_row("Requires Normals", "Yes" if info.requires_normals else "No")
    table.add_row("Format", info.format)
    table.add_row(
        "Accuracy",
        f"{info.modelnet40_accuracy:.1%}" if info.modelnet40_accuracy else "N/A",
    )
    table.add_row("Weights URL", info.weights_url)
    table.add_row(
        "Local Path",
        str(info.weights_path) if info.weights_path else "Not downloaded",
    )

    downloaded = info.weights_path and info.weights_path.exists()
    table.add_row(
        "Status",
        "[green]Downloaded[/green]" if downloaded else "[yellow]Not downloaded[/yellow]",
    )

    console.print(table)
