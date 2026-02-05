"""
CLI entry point for Destill3D.

Provides the main Typer application and subcommand registration.
"""

import typer
from rich.console import Console
from rich.panel import Panel

from destill3d import __version__

# Create main app
app = typer.Typer(
    name="destill3d",
    help="3D model feature extraction and classification toolkit.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold green]destill3d[/bold green] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """
    Destill3D - 3D Model Feature Extraction & Classification

    Transform 3D files into compact, classification-ready snapshots.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Register subcommands
# ─────────────────────────────────────────────────────────────────────────────

from destill3d.cli.commands import extract, classify, db, config, acquire, models, server

app.add_typer(acquire.app, name="acquire", help="Acquire 3D models from platforms")
app.add_typer(extract.app, name="extract", help="Extract features from 3D files")
app.add_typer(classify.app, name="classify", help="Classify snapshots using ML models")
app.add_typer(db.app, name="db", help="Database operations and queries")
app.add_typer(models.app, name="models", help="Manage classification models")
app.add_typer(config.app, name="config", help="Configuration management")
app.add_typer(server.app, name="server", help="REST API server")


# ─────────────────────────────────────────────────────────────────────────────
# Quick commands (shortcuts)
# ─────────────────────────────────────────────────────────────────────────────


@app.command("info")
def info() -> None:
    """Show system information and capabilities."""
    from destill3d.core.config import Destill3DConfig, get_default_data_dir

    config_obj = Destill3DConfig()
    config_dir = get_default_data_dir()

    # Check for optional dependencies
    features = []

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            features.append("[green]✓[/green] GPU Inference (CUDA)")
        else:
            features.append("[yellow]○[/yellow] CPU Inference only")
    except ImportError:
        features.append("[yellow]○[/yellow] ONNX Runtime not installed")

    try:
        import OCC.Core
        features.append("[green]✓[/green] CAD Support (pythonocc)")
    except ImportError:
        features.append("[yellow]○[/yellow] CAD Support not available")

    try:
        import trimesh
        features.append(f"[green]✓[/green] Mesh Support (trimesh {trimesh.__version__})")
    except ImportError:
        features.append("[red]✗[/red] Trimesh not installed")

    try:
        import open3d as o3d
        features.append(f"[green]✓[/green] Open3D {o3d.__version__}")
    except ImportError:
        features.append("[yellow]○[/yellow] Open3D not available")

    info_text = f"""[bold]Version:[/bold] {__version__}
[bold]Config Dir:[/bold] {config_dir}
[bold]Models Dir:[/bold] {config_obj.models_dir}
[bold]Database:[/bold] {config_obj.database.path}

[bold]Capabilities:[/bold]
""" + "\n".join(f"  {f}" for f in features)

    console.print(Panel(info_text, title="Destill3D Info", border_style="blue"))


@app.command("quick")
def quick_extract(
    file_path: str = typer.Argument(..., help="Path to 3D file"),
    classify_flag: bool = typer.Option(
        False,
        "--classify",
        "-c",
        help="Also run classification",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for snapshot",
    ),
) -> None:
    """
    Quick extract (and optionally classify) a single 3D file.

    Shortcut for: destill3d extract file <path> [--classify]
    """
    from pathlib import Path
    from destill3d.extract import FeatureExtractor
    from destill3d.core.config import Destill3DConfig

    config_obj = Destill3DConfig()
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)

    console.print(f"[blue]Extracting features from:[/blue] {file_path_obj.name}")

    try:
        extractor = FeatureExtractor(config_obj.extraction)
        snapshot = extractor.extract_from_file(file_path_obj)
    except Exception as e:
        console.print(f"[red]Extraction failed:[/red] {e}")
        raise typer.Exit(1)

    # Save snapshot
    if output:
        output_path = Path(output)
    else:
        output_path = file_path_obj.with_suffix(".d3d")

    snapshot.save(output_path)
    console.print(f"[green]✓[/green] Saved snapshot: {output_path}")

    # Optionally classify
    if classify_flag:
        try:
            from destill3d.classify.inference import Classifier

            console.print("[blue]Running classification...[/blue]")
            classifier = Classifier(config_obj.data_dir / "models")
            predictions, embedding = classifier.classify(snapshot)

            if predictions:
                console.print(f"[green]✓[/green] Top prediction: {predictions[0].label} ({predictions[0].confidence:.1%})")
        except Exception as e:
            console.print(f"[yellow]Classification failed:[/yellow] {e}")


if __name__ == "__main__":
    app()
