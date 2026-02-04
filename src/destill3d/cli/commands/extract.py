"""
Extract commands for Destill3D CLI.

Handles feature extraction from 3D files.
"""

from pathlib import Path
from typing import Optional, List
import time

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from destill3d.core.config import Destill3DConfig, ExtractionConfig
from destill3d.core.exceptions import ExtractionError, FormatError
from destill3d.extract import FeatureExtractor
from destill3d.extract.loader import FormatDetector, FileFormat

app = typer.Typer(
    name="extract",
    help="Extract features from 3D files.",
    no_args_is_help=True,
)

console = Console()


# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".stl", ".obj", ".ply", ".off", ".gltf", ".glb",
    ".step", ".stp", ".iges", ".igs", ".brep", ".brp",
}


@app.command("file")
def extract_file(
    file_path: str = typer.Argument(..., help="Path to 3D file"),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for snapshot (default: <filename>.d3d)",
    ),
    store: bool = typer.Option(
        True,
        "--store/--no-store",
        help="Store snapshot in database",
    ),
    target_points: int = typer.Option(
        2048,
        "--points",
        "-n",
        help="Target number of points in point cloud",
    ),
    sampling: str = typer.Option(
        "hybrid",
        "--sampling",
        "-s",
        help="Sampling strategy: uniform, fps, hybrid, poisson",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
) -> None:
    """
    Extract features from a single 3D file.

    Supports STL, OBJ, PLY, GLTF, STEP, IGES formats.
    """
    file_path_obj = Path(file_path).expanduser().resolve()

    if not file_path_obj.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)

    if file_path_obj.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(f"[red]Error:[/red] Unsupported format: {file_path_obj.suffix}")
        console.print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        raise typer.Exit(1)

    # Build config
    config = Destill3DConfig()
    extraction_config = ExtractionConfig(
        target_points=target_points,
        sampling_strategy=sampling,
    )

    extractor = FeatureExtractor(extraction_config)

    # Detect format
    detector = FormatDetector()
    file_format = detector.detect(file_path_obj)

    console.print(f"[blue]File:[/blue] {file_path_obj.name}")
    console.print(f"[blue]Format:[/blue] {file_format.value}")
    console.print(f"[blue]Size:[/blue] {file_path_obj.stat().st_size / 1024:.1f} KB")

    # Extract
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Extracting features...", total=None)

        try:
            snapshot = extractor.extract_from_file(file_path_obj)
        except ExtractionError as e:
            console.print(f"[red]Extraction failed:[/red] {e}")
            raise typer.Exit(1)
        except FormatError as e:
            console.print(f"[red]Format error:[/red] {e}")
            raise typer.Exit(1)

    elapsed = time.time() - start_time

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = file_path_obj.with_suffix(".d3d")

    # Save snapshot
    snapshot.save(output_path)

    # Store in database if requested
    if store:
        try:
            from destill3d.core.database import Database
            db = Database(config.database.path)
            db.insert(snapshot)
            console.print(f"[green]✓[/green] Stored in database")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to store in database: {e}")

    # Output summary
    console.print(f"\n[green]✓[/green] Extraction complete in {elapsed:.2f}s")
    console.print(f"[green]✓[/green] Saved: {output_path}")

    if verbose:
        _print_snapshot_details(snapshot)


@app.command("dir")
def extract_directory(
    dir_path: str = typer.Argument(..., help="Directory containing 3D files"),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for snapshots (default: same as input)",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search subdirectories recursively",
    ),
    store: bool = typer.Option(
        True,
        "--store/--no-store",
        help="Store snapshots in database",
    ),
    target_points: int = typer.Option(
        2048,
        "--points",
        "-n",
        help="Target number of points",
    ),
    sampling: str = typer.Option(
        "hybrid",
        "--sampling",
        "-s",
        help="Sampling strategy",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of parallel workers",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip files that already have .d3d output",
    ),
) -> None:
    """
    Extract features from all 3D files in a directory.

    Processes all supported formats found in the directory.
    """
    dir_path_obj = Path(dir_path).expanduser().resolve()

    if not dir_path_obj.exists():
        console.print(f"[red]Error:[/red] Directory not found: {dir_path}")
        raise typer.Exit(1)

    if not dir_path_obj.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {dir_path}")
        raise typer.Exit(1)

    # Find all 3D files
    files: List[Path] = []
    pattern = "**/*" if recursive else "*"

    for ext in SUPPORTED_EXTENSIONS:
        files.extend(dir_path_obj.glob(f"{pattern}{ext}"))
        files.extend(dir_path_obj.glob(f"{pattern}{ext.upper()}"))

    # Remove duplicates and sort
    files = sorted(set(files))

    if not files:
        console.print(f"[yellow]No supported 3D files found in:[/yellow] {dir_path}")
        raise typer.Exit(0)

    console.print(f"[blue]Found {len(files)} 3D files[/blue]")

    # Setup output directory
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_path = None

    # Build config
    config = Destill3DConfig()
    extraction_config = ExtractionConfig(
        target_points=target_points,
        sampling_strategy=sampling,
    )

    extractor = FeatureExtractor(extraction_config)

    # Setup database if storing
    db = None
    if store:
        try:
            from destill3d.core.database import Database
            db = Database(config.database.path)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Database unavailable: {e}")

    # Process files
    success = 0
    failed = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(files))

        for file_path in files:
            # Determine output path
            if output_dir_path:
                output_path = output_dir_path / file_path.with_suffix(".d3d").name
            else:
                output_path = file_path.with_suffix(".d3d")

            # Skip if exists
            if skip_existing and output_path.exists():
                skipped += 1
                progress.update(task, advance=1, description=f"Skipped: {file_path.name}")
                continue

            progress.update(task, description=f"Processing: {file_path.name}")

            try:
                snapshot = extractor.extract_from_file(file_path)
                snapshot.save(output_path)

                if db:
                    try:
                        db.insert(snapshot)
                    except Exception:
                        pass  # Ignore DB errors for batch

                success += 1
            except Exception as e:
                failed += 1
                if progress.console.is_terminal:
                    progress.console.print(f"[red]Failed:[/red] {file_path.name}: {e}")

            progress.update(task, advance=1)

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  [green]✓ Success:[/green] {success}")
    if skipped > 0:
        console.print(f"  [yellow]○ Skipped:[/yellow] {skipped}")
    if failed > 0:
        console.print(f"  [red]✗ Failed:[/red] {failed}")


@app.command("info")
def file_info(
    file_path: str = typer.Argument(..., help="Path to 3D file"),
) -> None:
    """
    Show information about a 3D file without extracting.
    """
    file_path_obj = Path(file_path).expanduser().resolve()

    if not file_path_obj.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)

    # Detect format
    detector = FormatDetector()
    file_format = detector.detect(file_path_obj)

    console.print(f"\n[bold]File Information[/bold]")
    console.print(f"  Name: {file_path_obj.name}")
    console.print(f"  Path: {file_path_obj}")
    console.print(f"  Size: {file_path_obj.stat().st_size / 1024:.1f} KB")
    console.print(f"  Format: {file_format.value}")
    console.print(f"  Is Mesh: {file_format.is_mesh}")
    console.print(f"  Is CAD: {file_format.is_cad}")

    # Try to load and get mesh info
    if file_format.is_mesh or file_format.is_cad:
        try:
            from destill3d.extract.loader import load_geometry, get_mesh_info

            with console.status("Loading geometry..."):
                mesh = load_geometry(file_path_obj)
                info = get_mesh_info(mesh)

            console.print(f"\n[bold]Geometry[/bold]")
            console.print(f"  Vertices: {info['vertex_count']:,}")
            console.print(f"  Faces: {info['face_count']:,}")
            console.print(f"  Watertight: {info['is_watertight']}")
            console.print(f"  Surface Area: {info['surface_area']:.4f}")
            if info['is_watertight']:
                console.print(f"  Volume: {info['volume']:.4f}")
            console.print(f"  Dimensions: {info['dimensions'][0]:.4f} x {info['dimensions'][1]:.4f} x {info['dimensions'][2]:.4f}")

        except Exception as e:
            console.print(f"\n[yellow]Could not load geometry:[/yellow] {e}")


def _print_snapshot_details(snapshot) -> None:
    """Print detailed snapshot information."""
    table = Table(title="Snapshot Details")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Model ID", snapshot.model_id)
    table.add_row("Version", snapshot.version)
    table.add_row("Platform", snapshot.provenance.platform)

    if snapshot.geometry:
        table.add_row("Points", f"{len(snapshot.geometry.points):,}")
        table.add_row("Has Normals", str(snapshot.geometry.normals is not None))
        table.add_row("Has Curvature", str(snapshot.geometry.curvature is not None))

    if snapshot.features:
        table.add_row("Global Features", f"{len(snapshot.features.global_features)} dims")
        table.add_row("Surface Area", f"{snapshot.features.surface_area:.4f}")
        if snapshot.features.is_watertight:
            table.add_row("Volume", f"{snapshot.features.volume:.4f}")

    if snapshot.processing:
        table.add_row("Extraction Time", f"{snapshot.processing.extraction_time_ms:.1f} ms")
        if snapshot.processing.warnings:
            table.add_row("Warnings", str(len(snapshot.processing.warnings)))

    console.print(table)
