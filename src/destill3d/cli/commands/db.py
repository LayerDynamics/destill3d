"""
Database commands for Destill3D CLI.

Handles database operations, queries, and exports.
"""

from pathlib import Path
from typing import Optional, List
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from destill3d.core.config import Destill3DConfig
from destill3d.core.exceptions import DatabaseError, SnapshotNotFoundError

app = typer.Typer(
    name="db",
    help="Database operations and queries.",
    no_args_is_help=True,
)

console = Console()


@app.command("stats")
def show_stats() -> None:
    """
    Show database statistics.

    Displays counts, label distributions, and storage info.
    """
    config = Destill3DConfig()

    from destill3d.core.database import Database

    try:
        db = Database(config.database.path)
        stats = db.get_stats()
    except Exception as e:
        console.print(f"[red]Database error:[/red] {e}")
        raise typer.Exit(1)

    # Display stats
    info_text = f"""[bold]Total Snapshots:[/bold] {stats['total_snapshots']:,}
[bold]Classified:[/bold] {stats['classified_snapshots']:,}
[bold]Unclassified:[/bold] {stats['unclassified_snapshots']:,}
[bold]With Embeddings:[/bold] {stats['with_embeddings']:,}
[bold]Database Size:[/bold] {stats['db_size_mb']:.2f} MB
[bold]Database Path:[/bold] {config.database.path}"""

    console.print(Panel(info_text, title="Database Statistics", border_style="blue"))

    # Label distribution
    if stats.get('label_counts'):
        console.print("\n[bold]Label Distribution (Top 10):[/bold]")
        table = Table()
        table.add_column("Label", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right")

        total = sum(stats['label_counts'].values())
        for label, count in sorted(stats['label_counts'].items(), key=lambda x: -x[1])[:10]:
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            table.add_row(label, str(count), f"{pct:.1f}% {bar}")

        console.print(table)

    # Platform distribution
    if stats.get('platform_counts'):
        console.print("\n[bold]Platform Distribution:[/bold]")
        for platform, count in stats['platform_counts'].items():
            console.print(f"  {platform}: {count:,}")


@app.command("query")
def query_snapshots(
    label: Optional[str] = typer.Option(
        None,
        "--label",
        "-l",
        help="Filter by classification label",
    ),
    platform: Optional[str] = typer.Option(
        None,
        "--platform",
        "-p",
        help="Filter by source platform",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Filter by tag",
    ),
    min_confidence: Optional[float] = typer.Option(
        None,
        "--min-confidence",
        help="Minimum classification confidence",
    ),
    unclassified: bool = typer.Option(
        False,
        "--unclassified",
        "-u",
        help="Show only unclassified snapshots",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum results to show",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output results to JSON file",
    ),
) -> None:
    """
    Query snapshots from the database.

    Filter by label, platform, tags, or confidence.
    """
    config = Destill3DConfig()

    from destill3d.core.database import Database

    try:
        db = Database(config.database.path)

        if unclassified:
            results = db.query_unclassified(limit=limit)
        else:
            results = db.query(
                label=label,
                platform=platform,
                tag=tag,
                min_confidence=min_confidence,
                limit=limit,
            )
    except Exception as e:
        console.print(f"[red]Query error:[/red] {e}")
        raise typer.Exit(1)

    if not results:
        console.print("[yellow]No snapshots found matching criteria[/yellow]")
        raise typer.Exit(0)

    console.print(f"[blue]Found {len(results)} snapshots[/blue]\n")

    # Display results
    table = Table()
    table.add_column("Model ID", style="cyan", max_width=25)
    table.add_column("Title", style="white", max_width=30)
    table.add_column("Platform", style="blue")
    table.add_column("Top Label", style="green")
    table.add_column("Confidence", justify="right")

    for snapshot in results:
        title = snapshot.provenance.title or "-"
        if len(title) > 28:
            title = title[:25] + "..."

        platform_str = snapshot.provenance.platform or "-"

        if snapshot.predictions:
            top_pred = snapshot.predictions[0]
            label_str = top_pred.label
            conf_str = f"{top_pred.confidence:.1%}"
        else:
            label_str = "[dim]-[/dim]"
            conf_str = "[dim]-[/dim]"

        table.add_row(
            snapshot.model_id,
            title,
            platform_str,
            label_str,
            conf_str,
        )

    console.print(table)

    # Output to JSON if requested
    if output:
        output_path = Path(output)
        data = [
            {
                "model_id": s.model_id,
                "title": s.provenance.title,
                "platform": s.provenance.platform,
                "top_label": s.predictions[0].label if s.predictions else None,
                "top_confidence": s.predictions[0].confidence if s.predictions else None,
            }
            for s in results
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"\n[green]✓[/green] Saved to: {output_path}")


@app.command("show")
def show_snapshot(
    model_id: str = typer.Argument(..., help="Model ID of the snapshot"),
) -> None:
    """
    Show detailed information about a snapshot.
    """
    config = Destill3DConfig()

    from destill3d.core.database import Database

    try:
        db = Database(config.database.path)
        snapshot = db.get(model_id)
    except SnapshotNotFoundError:
        console.print(f"[red]Snapshot not found:[/red] {model_id}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Provenance
    console.print(Panel(f"""[bold]Model ID:[/bold] {snapshot.model_id}
[bold]Version:[/bold] {snapshot.version}
[bold]Platform:[/bold] {snapshot.provenance.platform}
[bold]Title:[/bold] {snapshot.provenance.title or '-'}
[bold]Author:[/bold] {snapshot.provenance.author or '-'}
[bold]Source URL:[/bold] {snapshot.provenance.source_url or '-'}
[bold]Original File:[/bold] {snapshot.provenance.original_filename or '-'}
[bold]Original Format:[/bold] {snapshot.provenance.original_format or '-'}
[bold]File Size:[/bold] {snapshot.provenance.original_file_size or 0:,} bytes
[bold]Tags:[/bold] {', '.join(snapshot.provenance.tags) if snapshot.provenance.tags else '-'}""",
    title="Provenance", border_style="blue"))

    # Geometry
    if snapshot.geometry:
        console.print(Panel(f"""[bold]Points:[/bold] {len(snapshot.geometry.points):,}
[bold]Has Normals:[/bold] {snapshot.geometry.normals is not None}
[bold]Has Curvature:[/bold] {snapshot.geometry.curvature is not None}
[bold]Centroid:[/bold] {snapshot.geometry.centroid}
[bold]Scale:[/bold] {snapshot.geometry.scale:.6f}""",
        title="Geometry", border_style="green"))

    # Features
    if snapshot.features:
        console.print(Panel(f"""[bold]Global Features:[/bold] {len(snapshot.features.global_features)} dimensions
[bold]Surface Area:[/bold] {snapshot.features.surface_area:.6f}
[bold]Volume:[/bold] {snapshot.features.volume:.6f}
[bold]Is Watertight:[/bold] {snapshot.features.is_watertight}
[bold]Original Vertices:[/bold] {snapshot.features.original_vertex_count:,}
[bold]Original Faces:[/bold] {snapshot.features.original_face_count:,}""",
        title="Features", border_style="yellow"))

    # Predictions
    if snapshot.predictions:
        pred_lines = []
        for pred in snapshot.predictions[:5]:
            pred_lines.append(f"  {pred.rank}. {pred.label}: {pred.confidence:.1%}")
        console.print(Panel("\n".join(pred_lines), title="Predictions", border_style="magenta"))

    # Processing
    if snapshot.processing:
        console.print(Panel(f"""[bold]Extraction Time:[/bold] {snapshot.processing.extraction_time_ms:.1f} ms
[bold]Classification Time:[/bold] {snapshot.processing.classification_time_ms or 0:.1f} ms
[bold]Target Points:[/bold] {snapshot.processing.target_points}
[bold]Sampling Strategy:[/bold] {snapshot.processing.sampling_strategy}
[bold]Warnings:[/bold] {len(snapshot.processing.warnings) if snapshot.processing.warnings else 0}""",
        title="Processing", border_style="cyan"))


@app.command("export")
def export_data(
    output: str = typer.Argument(..., help="Output file path"),
    format: str = typer.Option(
        "hdf5",
        "--format",
        "-f",
        help="Export format: hdf5, numpy, parquet, json",
    ),
    label: Optional[str] = typer.Option(
        None,
        "--label",
        "-l",
        help="Filter by label",
    ),
    include_geometry: bool = typer.Option(
        True,
        "--geometry/--no-geometry",
        help="Include point cloud geometry",
    ),
    include_embeddings: bool = typer.Option(
        True,
        "--embeddings/--no-embeddings",
        help="Include embeddings",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Maximum snapshots to export",
    ),
) -> None:
    """
    Export snapshots to various formats.

    Supports HDF5, NumPy, Parquet, and JSON.
    """
    config = Destill3DConfig()
    output_path = Path(output)

    from destill3d.core.database import Database

    try:
        db = Database(config.database.path)
    except Exception as e:
        console.print(f"[red]Database error:[/red] {e}")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Exporting to {format}...", total=None)

        try:
            export_path = db.export(
                output_path=output_path,
                format=format,
                label=label,
                include_geometry=include_geometry,
                include_embeddings=include_embeddings,
                limit=limit,
            )
            progress.update(task, description="Export complete")
        except Exception as e:
            console.print(f"[red]Export failed:[/red] {e}")
            raise typer.Exit(1)

    console.print(f"[green]✓[/green] Exported to: {export_path}")

    # Show file size
    if export_path.exists():
        size_mb = export_path.stat().st_size / (1024 * 1024)
        console.print(f"[blue]File size:[/blue] {size_mb:.2f} MB")


@app.command("delete")
def delete_snapshot(
    model_id: str = typer.Argument(..., help="Model ID to delete"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """
    Delete a snapshot from the database.
    """
    config = Destill3DConfig()

    from destill3d.core.database import Database

    try:
        db = Database(config.database.path)
    except Exception as e:
        console.print(f"[red]Database error:[/red] {e}")
        raise typer.Exit(1)

    # Check if exists
    try:
        snapshot = db.get(model_id)
    except SnapshotNotFoundError:
        console.print(f"[red]Snapshot not found:[/red] {model_id}")
        raise typer.Exit(1)

    # Confirm
    if not force:
        title = snapshot.provenance.title or model_id
        confirm = typer.confirm(f"Delete snapshot '{title}'?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    # Delete
    try:
        db.delete(model_id)
        console.print(f"[green]✓[/green] Deleted: {model_id}")
    except Exception as e:
        console.print(f"[red]Delete failed:[/red] {e}")
        raise typer.Exit(1)


@app.command("vacuum")
def vacuum_database() -> None:
    """
    Optimize the database (reclaim space).
    """
    config = Destill3DConfig()

    from destill3d.core.database import Database

    # Get size before
    db_path = Path(config.database.path)
    size_before = db_path.stat().st_size if db_path.exists() else 0

    try:
        db = Database(config.database.path)
        db.vacuum()
    except Exception as e:
        console.print(f"[red]Vacuum failed:[/red] {e}")
        raise typer.Exit(1)

    # Get size after
    size_after = db_path.stat().st_size if db_path.exists() else 0

    saved = size_before - size_after
    console.print(f"[green]✓[/green] Database optimized")
    console.print(f"[blue]Space saved:[/blue] {saved / 1024:.1f} KB")


@app.command("import")
def import_snapshots(
    input_path: str = typer.Argument(..., help="Directory or file to import from"),
    pattern: str = typer.Option(
        "*.d3d",
        "--pattern",
        "-p",
        help="Glob pattern for snapshot files",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search recursively",
    ),
    skip_duplicates: bool = typer.Option(
        True,
        "--skip-duplicates/--replace",
        help="Skip existing snapshots or replace them",
    ),
) -> None:
    """
    Import snapshots from files into the database.
    """
    config = Destill3DConfig()
    input_path_obj = Path(input_path).expanduser().resolve()

    if not input_path_obj.exists():
        console.print(f"[red]Path not found:[/red] {input_path}")
        raise typer.Exit(1)

    from destill3d.core.database import Database
    from destill3d.core.snapshot import Snapshot

    try:
        db = Database(config.database.path)
    except Exception as e:
        console.print(f"[red]Database error:[/red] {e}")
        raise typer.Exit(1)

    # Find snapshot files
    if input_path_obj.is_file():
        files = [input_path_obj]
    else:
        glob_pattern = f"**/{pattern}" if recursive else pattern
        files = list(input_path_obj.glob(glob_pattern))

    if not files:
        console.print(f"[yellow]No snapshot files found matching:[/yellow] {pattern}")
        raise typer.Exit(0)

    console.print(f"[blue]Found {len(files)} snapshot files[/blue]")

    # Import
    imported = 0
    skipped = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Importing...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"Importing: {file_path.name}")

            try:
                snapshot = Snapshot.load(file_path)

                if skip_duplicates:
                    try:
                        db.get(snapshot.model_id)
                        skipped += 1
                        progress.update(task, advance=1)
                        continue
                    except SnapshotNotFoundError:
                        pass

                db.insert(snapshot)
                imported += 1
            except Exception as e:
                failed += 1
                console.print(f"[red]Failed:[/red] {file_path.name}: {e}")

            progress.update(task, advance=1)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  [green]✓ Imported:[/green] {imported}")
    if skipped > 0:
        console.print(f"  [yellow]○ Skipped (duplicates):[/yellow] {skipped}")
    if failed > 0:
        console.print(f"  [red]✗ Failed:[/red] {failed}")
