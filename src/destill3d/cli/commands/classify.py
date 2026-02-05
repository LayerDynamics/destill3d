"""
Classify commands for Destill3D CLI.

Handles ML-based classification of snapshots.
"""

from pathlib import Path
from typing import Optional, List
import time

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from destill3d.core.config import Destill3DConfig
from destill3d.core.snapshot import Snapshot
from destill3d.core.exceptions import (
    ClassificationError,
    ModelNotFoundError,
    InferenceError,
)

app = typer.Typer(
    name="classify",
    help="Classify snapshots using ML models.",
    no_args_is_help=True,
)

console = Console()


@app.command("snapshot")
def classify_snapshot(
    snapshot_path: str = typer.Argument(..., help="Path to .d3d snapshot file"),
    model: str = typer.Option(
        "pointnet2_ssg_mn40",
        "--model",
        "-m",
        help="Model to use for classification",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of top predictions to show",
    ),
    store: bool = typer.Option(
        True,
        "--store/--no-store",
        help="Store classification result in database",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save updated snapshot to path",
    ),
) -> None:
    """
    Classify a single snapshot file.

    Loads the snapshot, runs inference, and displays predictions.
    """
    config = Destill3DConfig()
    snapshot_path_obj = Path(snapshot_path).expanduser().resolve()

    if not snapshot_path_obj.exists():
        console.print(f"[red]Error:[/red] Snapshot not found: {snapshot_path}")
        raise typer.Exit(1)

    # Load snapshot
    console.print(f"[blue]Loading snapshot:[/blue] {snapshot_path_obj.name}")

    try:
        snapshot = Snapshot.load(snapshot_path_obj)
    except Exception as e:
        console.print(f"[red]Failed to load snapshot:[/red] {e}")
        raise typer.Exit(1)

    # Initialize classifier
    from destill3d.classify.inference import Classifier

    try:
        with console.status("Loading classification model..."):
            classifier = Classifier(
                models_dir=config.data_dir / "models",
                device=config.classification.device,
            )
    except ModelNotFoundError as e:
        console.print(f"[red]Model not found:[/red] {e}")
        console.print("[yellow]Hint:[/yellow] Run 'destill3d classify models --download' to download models")
        raise typer.Exit(1)

    # Run classification
    console.print(f"[blue]Classifying with model:[/blue] {model}")
    start_time = time.time()

    try:
        predictions, embedding = classifier.classify(snapshot, model_id=model, top_k=top_k)
    except InferenceError as e:
        console.print(f"[red]Classification failed:[/red] {e}")
        raise typer.Exit(1)

    elapsed = time.time() - start_time

    # Update snapshot with predictions
    snapshot.predictions = predictions
    if embedding is not None:
        snapshot.embedding = embedding

    # Display results
    console.print(f"\n[green]✓[/green] Classification complete in {elapsed:.2f}s\n")

    table = Table(title="Predictions")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Label", style="white")
    table.add_column("Confidence", justify="right", style="green")

    for pred in predictions:
        confidence_bar = "█" * int(pred.confidence * 20) + "░" * (20 - int(pred.confidence * 20))
        table.add_row(
            str(pred.rank),
            pred.label,
            f"{pred.confidence:.1%} {confidence_bar}",
        )

    console.print(table)

    if predictions and predictions[0].uncertainty is not None:
        uncertainty = predictions[0].uncertainty
        if uncertainty < 0.3:
            confidence_label = "[green]High confidence[/green]"
        elif uncertainty < 0.6:
            confidence_label = "[yellow]Medium confidence[/yellow]"
        else:
            confidence_label = "[red]Low confidence[/red]"
        console.print(f"\nUncertainty: {uncertainty:.3f} ({confidence_label})")

    # Save updated snapshot
    if output:
        output_path = Path(output)
        snapshot.save(output_path)
        console.print(f"\n[green]✓[/green] Saved: {output_path}")

    # Store in database
    if store:
        try:
            from destill3d.core.database import Database
            db = Database(config.database.path)
            db.update_classification(snapshot)
            console.print(f"[green]✓[/green] Stored in database")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to store: {e}")


@app.command("all")
def classify_all(
    model: str = typer.Option(
        "pointnet2_ssg_mn40",
        "--model",
        "-m",
        help="Model to use for classification",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of predictions per snapshot",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for inference",
    ),
    unclassified_only: bool = typer.Option(
        True,
        "--unclassified/--all",
        help="Only classify snapshots without predictions",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Maximum number of snapshots to classify",
    ),
) -> None:
    """
    Classify all snapshots in the database.

    By default, only processes snapshots without existing predictions.
    """
    config = Destill3DConfig()

    # Get snapshots from database
    from destill3d.core.database import Database

    try:
        db = Database(config.database.path)
    except Exception as e:
        console.print(f"[red]Database error:[/red] {e}")
        raise typer.Exit(1)

    # Query snapshots
    if unclassified_only:
        snapshots = db.query_unclassified(limit=limit)
    else:
        snapshots = db.query_all(limit=limit)

    if not snapshots:
        console.print("[yellow]No snapshots to classify[/yellow]")
        raise typer.Exit(0)

    console.print(f"[blue]Found {len(snapshots)} snapshots to classify[/blue]")

    # Initialize classifier
    from destill3d.classify.inference import Classifier

    try:
        with console.status("Loading classification model..."):
            classifier = Classifier(
                models_dir=config.data_dir / "models",
                device=config.classification.device,
            )
    except ModelNotFoundError as e:
        console.print(f"[red]Model not found:[/red] {e}")
        raise typer.Exit(1)

    # Classify in batches
    success = 0
    failed = 0
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Classifying...", total=len(snapshots))

        # Process in batches
        for i in range(0, len(snapshots), batch_size):
            batch = snapshots[i:i + batch_size]
            progress.update(task, description=f"Batch {i // batch_size + 1}")

            try:
                results = classifier.classify_batch(batch, model_id=model, batch_size=batch_size, top_k=top_k)

                for snapshot, (predictions, embedding) in zip(batch, results):
                    snapshot.predictions = predictions
                    if embedding is not None:
                        snapshot.embedding = embedding

                    try:
                        db.update_classification(snapshot)
                        success += 1
                    except Exception:
                        failed += 1

            except Exception as e:
                console.print(f"[red]Batch failed:[/red] {e}")
                failed += len(batch)

            progress.update(task, advance=len(batch))

    elapsed = time.time() - start_time

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total time: {elapsed:.1f}s")
    console.print(f"  Throughput: {success / elapsed:.1f} snapshots/sec")
    console.print(f"  [green]✓ Success:[/green] {success}")
    if failed > 0:
        console.print(f"  [red]✗ Failed:[/red] {failed}")


@app.command("models")
def list_models(
    download: bool = typer.Option(
        False,
        "--download",
        "-d",
        help="Download all available models",
    ),
) -> None:
    """
    List available classification models.

    Shows model info and download status.
    """
    config = Destill3DConfig()

    from destill3d.classify.registry import ModelRegistry

    registry = ModelRegistry(config.data_dir / "models")
    models = registry.list_models()

    table = Table(title="Available Models")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Taxonomy", style="blue")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Downloaded", justify="center")

    for model in models:
        is_downloaded = registry.is_downloaded(model.model_id)
        status = "[green]✓[/green]" if is_downloaded else "[dim]○[/dim]"
        accuracy = f"{model.modelnet40_accuracy:.1%}" if model.modelnet40_accuracy else "-"

        table.add_row(
            model.model_id,
            model.name,
            model.taxonomy,
            accuracy,
            status,
        )

    console.print(table)

    # Download if requested
    if download:
        console.print("\n[blue]Downloading models...[/blue]")

        for model in models:
            if not registry.is_downloaded(model.model_id):
                try:
                    with console.status(f"Downloading {model.model_id}..."):
                        registry.download_model(model.model_id)
                    console.print(f"[green]✓[/green] Downloaded: {model.model_id}")
                except Exception as e:
                    console.print(f"[red]✗[/red] Failed to download {model.model_id}: {e}")


@app.command("taxonomies")
def list_taxonomies() -> None:
    """
    List available classification taxonomies.

    Shows class labels for each taxonomy.
    """
    from destill3d.classify.registry import TAXONOMIES

    for name, taxonomy in TAXONOMIES.items():
        console.print(f"\n[bold cyan]{name}[/bold cyan] ({taxonomy.num_classes} classes)")

        # Display labels in columns
        labels = taxonomy.labels
        cols = 5
        rows = (len(labels) + cols - 1) // cols

        for row in range(rows):
            row_labels = []
            for col in range(cols):
                idx = row + col * rows
                if idx < len(labels):
                    row_labels.append(f"{idx:2d}. {labels[idx]:15s}")
            console.print("  " + " ".join(row_labels))


@app.command("zero-shot")
def classify_zero_shot(
    snapshot_path: str = typer.Argument(..., help="Path to .d3d snapshot file"),
    classes: List[str] = typer.Option(
        ...,
        "--class",
        "-c",
        help="Class labels (repeat for multiple)",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of top predictions to show",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save updated snapshot to path",
    ),
) -> None:
    """
    Zero-shot classify a snapshot with arbitrary class names.

    Uses OpenShape/CLIP-based embeddings for open-vocabulary classification.
    """
    snapshot_path_obj = Path(snapshot_path).expanduser().resolve()

    if not snapshot_path_obj.exists():
        console.print(f"[red]Error:[/red] Snapshot not found: {snapshot_path}")
        raise typer.Exit(1)

    console.print(f"[blue]Loading snapshot:[/blue] {snapshot_path_obj.name}")

    try:
        snapshot = Snapshot.load(snapshot_path_obj)
    except Exception as e:
        console.print(f"[red]Failed to load snapshot:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[blue]Zero-shot classifying with {len(classes)} classes[/blue]")

    try:
        from destill3d.classify.zero_shot import ZeroShotClassifier

        with console.status("Running zero-shot classification..."):
            zs = ZeroShotClassifier()
            result = zs.classify(
                snapshot.geometry.points,
                classes,
                top_k=top_k,
            )
    except Exception as e:
        console.print(f"[red]Classification failed:[/red] {e}")
        raise typer.Exit(1)

    # Display results
    table = Table(title="Zero-Shot Predictions")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Label", style="white")
    table.add_column("Probability", justify="right", style="green")

    for i, (cls, prob) in enumerate(
        zip(result.classes[:top_k], result.probabilities[:top_k])
    ):
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        table.add_row(str(i + 1), cls, f"{prob:.1%} {bar}")

    console.print(table)

    # Save updated snapshot
    if output:
        from destill3d.core.snapshot import Prediction

        snapshot.predictions = [
            Prediction(
                label=cls,
                confidence=float(prob),
                taxonomy="zero-shot",
                model_name="openshape",
                rank=i + 1,
            )
            for i, (cls, prob) in enumerate(
                zip(result.classes[:top_k], result.probabilities[:top_k])
            )
        ]
        if result.embedding_3d is not None:
            snapshot.embedding = result.embedding_3d

        output_path = Path(output)
        snapshot.save(output_path)
        console.print(f"\n[green]✓[/green] Saved: {output_path}")


@app.command("batch")
def classify_batch_files(
    snapshot_paths: List[str] = typer.Argument(..., help="Paths to .d3d snapshot files"),
    model: str = typer.Option(
        "pointnet2_ssg_mn40",
        "--model",
        "-m",
        help="Model to use",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for updated snapshots",
    ),
) -> None:
    """
    Classify multiple snapshot files at once.
    """
    config = Destill3DConfig()

    # Load snapshots
    snapshots: List[Snapshot] = []
    paths: List[Path] = []

    for path_str in snapshot_paths:
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            console.print(f"[yellow]Warning:[/yellow] Skipping missing file: {path}")
            continue

        try:
            snapshot = Snapshot.load(path)
            snapshots.append(snapshot)
            paths.append(path)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to load {path}: {e}")

    if not snapshots:
        console.print("[red]No valid snapshots to classify[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Classifying {len(snapshots)} snapshots[/blue]")

    # Initialize classifier
    from destill3d.classify.inference import Classifier

    classifier = Classifier(
        models_dir=config.data_dir / "models",
        device=config.classification.device,
    )

    # Classify
    start_time = time.time()

    with console.status("Running batch classification..."):
        results = classifier.classify_batch(snapshots, model_id=model)

    elapsed = time.time() - start_time

    # Update and save
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for snapshot, path, (predictions, embedding) in zip(snapshots, paths, results):
        snapshot.predictions = predictions
        if embedding is not None:
            snapshot.embedding = embedding

        if output_dir:
            save_path = output_path / path.name
        else:
            save_path = path

        snapshot.save(save_path)

        # Display result
        top_pred = predictions[0] if predictions else None
        if top_pred:
            console.print(f"[green]✓[/green] {path.name}: {top_pred.label} ({top_pred.confidence:.1%})")

    console.print(f"\n[green]✓[/green] Classified {len(snapshots)} snapshots in {elapsed:.2f}s")
