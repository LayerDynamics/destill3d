"""
Acquire commands for Destill3D CLI.

Handles model acquisition from platforms.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from destill3d.core.config import Destill3DConfig
from destill3d.core.database import Database

app = typer.Typer(
    name="acquire",
    help="Acquire 3D models from platforms.",
    no_args_is_help=True,
)

console = Console()


def _get_db() -> Database:
    """Get database instance from default config."""
    config = Destill3DConfig.load()
    return Database(config.database.path)


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    platform: str = typer.Option("thingiverse", "-p", "--platform"),
    limit: int = typer.Option(20, "-l", "--limit"),
    license_type: Optional[str] = typer.Option(None, "--license"),
    queue: bool = typer.Option(
        False, "-q", "--queue", help="Add results to download queue"
    ),
) -> None:
    """Search platforms for 3D models and optionally queue results."""
    from destill3d.acquire.base import SearchFilters
    from destill3d.acquire.models import PlatformRegistry

    registry = PlatformRegistry()
    adapter = registry.get(platform)
    if adapter is None:
        console.print(f"[red]Unknown platform: {platform}[/red]")
        raise typer.Exit(1)

    filters = SearchFilters(license=[license_type]) if license_type else SearchFilters()

    console.print(f"[blue]Searching {platform} for:[/blue] {query}")

    try:
        results = asyncio.run(
            adapter.search(query, filters=filters, page=1)
        )
    except Exception as e:
        console.print(f"[red]Search failed:[/red] {e}")
        raise typer.Exit(1)

    if not results.items:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"Search Results ({results.total_count} total)")
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Author")
    table.add_column("License")
    table.add_column("Downloads", justify="right")

    for result in results.items[:limit]:
        table.add_row(
            result.model_id,
            result.title[:50],
            result.author or "",
            result.license or "",
            str(result.download_count or 0),
        )

    console.print(table)

    if queue and results.items:
        from destill3d.acquire.queue import DownloadQueue

        db = _get_db()
        q = DownloadQueue(db)
        for result in results.items[:limit]:
            asyncio.run(q.add(result.url))
        console.print(
            f"[green]Queued {min(len(results.items), limit)} items for download[/green]"
        )


@app.command("url")
def add_url_cmd(
    url: str = typer.Argument(..., help="URL to add to download queue"),
    priority: int = typer.Option(0, "-p", "--priority"),
) -> None:
    """Add a specific URL to the download queue."""
    from destill3d.acquire.queue import DownloadQueue

    db = _get_db()
    q = DownloadQueue(db)
    asyncio.run(q.add(url, priority=priority))
    console.print(f"[green]Added to queue:[/green] {url}")


@app.command("list")
def list_queue_cmd(
    status: Optional[str] = typer.Option(None, "-s", "--status"),
    limit: int = typer.Option(50, "-l", "--limit"),
) -> None:
    """List entries in the download queue."""
    from destill3d.acquire.queue import DownloadQueue, QueueStatus

    db = _get_db()
    q = DownloadQueue(db)

    status_filter = None
    if status:
        try:
            status_filter = QueueStatus(status)
        except ValueError:
            console.print(f"[red]Invalid status: {status}[/red]")
            raise typer.Exit(1)

    entries = asyncio.run(q.get_pending(limit=limit))

    table = Table(title="Download Queue")
    table.add_column("URL", style="cyan")
    table.add_column("Status")
    table.add_column("Priority", justify="right")

    count = 0
    for entry in entries:
        if status_filter and entry.status != status_filter:
            continue
        table.add_row(
            entry.source_url[:60] + ("..." if len(entry.source_url) > 60 else ""),
            entry.status.value,
            str(entry.priority),
        )
        count += 1
        if count >= limit:
            break

    console.print(table)


@app.command("run")
def run_queue_cmd(
    concurrency: int = typer.Option(4, "-c", "--concurrency"),
    rate_limit: Optional[str] = typer.Option(None, "-r", "--rate-limit"),
    extract: bool = typer.Option(True, "--extract/--no-extract"),
    classify: bool = typer.Option(False, "--classify/--no-classify"),
) -> None:
    """Process the download queue."""
    from destill3d.acquire.queue import DownloadQueue

    db = _get_db()
    q = DownloadQueue(db)
    pending = asyncio.run(q.get_pending())

    if not pending:
        console.print("[yellow]Queue is empty[/yellow]")
        return

    console.print(f"[blue]Processing {len(pending)} items (concurrency={concurrency})[/blue]")

    async def run():
        failed = 0
        async for progress in q.process(concurrency=concurrency):
            failed = progress.total - progress.completed
            console.print(
                f"  Progress: {progress.completed}/{progress.total} "
                f"(failed: {failed})"
            )

    asyncio.run(run())
    console.print("[green]Queue processing complete[/green]")


@app.command("platforms")
def list_platforms_cmd() -> None:
    """List available platforms and their status."""
    from destill3d.acquire.models import PlatformRegistry

    registry = PlatformRegistry()
    platforms = registry.list_platforms()

    table = Table(title="Available Platforms")
    table.add_column("Platform", style="cyan")
    table.add_column("Status")

    for name in platforms:
        table.add_row(name, "[green]Available[/green]")

    console.print(table)


@app.command("credentials")
def manage_credentials_cmd(
    platform: str = typer.Argument(..., help="Platform name"),
    action: str = typer.Option(
        "show", "-a", "--action", help="show|set|delete"
    ),
    value: Optional[str] = typer.Option(None, "-v", "--value"),
) -> None:
    """Manage platform API credentials."""
    from destill3d.acquire.credentials import CredentialManager

    cm = CredentialManager()

    if action == "show":
        cred = cm.retrieve(platform)
        if cred:
            masked = cred[:4] + "*" * (len(cred) - 4) if len(cred) > 4 else "****"
            console.print(f"[green]{platform}:[/green] {masked}")
        else:
            console.print(f"[yellow]No credentials stored for {platform}[/yellow]")

    elif action == "set":
        if not value:
            console.print("[red]Must provide --value with set action[/red]")
            raise typer.Exit(1)
        cm.store(platform, value)
        console.print(f"[green]Credentials stored for {platform}[/green]")

    elif action == "delete":
        cm.delete(platform)
        console.print(f"[green]Credentials deleted for {platform}[/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)
