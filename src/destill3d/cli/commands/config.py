"""
Config commands for Destill3D CLI.

Handles configuration management.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from destill3d.core.config import Destill3DConfig

app = typer.Typer(
    name="config",
    help="Configuration management.",
    no_args_is_help=True,
)

console = Console()


@app.command("show")
def show_config(
    section: Optional[str] = typer.Argument(
        None,
        help="Config section to show (database, extraction, classification)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, yaml, json",
    ),
) -> None:
    """
    Show current configuration.

    Displays all settings or a specific section.
    """
    config = Destill3DConfig()

    if format == "yaml":
        import yaml
        config_dict = {
            "database": {
                "db_path": str(config.database.path),
                "echo": config.database.echo,
            },
            "extraction": {
                "target_points": config.extraction.target_points,
                "sampling_strategy": config.extraction.sampling_strategy,
                "oversample_ratio": config.extraction.oversample_ratio,
                "normal_estimation_k": config.extraction.normal_estimation_k,
                "curvature_estimation_k": config.extraction.curvature_estimation_k,
            },
            "classification": {
                "default_model": config.classification.default_model,
                "device": config.classification.device,
                "batch_size": config.classification.batch_size,
                "top_k": config.classification.top_k,
            },
            "tessellation": {
                "linear_deflection": config.tessellation.linear_deflection,
                "angular_deflection": config.tessellation.angular_deflection,
                "relative": config.tessellation.relative,
            },
        }

        if section:
            if section in config_dict:
                config_dict = {section: config_dict[section]}
            else:
                console.print(f"[red]Unknown section:[/red] {section}")
                console.print(f"Available: {', '.join(config_dict.keys())}")
                raise typer.Exit(1)

        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        console.print(Syntax(yaml_str, "yaml", theme="monokai"))
        return

    if format == "json":
        import json
        config_dict = {
            "database": {
                "db_path": str(config.database.path),
                "echo": config.database.echo,
            },
            "extraction": {
                "target_points": config.extraction.target_points,
                "sampling_strategy": config.extraction.sampling_strategy,
                "oversample_ratio": config.extraction.oversample_ratio,
            },
            "classification": {
                "default_model": config.classification.default_model,
                "device": config.classification.device,
                "batch_size": config.classification.batch_size,
            },
        }

        if section and section in config_dict:
            config_dict = {section: config_dict[section]}

        console.print(Syntax(json.dumps(config_dict, indent=2), "json", theme="monokai"))
        return

    # Table format
    if section is None or section == "paths":
        console.print(Panel(f"""[bold]Config Directory:[/bold] {config.config_dir}
[bold]Data Directory:[/bold] {config.data_dir}
[bold]Config File:[/bold] {config.config_dir / 'config.yaml'}""",
        title="Paths", border_style="blue"))

    if section is None or section == "database":
        table = Table(title="Database Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("db_path", str(config.database.path))
        table.add_row("echo", str(config.database.echo))
        console.print(table)

    if section is None or section == "extraction":
        table = Table(title="Extraction Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("target_points", str(config.extraction.target_points))
        table.add_row("sampling_strategy", config.extraction.sampling_strategy)
        table.add_row("oversample_ratio", str(config.extraction.oversample_ratio))
        table.add_row("normal_estimation_k", str(config.extraction.normal_estimation_k))
        table.add_row("curvature_estimation_k", str(config.extraction.curvature_estimation_k))
        console.print(table)

    if section is None or section == "classification":
        table = Table(title="Classification Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("default_model", config.classification.default_model)
        table.add_row("device", config.classification.device)
        table.add_row("batch_size", str(config.classification.batch_size))
        table.add_row("top_k", str(config.classification.top_k))
        console.print(table)

    if section is None or section == "tessellation":
        table = Table(title="Tessellation Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("linear_deflection", str(config.tessellation.linear_deflection))
        table.add_row("angular_deflection", str(config.tessellation.angular_deflection))
        table.add_row("relative", str(config.tessellation.relative))
        console.print(table)


@app.command("init")
def init_config(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config",
    ),
) -> None:
    """
    Initialize configuration file with defaults.

    Creates config.yaml in the config directory.
    """
    config = Destill3DConfig()
    config_path = config.config_dir / "config.yaml"

    if config_path.exists() and not force:
        console.print(f"[yellow]Config file already exists:[/yellow] {config_path}")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    # Create config directory
    config.config_dir.mkdir(parents=True, exist_ok=True)

    # Create data directory
    config.data_dir.mkdir(parents=True, exist_ok=True)

    # Create models directory
    (config.data_dir / "models").mkdir(parents=True, exist_ok=True)

    # Save default config
    config.save(config_path)

    console.print(f"[green]✓[/green] Created config file: {config_path}")
    console.print(f"[green]✓[/green] Created data directory: {config.data_dir}")
    console.print(f"\n[blue]Edit {config_path} to customize settings[/blue]")


@app.command("set")
def set_config(
    key: str = typer.Argument(..., help="Config key (e.g., extraction.target_points)"),
    value: str = typer.Argument(..., help="New value"),
) -> None:
    """
    Set a configuration value.

    Updates the config file with the new value.
    """
    config = Destill3DConfig()
    config_path = config.config_dir / "config.yaml"

    # Parse key
    parts = key.split(".")
    if len(parts) != 2:
        console.print("[red]Key format:[/red] section.setting (e.g., extraction.target_points)")
        raise typer.Exit(1)

    section, setting = parts

    # Load existing config
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}
    else:
        config_dict = {}

    # Ensure section exists
    if section not in config_dict:
        config_dict[section] = {}

    # Parse value based on type
    try:
        # Try int
        parsed_value = int(value)
    except ValueError:
        try:
            # Try float
            parsed_value = float(value)
        except ValueError:
            # Try bool
            if value.lower() in ("true", "yes", "1"):
                parsed_value = True
            elif value.lower() in ("false", "no", "0"):
                parsed_value = False
            else:
                parsed_value = value

    # Update
    old_value = config_dict[section].get(setting)
    config_dict[section][setting] = parsed_value

    # Save
    import yaml
    config.config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Updated {key}")
    if old_value is not None:
        console.print(f"  Old: {old_value}")
    console.print(f"  New: {parsed_value}")


@app.command("get")
def get_config(
    key: str = typer.Argument(..., help="Config key (e.g., extraction.target_points)"),
) -> None:
    """
    Get a specific configuration value.
    """
    config = Destill3DConfig()

    # Parse key
    parts = key.split(".")
    if len(parts) != 2:
        console.print("[red]Key format:[/red] section.setting")
        raise typer.Exit(1)

    section, setting = parts

    # Get value
    try:
        section_obj = getattr(config, section)
        value = getattr(section_obj, setting)
        console.print(f"{key} = {value}")
    except AttributeError:
        console.print(f"[red]Unknown key:[/red] {key}")
        raise typer.Exit(1)


@app.command("path")
def show_paths() -> None:
    """
    Show configuration and data paths.
    """
    config = Destill3DConfig()

    console.print(f"[bold]Config directory:[/bold] {config.config_dir}")
    console.print(f"[bold]Data directory:[/bold] {config.data_dir}")
    console.print(f"[bold]Database:[/bold] {config.database.path}")
    console.print(f"[bold]Models:[/bold] {config.data_dir / 'models'}")

    # Check what exists
    console.print("\n[bold]Status:[/bold]")

    config_file = config.config_dir / "config.yaml"
    status = "[green]✓[/green]" if config_file.exists() else "[dim]○[/dim]"
    console.print(f"  {status} Config file: {config_file}")

    db_file = Path(config.database.path)
    status = "[green]✓[/green]" if db_file.exists() else "[dim]○[/dim]"
    console.print(f"  {status} Database: {db_file}")

    models_dir = config.data_dir / "models"
    status = "[green]✓[/green]" if models_dir.exists() else "[dim]○[/dim]"
    console.print(f"  {status} Models directory: {models_dir}")


@app.command("reset")
def reset_config(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """
    Reset configuration to defaults.

    Removes the config file (data is preserved).
    """
    config = Destill3DConfig()
    config_path = config.config_dir / "config.yaml"

    if not config_path.exists():
        console.print("[yellow]No config file to reset[/yellow]")
        raise typer.Exit(0)

    if not force:
        confirm = typer.confirm("Reset configuration to defaults?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    config_path.unlink()
    console.print(f"[green]✓[/green] Removed: {config_path}")
    console.print("[blue]Config will use defaults until re-initialized[/blue]")


@app.command("env")
def show_env_vars() -> None:
    """
    Show environment variable configuration.

    Lists environment variables that can be used to configure Destill3D.
    """
    env_vars = [
        ("DESTILL3D_CONFIG_DIR", "Configuration directory"),
        ("DESTILL3D_DATA_DIR", "Data storage directory"),
        ("DESTILL3D_DB_PATH", "Database file path"),
        ("DESTILL3D_DEVICE", "Inference device (cuda/cpu/auto)"),
        ("DESTILL3D_TARGET_POINTS", "Default target points for extraction"),
        ("DESTILL3D_SAMPLING_STRATEGY", "Sampling strategy (uniform/fps/hybrid)"),
    ]

    import os

    table = Table(title="Environment Variables")
    table.add_column("Variable", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Current Value", style="green")

    for var, desc in env_vars:
        value = os.environ.get(var, "[dim]not set[/dim]")
        table.add_row(var, desc, value)

    console.print(table)

    console.print("\n[blue]Set these in your shell profile to customize defaults[/blue]")
