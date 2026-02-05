"""
CLI commands for Destill3D.

Subcommands: extract, classify, db, config, acquire, models, server
"""

from destill3d.cli.commands import extract, classify, db, config, acquire, models, server

__all__ = ["extract", "classify", "db", "config", "acquire", "models", "server"]
