"""Alembic environment configuration with async SQLAlchemy support.

This module configures Alembic to work with the application's async
database engine and SQLAlchemy models. It supports both online (connected)
and offline (SQL script generation) migration modes.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import the application's Base metadata and settings
from app.storage.postgres import Base

# This is the Alembic Config object
config = context.config

# Set up logging from the alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata


def get_database_url() -> str:
    """Get the database URL from application settings.

    Falls back to alembic.ini sqlalchemy.url if app config is unavailable.

    Returns:
        Database connection URL string.
    """
    try:
        from app.config import get_settings
        settings = get_settings()
        # Alembic needs the sync URL (without +asyncpg)
        return settings.database.sync_postgres_url
    except Exception:
        # Fallback to alembic.ini value
        return config.get_main_option("sqlalchemy.url", "")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This generates SQL scripts without requiring a database connection.
    Useful for generating migration SQL for review or manual execution.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations within a database connection.

    Args:
        connection: Active SQLAlchemy database connection.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using an async engine.

    Creates an async engine from the application configuration,
    connects, and runs migrations synchronously within the connection.
    """
    # Build configuration dict for async engine
    configuration = config.get_section(config.config_ini_section, {})

    # Override with application database URL
    db_url = get_database_url()
    # Convert sync URL to async for the engine
    async_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    configuration["sqlalchemy.url"] = async_url

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with async support.

    Creates an async engine and runs migrations within an async context.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
