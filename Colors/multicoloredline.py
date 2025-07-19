import json
import time
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

console: Console = Console()

# Fancy separator
console.rule("[bold]Welcome to Rich Terminal[/bold]", style="rainbow")

# Define some JSON data
json_data: dict[str, Any] = {
    "message": "Hello, World!",
    "status": "success",
    "code": 200
}

# Print JSON with syntax highlighting
syntax: Syntax = Syntax(json.dumps(json_data, indent=4), "json", theme="monokai", line_numbers=True)
console.print(syntax)

# Simulating a progress bar
console.print("\n[bold cyan]Processing data...[/bold cyan]\n")

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.percentage:>3.0f}%"),
    console=console,
) as progress:
    task_id: int = progress.add_task("[cyan]Loading...", total=100)
    for _ in range(100):
        time.sleep(0.02)
        progress.update(task_id, advance=1)

# Create a rich table
console.print("\n[bold magenta]Results Summary:[/bold magenta]\n")

table: Table = Table(title="System Report", show_header=True, header_style="bold cyan")
table.add_column("Metric", style="bold yellow")
table.add_column("Value", justify="right", style="bold green")

table.add_row("CPU Usage", "12.5%")
table.add_row("Memory Usage", "68.3%")
table.add_row("Disk Space", "45.7% free")

console.print(table)

# Success message
console.print("\n[bold green]🎉 Process completed successfully![/bold green]\n")
console.rule(style="rainbow")