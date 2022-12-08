"""
有些功能在新版的Rich中已经实现，但是老版本还没有，这里手动搓一下
"""

from typing import Any, Optional, Tuple
from rich.progress import (
    Progress, GetTimeCallable, TaskID, ProgressColumn, 
    Task, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn
)
from rich.console import Console
from rich.text import Text
from rich.table import Column


class MofNCompleteColumn(ProgressColumn):
    """Renders completed count/total, e.g. '  10/1000'.

    Best for bounded tasks with int quantities.

    Space pads the completed count so that progress length does not change as task progresses
    past powers of 10.

    Args:
        separator (str, optional): Text to separate completed and total values. Defaults to "/".
    """

    def __init__(self, separator: str = "/", table_column: Optional[Column] = None):
        self.separator = separator
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> Text:
        """Show completed/total."""
        completed = int(task.completed)
        total = int(task.total)
        total_width = len(str(total))
        return Text(
            f"{completed:{total_width}d}{self.separator}{total}",
            style="progress.download",
        )

class CompletedColumn(ProgressColumn):
    def __init__(self, table_column: Optional[Column] = None):
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> Text:
        """Show completed/total."""
        return Text(
            str(task.completed),
            style="progress.download",
        )

def get_default_columns() -> Tuple[ProgressColumn, ...]:
    """Get the default columns used for a new Progress instance:
        - a text column for the description (TextColumn)
        - the bar itself (BarColumn)
        - a text column showing completion percentage (TextColumn)
        - an estimated-time-remaining column (TimeRemainingColumn)
    If the Progress instance is created without passing a columns argument,
    the default columns defined here will be used.

    You can also create a Progress instance using custom columns before
    and/or after the defaults, as in this example:

        progress = Progress(
            SpinnerColumn(),
            *Progress.default_columns(),
            "Elapsed:",
            TimeElapsedColumn(),
        )

    This code shows the creation of a Progress display, containing
    a spinner to the left, the default columns, and a labeled elapsed
    time column.
    """
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )

def no_total_columns() -> Tuple[ProgressColumn, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        CompletedColumn(),
        TimeElapsedColumn(),
    )

def full_columns() -> Tuple[ProgressColumn, ...]:
    """这是我自己觉得比较舒服的组合
    """
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


class ProgressWOTotal(Progress):
    def __init__(
        self, 
        console: Optional[Console] = None, 
        auto_refresh: bool = True,
        refresh_per_second: float = 10, 
        speed_estimate_period: float = 30, 
        transient: bool = False, 
        redirect_stdout: bool = True, 
        redirect_stderr: bool = True, 
        get_time: Optional[GetTimeCallable] = None,
        disable: bool = False, 
        expand: bool = False
    ) -> None:
        super().__init__(
            *no_total_columns(), 
            console=console, 
            auto_refresh=auto_refresh, 
            refresh_per_second=refresh_per_second, 
            speed_estimate_period=speed_estimate_period, 
            transient=transient, 
            redirect_stdout=redirect_stdout, 
            redirect_stderr=redirect_stderr, 
            get_time=get_time, 
            disable=disable, 
            expand=expand,
        )

    def add_task(
        self,
        description: str, 
        start: bool = True,
        total: float = float('inf'), 
        completed: int = 0, 
        visible: bool = True, 
        **fields: Any
    ) -> TaskID:
        return super().add_task(description, start, total, completed, visible, **fields)
