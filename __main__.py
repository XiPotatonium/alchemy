from pathlib import Path
from typing import List, Optional
import typer


app = typer.Typer()


@app.command()
def run(
    cfgs: List[str],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    file: bool = True,
    force_mp: bool = False,
    task_per_device: int = 1,
):
    from . import run as _run
    from . import prepare_cfg
    _run(
        cfgs=[prepare_cfg(Path(cfg)) for cfg in cfgs],
        device=device, user_dir=user_dir, desc=desc, debug=debug, no_file=not file, force_mp=force_mp,
        task_per_device=task_per_device,
    )


@app.command()
def serve(
    host: Optional[str] = "127.0.0.1",
    port: Optional[int] = 8000,
):
    import uvicorn
    from .web import init
    init()
    uvicorn.run("alchemy.web.backend:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()