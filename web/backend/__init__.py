from dataclasses import dataclass
import tomlkit
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from starlette.responses import FileResponse, PlainTextResponse
from pathlib import Path

app = FastAPI()


ARGS: Dict[str, Any] = {}
RECORDS_DIR = Path("records")
# 这样设置好吗?
PUBLIC_DIR = Path(__file__).parent.with_name("public")


@app.get("/")
async def get_home():
    # return PlainTextResponse("alchemy")
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/assets/{whatever:path}")
async def get_static_files(whatever: str):
    path = PUBLIC_DIR / "assets" / whatever
    if not path.exists():
        raise HTTPException(status_code=404, detail="{} not found".format(path))
    if path.is_file():
        return FileResponse(path)
    return PlainTextResponse("alchemy")
    # return FileResponse(PUBLIC_DIR / "index.html")


@dataclass
class FileInfo:
    ty: str
    path: List[str]
    ctime: str
    mtime: str


def read_file_info(path: Path) -> FileInfo:
    import time
    ty = "missing" if not path.exists() else "file" if path.is_file() else "folder"
    plist = list(path.parts)
    ctime = time.ctime(path.stat().st_ctime)
    mtime = time.ctime(path.stat().st_mtime)
    return FileInfo(ty=ty, path=plist, ctime=ctime, mtime=mtime)


@app.get("/records/")
async def read_record_root():
    subitems = [read_file_info(r) for r in RECORDS_DIR.iterdir()]
    return {"path": ["records"], "subitems": subitems}


@app.get("/records/{p:path}")
async def read_record(p: str):
    path = RECORDS_DIR / p

    if path.is_file():
        return FileResponse(path)

    subitems = [read_file_info(r) for r in path.iterdir()]

    ret = {"path": list(path.parts), "subitems": subitems}

    cfg_file = path / "cfg.toml"
    readme_file = path / "README.md"
    if cfg_file.exists():
        with cfg_file.open('r', encoding="utf8") as f:
            cfg = tomlkit.load(f).value
        ret["cfg"] = cfg
    if readme_file.exists():
        doc = readme_file.read_text(encoding="utf8")
        ret["doc"] = doc

    return ret
