from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from starlette.responses import FileResponse, PlainTextResponse
from pathlib import Path

app = FastAPI()


ARGS: Dict[str, Any] = {}
RECORDS_DIR = Path("records")
PUBLIC_DIR = Path(__file__).parent.with_name("frontend")


@app.get("/")
async def get_home():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/favicon.ico")
async def get_favicon():
    path = PUBLIC_DIR / "favicon.ico"
    if not path.exists():
        raise HTTPException(status_code=404, detail="{} not found".format(path))
    return FileResponse(path)


@app.get("/assets/{whatever:path}")
async def get_assets(whatever: str):
    path = PUBLIC_DIR / "assets" / whatever
    if not path.exists():
        raise HTTPException(status_code=404, detail="{} not found".format(path))
    if path.is_file():
        if path.suffix == ".js":
            # 默认media_type是text/plain，这使得浏览器会认为返回的是文本文件，这样返回的js文件不会执行
            # 而assets中实际上是js模块，需要手动指定
            return FileResponse(path, media_type="text/javascript")
        else:
            return FileResponse(path)
    return FileResponse(PUBLIC_DIR / "index.html")


@dataclass
class FileInfo:
    ty: str
    path: List[str]
    ctime: str
    mtime: str


def read_finfo(path: Path) -> FileInfo:
    import time
    ty = "missing" if not path.exists() else "file" if path.is_file() else "folder"
    plist = list(path.parts)
    ctime = time.ctime(path.stat().st_ctime)
    mtime = time.ctime(path.stat().st_mtime)
    return FileInfo(ty=ty, path=plist, ctime=ctime, mtime=mtime)


@app.get("/api/lsFiles/")
async def ls_root():
    subitems = [read_finfo(r) for r in RECORDS_DIR.iterdir()]
    return {"path": list(RECORDS_DIR.parts), "subitems": subitems}


@app.get("/api/lsFiles/{p:path}")
async def ls_files(p: str):
    path = RECORDS_DIR / p
    if not path.exists():
        pass
    subitems = [read_finfo(r) for r in path.iterdir()]
    return {"path": list(path.parts), "subitems": subitems }


@app.get("/api/getFile/{p:path}")
async def get_file(p: str):
    path = RECORDS_DIR / p
    if not path.exists():
        pass
    if path.suffix == ".html":
        return HTMLResponse(path.read_text(encoding="utf8"))
    else:
        return FileResponse(path)
