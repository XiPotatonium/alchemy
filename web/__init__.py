from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict
from ..util.sym import sym_tbl


ARGS: Dict[str, Any] = {}
LOCAL_DIRNAME = Path(".alchemy-web")
SETTING_FILE = LOCAL_DIRNAME / "settings.json"
MAILS_DIR = LOCAL_DIRNAME / "mails"


@dataclass
class Setting():
    theme: str
    mail_fetch_period: int      # in second


DEFAULT_SETTINGS = Setting(
    theme="light",
    mail_fetch_period=10,
)


def init():
    if LOCAL_DIRNAME.exists():
        return
    LOCAL_DIRNAME.mkdir()
    MAILS_DIR.mkdir()
    with SETTING_FILE.open('w', encoding="utf8") as wf:
        json.dump(DEFAULT_SETTINGS.__dict__, wf, ensure_ascii=False)


@dataclass
class Sender:
    tag: str
    category: str       # runner type
    ctime: str
    dir: str


@dataclass
class Mail:
    id: str
    title: str
    text: str
    time: str
    sender: Sender
    level: str
    read: str


class Level(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    OTHER = "other"


def mail(title: str, text: str = "", level: Level = Level.INFO):
    import uuid
    from datetime import datetime

    sender = Sender(
        tag=sym_tbl().cfg["tag"],
        category=sym_tbl().cfg["runner"],
        ctime=sym_tbl().ctime.isoformat(),
        dir=str(sym_tbl().record_dir)
    )
    mail = Mail(
        id=str(uuid.uuid1()),
        title=title,
        text=text,
        time=datetime.now().isoformat(),
        sender=sender,
        level=str(level),
        read=False
    )
    init()
    with (MAILS_DIR / "{}.json".format(mail.id)).open('w', encoding="utf8") as wf:
        json.dump(asdict(mail), wf, ensure_ascii=False)
