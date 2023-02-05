from typing import List, Optional
from ..plugins import AlchemyPlugin
from ..util.sym import sym_tbl
from . import mail, Level, init
import traceback


@AlchemyPlugin.register()
class MailOnFinish(AlchemyPlugin):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __enter__(self):
        init()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sym_tbl().exception is None:
            mail(
                title="Run finished",
                text="",
                level=Level.INFO
            )
        else:
            mail(
                title="[EXCEPTION] {}".format(sym_tbl().exception),
                text=traceback.format_exc(),
                level=Level.ERROR
            )
        return super().__exit__(exc_type, exc_val, exc_tb)
