import sys
from typing import List, Optional
from ..plugins import AlchemyPlugin
from . import mail, Level, init
import traceback


@AlchemyPlugin.register()
class MailOnFinish(AlchemyPlugin):
    """Send mail to alchemy-web when runner finishes.
    Will send "Run finish" when runner successfully finishes.
    Will send exception and its trackback when an exception is raised during running.

    Args:
        AlchemyPlugin (_type_): _description_
    """
    def __init__(self, send_ok: bool = True, **kwargs) -> None:
        """_summary_

        Args:
            send_ok (bool, optional): Send "Run finish" when runner successfully finishes. Defaults to True.
        """
        super().__init__()
        self.send_ok = send_ok

    def __enter__(self):
        init()

    def __exit__(self, exc_type, exc_val, exc_tb):
        etype, e, tb = sys.exc_info()           # NOTE: in python >= 3.11, you may use sys.exception()
        if e is None:
            if self.send_ok:
                mail(
                    title="Run finished",
                    text="",
                    level=Level.INFO
                )
        else:
            mail(
                title="[EXCEPTION] {}".format(e),
                text=traceback.format_exc(),
                level=Level.ERROR
            )
        return super().__exit__(exc_type, exc_val, exc_tb)
