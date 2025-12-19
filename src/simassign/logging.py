import logging
import sys

LEVEL = 15 # More than debug less than info
logging.addLevelName(LEVEL, "DETAILS")

def details(self, message, *args, **kws):
    if self.isEnabledFor(LEVEL):
        self._log(LEVEL, message, args, **kws)
logging.Logger.details = details
log = logging.getLogger(__name__)
log.setLevel(LEVEL-1)

# I need to log things instead of print because of the way multiprocessing
# works: all the text printed will be hijacked until the end of the script
# but it's actually of debug benefit to have it in the right place.
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

def get_log():
    return log