import re
from enum import Enum


class Label(Enum):
    MALICIOUS = "Malicious"
    BENIGN = "Benign"
    BACKGROUND = "Background"


BENIGN = Label.BENIGN.value
MALICIOUS = Label.MALICIOUS.value
BACKGROUND = Label.BACKGROUND.value

# "MAL" must start a word: Malicious, Malware, ATTACK_MALWARE -- but not Normal or Anomaly
_MALICIOUS_WORD = re.compile(r"(?:^|[^A-Z0-9])MAL")


def is_malicious_label(value) -> bool:
    """True for labels such as 'Malicious', 'Malware', 'ATTACK_MALWARE' or '1'."""
    text = str(value).strip().upper()
    return text == "1" or bool(_MALICIOUS_WORD.search(text))
