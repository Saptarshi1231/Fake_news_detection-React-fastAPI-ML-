"""Text preprocessing shared by model training and API inference."""
from __future__ import annotations

import re

from nltk.stem.porter import PorterStemmer

# A bundled stop-word set keeps training and inference reproducible and avoids
# requiring an NLTK corpus download at application startup.
DEFAULT_STOP_WORDS = set("""
a about above after again against ain all am an and any are aren aren't as at be
because been before being below between both but by can could couldn couldn't d did
didn didn't do does doesn doesn't doing don don't down during each few for from
further had hadn hadn't has hasn hasn't have haven haven't having he he'd he'll he's
her here hers herself him himself his how i i'd i'll i'm i've if in into is isn isn't
it it's its itself just ll m ma me mightn mightn't more most must mustn't my myself
needn needn't no nor not now o of off on once only or other our ours ourselves out
over own re s same shan shan't she she'd she'll she's should shouldn shouldn't so
some such t than that that'll the their theirs them themselves then there these they
they'd they'll they're they've this those through to too under until up ve very was
wasn wasn't we we'd we'll we're we've were weren weren't what when where which while
who whom why will won won't wouldn wouldn't y you you'd you'll you're you've your
yours yourself yourselves
""".split())

# Negation carries meaning in news claims, so retain these terms as features.
DEFAULT_STOP_WORDS.difference_update({"no", "nor", "not"})

_STEMMER = PorterStemmer()


def stemming(content: str, stop_words: set[str] | None = None) -> str:
    """Lowercase, remove non-letters, remove stop words, and Porter-stem."""
    if not isinstance(content, str):
        content = str(content or "")
    words_to_remove = DEFAULT_STOP_WORDS if stop_words is None else stop_words
    words = re.sub(r"[^a-zA-Z]", " ", content).lower().split()
    return " ".join(_STEMMER.stem(word) for word in words if word not in words_to_remove)
