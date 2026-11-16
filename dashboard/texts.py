"""English display strings.

The catalogs live in ``dashboard.locale``. This module re-exports the English
set so older imports keep working and there is not a second copy of the text.
"""

from dashboard.locale import bundle

_EN = bundle("en")
FEATURE_DESCRIPTIONS = _EN["feature_descriptions"]
GLOSSARY = _EN["glossary"]
TAB_INTROS = _EN["tab_intros"]
METHOD_STEPS = _EN["method_steps"]
HYPERPARAMS = _EN["hyperparams"]
WHATIF_HELP = _EN["whatif_help"]
FIGURES = _EN["figures"]
