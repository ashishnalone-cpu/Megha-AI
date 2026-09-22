"""
date_utils.py

Resolves relative date words ("today", "tomorrow", a specific date, ...)
into real datetime.date objects, using IST "now" -- not the LLM, which
doesn't reliably know the real current date/time.
"""

import re
from datetime import datetime, timedelta, date as date_cls

try:
    from dateutil import parser as _dateutil_parser
    _HAVE_DATEUTIL = True
except ImportError:
    _HAVE_DATEUTIL = False

IST_OFFSET = timedelta(hours=5, minutes=30)

RELATIVE_TERMS = {
    "today": 0,
    "tomorrow": 1,
    "day after tomorrow": 2,
    "yesterday": -1,
}


def now_ist():
    """Current time in IST, computed from UTC (not the machine's local clock)."""
    return datetime.utcnow() + IST_OFFSET


def resolve_date_term(term):
    """Returns a datetime.date, or None if term is None/unrecognized."""
    if not term:
        return None
    t = str(term).strip().lower()

    if t in RELATIVE_TERMS:
        return (now_ist() + timedelta(days=RELATIVE_TERMS[t])).date()

    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", t)  # DD-MM-YYYY
    if m:
        d, mo, y = map(int, m.groups())
        return date_cls(y, mo, d)

    m = re.match(r"^(\d{2})[/.](\d{2})[/.](\d{4})$", t)  # DD/MM/YYYY or DD.MM.YYYY
    if m:
        d, mo, y = map(int, m.groups())
        return date_cls(y, mo, d)

    # Fallback: flexible natural-language dates ("15 August", "Aug 15 2026", ...)
    if _HAVE_DATEUTIL:
        try:
            dt = _dateutil_parser.parse(t, dayfirst=True, fuzzy=True, default=now_ist())
            return dt.date()
        except (ValueError, OverflowError, TypeError):
            return None

    return None


def get_date_range(start_date, n_days):
    return [start_date + timedelta(days=i) for i in range(n_days)]


def day_label_for_date(target_date, today=None):
    if today is None:
        today = now_ist().date()
    diff = (target_date - today).days
    if diff == 0:
        return "Today"
    if diff == 1:
        return "Tomorrow"
    if diff == 2:
        return "Day after tomorrow"
    return target_date.strftime("%A, %d %b")
