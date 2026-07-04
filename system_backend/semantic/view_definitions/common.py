from decimal import Decimal, InvalidOperation
import re

from ..models import AreaDomain

__all__ = [
    "_blank_to_none",
    "_decimal_or_none",
    "_int_or_none",
    "_domain_by_name_or_code",
    "_range_values",
]


def _blank_to_none(value):
    return None if value == "" or value is None else value


def _decimal_or_none(value):
    value = _blank_to_none(value)
    if value is None:
        return None
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            value = match.group(0)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{value} is not a valid decimal value.")


def _int_or_none(value):
    value = _blank_to_none(value)
    if value is None:
        return None
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            value = match.group(0)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"{value} is not a valid integer value.")


def _domain_by_name_or_code(value):
    value = str(value or "").strip()
    if not value:
        return None
    return (
        AreaDomain.objects.filter(name=value, is_active=True).first()
        or AreaDomain.objects.filter(code=value, is_active=True).first()
    )


def _range_values(payload, prefix, decimal=False):
    fixed_value = payload.get(f"{prefix}Value")
    min_value = payload.get(f"{prefix}Min")
    max_value = payload.get(f"{prefix}Max")

    if payload.get(f"{prefix}Mode") == "fixed" and fixed_value not in ("", None):
        min_value = fixed_value
        max_value = fixed_value

    parser = _decimal_or_none if decimal else _int_or_none
    return parser(min_value), parser(max_value)
