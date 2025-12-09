"""
Utilities for emission calculations.
Default emission factor: 0.475 kg CO2e per kWh (example value; you should adjust per-country or per-grid).
Source-specific factors should be used in production.
"""

DEFAULT_EMISSION_FACTOR_KG_PER_KWH = 0.475  # kg CO2e per kWh (example)

def calculate_emission_from_kwh(kwh: float, emission_factor: float = None) -> float:
    """
    Return CO2e in kilograms.
    """
    factor = emission_factor if emission_factor is not None else DEFAULT_EMISSION_FACTOR_KG_PER_KWH
    try:
        factor = float(factor)
    except Exception:
        factor = DEFAULT_EMISSION_FACTOR_KG_PER_KWH
    return round(kwh * factor, 6)
