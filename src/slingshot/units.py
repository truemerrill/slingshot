from typing import Dict, Literal
import numpy as np


# Length
METER = 1.0
CENTIMETER = 0.01 * METER
MILLIMETER = 0.001 * METER
INCH = 0.0254 * METER
FOOT = 0.3048 * METER
YARD = 0.9144 * METER
MILE = 1609.34 * METER

# Time
SECOND = 1.0
HOUR = 3600 * SECOND

# Mass
KILOGRAM = 1.0
GRAM = 0.001 * KILOGRAM
MILLIGRAM = 0.001 * GRAM
OUNCE = 28.3495 * GRAM
POUND = 16 * OUNCE
GRAIN = 64.79891 * MILLIGRAM

# Force
NEWTON = KILOGRAM * METER / (SECOND**2)

# Energy
JOULE = NEWTON * METER
FOOT_POUND = 1.355818 * JOULE

# Pressure
PASCAL = NEWTON / (METER**2)
ATMOSPHERE = 101325 * PASCAL
BAR = 100000 * PASCAL
MILLIBAR = 0.001 * BAR
PSI = 6894.76 * PASCAL

# Temperature
KELVIN = 1.0

# Angle
RADIAN = 1.0
DEGREE = (np.pi / 180) * RADIAN


def fahrenheit(degrees_fahrenheit: float) -> float:
    return celsius((degrees_fahrenheit - 32.0) * 5 / 9)


def celsius(degrees_celsius: float) -> float:
    return degrees_celsius + FREEZING_POINT_OF_WATER


def to_celsius(degrees_kelvin: float) -> float:
    return degrees_kelvin - FREEZING_POINT_OF_WATER


def to_fahrenheit(degrees_kelvin: float) -> float:
    c = to_celsius(degrees_kelvin)
    return (9 / 5) * c + 32.0


# Constants
MOLE = 1.0
MOLAR_MASS_OF_AIR = 0.02897 * KILOGRAM / MOLE

FREEZING_POINT_OF_WATER = 273.15 * KELVIN
STANDARD_TEMPERATURE = FREEZING_POINT_OF_WATER + 25 * KELVIN
IDEAL_GAS_CONST = 8.31446261815324 * JOULE / (MOLE * KELVIN)

STANDARD_GRAVITY = 9.80665 * METER / (SECOND**2)

# Material properties
Material = Literal["steel", "lead", "clay"]

MATERIAL_DENSITY: Dict[Material, float] = {
    "steel": 7850 * KILOGRAM / (METER**3),
    "lead": 11_340 * KILOGRAM / (METER**3),
    "clay": 1800 * KILOGRAM / (METER**3),
}
