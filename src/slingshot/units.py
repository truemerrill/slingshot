from typing import Dict, Literal
import numpy as np


# Length
METER = 1.0
CENTIMETER = 0.01 * METER
MILIMETER = 0.001 * METER
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
MILIGRAM = 0.001 * GRAM
OUNCE = 28.3495 * GRAM
POUND = 16 * OUNCE
GRAIN = 64.79891 * MILIGRAM

# Force
NEWTON = KILOGRAM * METER / (SECOND**2)

# Energy
JOULE = NEWTON * METER
FOOT_POUND = 1.355818 * JOULE

# Pressure
PASCAL = NEWTON / (METER**2)
ATMOSPHERE = 101325 * PASCAL
BAR = 100000 * PASCAL
PSI = 6894.76 * PASCAL

# Temperature
KELVIN = 1.0

# Angle
RADIAN = 1.0
DEGREE = (np.pi / 180) * RADIAN


def fahrenheit(degrees_fahrenheit: float) -> float:
    return celcius((degrees_fahrenheit - 32.0) * 5 / 9)


def celcius(degrees_celcius: float) -> float:
    return degrees_celcius + FREEZING_POINT_OF_WATER


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
