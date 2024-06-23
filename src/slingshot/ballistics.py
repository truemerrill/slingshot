import traceback
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast
from math import sqrt

import numpy as np
from numpy.typing import NDArray

# Note: Scipy hasn't shipped modern type hints yet
from scipy.integrate import solve_ivp  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from scipy.optimize import root_scalar, minimize_scalar  # type: ignore

from .units import (
    ATMOSPHERE,
    DEGREE,
    IDEAL_GAS_CONST,
    KELVIN,
    MATERIAL_DENSITY,
    METER,
    MOLAR_MASS_OF_AIR,
    PASCAL,
    SECOND,
    STANDARD_GRAVITY,
    STANDARD_TEMPERATURE,
    Material,
)

Vec = NDArray[np.float64]
OdeFunction = Callable[[float, Vec], Vec]
EventFunction = Callable[[float, Vec], float]


# Note: Drag coefficient versus Reynolds Number for a smooth sphere.
#
# Experimental data was extracted from a plot from Munson, Young, and Okiishi
# "Fundamentals of Fluid Mechanics", Wiley, 1990 using the WebPlotDigitizer
# tool.  The logarithms of the extracted data were then smoothed using Locally
# Weighted Scatterplot Smoothing (Lowess).  The smoothed logarithms were then
# converted back to linear space and stored here as a static array of
# coefficients.

_DRAG_COEFF_DATA_SMOOTHED = np.array(
    [
        [1.00000000e-01, 2.40782512e02],
        [1.08344157e-01, 2.22146545e02],
        [1.34159105e-01, 1.79264220e02],
        [1.66124928e-01, 1.44644924e02],
        [1.79986653e-01, 1.33476850e02],
        [2.35103244e-01, 1.02661213e02],
        [3.11227423e-01, 7.84550728e01],
        [4.58463134e-01, 5.47977085e01],
        [5.17026332e-01, 4.90306601e01],
        [5.38164951e-01, 4.71578884e01],
        [5.52735352e-01, 4.59368954e01],
        [5.98856459e-01, 4.24296689e01],
        [6.31722582e-01, 4.02469733e01],
        [7.02965044e-01, 3.63594858e01],
        [7.41544799e-01, 3.46287807e01],
        [9.06047847e-01, 2.90923009e01],
        [1.00822732e00, 2.65425767e01],
        [1.12193007e00, 2.40873257e01],
        [1.48520028e00, 1.81925878e01],
        [1.56671026e00, 1.72897316e01],
        [1.81467461e00, 1.52130793e01],
        [1.86380552e00, 1.48971632e01],
        [2.33892430e00, 1.24571786e01],
        [2.60269629e00, 1.14873929e01],
        [3.09624559e00, 9.97764141e00],
        [4.32372734e00, 7.73033125e00],
        [4.94159853e00, 6.98473634e00],
        [6.28469124e00, 5.90551489e00],
        [7.78213215e00, 5.13349263e00],
        [1.03019120e01, 4.20208136e00],
        [1.13116054e01, 3.91771699e00],
        [1.17740804e01, 3.80080102e00],
        [1.25872708e01, 3.61033714e00],
        [1.45794671e01, 3.24506982e00],
        [1.90440788e01, 2.71888629e00],
        [1.90450788e01, 2.71878602e00],
        [2.11917732e01, 2.55561965e00],
        [2.29600481e01, 2.44005087e00],
        [2.45458103e01, 2.34740814e00],
        [3.16369623e01, 2.00520862e00],
        [4.59854482e01, 1.58951248e00],
        [4.98225463e01, 1.51962557e00],
        [5.18595408e01, 1.48559329e00],
        [7.95166322e01, 1.17490322e00],
        [8.96739337e01, 1.10434580e00],
        [1.37497731e02, 8.94619378e-01],
        [1.89459992e02, 7.72117585e-01],
        [2.31489477e02, 7.07371609e-01],
        [3.06443553e02, 6.44114525e-01],
        [4.69872253e02, 5.51059014e-01],
        [5.81827782e02, 5.17467622e-01],
        [6.56149344e02, 4.98306233e-01],
        [7.91071106e02, 4.69078229e-01],
        [1.07556482e03, 4.30378553e-01],
        [1.14984994e03, 4.23467518e-01],
        [2.09740542e03, 4.01945491e-01],
        [2.12560862e03, 4.00947230e-01],
        [2.36532396e03, 3.94144329e-01],
        [2.42936328e03, 3.92541856e-01],
        [2.96828913e03, 3.81632425e-01],
        [3.48431326e03, 3.74052711e-01],
        [3.98223012e03, 3.71271247e-01],
        [4.49091253e03, 3.71835922e-01],
        [5.27163810e03, 3.72250882e-01],
        [5.71151187e03, 3.72625910e-01],
        [6.44108940e03, 3.72690284e-01],
        [6.88595067e03, 3.73040897e-01],
        [8.30188533e03, 3.74998430e-01],
        [8.41351835e03, 3.75388921e-01],
        [1.05582810e04, 3.81444308e-01],
        [1.37915010e04, 3.93742080e-01],
        [1.75399368e04, 4.07859385e-01],
        [2.11466145e04, 4.19615308e-01],
        [2.48228612e04, 4.29697939e-01],
        [2.79936857e04, 4.35122599e-01],
        [3.03295229e04, 4.37054838e-01],
        [3.37499323e04, 4.37652063e-01],
        [3.80610837e04, 4.37826039e-01],
        [4.35001054e04, 4.39986405e-01],
        [4.90567251e04, 4.46219183e-01],
        [5.91440929e04, 4.59651378e-01],
        [6.94260357e04, 4.57732203e-01],
        [8.71240265e04, 4.43121779e-01],
        [1.12293687e05, 4.23703014e-01],
        [1.37204729e05, 4.00999127e-01],
        [1.61057173e05, 3.87694074e-01],
        [1.89056261e05, 3.72491745e-01],
        [2.16072861e05, 3.31902036e-01],
        [2.50270867e05, 2.99095039e-01],
        [2.64006101e05, 2.72355535e-01],
        [2.78495144e05, 2.31550054e-01],
        [2.82239986e05, 2.10766359e-01],
        [2.93779367e05, 1.64741017e-01],
        [3.01733216e05, 1.42797678e-01],
        [3.18292779e05, 1.09314297e-01],
        [3.54188221e05, 9.84181191e-02],
        [4.04802581e05, 8.98382420e-02],
        [4.62649858e05, 7.98737360e-02],
    ]
)

_DRAG_COEFF_INTERPOLATOR = interp1d(
    _DRAG_COEFF_DATA_SMOOTHED[:, 0], _DRAG_COEFF_DATA_SMOOTHED[:, 1], kind="cubic"
)


def density(
    temperature: float = STANDARD_TEMPERATURE, pressure: float = ATMOSPHERE
) -> float:
    """Calculate the density of air using the ideal gas law.

    Args:
        temperature (float, optional): the temperature in Kelvin.  Defaults to
            STANDARD_TEMPERATURE.
        pressure (float, optional): the pressure in Pascals.  Defaults to
            ATMOSPEHRE.

    Returns:
        float: the air density in kilograms per cubic meter (also equal to g/L)
    """
    return pressure * MOLAR_MASS_OF_AIR / (IDEAL_GAS_CONST * temperature)


def viscosity(
    temperature: float = STANDARD_TEMPERATURE, pressure: float = ATMOSPHERE
) -> float:
    """Calculate the air dynamic viscosity with the extended Sutherland formula

    Args:
         temperature (float, optional): the temperature in Kelvin.  Defaults to
            STANDARD_TEMPERATURE.
        pressure (float, optional): the pressure in Pascals.  Defaults to
            ATMOSPEHRE.

    Returns:
        float: the air dynamic viscosity in units of Pascal seconds.
    """
    MU_0 = 1.827 * 1e-5 * PASCAL * SECOND
    T_0 = 291.15 * KELVIN
    P_0 = ATMOSPHERE
    S_MU = 110.4 * KELVIN

    sutherland = (
        MU_0 * (temperature / T_0) ** (3 / 2) * (T_0 + S_MU) / (temperature + S_MU)
    )

    return sutherland * (pressure / P_0)


def reynolds_number(
    flow_speed: float,
    diameter: float,
    temperature: float = STANDARD_TEMPERATURE,
    pressure: float = ATMOSPHERE,
) -> float:
    """Calculate the Reynolds number of a sphere in dry air

    Args:
        flow_speed (float): the flow speed in meters per second
        diameter (float): the diameter of the sphere in meters
        temperature (float, optional): the temperature in Kelvin.  Defaults to
            STANDARD_TEMPERATURE.
        pressure (float, optional): the pressure in Pascals.  Defaults to
            ATMOSPEHRE.

    Returns:
        float: the Reynolds number (dimensionless)
    """
    rho = density(temperature, pressure)
    mu = viscosity(temperature, pressure)
    return rho * flow_speed * diameter / mu


def drag_coefficient_interpolate(reynolds_number: float) -> float:
    """Estimate the drag coefficient of a smooth sphere from empirical data

    Note:
        This function uses interpolation of experimental data and is valid over
        the range 0.1 <= Re < 4.62e5.  This covers the expected range of values
        produced by slingshot ammunition and stops soon after the "drag crisis"
        at ~2.5e5 where the boundary layer transitions to turbulent flow.

    Args:
        reynolds_number (float): the Reynolds number (dimensionless)

    Raises:
        ValueError: if the Reynolds number is outside the interpolation range

    Returns:
        float: the drag coefficient (dimensionless)
    """
    if reynolds_number < 0.1:
        raise ValueError("Below minimum Reynolds number")
    elif reynolds_number >= 4.62e05:
        raise ValueError("Above maximum Reynolds number")
    return float(_DRAG_COEFF_INTERPOLATOR(reynolds_number))  # type: ignore


def drag_coefficient(reynolds_number: float) -> float:
    """Estimate the drag coefficient of a smooth sphere using Cheng's model

    Args:
        reynolds_number (float): the Reynolds number (dimensionless)

    Returns:
        float: the drag coefficient (dimensionless)
    """
    if reynolds_number == 0:
        return float("inf")

    C_d_low = (24 / reynolds_number) * (1 + 0.15 * reynolds_number**0.687)
    C_d_high = 0.42 / (1 + 4.25e4 * reynolds_number**-1.16)
    return C_d_low + C_d_high


def drag_force(
    flow_velocity: Vec,
    diameter: float,
    temperature: float = STANDARD_TEMPERATURE,
    pressure: float = ATMOSPHERE,
    threshold_speed: float = 1e-8 * METER / SECOND,
) -> Vec:
    """Calculate the drag force on a smooth sphere in dry air

    Args:
        flow_velocity (Vec): the velocity vector describing the flow of air
            around the sphere
        diameter (float): the diameter of the sphere
        temperature (float, optional): the temperature in Kelvin.  Defaults to
            STANDARD_TEMPERATURE.
        pressure (float, optional): the pressure in Pascals.  Defaults to
            ATMOSPEHRE.
        threshold_speed (float, optional): the speed threshold below which the
            drag force is exactly zero.  The threshold is a numerical technique
            to make sure the drag force has the correct limiting behavior and
            should not typically be modified by a user.  Defaults to
            `1e-8 * METER / SECOND`.

    Returns:
        Vec: the drag force vector
    """
    flow_speed = float(np.linalg.norm(flow_velocity))

    if flow_speed < threshold_speed:
        return 0.0 * flow_velocity
    else:
        rho = density(temperature, pressure)
        A = np.pi * ((diameter / 2.0) ** 2)
        re = reynolds_number(flow_speed, diameter, temperature, pressure)
        cd = drag_coefficient(re)
        return (rho * flow_speed * cd * A / 2) * flow_velocity


def force(
    state: Vec,
    mass: float,
    diameter: float,
    temperature: float = STANDARD_TEMPERATURE,
    pressure: float = ATMOSPHERE,
) -> Vec:
    """The total force vector

    Args:
        state (Vec): the 4-dimensional state vector (x, y, vx, vy)
        mass (float): the ammunition mass
        diameter (float): the ammunition diameter
        temperature (float, optional): the temperature in Kelvin.  Defaults to
            STANDARD_TEMPERATURE.
        pressure (float, optional): the pressure in Pascals.  Defaults to
            ATMOSPEHRE.

    Returns:
        Vec: the 2-dimensional force vector (Fx, Fy)
    """
    velocity = state[2:]
    Fg = -mass * STANDARD_GRAVITY * np.array([0, 1])
    Fd = -drag_force(velocity, diameter, temperature, pressure)
    return Fg + Fd


def kinetic_energy(state: Vec, mass: float) -> float:
    """The kinetic energy

    Args:
        state (Vec): the 4-dimensional state vector (x, y, vx, vy)
        mass (float): the ammunition mass

    Returns:
        float: the kinetic energy in Joules
    """
    v = float(np.linalg.norm(state[2:]))
    return mass * v**2 / 2.0


def potential_energy(state: Vec, mass: float) -> float:
    """The potential energy (relative to y = 0)

    Args:
        state (Vec): the 4-dimensional state vector (x, y, vx, vy)
        mass (float): the ammunition mass

    Returns:
        float: the potential energy in Joules
    """
    h = float(state[1])
    return mass * STANDARD_GRAVITY * h


def ballistic_coefficient(
    flow_speed: float,
    mass: float,
    diameter: float,
    temperature: float = STANDARD_TEMPERATURE,
    pressure: float = ATMOSPHERE
) -> float:
    """The ballistic coefficient as used in physics and engineering

    Args:
        flow_speed (float): the flow speed in meters per second
        mass (float): the mass of the ammunition in kilograms
        diameter (float): the diameter of the ammunition in meters
        temperature (float, optional): the temperature in Kelvin. Defaults to
            STANDARD_TEMPERATURE.
        pressure (float, optional): the pressure in Pascals. Defaults to
            ATMOSPHERE.

    Returns:
        float: the ballistic coefficient
    """
    re = reynolds_number(flow_speed, diameter, temperature, pressure)
    cd = drag_coefficient(re)
    area = np.pi * ((diameter / 2) ** 2)
    return mass / (cd * area)


def derivative(
    t: float,
    state: Vec,
    mass: float,
    diameter: float,
    temperature: float = STANDARD_TEMPERATURE,
    pressure: float = ATMOSPHERE,
) -> Vec:
    """The time derivative of the state vector

    Args:
        t (float): the time
        state (Vec): the 4-dimensional state vector (x, y, vx, vy)
        mass (float): the ammunition mass
        diameter (float): the ammunition diameter
        temperature (float, optional): the temperature in Kelvin.  Defaults to
            STANDARD_TEMPERATURE.
        pressure (float, optional): the pressure in Pascals.  Defaults to
            ATMOSPEHRE.

    Returns:
        Vec: the derivative of the state vector
    """
    F = force(state, mass, diameter, temperature, pressure)
    return np.array(
        [
            state[2],  # Vx
            state[3],  # Vy
            F[0] / mass,  # Fx / m
            F[1] / mass,  # Fy / m
        ]
    )


class Event:
    """A helper class to use in solve_inp without breaking the type checker"""

    def __init__(self, fun: OdeFunction, terminal: bool = False):
        self.fun = fun
        self.terminal = terminal

    def __call__(self, t: float, y: Vec, *args: Any) -> Vec:
        return self.fun(t, y, *args)


@dataclass
class OdeResult:
    """A helper class to type check the return value of solve_ivp"""

    status: int
    message: str
    t_events: list[Vec]
    y_events: list[Vec]


@dataclass
class OptimizeResult:
    """A helper class to type check the return value of minimize_scalar"""

    status: int
    message: str
    x: float


@dataclass
class RootResult:
    """A helper class to type check the return value of root_scalar"""

    converged: bool
    root: float


class Ammunition:
    """A type representing the kind of ammunition used"""

    def __init__(self, material: Material, diameter: float):
        self.material = material
        self.diameter = diameter

        rho = MATERIAL_DENSITY[material]
        self.mass = rho * self.volume

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def volume(self) -> float:
        return 4 / 3 * np.pi * (self.radius**3)


@dataclass
class Launch:
    """A type representing the launch height, speed, and angle"""

    height: float
    speed: float
    angle: float

    def state(self) -> Vec:
        """The initial state vector.

        Return:
            NDArray[np.float64]: the state vector (x, y, vx, vy)
        """
        return np.array(
            [
                0,
                self.height,
                self.speed * np.cos(self.angle),
                self.speed * np.sin(self.angle),
            ]
        )


class Environment:
    def __init__(
        self, termperature: float = STANDARD_TEMPERATURE, pressure: float = ATMOSPHERE
    ):
        """Initialize the environmental variables

        Args:
            termperature (float, optional): the air temperature. Defaults to
                STANDARD_TEMPERATURE.
            pressure (float, optional): the air pressure. Defaults to
                ATMOSPEHRE.
        """
        self.temperature = termperature
        self.pressure = pressure


class TargetMissError(ValueError):
    """Raised when the target was never reached"""

    pass


def _solve_event(
    event: Event,
    t_max: float,
    launch: Launch,
    ammo: Ammunition,
    environment: Environment,
    solution: Literal["first", "last"] = "last",
) -> tuple[float, Vec]:
    """Private function to write DRY code ..."""
    args = (
        ammo.mass,
        ammo.diameter,
        environment.temperature,
        environment.pressure,
    )

    # Solve ODE
    res = cast(
        OdeResult,
        solve_ivp(
            fun=derivative,
            args=args,
            t_span=(0, t_max),
            y0=launch.state(),
            method="RK45",
            events=event,
        ),
    )

    if res.status < 0:
        raise ValueError(res.message)

    # Check that at least one event was reached
    assert len(res.t_events) == 1 and len(res.y_events) == 1
    t_events = res.t_events[0]
    y_events = res.y_events[0]

    if len(t_events) == 0:
        raise TargetMissError()
    if solution == "first":
        return float(t_events[0]), y_events[0, :]
    else:
        return float(t_events[-1]), y_events[-1, :]


def solve_distance(
    target_height: float,
    launch: Launch,
    ammo: Ammunition,
    environment: Environment = Environment(),
    solution: Literal["first", "last"] = "last",
) -> tuple[float, Vec]:
    """Calculate the distance the ammo travels before it reaches `target_height`

    Args:
        target_height (float): the target height
        launch (Launch): the slingshot launch parameters
        ammo (Ammunition): the slingshot ammunition
        environment (Environment, optional): the environmental parameters.
            Defaults to Environment().
        solution (Literal["first", "last"], optional): optional flag
            controlling which solution point to return.  In a parabolic flight,
            a projectile can reach a target height zero, one, or two times.  If
            the trajectory reaches the height twice, this flag controls whether
            to return the first or second crossing point.  Defaults to "last".

    Raises:
        TargetMissError: raised if the target was never reached.

    Returns:
        tuple[float, Vec]: the time and final state
    """

    def event_func(t: float, state: Vec, *args: float) -> Vec:
        """Function with a root where y == target_height"""
        return np.array([state[1] - target_height])

    event = Event(event_func, terminal=False)

    # Safe upper bound on time to reach target
    t1 = 2 * launch.speed / STANDARD_GRAVITY
    t2 = (launch.height - target_height) / launch.speed if launch.speed != 0 else 0.0
    t3 = (
        sqrt(2 * (launch.height - target_height) / STANDARD_GRAVITY)
        if launch.height > target_height
        else 0.0
    )
    t_max = (t1 + t2 + t3) * 10

    return _solve_event(event, t_max, launch, ammo, environment, solution)


def solve_max_distance(
    target_height: float,
    launch: Launch,
    ammo: Ammunition,
    environment: Environment = Environment(),
) -> tuple[Launch, float, Vec]:
    L = Launch(launch.height, launch.speed, launch.angle)

    def opt_func(angle: float) -> float:
        """Function with a minimum at the optimum angle"""
        L.angle = angle
        _, y = solve_distance(
            target_height=target_height, launch=L, ammo=ammo, environment=environment
        )
        return -float(y[0])

    res = cast(
        OptimizeResult,
        minimize_scalar(
            fun=opt_func,
            bracket=(0 * DEGREE, 45 * DEGREE),
            bounds=(0 * DEGREE, 45 * DEGREE),
        ),
    )

    if res.status != 0:
        raise TargetMissError()

    L.angle = float(res.x)
    t, y = solve_distance(
        target_height=target_height, launch=L, ammo=ammo, environment=environment
    )

    return L, t, y


def solve_height(
    target_distance: float,
    launch: Launch,
    ammo: Ammunition,
    environment: Environment = Environment(),
) -> tuple[float, Vec]:
    """Calculate the height of the ammo once it travels `target_distance`

    Args:
        target_distance (float): the target distance
        launch (Launch): the slingshot launch parameters
        ammo (Ammunition): the slingshot ammunition
        environment (Environment, optional): the environmental parameters.
            Defaults to Environment().

    Raises:
        TargetMissError: raised if the target was never reached.

    Returns:
        tuple[float, Vec]: the time and final state
    """

    def event_func(t: float, state: Vec, *args: float) -> Vec:
        """Function with a root where x == target_distance"""
        return np.array([state[0] - target_distance])

    event = Event(event_func, terminal=True)

    # Safe upper bound on time to reach target
    t_max = (target_distance / launch.speed) * 10

    return _solve_event(event, t_max, launch, ammo, environment)


def solve_angle(
    target_distance: float,
    target_height: float,
    launch: Launch,
    ammo: Ammunition,
    environment: Environment = Environment(),
) -> tuple[Launch, float, Vec]:
    """Calculate the launch angle required to hit a target.

    Args:
        target_distance (float): the target distance
        target_height (float): the target height
        launch (Launch): the trial slingshot launch parameters.  This object
            is not mutated.
        ammo (Ammunition): the slingshot ammunition
        environment (Environment, optional): the environmental parameters.
            Defaults to Environment().

    Raises:
        TargetMissError: raised if the target cannot be reached

    Returns:
        tuple[Launch, float, Vec]: the optimized Launch parameters, the time,
            and the final state
    """
    L = Launch(launch.height, launch.speed, launch.angle)

    def opt_func(angle: float) -> float:
        """A function with a root at the optimum launch angle"""
        L.angle = angle
        _, y = solve_height(
            target_distance=target_distance,
            launch=L,
            ammo=ammo,
            environment=environment,
        )
        return float(y[1] - target_height)

    try:
        res = cast(
            RootResult,
            root_scalar(f=opt_func, bracket=(0 * DEGREE, 45 * DEGREE), method="brentq"),
        )
    except ValueError as e:
        # Scipy doesn't return a custom exception type, so we are inspecting
        # the ValueError here to confirm it's being raised by the brentq
        # solver.  Scipy needs to be updated to modern Python style.

        tb_list = traceback.extract_tb(e.__traceback__)
        fn_name = tb_list[-1].name
        if fn_name == "brentq":
            raise TargetMissError()
        raise e

    if res.converged is False:
        raise TargetMissError()

    L.angle = float(res.root)
    t, y = solve_distance(
        target_height=target_height, launch=L, ammo=ammo, environment=environment
    )

    return L, t, y
