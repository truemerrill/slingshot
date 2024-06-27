from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable

import jinja2
import numpy as np
import pandas as pd
import plotly.express as px
from numpy.typing import NDArray

from .ballistics import (
    Ammunition,
    Environment,
    Launch,
    TargetMissError,
    Vec,
    speed,
    kinetic_energy,
    potential_energy,
    solve_angle,
    solve_max_distance,
    solve_trajectory,
)
from .components import Dataframe, Table, Report, ReportSection, Figure, render
from .units import (
    ATMOSPHERE,
    DEGREE,
    FOOT,
    FOOT_POUND,
    GRAIN,
    GRAM,
    INCH,
    JOULE,
    METER,
    MILLIBAR,
    MILLIMETER,
    SECOND,
    to_celsius,
    to_fahrenheit,
)


@dataclass
class _TrajectorySolution:
    time: NDArray[np.float64]
    state: NDArray[np.float64]


@dataclass
class _DistanceTableSolution:
    range: float
    time: float
    state: Vec
    launch: Launch
    ammo: Ammunition
    env: Environment
    trajectory: _TrajectorySolution


def _iter_distance_table_solutions(
    launch_speed: float,
    ammo: Ammunition,
    environment: Environment = Environment(),
    step: float = 10 * METER,
) -> Generator[_DistanceTableSolution, None, None]:
    """Iterate over the numerical solutions used to build a distance table

    Args:
        launch_speed (float): the launch speed in meters per second
        ammo (Ammunition): the ammunition
        environment (Environment, optional): the environment. Defaults to
            Environment().
        step (float, optional): the step size. Defaults to 10*METER.

    Yields:
        Generator[_DistanceTableSolution, None, None]: the solutions
    """
    launch = Launch(height=0 * METER, speed=launch_speed, angle=0 * DEGREE)

    # Solve for the maximum distance
    L_max, t_max, y_max = solve_max_distance(
        target_height=0 * METER, launch=launch, ammo=ammo, environment=environment
    )
    distance_max = y_max[0]

    # Iterate over distances up to the max distance
    for d in np.arange(start=step, stop=distance_max, step=step):
        try:
            L, t, y = solve_angle(
                target_distance=d,
                target_height=0 * METER,
                launch=launch,
                ammo=ammo,
                environment=environment,
            )
        except TargetMissError:
            continue

        times = np.linspace(0, t, 128)
        t_traj, y_traj = solve_trajectory(times, L, ammo, environment)
        yield _DistanceTableSolution(
            range=d,
            time=t,
            state=y,
            launch=L,
            ammo=ammo,
            env=environment,
            trajectory=_TrajectorySolution(time=t_traj, state=y_traj),
        )

    # Yield the max range solution
    times = np.linspace(0, t_max, 128)
    t_traj, y_traj = solve_trajectory(times, L_max, ammo, environment)

    yield _DistanceTableSolution(
        range=distance_max,
        time=t_max,
        state=y_max,
        launch=L_max,
        ammo=ammo,
        env=environment,
        trajectory=_TrajectorySolution(time=t_traj, state=y_traj),
    )


def _range_table_dataframe(solutions: Iterable[_DistanceTableSolution]) -> pd.DataFrame:
    """Construct a range table dataframe

    Args:
        solutions (Iterable[_DistanceTableSolution]): sequence of solutions

    Returns:
        pd.DataFrame: the range table dataframe
    """

    def convert(solution: _DistanceTableSolution) -> dict[str, Any]:
        energy = kinetic_energy(solution.state, solution.ammo.mass)
        return {
            "Range (m)": solution.range / METER,
            "Range (ft)": solution.range / FOOT,
            "Angle (deg)": solution.launch.angle / DEGREE,
            "Time (s)": solution.time / SECOND,
            "Energy (J)": energy / JOULE,
            "Energy (FPE)": energy / FOOT_POUND,
        }

    return pd.DataFrame([convert(s) for s in solutions])


def _range_table(solutions: Iterable[_DistanceTableSolution]) -> Dataframe:
    return Dataframe(_range_table_dataframe(solutions), id="range-table")


def _parameter_table(
    launch_speed: float, ammo: Ammunition, environment: Environment
) -> Table:
    def convert(value: float, m1: float, s1: str, m2: float, s2: str) -> str:
        return f"{(value / m1):.4g} {s1} ({(value / m2):.4g} {s2})"

    energy = ammo.mass * (launch_speed**2) / 2
    T_celcius = to_celsius(environment.temperature)
    T_fahr = to_fahrenheit(environment.temperature)

    return Table(
        headers=("Parameter", "Value"),
        rows=(
            ("Ammo material", ammo.material),
            ("Ammo diameter", convert(ammo.diameter, MILLIMETER, "mm", INCH, "in")),
            ("Ammo mass", convert(ammo.mass, GRAM, "g", GRAIN, "gr")),
            (
                "Launch speed",
                convert(launch_speed, METER / SECOND, "m/s", FOOT / SECOND, "FPS"),
            ),
            ("Launch energy", convert(energy, JOULE, "J", FOOT_POUND, "FPE")),
            ("Air temperature", f"{T_celcius} C ({T_fahr} F)"),
            (
                "Air pressure",
                convert(environment.pressure, MILLIBAR, "mbar", ATMOSPHERE, "atm"),
            ),
        ),
    )


def _trajectory_dataframe(solutions: Iterable[_DistanceTableSolution]) -> pd.DataFrame:
    def solution_dataframe(solution: _DistanceTableSolution) -> pd.DataFrame:
        s = np.array([speed(y) for y in solution.trajectory.state.T])
        ke = np.array(
            [kinetic_energy(y, solution.ammo.mass) for y in solution.trajectory.state.T]
        )
        pe = np.array(
            [
                potential_energy(y, solution.ammo.mass)
                for y in solution.trajectory.state.T
            ]
        )
        return pd.DataFrame(
            {
                "Distance (m)": solution.trajectory.state[0, :] / METER,
                "Height (m)": solution.trajectory.state[1, :] / METER,
                "Energy (J)": (ke + pe) / JOULE,
                "Speed (m/s)": s / (METER / SECOND),
                "Angle (deg)": solution.launch.angle / DEGREE,
                "Range (m)": solution.range / METER,
            }
        )

    components = [solution_dataframe(s) for s in solutions]
    return pd.concat(components, axis=0).reset_index(drop=True)  # type: ignore


def _trajectory_figure(solutions: Iterable[_DistanceTableSolution]) -> Figure:
    df = _trajectory_dataframe(solutions)
    fig = px.line(  # type: ignore
        df,
        x="Distance (m)",
        y="Height (m)",
        hover_data={"Energy (J)": True, "Speed (m/s)": True},
        line_group="Range (m)",
        color="Range (m)"
    )
    fig.update_layout(  # type: ignore
        showlegend=False,
    )
    return Figure(fig)


class BallisticReportPage:
    """A page for ballistics reports"""

    def __init__(
        self,
        ammo_name: str,
        ammo: Ammunition,
        launch_speed: float,
        environment: Environment = Environment(),
        step: float = 10 * METER,
    ):
        """Initialize the page builder

        Args:
            ammo_name (float): the name of the ammunition.
            ammo (Ammunition): the ammunition.
            launch_speed (float): the launch speed in meters per second.
            environment (Environment, optional): the environment. Defaults to
                Environment().
            step (float, optional): the step size for the range table. Defaults
                to 10*METER.
        """
        self.lauch_speed = launch_speed
        self.ammo = ammo
        self.ammo_name = ammo_name
        self.environment = environment
        self.step = step
        self.solutions = list(
            _iter_distance_table_solutions(launch_speed, ammo, environment, step)
        )

    def __call__(self, filename: Path, env: jinja2.Environment) -> None:
        """Write a ballistics report

        Args:
            filename (Path): the path to the file to write
            env (jinja2.Environment): the Jinja2 environment
        """
        speed = self.lauch_speed / (FOOT / SECOND)
        title = f"Ballistic report for {self.ammo_name} @ {speed:4.4g} FPS"

        report = Report(
            title=title,
            sections=[
                ReportSection(
                    "Parameters",
                    children=[
                        _parameter_table(self.lauch_speed, self.ammo, self.environment)
                    ],
                ),
                ReportSection(
                    "Ballistic table", children=[_range_table(self.solutions)]
                ),
                ReportSection(
                    "Trajectories", children=[_trajectory_figure(self.solutions)]
                ),
            ],
        )

        with open(filename, "w") as f:
            f.write(render(report, env))
