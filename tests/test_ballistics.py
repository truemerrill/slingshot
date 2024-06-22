import pytest

from math import isclose, sqrt
from slingshot.ballistics import (
    drag_coefficient,
    solve_distance,
    solve_max_distance,
    solve_height,
    solve_angle,
    Launch,
    Ammunition,
    TargetMissError
)
from slingshot.units import METER, FOOT, INCH, DEGREE, SECOND, STANDARD_GRAVITY


steel_5_16 = Ammunition(material="steel", diameter=(5 / 16) * INCH)
steel_3_8 = Ammunition(material="steel", diameter=(3 / 8) * INCH)


def test_drag_coefficient_lower_limit() -> None:
    """That as Re -> 0 then Cd approaches limiting value

    Note:
        For a sphere in the viscous flow regime (very low Re), the drag
        coefficient is `C_d = 24 / Re`.  At `Re == 0` the drag coefficient
        reaches positive infinity.
    """

    def drag_coeff_expected(re: float) -> float:
        return 24 / re

    # These values should use interpolated experimental data
    reynolds_numbers = [0.1, 0.2, 0.3, 0.4, 0.5]

    for re in reynolds_numbers:
        cd = drag_coefficient(re)
        cd_expected = drag_coeff_expected(re)
        assert isclose(cd, cd_expected, rel_tol=0.1)


def test_solve_distance_freefall() -> None:
    """That dropped ammo hits the ground at the expected time"""

    # Time to drop 1 meter
    height = 1 * METER
    t_expected = sqrt(2 * height / STANDARD_GRAVITY)

    t, y = solve_distance(
        target_height=0 * METER,
        launch=Launch(height=height, speed=0 * FOOT / SECOND, angle=0 * DEGREE),
        ammo=steel_5_16,
    )

    # Assert that we hit the ground right below the drop point
    assert isclose(0, y[0], abs_tol=1e-9)
    assert isclose(0, y[1], abs_tol=1e-9)

    # The actual drop time is slightly longer than the expected drop time
    # because air drag
    assert isclose(t, t_expected, rel_tol=0.01)
    assert t > t_expected


def test_solve_distance_fire_down() -> None:
    """That ammo fired straight down reaches the ground at the expected time"""

    height = 1 * METER
    speed = 100 * FOOT / SECOND
    t_expected = height / speed  # Ignoring gravity

    t, y = solve_distance(
        target_height=0 * METER,
        launch=Launch(height=height, speed=speed, angle=-90 * DEGREE),
        ammo=steel_5_16,
    )

    # Assert that we hit the ground right below the drop point
    assert isclose(0, y[0], abs_tol=1e-9)
    assert isclose(0, y[1], abs_tol=1e-9)

    # t < t_expected because of the influence of gravity
    assert isclose(t, t_expected, rel_tol=0.01)
    assert t < t_expected


def test_solve_distance_unreachable_target() -> None:
    """That solve_distance raises a TargetMissError"""

    with pytest.raises(TargetMissError):
        # Cannot reach target elevation, firing below the target.
        solve_distance(
            target_height=1 * METER,
            launch=Launch(height=0 * METER, speed=100 * FOOT / SECOND, angle=0 * DEGREE),
            ammo=steel_5_16,
        )


def test_solve_distance_two_solutions() -> None:
    speed = 1 * METER / SECOND

    # By default, return the second solution
    t, y = solve_distance(
        target_height=0 * METER,
        launch=Launch(height=0 * METER, speed=speed, angle=45 * DEGREE),
        ammo=steel_5_16,
    )

    assert t > 0.1 * SECOND
    assert y[0] > 0.1 * METER
    assert isclose(0 * METER, y[1], abs_tol=1e-9)

    # But if we request it, return the first solution
    t, y = solve_distance(
        target_height=0 * METER,
        launch=Launch(height=0 * METER, speed=speed, angle=45 * DEGREE),
        ammo=steel_5_16,
        solution="first"
    )

    assert t == 0
    assert isclose(0 * METER, y[0], abs_tol=1e-9)
    assert isclose(0 * METER, y[1], abs_tol=1e-9)


def test_solve_max_distance() -> None:
    speed = 200 * FOOT / SECOND

    L, _, y = solve_max_distance(
        target_height=0 * METER,
        launch=Launch(height=0 * METER, speed=speed, angle=0 * DEGREE),
        ammo=steel_5_16
    )

    distance = y[0]

    assert isclose(L.angle, 38.64 * DEGREE, abs_tol=1e-2)
    assert isclose(distance, 151 * METER, abs_tol=1e-1)


def test_solve_height() -> None:

    speed = 200 * FOOT / SECOND
    distance = 10 * METER

    # Calculate the expected drop from horizontal flight
    t_expected = distance / speed
    drop_expected = - STANDARD_GRAVITY * (t_expected ** 2) / 2.0

    t, y = solve_height(
        target_distance=distance,
        launch=Launch(height=0 * METER, speed=speed, angle=0 * DEGREE),
        ammo=steel_5_16,
    )

    assert isclose(t, t_expected, rel_tol=0.1)
    assert isclose(distance, y[0], abs_tol=1e-9)
    assert isclose(y[1], drop_expected, rel_tol=0.1)


def test_solve_height_unreachable_target() -> None:
    """That solve_height raises a TargetMissError"""
    speed = 200 * FOOT / SECOND

    with pytest.raises(TargetMissError):
        # Cannot reach target distance, firing backwards.
        solve_height(
            target_distance=10 * METER,
            launch=Launch(height=0 * METER, speed=speed, angle=180 * DEGREE),
            ammo=steel_5_16,
        )


def test_solve_angle() -> None:
    """That solve_angle finds the optimum launch angle to hit a target"""
    speed = 200 * FOOT / SECOND

    L, t, y = solve_angle(
        target_distance=100 * METER,
        target_height=0 * METER,
        launch=Launch(height=0 * METER, speed=speed, angle=0 * DEGREE),
        ammo=steel_5_16
    )

    assert isclose(L.angle, 12.91 * DEGREE, abs_tol=1e-2)
    assert isclose(y[0], 100 * METER, abs_tol=1e-9)
    assert isclose(y[1], 0 * METER, abs_tol=1e-9)


def test_solve_angle_out_of_range() -> None:
    speed = 200 * FOOT / SECOND

    with pytest.raises(TargetMissError):
        solve_angle(
            target_distance=160 * METER,   # About 9 meters outside max range
            target_height=0 * METER,
            launch=Launch(height=0 * METER, speed=speed, angle=0 * DEGREE),
            ammo=steel_5_16
        )
