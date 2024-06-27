from slingshot.site import Site
from slingshot.reports import BallisticReportPage
from slingshot.ballistics import Ammunition
from slingshot.units import INCH, MILLIMETER, FOOT, SECOND


AMMUNITION = {
    "Steel BB": Ammunition("steel", 4.5 * MILLIMETER),
    'Steel 5/16"': Ammunition("steel", 5/16 * INCH),
    'Steel 3/8"': Ammunition("steel", 3/8 * INCH),
    'Steel 10 mm': Ammunition("steel", 10 * MILLIMETER),
    'Steel 12 mm': Ammunition("steel", 12 * MILLIMETER),
    'Steel 16 mm': Ammunition("steel", 16 * MILLIMETER),
    'Lead 7.5 mm': Ammunition("lead", 7.5 * MILLIMETER),
    'Lead 10 mm': Ammunition("lead", 10 * MILLIMETER),
    'Lead 12 mm': Ammunition("lead", 12 * MILLIMETER),
    'Lead 16 mm': Ammunition("lead", 16 * MILLIMETER),
    'Clay 8 mm': Ammunition("clay", 8 * MILLIMETER),
    'Clay 10 mm': Ammunition("clay", 10 * MILLIMETER),
    'Clay 12 mm': Ammunition("clay", 12 * MILLIMETER)
}

SPEED = (
    100 * FOOT / SECOND,
    150 * FOOT / SECOND,
    200 * FOOT / SECOND,
    250 * FOOT / SECOND,
    300 * FOOT / SECOND
)


def report_name(ammo_name: str, speed: float) -> str:
    speed_str = f"{speed / (FOOT / SECOND):.3g}FPS"
    ammo_str = ammo_name.replace('"', "").replace(" ", "_").replace("/", "-")
    return f"reports/ballistic/{ammo_str}_{speed_str}.html"


site = Site()

for ammo_name, ammo in AMMUNITION.items():
    for speed in SPEED:
        name = report_name(ammo_name, speed)
        page = BallisticReportPage(
            ammo_name=ammo_name,
            ammo=ammo,
            launch_speed=speed
        )
        site.publish(name, page)


site.build("build")
