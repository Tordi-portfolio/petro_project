from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand

from petrophysics.las import parse_las
from petrophysics.synthetic import make_synthetic_las
from wells.models import Well


class Command(BaseCommand):
    help = "Create a synthetic demo well (gas cap / oil leg / water leg, tight limestone, washout) or write its LAS to a path."

    def add_arguments(self, parser):
        parser.add_argument("--write", metavar="PATH", help="Only write the synthetic LAS file to PATH.")

    def handle(self, *args, **opts):
        text, _ = make_synthetic_las()
        if opts["write"]:
            with open(opts["write"], "w") as fh:
                fh.write(text)
            self.stdout.write(self.style.SUCCESS(f"Wrote {opts['write']}"))
            return
        las = parse_las(text)
        well = Well(name="SYN-1 (synthetic demo)", curve_names=list(las.curves),
                    top=float(las.depth.min()), base=float(las.depth.max()), depth_unit="m")
        well.las_file.save("SYN-1.las", ContentFile(text.encode()), save=True)
        self.stdout.write(self.style.SUCCESS(
            "Created demo well. Suggested parameters: surface temp 80 F, BHT 210 F, Rmf 0.35 ohm.m at 75 F."))
