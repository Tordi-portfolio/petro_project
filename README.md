# Petrolog - petrophysical evaluation from LAS files (Django)

Upload a LAS file, set (or leave blank) the interpretation parameters, and get a written evaluation:

1. **Log quality analysis** - completeness, plausible ranges, spikes, flat-lines, depth issues, washout / mudcake / DRHO, depth-match and cross-curve checks, per-curve grades.
2. **Lithology determination** - Vsh from GR (linear, Larionov, Clavier, Steiber), then sand/shale/limestone/dolomite/coal/anhydrite from GR + PEF + neutron-density separation.
3. **Fluid identification** - gas / oil / water / transition from Sw plus density-neutron crossover; zones, contacts (GOC/OWC/GWC), net pay.
4. **Saturation analysis** - Archie, Simandoux, Indonesian (all computed, one selected), Rw from Rwa or user input, Arps temperature correction, Sxo, BVW, full parameter table and assumptions.
5. **Well selection** - every well gets a suitability score (needs gas, oil and water zones + good key logs); the Wells page ranks them.

## Run it

```bash
pip install -r requirements.txt
python manage.py makemigrations wells
python manage.py migrate
python manage.py make_sample        # optional: loads a synthetic demo well
python manage.py runserver
```

Open http://127.0.0.1:8000/ . For the demo well use surface temp 80 F, BHT 210 F, Rmf 0.35 (the form defaults to blanks for these).

## Tests

```bash
python -m unittest discover -s petrophysics/tests -t .   # engine, no Django needed
python manage.py test wells                              # web layer
```

## Layout

- `petrophysics/` - framework-free engine (`las.py`, `curves.py`, `qc.py`, `lithology.py`, `porosity.py`, `saturation.py`, `fluids.py`, `pipeline.py`, `report.py`, `plotting.py`, `synthetic.py`)
- `wells/` - Django app (models, upload + parameter forms, views, templates)

## Limits you should know about

- Validated only on a **synthetic** well (the forward model uses the same equations the code inverts), so it proves the code path, not field accuracy. Test on your own wells before relying on any number.
- Vertical well assumed; no depth shifting, borehole/invasion/thin-bed corrections; LAS 1.2/2.0 only.
- Rw is estimated if not supplied and assumes a water-bearing clean sand is in the interval.
- Log-only fluid calls need confirmation from pressure data, samples or tests.
- Analysis runs inside the web request; fine for typical wells, move to a task queue for very large files.
- Development settings (DEBUG on, SQLite, insecure key by default) - set `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=0`, `DJANGO_ALLOWED_HOSTS` before deploying.
"# petro_project" 
