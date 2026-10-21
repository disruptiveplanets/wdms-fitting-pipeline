# Model tables shipped with the repository

These small tables (≈ 3 MB in total) are read directly from this folder by `main_all.py` (`TABLES_DIR`). Nothing has to be downloaded for them.

They were assembled by the project author from published evolutionary sequences, in the formats the code expects. Cite the original works when you use results that depend on them. This is a private repository: ask the owner before redistributing these files.

| File | Content | Used by | Original work (as named in the files and the code) |
|---|---|---|---|
| `bedard2020/Table_DA.csv` | White-dwarf cooling sequences with thick hydrogen layers (`hypothesis = thick`; 23 masses from 0.20 to 1.30 M☉, `source_file` = `seq_<mass>_thick.txt`). Columns include `MassMsun`, `Teff`, `Logg`, `RadiusRsun`, `CoolingAgeGyr`. | white-dwarf gravity of every stage (`sed_fit._WDCoolingLogg`); CO-core masses (`mass.load_bedard_grid`) | Bédard et al. 2020 (Montreal cooling sequences) |
| `bedard2020/Table_DB.csv` | Same format, thin hydrogen layers (`hypothesis = thin`). | only when `WD_ATMO_TYPE = "DB"` (see docs/KNOWN_ISSUES.md) | Bédard et al. 2020 |
| `BHAC2015/BHAC15_tracks.csv` | Low-mass stellar evolutionary tracks; columns `M/Ms`, `log t(yr)`, `Teff`, `g`, `R/Rs`. | BHAC15 radius–temperature relation (dilution, ideal joint fit), M-dwarf mass–radius relation (`mass.py`) | Baraffe et al. 2015 (BHAC15) |
| `laplata2013/DAwhitedwarfsZ0.02.dat` | La Plata DA white-dwarf sequences; columns log g, log Teff, M, cooling age, R, core (`He`, `CO`, `ONe`), … | He-core masses in the mass stage (`mass.load_laplata_grid`) | He cores: Althaus et al. 2013; CO cores: Camisassa et al. 2016; ONe cores: Camisassa et al. 2019 (from the file header) |

To use other copies, set `TABLES_DIR` near the top of `main_all.py`. The Koester DA spectra are not here: they are downloaded into `<WORKSPACE>/models/koester2/` (see the main README).
