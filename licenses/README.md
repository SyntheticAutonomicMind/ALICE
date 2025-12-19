<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->\n<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->\n\n# ALICE Licensing

ALICE uses a dual licensing approach:

## Code (GPLv3)

All source code in this repository is licensed under the **GNU General Public License v3.0** (GPLv3).

- **License File:** `../LICENSE`
- **Applies To:** All Python code in `src/`, shell scripts in `scripts/`, and other executable code
- **Copyright:** Copyright (c) 2025 Andrew Wyatt (Fewtarius)

### SPDX Headers for Code

All code files include:
```
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)
```

## Documentation (CC-BY-NC-4.0)

All documentation in this repository is licensed under **Creative Commons Attribution-NonCommercial 4.0 International** (CC BY-NC 4.0).

- **License File:** `CC-BY-NC-4.0.txt`
- **Applies To:** All Markdown files in `docs/`, README files, and other documentation
- **Copyright:** Copyright (c) 2025 Andrew Wyatt (Fewtarius)

### SPDX Headers for Documentation

All documentation files include:
```
<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->
```

## Why Dual Licensing?

- **Code (GPLv3):** Ensures the software remains free and open source, protecting user freedoms
- **Documentation (CC-BY-NC-4.0):** Prevents commercial exploitation of documentation while allowing sharing and adaptation for non-commercial purposes

## External Dependencies

ALICE depends on several third-party libraries, each with their own licenses:

- **FastAPI:** MIT License
- **Diffusers:** Apache License 2.0
- **PyTorch:** BSD-style license
- **Stable Diffusion Models:** Various (check model card before use)

See `requirements.txt` for the complete list of Python dependencies.
