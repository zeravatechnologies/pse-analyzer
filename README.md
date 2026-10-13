# PSE Analyzer

The anatomy of a model's weights. PSE Analyzer opens a GGUF file, measures the
eigenvalue spectrum of every weight matrix, compares one precision of a model
against another, and seals what it finds in a dossier. It runs on an ordinary
CPU, and nothing leaves your computer. Designed by Andrej Ristić.

- Product page and downloads: https://www.zeravatech.ca/products/pse-analyzer/
- Getting started: https://www.zeravatech.ca/docs/pse-analyzer/getting-started/

## Layout

    src/pse_a/        the program (numpy and scipy; torch only for lens, sae, jspace, ppl)
    src/requirements.txt
    build/            Windows (PyInstaller) and Ubuntu (.deb) packaging
    .github/workflows/release.yml   builds both and publishes a release

Run from source: `pip install -r src/requirements.txt`, then, from `src/`,
`python -m pse_a.cli doctor`.

## Licence

Copyright (C) 2026 Andrej Ristic / Zerava Technologies Inc. Free and open source
under the GNU General Public License, version 3 or later: see
[`LICENSE.md`](LICENSE.md). The authorship notice in [`NOTICE`](NOTICE) is an
additional term under section 7(b) of the GPL and must be kept in every copy.
