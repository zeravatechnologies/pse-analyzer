PSE Analyzer __VERSION__ (Windows)

Two downloads exist; this file travels with both.
  PSE-Analyzer-__VERSION__-windows-x64.exe          one file. Unpacks to a temp folder on
                                                    every start, which costs about 12 seconds.
  PSE-Analyzer-__VERSION__-windows-x64-folder.zip   a folder. Starts in half a second. Keep
                                                    the whole folder together; the .exe inside
                                                    needs its neighbours.
Neither needs Python installed, and neither needs the internet.

1. This build is not code-signed. Check the SHA-256 against SHA256SUMS.txt from the
   same release before you run it:
       Get-FileHash .\PSE-Analyzer-__VERSION__-windows-x64.exe -Algorithm SHA256
   Windows then shows "Windows protected your PC" on the first run: click
   "More info", then "Run anyway".

2. Open a terminal in the folder that holds the executable (below, PSE-A.exe stands
   for whichever file you have):
       PSE-A.exe doctor         measures this machine: CPU, RAM, BLAS speed
       PSE-A.exe selftest       checks the invariants; all must say OK
       PSE-A.exe gui            opens the web view in your browser, 127.0.0.1 only
   Double-clicking the executable with no arguments does the same as gui.

3. Where your data lives:
       %LOCALAPPDATA%\pse-a     dossiers, sealed runs, the shared library, harvests
       %APPDATA%\pse-a          API keys and the torch setting
   Harvests run to tens of gigabytes. Put the data root on a drive with room:
       setx PSE_A_DATA D:\pse-a

4. torch is not included. doctor, info, l1, qdiff and gui work without it. lens, sae,
   jspace and ppl need it:
       PSE-A.exe torch setup                  builds an environment alongside (needs Python
                                              3.10+ and internet, once)
       PSE-A.exe torch use "C:\path\to\python.exe"   adopts a Python that already has torch

Designed by Andrej Ristic, Zerava Technologies Inc.
Licence: GPL-3.0-or-later (LICENSE.md); authorship notice in NOTICE.
Source: https://github.com/zeravatechnologies/pse-analyzer
Third-party components and their licences: THIRD-PARTY-NOTICES.txt.
Downloads and checksums: https://github.com/zeravatechnologies/releases
Documentation: https://www.zeravatech.ca/products/pse-analyzer/
