"""The first-run torch notice.

Shown once, when the web view is started on a machine where torch is not
available. It says what works without torch, what torch would add, and offers
the two honest ways to get it: build an environment alongside, or point at a
Python that already carries torch. If tkinter is missing the same text is
printed to the console instead -- the notice is never a hard dependency.
"""
from __future__ import annotations

import threading
import webbrowser

from . import torch_env as te

_INTRO = (
    "PSE-A works without torch. The whole weight-anatomy side of the tool -- "
    "doctor, info, the L1 spectral sweep, qdiff and this web view -- is pure "
    "numpy and scipy, and runs inside the application itself."
)

_WHY = (
    "torch unlocks the activation-physiology missions, which read the model "
    "while it is running:"
)

_HOW = (
    "torch cannot be installed into the application itself: a frozen build has "
    "no pip. It is kept alongside instead, in its own small environment. That "
    "needs Python 3.10+ and about 300 MB downloaded once.\n\n"
    "Already have torch somewhere? Use the second button and point at that "
    "Python -- nothing will be downloaded or installed."
)


def should_show() -> bool:
    if te.available():
        return False
    return not te.load_state().get("notice_dismissed")


def show(force: bool = False) -> None:
    """Show the notice. Never raises: a popup must not take the app down."""
    if not force and not should_show():
        return
    try:
        _show_tk()
    except Exception:
        print("\n" + te.notice_text() + "\n")


def _show_tk() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("PSE-A - torch is optional")
    root.resizable(False, False)
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    pad = {"padx": 16}
    frm = ttk.Frame(root, padding=16)
    frm.grid(sticky="nsew")

    ttk.Label(frm, text="torch is not installed", font=("Segoe UI", 13, "bold")) \
        .grid(row=0, column=0, columnspan=3, sticky="w")
    ttk.Label(frm, text=_INTRO, wraplength=560, justify="left") \
        .grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 10))
    ttk.Label(frm, text=_WHY, wraplength=560, justify="left") \
        .grid(row=2, column=0, columnspan=3, sticky="w")

    box = ttk.Frame(frm)
    box.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 10), **pad)
    for i, b in enumerate(te.BENEFITS):
        name, _, rest = b.partition("-")
        ttk.Label(box, text=name.strip(), font=("Consolas", 9, "bold")) \
            .grid(row=i, column=0, sticky="nw")
        ttk.Label(box, text=rest.strip(), wraplength=470, justify="left") \
            .grid(row=i, column=1, sticky="w", padx=(10, 0))

    ttk.Label(frm, text=_HOW, wraplength=560, justify="left") \
        .grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 6))

    link = ttk.Label(frm, text=te.TORCH_HOMEPAGE, foreground="#0a58ca", cursor="hand2")
    link.grid(row=5, column=0, columnspan=3, sticky="w")
    link.bind("<Button-1>", lambda _e: webbrowser.open(te.TORCH_HOMEPAGE))

    status = ttk.Label(frm, text="", wraplength=560, justify="left", foreground="#555")
    status.grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 0))
    bar = ttk.Progressbar(frm, mode="indeterminate", length=560)

    again = tk.BooleanVar(value=False)
    chk = ttk.Checkbutton(frm, text="Do not show this again", variable=again)
    chk.grid(row=8, column=0, sticky="w", pady=(12, 0))

    btns = ttk.Frame(frm)
    btns.grid(row=8, column=1, columnspan=2, sticky="e", pady=(12, 0))

    state = {"busy": False}

    def finish() -> None:
        if again.get() and not te.available():
            te.dismiss_notice()
        root.destroy()

    def set_busy(on: bool, msg: str = "") -> None:
        state["busy"] = on
        for b in (b_setup, b_use, b_later, chk):
            b.configure(state="disabled" if on else "normal")
        status.configure(text=msg)
        if on:
            bar.grid(row=7, column=0, columnspan=3, sticky="we", pady=(8, 0))
            bar.start(12)
        else:
            bar.stop()
            bar.grid_forget()

    def report(ok: bool, res: dict, how: str) -> None:
        set_busy(False)
        if ok:
            te.dismiss_notice()
            messagebox.showinfo(
                "torch is ready",
                f"torch {res.get('version')} is now available {how}.\n\n"
                f"Interpreter:\n{res.get('python')}\n\n"
                "The L2-L4 missions (lens, sae, jspace, ppl) will be handed to it "
                "automatically. Run 'pse-a doctor' to see the full picture.",
                parent=root)
            root.destroy()
        else:
            status.configure(text="")
            messagebox.showerror("torch is not available",
                                 str(res.get("error") or "unknown error"), parent=root)

    def do_setup() -> None:
        if state["busy"]:
            return
        set_busy(True, "working ... this can take several minutes and needs internet")

        def work() -> None:
            def prog(msg: str) -> None:
                root.after(0, lambda: status.configure(text=str(msg)))
            res = te.create_sidecar(progress=prog)
            root.after(0, lambda: report(bool(res.get("ok")), res, "alongside PSE-A"))

        threading.Thread(target=work, daemon=True).start()

    def do_use() -> None:
        if state["busy"]:
            return
        picked = filedialog.askopenfilename(
            parent=root, title="Select a Python interpreter that already has torch",
            filetypes=[("Python interpreter", "python.exe python3.exe python python3"),
                       ("All files", "*.*")])
        if not picked:
            return
        set_busy(True, f"checking {picked} ...")

        def work() -> None:
            res = te.use_external(picked)
            root.after(0, lambda: report(bool(res.get("ok")), res, "from your own Python"))

        threading.Thread(target=work, daemon=True).start()

    b_setup = ttk.Button(btns, text="Set up torch alongside", command=do_setup)
    b_use = ttk.Button(btns, text="I already have torch...", command=do_use)
    b_later = ttk.Button(btns, text="Not now", command=finish)
    b_setup.grid(row=0, column=0, padx=(0, 6))
    b_use.grid(row=0, column=1, padx=(0, 6))
    b_later.grid(row=0, column=2)

    root.protocol("WM_DELETE_WINDOW", finish)
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 3
    root.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    root.mainloop()
