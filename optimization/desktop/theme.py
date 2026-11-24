import tkinter.font as tkfont
from tkinter import ttk


NAVY = "#132338"
HEADER = "#0B1728"
SIDE_LINE = "#2F445D"
SIDE_TEXT = "#EFF5FC"
SIDE_MUTED = "#A7BAD1"
PAPER = "#FFFFFF"
BASE = "#F2F6FB"
INK = "#142338"
MUTED = "#536982"
BORDER = "#D9E3EE"
TEAL = "#0E7367"
MINT = "#69DEC2"
SUCCESS = "#046850"
WARNING = "#92431C"


def install(root):
    families = {name.casefold(): name for name in tkfont.families(root)}
    default_family = tkfont.nametofont("TkDefaultFont", root).actual("family")
    family = next((families[name] for name in
                   ("segoe ui", "inter", "dejavu sans", "liberation sans", "arial", "helvetica")
                   if name in families), default_family)
    text = (family, 10)
    bold = (family, 10, "bold")
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        tkfont.nametofont(name, root).configure(family=family, size=10)
    root.option_add("*TCombobox*Listbox.font", text)
    root.option_add("*TCombobox*Listbox.selectBackground", "#DAECEA")
    root.option_add("*TCombobox*Listbox.selectForeground", INK)
    root.configure(background=BASE)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=text, background=BASE, foreground=INK)
    style.configure("TFrame", background=BASE)
    style.configure("White.TFrame", background=PAPER)
    style.configure("Side.TFrame", background=NAVY)
    style.configure("Header.TFrame", background=HEADER)
    style.configure("Footer.TFrame", background="#E8F0F7")

    style.configure("TLabel", background=BASE, foreground=INK)
    style.configure("Muted.TLabel", background=BASE, foreground=MUTED, font=(family, 9))
    style.configure("PageTitle.TLabel", background=BASE, foreground=INK, font=(family, 15, "bold"))
    style.configure("White.TLabel", background=PAPER, foreground=INK)
    style.configure("Kicker.TLabel", background=PAPER, foreground=MUTED, font=(family, 9, "bold"))
    style.configure("Stat.TLabel", background=PAPER, foreground=INK, font=(family, 17, "bold"))
    style.configure("CardHint.TLabel", background=PAPER, foreground=MUTED, font=(family, 9))
    style.configure("Side.TLabel", background=NAVY, foreground=SIDE_TEXT)
    style.configure("SideMuted.TLabel", background=NAVY, foreground=SIDE_MUTED, font=(family, 9))
    style.configure("SideTitle.TLabel", background=NAVY, foreground=SIDE_TEXT, font=(family, 14, "bold"))
    style.configure("SideLabel.TLabel", background=NAVY, foreground=SIDE_TEXT, font=bold)
    style.configure("Brand.TLabel", background=HEADER, foreground=PAPER, font=(family, 20, "bold"))
    style.configure("BrandNote.TLabel", background=HEADER, foreground="#A9BED5", font=(family, 9))
    style.configure("Footer.TLabel", background="#E8F0F7", foreground=INK, font=(family, 9))

    style.configure("TButton", padding=(10, 6), relief="flat", borderwidth=1,
                    bordercolor=BORDER, background=PAPER, foreground=INK)
    style.map("TButton", background=[("active", "#E7F0F8")], foreground=[("disabled", "#8493A3")])
    style.configure("Primary.TButton", font=bold, padding=(12, 8), background=MINT,
                    foreground="#08372D", bordercolor=MINT)
    style.map("Primary.TButton", background=[("disabled", "#39596A"), ("active", "#9EF1DB")],
              foreground=[("disabled", "#C1D1DB")])
    style.configure("Side.TButton", background="#21364F", foreground=SIDE_TEXT, bordercolor=SIDE_LINE)
    style.map("Side.TButton", background=[("active", "#324C6B")], foreground=[("disabled", "#738AA0")])
    style.configure("Preset.TButton", background="#21364F", foreground=SIDE_TEXT, bordercolor=SIDE_LINE,
                    padding=(5, 4), font=(family, 9))
    style.map("Preset.TButton", background=[("active", "#324C6B")])
    style.configure("Header.TButton", background="#233950", bordercolor="#3A536C",
                    foreground=PAPER, font=bold, padding=(12, 7))
    style.map("Header.TButton", background=[("active", "#365875")], foreground=[("disabled", "#93AAC1")])

    style.configure("TEntry", fieldbackground=PAPER, bordercolor=BORDER, lightcolor=BORDER,
                    darkcolor=BORDER, padding=(7, 4), foreground=INK, insertcolor=INK)
    style.configure("TCombobox", fieldbackground=PAPER, background=PAPER, foreground=INK,
                    arrowcolor=INK, bordercolor=BORDER, padding=(7, 4))
    style.map("TCombobox", fieldbackground=[("readonly", PAPER)], foreground=[("readonly", INK)],
              selectbackground=[("readonly", PAPER)], selectforeground=[("readonly", INK)])
    style.configure("TCheckbutton", background=BASE, foreground=INK, padding=(1, 4))
    style.configure("TNotebook", borderwidth=0, background=BASE, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", font=bold, padding=(14, 9), background="#E8EFF6", foreground=MUTED,
                    bordercolor=BORDER)
    style.map("TNotebook.Tab", background=[("selected", PAPER), ("active", "#F4F9FC")],
              foreground=[("selected", TEAL)])
    style.configure("Treeview", background=PAPER, fieldbackground=PAPER, foreground=INK,
                    rowheight=28, bordercolor=BORDER, borderwidth=1, font=(family, 9))
    style.configure("Treeview.Heading", background="#E9F0F7", foreground=INK, font=(family, 9, "bold"),
                    relief="flat", padding=(6, 6))
    style.map("Treeview", background=[("selected", "#D7EFEB")], foreground=[("selected", "#104A40")])
    style.configure("Horizontal.TProgressbar", background="#21A794", troughcolor="#D1DEE8",
                    bordercolor="#D1DEE8")
