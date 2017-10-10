import cx_Freeze
import sys

base = None

if sys.platform == 'win32':
    base = "Win32GUI"

executables = [cx_Freeze.Executable("jk-music-player.py", base=base)]

cx_Freeze.setup(
    name= "abcd",
    options = {"build_exe": {"packages":["tkinter", "bs4"]}},
    version = '0.1',
    description = "",
    executables = executables
    )
