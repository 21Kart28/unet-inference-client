# PyInstaller spec: `pyinstaller unet-client.spec` builds dist/unet-client/unet-client.exe.
# One-folder build: SimpleITK and Qt ship large native libraries, and a single-file exe
# would unpack them on every start.

from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

a = Analysis(
    ["unet_inference_client/__main__.py"],
    pathex=[],
    binaries=collect_dynamic_libs("SimpleITK"),
    datas=[],
    hiddenimports=["unet_inference_client.app"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Qt modules the client never imports; leaving them out roughly halves the bundle.
        "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets", "PySide6.QtMultimedia", "PySide6.Qt3DCore", "PySide6.QtCharts",
        "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtSql", "PySide6.QtTest",
        "tkinter", "matplotlib", "scipy", "pandas", "IPython",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="unet-client",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="unet-client")
