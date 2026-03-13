# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Auto Video Editor."""

import platform
from pathlib import Path

block_cipher = None

PROJECT_ROOT = Path(SPECPATH)
SRC_DIR = PROJECT_ROOT / 'src'
TOOLS_DIR = PROJECT_ROOT / 'tools'

# Collect FFmpeg binaries from tools/ directory
ffmpeg_binaries = []
if platform.system() == 'Windows':
    _exts = ['.exe']
else:
    _exts = ['']

for _name in ('ffmpeg', 'ffprobe'):
    for _ext in _exts:
        _path = TOOLS_DIR / f'{_name}{_ext}'
        if _path.is_file():
            ffmpeg_binaries.append((str(_path), 'tools'))

a = Analysis(
    [str(SRC_DIR / 'main.py')],
    pathex=[str(SRC_DIR)],
    binaries=ffmpeg_binaries,
    datas=[],
    hiddenimports=[
        # --- librosa and its dependency chain ---
        'librosa',
        'librosa.core',
        'librosa.core.audio',
        'librosa.core.spectrum',
        'librosa.core.pitch',
        'librosa.core.constantq',
        'librosa.core.notation',
        'librosa.core.convert',
        'librosa.util',
        'librosa.util.utils',
        'librosa.util.decorators',
        'librosa.util.exceptions',
        'librosa.feature',
        'librosa.feature.spectral',
        'librosa.feature.rhythm',
        'librosa.onset',
        'librosa.beat',
        'librosa.filters',
        # --- numba (JIT compiler used by librosa) ---
        'numba',
        'numba.core',
        'numba.core.types',
        'numba.np.ufunc',
        'numba.cpython',
        # --- llvmlite (numba backend) ---
        'llvmlite',
        'llvmlite.binding',
        # --- scipy submodules used by librosa ---
        'scipy',
        'scipy.signal',
        'scipy.signal.windows',
        'scipy.fft',
        'scipy.fft._pocketfft',
        'scipy.ndimage',
        'scipy.interpolate',
        'scipy.sparse',
        'scipy.sparse.csgraph',
        'scipy.special',
        'scipy.linalg',
        # --- scikit-learn (used by librosa) ---
        'sklearn',
        'sklearn.utils',
        'sklearn.utils._cython_blas',
        'sklearn.utils._typedefs',
        'sklearn.neighbors',
        'sklearn.neighbors._partition_nodes',
        'sklearn.tree',
        'sklearn.tree._utils',
        # --- audio I/O ---
        'audioread',
        'soundfile',
        'soxr',
        # --- other librosa deps ---
        'joblib',
        'decorator',
        'pooch',
        'platformdirs',
        'lazy_loader',
        # --- numpy ---
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'numpy.lib',
        'numpy.lib.format',
        # --- sometimes missed ---
        'pkg_resources',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'test',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoVideoEditor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutoVideoEditor',
)
