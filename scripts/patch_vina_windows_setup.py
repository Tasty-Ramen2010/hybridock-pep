#!/usr/bin/env python
"""Patch AutoDock Vina's setup.py to build on native Windows against MSYS2's
mingw-w64 GCC + Boost.

vina's own sdist has no Windows support (its classifiers list even has
'Operating System :: Microsoft :: Windows' commented out) - pip installing it
on Windows means building from source, and its setup.py hardcodes assumptions
that only hold on Linux/macOS. This patches the four real, confirmed root
causes found by actually reproducing the build locally with MSYS2 (2026-08-03):

1. Hardcoded -std=c++11: MSYS2's current Boost (Boost.Math specifically)
   requires C++14+; its own header #warns this. Bump to C++17.
2. Boost.Atomic's Windows backend needs _WIN32_WINNT >= Windows 8 (0x0602) to
   see WaitOnAddress/WakeByAddress* from synchapi.h; MinGW's default target is
   older, so boost/atomic/detail/wait_ops_windows.hpp fails with "'WaitOnAddress'
   is not a member of 'boost::winapi'".
3. extra_link_args hardcodes bare -lboost_thread/-lboost_serialization/etc.
   MSYS2's mingw-w64 Boost packages always ship the -mt (multi-threaded)
   tagged names (libboost_thread-mt.a), so the linker fails with "cannot find
   -lboost_thread". Link the static .a archives directly (by full path,
   bypassing -l lookup entirely) instead of the bare names, both to fix the
   naming mismatch and to avoid a runtime dependency on Boost's DLLs - CPython
   >= 3.8 no longer searches PATH for an extension module's dependent DLLs, so
   a dynamically-linked .pyd would silently fail to import for anyone who
   just ran `pip install vina` without MSYS2 on their PATH.
4. Even with Boost statically linked, libwinpthread-1.dll remains a genuine
   runtime dependency (MinGW's C++11 threading support) that no link-flag
   combination avoids. Vendor it directly next to the built .pyd - Windows'
   DLL loader always searches a loaded module's own directory - so the wheel
   works with a plain `pip install`, matching what tools like delvewheel do.

Usage: python patch_vina_windows_setup.py <path-to-vina's-setup.py>
"""

import sys


def patch(text: str) -> str:
    if "platform.system() == 'Windows'" in text:
        raise RuntimeError("setup.py already looks patched - refusing to double-patch")

    # 1. C++ standard version
    text = text.replace('"-std=c++11"', '"-std=c++17"', 1)
    if '"-std=c++17"' not in text:
        raise RuntimeError("could not find -std=c++11 to patch")

    # 2. _WIN32_WINNT / WINVER
    old_opts = (
        '                               "-std=c++17",\n'
        '                               "-Wno-long-long",\n'
        '                               "-pedantic",\n'
        "                               '-DBOOST_ERROR_CODE_HEADER_ONLY'\n"
        '                              ]'
    )
    new_opts = (
        '                               "-std=c++17",\n'
        '                               "-Wno-long-long",\n'
        '                               "-pedantic",\n'
        "                               '-DBOOST_ERROR_CODE_HEADER_ONLY',\n"
        "                               '-D_WIN32_WINNT=0x0602',\n"
        "                               '-DWINVER=0x0602',\n"
        '                              ]'
    )
    if old_opts not in text:
        raise RuntimeError("could not find vina_compiler_options block to patch")
    text = text.replace(old_opts, new_opts, 1)

    # 3 & 4. Windows-specific link handling + winpthread vendoring
    anchor = (
        "            # To get the right @rpath on macos for libraries\n"
        "            self.extensions[0].extra_link_args.append('-Wl,-rpath,' + self.library_dirs[0])\n"
        "            self.extensions[0].extra_link_args.append('-Wl,-rpath,' + '/usr/lib')\n"
    )
    if anchor not in text:
        raise RuntimeError("could not find macOS rpath block to anchor the Windows patch")

    windows_link_block = anchor + (
        "        elif platform.system() == 'Windows':\n"
        "            # See module docstring points 3 & 4.\n"
        "            static_libs = []\n"
        "            for arg in self.extensions[0].extra_link_args:\n"
        "                if arg.startswith('-lboost_'):\n"
        "                    libname = 'lib' + arg[2:] + '-mt.a'\n"
        "                    static_libs.append(os.path.join(self.library_dirs[-1], libname))\n"
        "                else:\n"
        "                    static_libs.append(arg)\n"
        "            self.extensions[0].extra_link_args = static_libs\n"
        "            self.extensions[0].extra_link_args += [\n"
        "                '-static-libgcc', '-static-libstdc++',\n"
        "                '-Wl,-Bstatic', '-lwinpthread', '-Wl,-Bdynamic',\n"
        "            ]\n"
    )
    text = text.replace(anchor, windows_link_block, 1)

    dll_vendor_block = (
        "\n"
        "        if platform.system() == 'Windows':\n"
        "            # See module docstring point 4.\n"
        "            dll_name = 'libwinpthread-1.dll'\n"
        "            search_dirs = os.environ.get('PATH', '').split(os.pathsep) + [self.boost_library_dir]\n"
        "            for d in search_dirs:\n"
        "                candidate = os.path.join(d, dll_name)\n"
        "                if os.path.isfile(candidate):\n"
        "                    dest_dir = os.path.join(self.build_lib, 'vina')\n"
        "                    os.makedirs(dest_dir, exist_ok=True)\n"
        "                    shutil.copy2(candidate, os.path.join(dest_dir, dll_name))\n"
        "                    print('Vendored %s into %s' % (candidate, dest_dir))\n"
        "                    break\n"
        "            else:\n"
        "                raise RuntimeError(\n"
        "                    '%s not found on PATH - cannot produce a self-contained '\n"
        "                    'Windows wheel without it.' % dll_name)\n"
    )
    build_ext_call = "\n        build_ext.build_extensions(self)"
    if build_ext_call not in text:
        raise RuntimeError("could not find build_ext.build_extensions(self) call to anchor DLL vendoring")
    text = text.replace(build_ext_call, build_ext_call + dll_vendor_block, 1)

    return text


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        original = f.read()

    patched = patch(original)

    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"Patched {path} for native Windows (MSYS2 mingw-w64) build.")


if __name__ == "__main__":
    main()
