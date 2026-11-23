# LITE-008D-R1 CMake Qualification

## Result

CMake recovery stopped before installation because the normal Termux transaction would configure unrelated pre-existing X11 packages. The directive requires stopping when the transaction is not limited to CMake and strictly necessary dependencies.

- Repository: `NewportRiver/memorilla`
- Branch: `feature/memorilla-lite-android`
- Required baseline: `cfcfb69`
- Baseline: verified
- Working tree: clean before this documentation
- CMake installed: no
- Model/source acquisition: none
- Package installation: none

## Repository and package provenance

Configured repositories:

- Main: `https://termux.net`, suite `stable`, component `main`, architecture `aarch64`
- Signature key: `/data/data/com.termux/files/usr/etc/apt/trusted.gpg.d/termux-packages.gpg`
- X11: `https://x11-packages.termux.net/`, suite `x11`, component `main`, architecture `aarch64`

`apt-cache policy cmake` selected:

- Version: `4.3.4`
- Origin: `https://termux.net stable/main aarch64`
- Installed: none

The configured Termux source uses signed repository metadata through the configured `Signed-By` key. No repository change was made.

## Package metadata

CMake package:

- Package: `cmake`
- Version: `4.3.4`
- Architecture: `aarch64`
- Archive: `debs/stable/aarch64/c/cmake/cmake_4.3.4_aarch64.deb`
- Download size: `10,061,972` bytes
- Installed size: `52,616 KiB`
- Repository SHA-256: `c674ce9eeabe36e69f3586dd958f5455877a8215a71eaf5867caa417aa312e7`

Strict CMake dependencies proposed by apt:

| Package | Version | Download bytes | Installed KiB | Repository SHA-256 |
|---|---:|---:|---:|---|
| `libarchive` | `3.8.8` | `392,732` | `1,020` | `d0409f55c5b745cf113f7d901be90abfcea108a4255d3778221579efc0f51f33` |
| `jsoncpp` | `1.9.8` | `96,204` | `452` | `9089651dddba30095f4406b596549765c323b295d8cec7ae59619ebb1c880c21` |
| `libuv` | `1.52.1` | `82,188` | `352` | `203b0bff16017016a47c1023e827a7ad6b484b4f3b83b331098f09591c69de5e` |
| `rhash` | `1.4.6-1` | `163,252` | `388` | `8a7dee7fde53702231c729bb89181fc6aa41322f819741b01b8a7a7e4b248bef` |

CMake also depends on already-installed or separately resolved core libraries (`libc++`, `libcurl`, `libexpat`, `zlib`) and recommends `clang` and `make`; no install was attempted.

## Simulated transaction

Command:

```sh
apt-get -s --no-remove --no-upgrade --no-install-recommends install cmake
```

Relevant result:

```text
The following NEW packages will be installed:
  cmake jsoncpp libarchive libuv rhash
0 upgraded, 5 newly installed, 0 to remove
9 not fully installed or removed
```

The simulation also proposes configuration of these unrelated existing X11 packages:

- `qt6-qtsvg`
- `qt6-qtdeclarative`
- `kf6-kwindowsystem`
- `liblxqt`
- `qt6-qtbase`
- `qt6-qtlanguageserver`
- `libqtxdg`
- `obconf-qt`
- `xdg-utils`

`dpkg --audit` confirmed these packages are unpacked or half-configured. Because configuring them would exceed the approved CMake-only scope, no transaction was run.

## Integrity and resource preflight

- MiniLM SHA-256 matched: `4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474`
- Associative checkpoint SHA-256: `fb9a273fd96bfe9d6cf35da2145e45492b414ad6a02cd35acfba999886587a7a`
- Direct checkpoint SHA-256: `a9e190cd7a0c81dc4ac070b9299819fced77d3b38432c9534577b8b19691d2c8`
- Physical RAM: `5,660,758,016` bytes
- Available RAM: `1,438,486,528` bytes
- Free swap: `5,379,932,160` bytes
- Free storage: `7,314,616 KiB`
- Decoder RSS ceiling: `268,435,456` bytes
- No SQLite or checkpoint mutation
- No holdout access

## Required approval boundary

Owner approval is required before either:

1. allowing the package manager to configure the nine unrelated X11 packages, or
2. authorizing a separately verified package-install method that installs only CMake and its strict dependencies without touching the pending X11 state.

No CMake rollback is applicable because no package was installed. No unrelated Termux file was deleted.
