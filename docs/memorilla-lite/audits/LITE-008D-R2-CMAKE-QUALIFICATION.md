# LITE-008D-R2 Controlled Termux Package Recovery

## Outcome

CMake 4.3.4 and its four strict dependencies were installed successfully using a bounded, supported package-manager recovery path. The nine unrelated pending X11 packages were not configured.

- Repository: `NewportRiver/memorilla`
- Branch: `feature/memorilla-lite-android`
- Baseline: `b055852`
- Package transaction: completed
- Decoder acquisition/build/execution: not started

## Initial package state

Before installation, `dpkg --audit` reported these incomplete packages:

- unpacked: `kf6-kwindowsystem`, `liblxqt`, `libqtxdg`, `obconf-qt`, `qt6-qtbase`, `qt6-qtdeclarative`, `qt6-qtlanguageserver`, `qt6-qtsvg`
- half-configured: `xdg-utils`

The normal `apt-get install cmake` simulation would configure these unrelated packages, so it was not used.

## Isolated supported method

The following supported package-manager flow was verified and executed:

1. `apt-get --download-only -y --no-install-recommends` resolved and downloaded only CMake plus its four strict dependencies into a temporary repository-local staging directory.
2. `dpkg --unpack` was invoked only with those five staged `.deb` files.
3. `dpkg --configure libarchive jsoncpp libuv rhash cmake` explicitly configured only those five packages.
4. The temporary staging directory was removed after checksum capture.

No `--force-*` option was used. No `.deb` was manually extracted. No package metadata was overwritten. No `apt-get -f`, `dpkg --configure -a`, repository change, upgrade, removal, or X11 configuration was performed.

Executed commands:

```sh
apt-get --download-only -y --no-install-recommends \
  -o Dir::Cache::archives=$PWD/.lite008d-r2-apt/archives install cmake

dpkg --unpack .lite008d-r2-apt/archives/libarchive_3.8.8_aarch64.deb \
  .lite008d-r2-apt/archives/jsoncpp_1.9.8_aarch64.deb \
  .lite008d-r2-apt/archives/libuv_1.52.1_aarch64.deb \
  .lite008d-r2-apt/archives/rhash_1.4.6-1_aarch64.deb \
  .lite008d-r2-apt/archives/cmake_4.3.4_aarch64.deb

dpkg --configure libarchive jsoncpp libuv rhash cmake
```

## Package verification

Installed successfully:

- `cmake 4.3.4 aarch64`
- `libarchive 3.8.8 aarch64`
- `jsoncpp 1.9.8 aarch64`
- `libuv 1.52.1 aarch64`
- `rhash 1.4.6-1 aarch64`

The staged package SHA-256 values matched repository metadata:

- CMake: `c674ce9eeabe36e69f3586dd958f5455877a8215a71eaf5867caa417aa312e7`
- libarchive: `d0409f55c5b745cf113f7d901be90abfcea108a4255d3778221579efc0f51f33`
- jsoncpp: `9089651dddba30095f4406b596549765c323b295d8cec7ae59619ebb1c880c21`
- libuv: `203b0bff16017016a47c1023e827a7ad6b484b4f3b83b331098f09591c69de5e`
- rhash: `8a7dee7fde53702231c729bb89181fc6aa41322f819741b01b8a7a7e4b248bef`

`dpkg -V` produced no integrity warnings for the five packages. `cmake --version` returned `4.3.4`. `file` verified an ARM64 Android ELF executable at `/data/data/com.termux/files/usr/bin/cmake`.

## Disk and rollback

- Free storage before transaction: `7,364,564 KiB`
- Free storage after install and staging cleanup: `7,251,664 KiB`
- Observed net change: `112,900 KiB`; this includes package installation and normal filesystem accounting, not just the package metadata estimate.
- CMake package declared installed size: `52,616 KiB`.

Rollback is package-scoped and feasible only after checking reverse dependencies:

```sh
dpkg-query -W -f='${binary:Package} ${Version} ${db:Status-Status}\n' \
  cmake libarchive jsoncpp libuv rhash
apt-cache rdepends --installed cmake libarchive jsoncpp libuv rhash
# If no newly introduced dependents exist:
dpkg --remove cmake rhash libuv jsoncpp libarchive
```

Do not run `dpkg --configure -a`, `apt-get -f install`, broad cache deletion, or removal of the unrelated X11 packages. The nine pending X11 packages remain pending after this recovery.

## Boundaries preserved

No model, llama.cpp source, Python package, decoder, SQLite store, neural checkpoint, holdout, latent token, or training state was accessed or modified.
