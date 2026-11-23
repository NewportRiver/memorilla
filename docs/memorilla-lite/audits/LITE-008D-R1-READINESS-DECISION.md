# LITE-008D-R1 Readiness Decision

## Decision: NOT READY — CONDITIONAL GO PENDING PACKAGE-SCOPE APPROVAL

The CMake package is available from the configured trusted Termux repository, but it was not installed. The normal package transaction would configure nine unrelated X11 packages that were already unpacked or half-configured. This violates the directive's requirement to stop when the transaction exceeds the approved CMake scope.

## Verified

- Branch and baseline: `feature/memorilla-lite-android` at `cfcfb69`
- Working tree: clean before documentation
- CMake candidate: `4.3.4`, `aarch64`
- Origin: `https://termux.net stable/main`
- Repository signing configuration: Termux `Signed-By` key at the configured trusted key path
- CMake package SHA-256 metadata: `c674ce9eeabe36e69f3586dd958f5455877a8215a71eaf5867caa417aa312e7`
- CMake download size: `10,061,972` bytes
- CMake installed size: `52,616 KiB`
- Strict dependencies: `libarchive`, `jsoncpp`, `libuv`, `rhash`, plus already-resolved core libraries
- No upgrades proposed
- No removals proposed
- MiniLM and representative neural checkpoints verified unchanged
- No model download, llama.cpp acquisition, Python installation, build, conversion, or execution

## Blocker

`apt-get -s --no-remove --no-upgrade --no-install-recommends install cmake` reports nine packages not fully installed or removed and proposes configuring them:

`qt6-qtsvg`, `qt6-qtdeclarative`, `kf6-kwindowsystem`, `liblxqt`, `qt6-qtbase`, `qt6-qtlanguageserver`, `libqtxdg`, `obconf-qt`, and `xdg-utils`.

Although these are not new upgrades or removals, configuring them is an unrelated side effect. The installation was therefore not attempted.

## Next approval required

Approve one of the following explicitly:

1. permit the normal transaction to configure the nine already-pending X11 packages, or
2. authorize a package-specific recovery method that installs CMake and only its strict dependencies while leaving the pending X11 packages untouched.

No package-specific rollback is needed because CMake was not installed. Any future rollback must remove only packages installed by the approved transaction and restore the pre-transaction package-state manifest; it must not delete unrelated Termux files.

## Scientific boundary

- Gate D: not passed.
- Gate E: not passed.
- Gate F: not passed.
- No decoder qualification occurred.
- No latent-memory integration occurred.
- No architecture, SQLite store, or checkpoint changed.

Stop at owner review. Decoder acquisition remains paused.
