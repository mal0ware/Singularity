---
layout: default
title: Download
permalink: /download/
description: Installers for macOS and Windows, plus the SmartScreen workaround.
---

# Download

Releases are published on GitHub:

**[github.com/mal0ware/Singularity/releases](https://github.com/mal0ware/Singularity/releases)**

| Platform | Artefact | Notes |
|---|---|---|
| macOS (Apple Silicon) | `Singularity-<ver>.dmg` | Ad-hoc signed; Developer ID notarization is gated on Apple Developer Program enrollment (planned summer 2026). |
| Windows 10/11 (x64) | `Singularity-<ver>-win64.msi` | Unsigned — see the SmartScreen workaround below. |

## Windows: SmartScreen workaround

Windows SmartScreen flags unsigned executables on first launch with a *"Microsoft Defender SmartScreen prevented an unrecognized app from starting"* dialog. The binary is intact; Singularity is shipped unsigned because OV/EV code-signing certificates run $70–$500/yr and the project has not reached a download volume that justifies that line item.

To run it:

1. Double-click the `.msi`.
2. When the SmartScreen dialog appears, click **"More info"**.
3. Click **"Run anyway"**.
4. The installer proceeds normally. After install, `singularity.exe` is located at `%ProgramFiles%\Singularity\bin\` with a Start-Menu shortcut.

A signing posture upgrade (OV first, possibly EV later) is planned once download volume justifies the cost. The release workflow already honours `SINGULARITY_WIN_SIGN_PFX` if a certificate is provisioned — see [`docs/NEXT_STEPS_WINDOWS.md`](https://github.com/mal0ware/Singularity/blob/main/docs/NEXT_STEPS_WINDOWS.md) §5 for the signtool wiring.

## macOS: Gatekeeper

The current `.dmg` is ad-hoc signed, not notarized; macOS Gatekeeper will refuse to launch it on first attempt. To run it:

1. Mount the `.dmg` and drag `Singularity.app` to `/Applications`.
2. Right-click `Singularity.app` → **Open** (not double-click).
3. Click **Open** in the Gatekeeper dialog.
4. Subsequent launches behave normally.

The notarization workflow is pre-wired in [`docs/NEXT_STEPS_MAC.md`](https://github.com/mal0ware/Singularity/blob/main/docs/NEXT_STEPS_MAC.md) and activates automatically once the Apple Developer Program membership is in place.

## Build from source

```bash
git clone --recurse-submodules https://github.com/mal0ware/Singularity.git
cd Singularity
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

CMake auto-selects the platform-native backend — Metal on macOS (requires Xcode for the metal shader compiler), Vulkan on Windows and Linux (requires the LunarG SDK for `dxc`). Full per-platform setup details are in [`NEXT_STEPS_MAC.md`](https://github.com/mal0ware/Singularity/blob/main/docs/NEXT_STEPS_MAC.md) and [`NEXT_STEPS_WINDOWS.md`](https://github.com/mal0ware/Singularity/blob/main/docs/NEXT_STEPS_WINDOWS.md).

## Browser demo

No install required: the WebGPU demo is embedded on the [home page]({{ '/' | relative_url }}). It runs the same Kerr integrator and physics, compiled to WebAssembly via Emscripten.
