# Third-party notices

## RoboDojo

RoboDojo-Turbo patches a user's RoboDojo checkout and mirrors a few lines of it (see NOTICE). RoboDojo's LICENSE file:

```
MIT License

Copyright (c) 2025 Yue Chen <yuechen020614@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Note: RoboDojo's LICENSE file is MIT, but its README states that RoboDojo is "Released under the RoboDojo Non-Commercial
Research License" for non-commercial research, education and evaluation, and that commercial use requires prior written
permission from its maintainers. Follow RoboDojo's current terms when you use RoboDojo, including evaluations patched with
this tool.

One file this tool edits in your RoboDojo checkout, env/camera_manager/capture/camera_view.py, carries an NVIDIA
proprietary notice ("All rights reserved ... without an express license agreement from NVIDIA CORPORATION is strictly
prohibited"). RoboDojo-Turbo contains only a one-line regex anchor from it (the `rgb3` patch) and edits your own copy; if
that matters to you, leave the patch out with `python -m robodojo_turbo apply --exclude rgb3`.

## XPolicyLab, NVIDIA Isaac Sim

Apache License 2.0; the full text is in LICENSE. See NOTICE for the adapted files.

## How patched code is marked

The patches rewrite files of your own RoboDojo checkout. Every injected block starts with a line carrying the marker
`[robodojo-turbo]`; to see everything that changed, run `git diff --submodule=diff` in the checkout (XPolicyLab is a
submodule), `git status` for the three installed `utils/rdturbo_*.py` modules, or `diff -r` against `.robodojo_turbo/backup`.

## Not affiliated

This project is not affiliated with or endorsed by the RoboDojo maintainers, XPolicyLab, Physical Intelligence or NVIDIA.

## Not included

NVIDIA Isaac Sim / Omniverse Kit binaries, NVIDIA materials and textures (scripts/mirror_nv_assets.sh downloads them from
NVIDIA to your machine; do not redistribute them), cuRobo, Isaac Lab, openpi, the pi0.5 weights (which build on
PaliGemma; Gemma terms may apply) and the RoboDojo datasets are obtained separately under their own licenses.
