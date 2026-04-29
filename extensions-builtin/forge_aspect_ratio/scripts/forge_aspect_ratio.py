"""
Forge Aspect Ratio Selector - Built-in Extension

Aspect-ratio calculation logic is adapted from:
  ControlAltAI-Nodes / flux_resolution_cal_node.py
  https://github.com/gseth/ControlAltAI-Nodes
  Copyright (c) 2024 ControlAltAI
  Used under the MIT License.

The above logic has been rewritten and integrated into the Forge WebUI
extension framework by the maintainers of this project.

---

MIT License (applies to the adapted calculation logic)

Copyright (c) 2024 ControlAltAI

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
"""

import gradio as gr
import modules.scripts as scripts
from modules.ui_components import ToolButton
from pathlib import Path
from math import gcd, sqrt

EXT_DIR = Path(scripts.basedir())


def _read_config_file(filename: str):
    """Parse a comma-separated config file. Returns list of (label, value_str, comment)."""
    file_path = EXT_DIR / filename
    if not file_path.exists():
        return []
    entries = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        comment = ""
        if "#" in line:
            line, comment = line.split("#", 1)
            comment = comment.strip()
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        entries.append((parts[0], parts[1], comment))
    return entries


def _write_default_aspect_ratios():
    (EXT_DIR / "aspect_ratios.txt").write_text(
        "1:1, 1.0\n"
        "3:2, 3/2\n"
        "4:3, 4/3\n"
        "16:9, 16/9\n"
        "9:16, 9/16\n",
        encoding="utf-8",
    )


def _write_default_resolutions():
    (EXT_DIR / "resolutions.txt").write_text(
        "512, 512, 512\n"
        "768x512, 768, 512\n"
        "512x768, 512, 768\n"
        "1024, 1024, 1024\n",
        encoding="utf-8",
    )


def _ratio_label(n: int, d: int) -> str:
    """Return reduced ratio label like '3:2'."""
    if n == d:
        return "1:1"
    div = gcd(n, d)
    return f"{n // div}:{d // div}"


def _calculate_dimensions(mp: float, ratio_str: str, round_to: int = 64):
    """Calculate width/height from megapixels and aspect ratio.

    Adapted from ControlAltAI-Nodes flux_resolution_cal_node.py.
    """
    width_ratio, height_ratio = map(int, ratio_str.split(":"))
    total_pixels = mp * 1_000_000
    dimension = sqrt(total_pixels / (width_ratio * height_ratio))
    width = int(dimension * width_ratio)
    height = int(dimension * height_ratio)
    width = round(width / round_to) * round_to
    height = round(height / round_to) * round_to
    return width, height


class ForgeAspectRatioScript(scripts.Script):
    def title(self):
        return "Aspect Ratio"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        prefix = "img2img" if is_img2img else "txt2img"

        # Ensure defaults exist
        if not (EXT_DIR / "aspect_ratios.txt").exists():
            _write_default_aspect_ratios()
        if not (EXT_DIR / "resolutions.txt").exists():
            _write_default_resolutions()

        ar_entries = _read_config_file("aspect_ratios.txt")
        res_entries = _read_config_file("resolutions.txt")

        with gr.Column(elem_id=f"{prefix}_forge_ar_container"):
            # Aspect ratio buttons
            with gr.Row(elem_id=f"{prefix}_forge_ar_row"):
                for label, value_str, _ in ar_entries:
                    try:
                        ratio = float(eval(value_str))
                    except Exception:
                        continue
                    btn = gr.Button(value=label, size="sm", variant="secondary")
                    btn.elem_classes = ["tool"]
                    if is_img2img:
                        inputs = [self.i2i_w, self.i2i_h]
                    else:
                        inputs = [self.t2i_w, self.t2i_h]

                    def apply_ar(w, h, r=ratio):
                        if r > 1.0:
                            return [round(r * h), h]
                        elif r < 1.0:
                            return [w, round(w / r)]
                        else:
                            m = min(w, h)
                            return [m, m]

                    btn.click(fn=apply_ar, inputs=inputs, outputs=inputs)

            # Resolution preset buttons
            with gr.Row(elem_id=f"{prefix}_forge_res_row"):
                for label, w_str, h_str in res_entries:
                    try:
                        w, h = int(w_str), int(h_str)
                    except Exception:
                        continue
                    btn = gr.Button(value=label, size="sm", variant="secondary")
                    btn.elem_classes = ["tool"]
                    if is_img2img:
                        outputs = [self.i2i_w, self.i2i_h]
                    else:
                        outputs = [self.t2i_w, self.t2i_h]
                    btn.click(fn=lambda w=w, h=h: [w, h], outputs=outputs)

            # Calculator toggle
            with gr.Row():
                calc_toggle = gr.Button(value="Calc", size="sm")
                calc_hide = gr.Button(value="Close", size="sm", visible=False)

            # Calculator panel (integrated with _calculate_dimensions)
            with gr.Column(visible=False) as calc_panel:
                gr.Markdown("#### Resolution Calculator")
                with gr.Row():
                    calc_mp = gr.Slider(
                        label="Megapixels",
                        minimum=0.1,
                        maximum=2.5,
                        step=0.1,
                        value=1.0,
                    )
                with gr.Row():
                    calc_ratio = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=[
                            "1:1",
                            "2:3",
                            "3:2",
                            "3:4",
                            "4:3",
                            "9:16",
                            "16:9",
                            "21:9",
                            "32:9",
                        ],
                        value="1:1",
                    )
                    calc_div = gr.Dropdown(
                        label="Divisible By",
                        choices=["8", "16", "32", "64"],
                        value="64",
                    )
                with gr.Row():
                    calc_btn = gr.Button("Calculate", variant="primary")
                with gr.Row():
                    calc_result = gr.Markdown("Resolution: **1024 x 1024**")
                with gr.Row():
                    calc_apply = gr.Button("Apply to Sliders")

                def do_calc(mp, ratio, div):
                    try:
                        w, h = _calculate_dimensions(
                            float(mp), ratio, int(div)
                        )
                        return f"Resolution: **{w} x {h}**"
                    except Exception:
                        return "Resolution: **error**"

                calc_btn.click(
                    fn=do_calc,
                    inputs=[calc_mp, calc_ratio, calc_div],
                    outputs=calc_result,
                )

                if is_img2img:
                    apply_out = [self.i2i_w, self.i2i_h]
                else:
                    apply_out = [self.t2i_w, self.t2i_h]

                def do_apply(mp, ratio, div):
                    try:
                        return list(_calculate_dimensions(float(mp), ratio, int(div)))
                    except Exception:
                        return [512, 512]

                calc_apply.click(
                    fn=do_apply,
                    inputs=[calc_mp, calc_ratio, calc_div],
                    outputs=apply_out,
                )

            calc_toggle.click(
                fn=lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)],
                outputs=[calc_panel, calc_toggle, calc_hide],
            )
            calc_hide.click(
                fn=lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
                outputs=[calc_panel, calc_toggle, calc_hide],
            )

    def after_component(self, component, **kwargs):
        eid = kwargs.get("elem_id")
        if eid == "txt2img_width":
            self.t2i_w = component
        elif eid == "txt2img_height":
            self.t2i_h = component
        elif eid == "img2img_width":
            self.i2i_w = component
        elif eid == "img2img_height":
            self.i2i_h = component
