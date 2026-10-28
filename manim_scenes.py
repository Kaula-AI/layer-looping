"""
Manim Community scenes for the per-token layer looping paper (talk / poster video).

Render (pip install manim):
  manim -pqh manim_scenes.py SecondPass
  manim -pqh manim_scenes.py DiagonalSwap
  manim -pqh manim_scenes.py ParetoFrontier
Replace the illustrative curves in ParetoFrontier with oracle_out/results.json once you have results.
"""
import json
import os

import numpy as np
from manim import (BLUE_C, DOWN, GREEN_C, GREY_D, LEFT, ORIGIN, RED_C, RIGHT, TEAL_C, UP, WHITE, YELLOW_C,
                   Axes, Create, CubicBezier, DashedVMobject, Dot, FadeIn, MathTex, MoveAlongPath, Rectangle,
                   RoundedRectangle, Scene, Square, Text, ValueTracker, VGroup, Write, always_redraw, linear)


def layer_stack(n=8, w=3.0, h=0.42, gap=0.14):
    g = VGroup(*[RoundedRectangle(width=w, height=h, corner_radius=0.06, stroke_color=BLUE_C,
                                  fill_color=BLUE_C, fill_opacity=0.1) for _ in range(n)])
    g.arrange(UP, buff=gap)
    labels = VGroup(*[MathTex(rf"\ell={i}", font_size=24, color=GREY_D).next_to(r, LEFT, buff=0.2)
                      for i, r in enumerate(g)])
    return g, labels


class SecondPass(Scene):
    def construct(self):
        title = Text("A second pass through one layer", font_size=34).to_edge(UP)
        stack, labels = layer_stack()
        stack.shift(DOWN * 0.3)
        labels.shift(DOWN * 0.3)
        self.play(Write(title))
        self.play(Create(stack), FadeIn(labels))
        dot = Dot(stack[0].get_bottom() + DOWN * 0.3, color=YELLOW_C, radius=0.1)
        self.play(FadeIn(dot))
        self.play(dot.animate.move_to(stack[4].get_top()), run_time=1.5)
        L4 = stack[4]
        p0, p3 = L4.get_right() + UP * 0.1, L4.get_right() + DOWN * 0.1
        loop = CubicBezier(p0, p0 + RIGHT * 1.8 + UP * 0.8, p3 + RIGHT * 1.8 + DOWN * 0.8, p3, color=YELLOW_C)
        k = MathTex("k=2", color=YELLOW_C).next_to(loop, RIGHT)
        self.play(L4.animate.set_stroke(YELLOW_C, 4), dot.animate.move_to(p0))
        self.play(Create(DashedVMobject(loop)), Write(k))
        self.play(MoveAlongPath(dot, loop), run_time=1.2)
        self.play(dot.animate.move_to(L4.get_top()), run_time=0.6)
        self.play(dot.animate.move_to(stack[-1].get_top() + UP * 0.3), run_time=1.2)
        eq = MathTex(r"h \leftarrow h + f_4(h)", r"\times 2", font_size=40).to_edge(LEFT).shift(UP)
        eq[1].set_color(YELLOW_C)
        nll = ValueTracker(2.41)
        nll_txt = always_redraw(lambda: MathTex(rf"\mathrm{{NLL}} = {nll.get_value():.2f}",
                                                color=GREEN_C if nll.get_value() < 2.4 else WHITE)
                                .next_to(eq, DOWN, buff=0.6).align_to(eq, LEFT))
        self.play(Write(eq), FadeIn(nll_txt))
        self.play(nll.animate.set_value(2.18), run_time=1.5)
        self.wait(1.5)


class DiagonalSwap(Scene):
    def construct(self):
        title = Text("Diagonal swap", font_size=36).to_edge(UP)
        self.play(Write(title))
        n, s = 8, 0.5
        cells = VGroup()
        for i in range(n):
            for j in range(i + 1):
                sq = Square(side_length=s * 0.9, stroke_width=0, fill_color=GREY_D, fill_opacity=0.9)
                sq.move_to(np.array([j * s, -i * s, 0]))
                sq.ij = (i, j)
                cells.add(sq)
        cells.move_to(LEFT * 3 + DOWN * 0.4)
        self.play(FadeIn(cells, lag_ratio=0.02))
        diag = [c for c in cells if c.ij[0] == c.ij[1]]
        self.play(*[c.animate.set_fill(YELLOW_C) for c in diag], run_time=1.2)
        row = [c for c in cells if c.ij[0] == 5]
        box = Rectangle(width=6 * s, height=s, color=BLUE_C).move_to(VGroup(*row))
        self.play(Create(box))
        lines = VGroup(
            Text("other positions keep base-pass K, V", font_size=24),
            Text("only t's own K, V change", font_size=24, color=YELLOW_C),
            MathTex(r"\text{one config} \approx \text{one forward pass}", font_size=34),
            Text("exact for token t's own loss (causal)", font_size=20, color=GREEN_C),
            Text("ignores the effect on later tokens", font_size=20, color=RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(RIGHT).shift(DOWN * 0.3)
        for ln in lines:
            self.play(FadeIn(ln, shift=RIGHT * 0.2), run_time=0.6)
        self.wait(2)


class ParetoFrontier(Scene):
    def construct(self):
        title = MathTex(r"\max\; -\Delta\mathrm{NLL} - \lambda \cdot \text{extra layers}", font_size=40).to_edge(UP)
        self.play(Write(title))
        ax = Axes(x_range=[0, 1, 0.5], y_range=[0, 1, 0.5], x_length=7, y_length=4.2,
                  tips=True, axis_config={"include_numbers": False}).shift(DOWN * 0.4 + LEFT * 1.2)
        xl = Text("extra layer applications per token", font_size=20).next_to(ax, DOWN)
        yl = Text("NLL saved", font_size=20).rotate(np.pi / 2).next_to(ax, LEFT)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))

        res_path = next((q for q in ["oracle_out_full/results.json", "oracle_out_quick/results.json",
                                     "oracle_out/results.json"] if os.path.exists(q)), None)
        curves = []
        if res_path:  # real results, normalised to the oracle's maximum
            fr = json.load(open(res_path))["frontiers"]
            ymax = max(-q["dnll"] for q in fr["oracle"]) or 1
            xmax = max(abs(q["extra_layers"]) for q in fr["oracle"]) or 1
            for name, col in [("oracle", YELLOW_C), ("same_interp", BLUE_C), ("lag_interp", GREEN_C), ("lag_raw", TEAL_C)]:
                pts = sorted((q["extra_layers"] / xmax, -q["dnll"] / ymax) for q in fr[name])
                curves.append((name, col, pts))
        else:  # illustrative shapes
            for name, col, A, k in [("oracle", YELLOW_C, 0.95, 0.13), ("same-token", BLUE_C, 0.72, 0.2),
                                    ("t-1 interp", GREEN_C, 0.62, 0.23), ("t-1 raw", TEAL_C, 0.52, 0.24)]:
                xs = np.linspace(0, 1, 60)
                curves.append((name, col, list(zip(xs, A * (1 - np.exp(-xs / k))))))
            self.add(Text("illustrative", font_size=18, color=GREY_D).next_to(ax, UP, buff=0.05).align_to(ax, RIGHT))
        for name, col, pts in curves:
            line = ax.plot_line_graph([p[0] for p in pts], [max(0, p[1]) for p in pts], line_color=col,
                                      add_vertex_dots=False, stroke_width=4)
            lab = Text(name, font_size=20, color=col).next_to(ax.c2p(1, max(0, pts[-1][1])), RIGHT, buff=0.2)
            self.play(Create(line), FadeIn(lab), run_time=1.0)
        lam = ValueTracker(0.9)
        g = curves[2][2]
        xs, ys = np.array([p[0] for p in g]), np.array([max(0, p[1]) for p in g])
        dot = always_redraw(lambda: Dot(ax.c2p(lam.get_value(), np.interp(lam.get_value(), xs, ys)), color=GREEN_C))
        tag = always_redraw(lambda: MathTex(r"\lambda", r"\uparrow", color=GREEN_C).next_to(dot, UP + RIGHT, buff=0.1))
        self.play(FadeIn(dot), FadeIn(tag))
        self.play(lam.animate.set_value(0.08), run_time=3, rate_func=linear)
        self.wait(1.5)
