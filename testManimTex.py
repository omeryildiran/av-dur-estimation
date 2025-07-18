from manim import *

class TestLatex(Scene):
    def construct(self):
        eq = MathTex(r"E = mc^2")
        self.play(Write(eq))
        self.wait()
