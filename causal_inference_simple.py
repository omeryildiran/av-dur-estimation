"""
Simple Manim Animation for Causal Inference Demo

This is a simplified version focusing on the core concepts.
"""

from manim import *
import numpy as np
from scipy.stats import norm, gaussian_kde


class CausalInferenceDemo(Scene):
    def construct(self):
        # Title
        title = Text("Causal Inference in Duration Perception", font_size=32, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # Show the core question
        question = Text("Should auditory and visual signals be combined?", 
                       font_size=24, color=WHITE)
        question.next_to(title, DOWN, buff=0.5)
        self.play(Write(question))
        self.wait(2)
        
        # Show mathematical formulation
        self.show_math_formulation()
        self.wait(2)
        
        # Clear and show simulation
        self.clear()
        self.show_simulation()
        
    def show_math_formulation(self):
        # Clear previous content except title
        if len(self.mobjects) > 1:
            self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])
        
        # Mathematical steps - using simpler LaTeX expressions
        step1 = Text("Step 1: Noisy Measurements", font_size=20, color=GREEN)
        eq1 = MathTex("m_a", "\\sim", "N(S_a, \\sigma_a^2)")
        eq2 = MathTex("m_v", "\\sim", "N(S_v, \\sigma_v^2)")
        
        step2 = Text("Step 2: Causal Inference", font_size=20, color=GREEN)
        eq3 = MathTex("P(C=1|m_a, m_v)", "=", "\\frac{L_1 \\cdot p}{L_1 \\cdot p + L_2 \\cdot (1-p)}")
        
        step3 = Text("Step 3: Final Estimate", font_size=20, color=GREEN)
        eq4 = MathTex("\\hat{S}", "=", "P(C=1) \\cdot S_{fused} + P(C=2) \\cdot S_{aud}")
        
        # Arrange equations
        equations = VGroup(step1, eq1, eq2, step2, eq3, step3, eq4)
        equations.arrange(DOWN, buff=0.3)
        equations.scale(0.8)
        equations.next_to(self.mobjects[0], DOWN, buff=0.5)
        
        # Animate equations
        for eq in equations:
            self.play(Write(eq))
            self.wait(0.5)
    
    def show_simulation(self):
        # Title
        title = Text("Simulation Results", font_size=32, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create axes
        axes = Axes(
            x_range=[0, 3, 0.5],
            y_range=[0, 3, 0.5],
            x_length=8,
            y_length=5,
            axis_config={"color": BLUE},
        )
        axes.next_to(title, DOWN, buff=0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("Duration (s)")
        y_label = axes.get_y_axis_label("Probability Density")
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Show different scenarios
        scenarios = [
            {"name": "Low Conflict", "S_a": 1.0, "conflict": 0.2, "color": GREEN},
            {"name": "High Conflict", "S_a": 1.0, "conflict": 0.8, "color": RED},
        ]
        
        for i, scenario in enumerate(scenarios):
            self.animate_scenario(axes, scenario, i)
            self.wait(2)
    
    def animate_scenario(self, axes, scenario, index):
        # Parameters
        S_a = scenario["S_a"]
        conflict = scenario["conflict"]
        S_v = S_a + conflict
        sigma_a = 0.2
        sigma_v = 0.2
        p_c = 0.7
        
        # Generate data
        n_samples = 1000
        m_a = np.random.normal(S_a, sigma_a, n_samples)
        m_v = np.random.normal(S_v, sigma_v, n_samples)
        
        # Compute causal inference estimates
        estimates = self.compute_causal_inference(S_a, S_v, m_a, m_v, sigma_a, sigma_v, p_c)
        
        # Create smooth curves using numpy arrays
        x_vals = np.linspace(0, 3, 100)
        
        # Auditory curve
        y_a = norm.pdf(x_vals, S_a, sigma_a)
        curve_a = axes.plot_line_graph(x_vals, y_a, line_color=TEAL, stroke_width=3)
        
        # Visual curve  
        y_v = norm.pdf(x_vals, S_v, sigma_v)
        curve_v = axes.plot_line_graph(x_vals, y_v, line_color=GREEN, stroke_width=3)
        
        # Estimate curve (approximate with normal distribution)
        y_est = norm.pdf(x_vals, np.mean(estimates), np.std(estimates))
        curve_est = axes.plot_line_graph(x_vals, y_est, line_color=scenario["color"], stroke_width=3)
        
        # Vertical lines for true values
        line_a = axes.get_vertical_line(axes.c2p(S_a, 0), color=TEAL, stroke_width=2)
        line_v = axes.get_vertical_line(axes.c2p(S_v, 0), color=GREEN, stroke_width=2)
        line_est = axes.get_vertical_line(axes.c2p(np.mean(estimates), 0), 
                                        color=scenario["color"], stroke_width=2)
        
        # Scenario label
        scenario_text = Text(scenario["name"], font_size=18, color=scenario["color"])
        scenario_text.to_edge(RIGHT).shift(UP * (1 - index))
        
        # Animate
        self.play(
            Create(curve_a),
            Create(curve_v), 
            Create(curve_est),
            Create(line_a),
            Create(line_v),
            Create(line_est),
            Write(scenario_text)
        )
    
    def compute_causal_inference(self, S_a, S_v, m_a, m_v, sigma_a, sigma_v, p_c):
        """Compute causal inference estimates"""
        # Fusion weights
        J_a = 1 / sigma_a**2
        J_v = 1 / sigma_v**2
        w_a = J_a / (J_a + J_v)
        w_v = 1 - w_a
        
        # Fused estimate
        fused = w_a * m_a + w_v * m_v
        
        # Likelihoods
        var_sum = sigma_a**2 + sigma_v**2
        likelihood_c1 = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))
        likelihood_c2 = norm.pdf(m_a, S_a, sigma_a) * norm.pdf(m_v, S_v, sigma_v)
        
        # Posterior probability of common cause
        posterior_c1 = (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))
        
        # Final estimate
        final_estimate = posterior_c1 * fused + (1 - posterior_c1) * m_a
        
        return final_estimate


class MathFormulationOnly(Scene):
    def construct(self):
        # Title
        title = Text("Causal Inference Mathematics", font_size=32, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Core equation - simplified
        main_eq = MathTex(
            "\\hat{S}_{final}", "=", "P(C=1) \\cdot \\hat{S}_{fused}", "+", "P(C=2) \\cdot \\hat{S}_{aud}"
        )
        main_eq.next_to(title, DOWN, buff=1)
        main_eq.scale(0.8)
        
        # Highlight the equation
        rect = SurroundingRectangle(main_eq, color=YELLOW, buff=0.2)
        
        self.play(Write(main_eq))
        self.play(Create(rect))
        self.wait(2)
        
        # Break down the components - using simpler LaTeX
        components = VGroup(
            Text("Where:", font_size=20, color=WHITE),
            MathTex("P(C=1)", "=", "\\text{Probability of common cause}"),
            MathTex("\\hat{S}_{fused}", "=", "w_a m_a + w_v m_v"),
            MathTex("\\hat{S}_{aud}", "=", "m_a"),
            MathTex("w_a", "=", "\\frac{J_a}{J_a + J_v}")
        )
        
        components.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        components.scale(0.7)
        components.next_to(main_eq, DOWN, buff=0.8)
        
        for comp in components:
            self.play(Write(comp))
            self.wait(0.5)
        
        self.wait(3)


class ParameterEffectsDemo(Scene):
    def construct(self):
        # Title
        title = Text("Parameter Effects on Causal Inference", font_size=28, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Parameter descriptions
        effects = [
            "Larger conflict → Lower probability of common cause",
            "Higher noise → More uncertainty in estimates", 
            "Higher prior P(C=1) → More tendency to fuse",
            "Asymmetric noise → Biased reliability weighting"
        ]
        
        effect_texts = VGroup()
        for i, effect in enumerate(effects):
            text = Text(effect, font_size=18, color=WHITE)
            if i == 0:
                text.next_to(title, DOWN, buff=0.5)
            else:
                text.next_to(effect_texts[-1], DOWN, buff=0.4)
            effect_texts.add(text)
        
        for text in effect_texts:
            self.play(Write(text))
            self.wait(1)
        
        # Summary equation - simplified
        summary = MathTex(
            "\\text{Final Estimate}", "=", "f(\\text{Conflict, Noise, Prior, Reliability})"
        )
        summary.next_to(effect_texts, DOWN, buff=0.8)
        summary.scale(0.8)
        
        self.play(Write(summary))
        self.wait(3)
