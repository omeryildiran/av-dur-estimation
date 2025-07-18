"""
Working Causal Inference Animation - Simplified Version
This version focuses on core functionality without complex features that might cause issues.
"""

from manim import *
import numpy as np
from scipy.stats import norm


class WorkingCausalInferenceDemo(Scene):
    def construct(self):
        # Title
        title = Text("Causal Inference in Duration Perception", font_size=30, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Show the simulation
        self.show_simulation()
        
    def show_simulation(self):
        # Create axes
        axes = Axes(
            x_range=[0, 3, 0.5],
            y_range=[0, 3, 0.5],
            x_length=9,
            y_length=4,
            axis_config={"color": BLUE},
        )
        axes.next_to(self.mobjects[0], DOWN, buff=0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("Duration (s)")
        y_label = axes.get_y_axis_label("Probability Density")
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Show three scenarios
        scenarios = [
            {"name": "No Conflict", "S_a": 0.8, "conflict": 0.0, "color": GREEN},
            {"name": "Medium Conflict", "S_a": 0.8, "conflict": 0.4, "color": YELLOW},
            {"name": "High Conflict", "S_a": 0.8, "conflict": 0.8, "color": RED},
        ]
        
        for i, scenario in enumerate(scenarios):
            if i > 0:
                self.wait(1)
                # Clear previous scenario
                for mob in self.mobjects[4:]:  # Keep title, axes, and labels
                    self.remove(mob)
            
            self.animate_scenario(axes, scenario)
            self.wait(2)
    
    def animate_scenario(self, axes, scenario):
        # Parameters
        S_a = scenario["S_a"]
        conflict = scenario["conflict"]
        S_v = S_a + conflict
        sigma = 0.2
        p_c = 0.8 if conflict < 0.2 else (0.6 if conflict < 0.6 else 0.3)
        
        # Generate data
        n_samples = 1000
        np.random.seed(42)
        m_a = np.random.normal(S_a, sigma, n_samples)
        m_v = np.random.normal(S_v, sigma, n_samples)
        
        # Compute causal inference
        estimates = self.compute_causal_inference(S_a, S_v, m_a, m_v, sigma, sigma, p_c)
        
        # Create smooth curves
        x_vals = np.linspace(0, 3, 100)
        
        # Theoretical curves
        y_a = norm.pdf(x_vals, S_a, sigma) * 2
        y_v = norm.pdf(x_vals, S_v, sigma) * 2
        y_est = norm.pdf(x_vals, np.mean(estimates), np.std(estimates)) * 2
        
        # Create curves
        curve_a = axes.plot(lambda x: norm.pdf(x, S_a, sigma) * 2, x_range=[0, 3], color=TEAL)
        curve_v = axes.plot(lambda x: norm.pdf(x, S_v, sigma) * 2, x_range=[0, 3], color=GREEN)
        curve_est = axes.plot(lambda x: norm.pdf(x, np.mean(estimates), np.std(estimates)) * 2, 
                             x_range=[0, 3], color=scenario["color"])
        
        # Vertical lines
        line_a = axes.get_vertical_line(axes.c2p(S_a, 0), color=TEAL, stroke_width=3)
        line_v = axes.get_vertical_line(axes.c2p(S_v, 0), color=GREEN, stroke_width=3)
        line_est = axes.get_vertical_line(axes.c2p(np.mean(estimates), 0), 
                                         color=scenario["color"], stroke_width=3)
        
        # Labels
        scenario_label = Text(scenario["name"], font_size=18, color=scenario["color"])
        scenario_label.next_to(axes, UP, buff=0.2)
        
        # Statistics
        stats_text = Text(f"S_a={S_a:.1f}, S_v={S_v:.1f}, P(fusion)≈{p_c:.1f}", 
                         font_size=14, color=WHITE)
        stats_text.next_to(scenario_label, DOWN, buff=0.2)
        
        # Animate
        self.play(Write(scenario_label))
        self.play(Write(stats_text))
        self.play(
            Create(curve_a),
            Create(curve_v),
            Create(line_a),
            Create(line_v),
            run_time=2
        )
        self.play(
            Create(curve_est),
            Create(line_est),
            run_time=1.5
        )
        
        # Add explanation
        if conflict == 0.0:
            explanation = Text("Perfect agreement → Strong fusion", font_size=12, color=GREEN)
        elif conflict < 0.6:
            explanation = Text("Moderate conflict → Partial fusion", font_size=12, color=YELLOW)
        else:
            explanation = Text("Large conflict → Weak fusion", font_size=12, color=RED)
        
        explanation.next_to(stats_text, DOWN, buff=0.3)
        self.play(Write(explanation))
    
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


class QuickMathDemo(Scene):
    def construct(self):
        # Title
        title = Text("Causal Inference Mathematics", font_size=28, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show key equation without LaTeX
        equation = Text("Final Estimate = P(Common) × Fused + P(Separate) × Auditory", 
                       font_size=16, color=YELLOW)
        equation.next_to(title, DOWN, buff=0.8)
        
        box = SurroundingRectangle(equation, color=YELLOW, buff=0.2)
        
        self.play(Write(equation))
        self.play(Create(box))
        self.wait(2)
        
        # Show components
        components = VGroup(
            Text("Where:", font_size=16, color=WHITE),
            Text("P(Common) = Probability signals come from same source", font_size=14, color=WHITE),
            Text("Fused = Reliability-weighted average of both signals", font_size=14, color=WHITE),
            Text("Auditory = Just the auditory measurement", font_size=14, color=WHITE),
            Text("", font_size=12, color=WHITE),  # spacer
            Text("Key insight: Higher conflict → Lower P(Common)", font_size=14, color=GREEN),
        )
        
        components.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        components.next_to(equation, DOWN, buff=0.8)
        
        for comp in components:
            self.play(Write(comp))
            self.wait(0.5)
        
        self.wait(3)


class QuickConceptDemo(Scene):
    def construct(self):
        # Title
        title = Text("The Causal Inference Question", font_size=28, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Central question
        question = Text("Are these signals from the same event?", font_size=20, color=WHITE)
        question.next_to(title, DOWN, buff=1)
        self.play(Write(question))
        
        # Two paths
        path1 = Text("YES → Combine the signals", font_size=18, color=GREEN)
        path2 = Text("NO → Keep them separate", font_size=18, color=RED)
        
        path1.next_to(question, DOWN, buff=0.8).shift(LEFT * 2)
        path2.next_to(question, DOWN, buff=0.8).shift(RIGHT * 2)
        
        # Arrows
        arrow1 = Arrow(question.get_bottom(), path1.get_top(), color=GREEN)
        arrow2 = Arrow(question.get_bottom(), path2.get_top(), color=RED)
        
        self.play(
            Write(path1),
            Write(path2),
            Create(arrow1),
            Create(arrow2)
        )
        self.wait(2)
        
        # Reality
        reality = Text("Reality: Use probabilistic weighting", font_size=20, color=YELLOW)
        reality.next_to(path1, DOWN, buff=1).shift(RIGHT)
        self.play(Write(reality))
        self.wait(3)
