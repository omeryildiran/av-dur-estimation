"""
Simplified Manim Animation for Causal Inference Demo

This version uses minimal LaTeX to avoid rendering issues.
"""

from manim import *
import numpy as np
from scipy.stats import norm


class SimpleCausalInferenceDemo(Scene):
    def construct(self):
        # Title
        title = Text("Causal Inference in Duration Perception", font_size=32, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # Core question
        question = Text("Should auditory and visual signals be combined?", 
                       font_size=24, color=WHITE)
        question.next_to(title, DOWN, buff=0.5)
        self.play(Write(question))
        self.wait(2)
        
        # Show the concept
        self.show_concept()
        self.wait(2)
        
        # Clear and show math
        self.clear()
        self.show_simple_math()
        
    def show_concept(self):
        # Clear question
        self.play(FadeOut(self.mobjects[1]))
        
        # Two scenarios
        scenario1 = Text("Scenario 1: Common source → Fuse signals", 
                        font_size=20, color=GREEN)
        scenario2 = Text("Scenario 2: Different sources → Keep separate", 
                        font_size=20, color=RED)
        
        scenario1.next_to(self.mobjects[0], DOWN, buff=0.2)
        scenario2.next_to(scenario1, DOWN, buff=0.35)
        
        self.play(Write(scenario1))
        self.wait(1)
        self.play(Write(scenario2))
        self.wait(2)
        
        # Solution
        solution = Text("Solution: Weight estimates by probability", 
                       font_size=24, color=YELLOW)
        solution.next_to(scenario2, DOWN, buff=0.5)
        self.play(Write(solution))
        self.wait(2)
    
    def show_simple_math(self):
        # Title
        title = Text("Mathematical Solution", font_size=32, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Step-by-step explanation
        step1 = Text("Step 1: Get noisy measurements", font_size=20, color=GREEN)
        step1.next_to(title, DOWN, buff=0.5)
        self.play(Write(step1))
        
        measurements = VGroup(
            Text("• Auditory measurement: m_a", font_size=18, color=WHITE),
            Text("• Visual measurement: m_v", font_size=18, color=WHITE)
        )
        measurements.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        measurements.next_to(step1, DOWN, buff=0.3)
        
        for m in measurements:
            self.play(Write(m))
            self.wait(0.5)
        
        # Step 2
        step2 = Text("Step 2: Compute probability of common cause", font_size=20, color=GREEN)
        step2.next_to(measurements, DOWN, buff=0.5)
        self.play(Write(step2))
        
        prob_text = Text("P(Common) = Based on how similar m_a and m_v are", 
                        font_size=18, color=WHITE)
        prob_text.next_to(step2, DOWN, buff=0.3)
        self.play(Write(prob_text))
        self.wait(1)
        
        # Step 3
        step3 = Text("Step 3: Combine estimates", font_size=20, color=GREEN)
        step3.next_to(prob_text, DOWN, buff=0.5)
        self.play(Write(step3))
        
        final_eq = Text("Final = P(Common) × Fused + P(Separate) × Auditory", 
                       font_size=18, color=YELLOW)
        final_eq.next_to(step3, DOWN, buff=0.3)
        self.play(Write(final_eq))
        self.wait(3)


class VisualDemo(Scene):
    def construct(self):
        # Title
        title = Text("Visual Demonstration", font_size=32, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create simple visualization
        self.show_distributions()
        
    def show_distributions(self):
        # Create axes
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 2, 0.5],
            x_length=8,
            y_length=4,
            axis_config={"color": BLUE},
        )
        axes.next_to(self.mobjects[0], DOWN, buff=0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("Duration (s)")
        y_label = axes.get_y_axis_label("Probability")
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Show different conflict scenarios
        scenarios = [
            {"name": "No Conflict", "Sa": 1.0, "Sv": 1.0, "color": GREEN},
            {"name": "Small Conflict", "Sa": 1.0, "Sv": 1.3, "color": YELLOW},
            {"name": "Large Conflict", "Sa": 1.0, "Sv": 1.8, "color": RED},
        ]
        
        for i, scenario in enumerate(scenarios):
            if i > 0:
                self.wait(1)
                # Clear previous curves
                self.play(*[FadeOut(mob) for mob in self.mobjects[4:]])
            
            self.animate_scenario(axes, scenario)
            self.wait(2)
    
    def animate_scenario(self, axes, scenario):
        # Parameters
        Sa = scenario["Sa"]
        Sv = scenario["Sv"]
        sigma = 0.15
        
        # Create simple curves using parametric functions
        def auditory_curve(t):
            x = Sa + 0.5 * np.cos(t)
            y = 0.8 * np.exp(-((Sa + 0.5 * np.cos(t) - Sa) ** 2) / (2 * sigma ** 2))
            return axes.c2p(x, y)
        
        def visual_curve(t):
            x = Sv + 0.5 * np.cos(t)
            y = 0.8 * np.exp(-((Sv + 0.5 * np.cos(t) - Sv) ** 2) / (2 * sigma ** 2))
            return axes.c2p(x, y)
        
        # Create curves
        auditory = ParametricFunction(auditory_curve, t_range=[0, 2*PI], color=TEAL)
        visual = ParametricFunction(visual_curve, t_range=[0, 2*PI], color=GREEN)
        
        # Vertical lines for true values
        line_a = axes.get_vertical_line(axes.c2p(Sa, 0), color=TEAL, stroke_width=3)
        line_v = axes.get_vertical_line(axes.c2p(Sv, 0), color=GREEN, stroke_width=3)
        
        # Scenario label
        scenario_text = Text(scenario["name"], font_size=18, color=scenario["color"])
        scenario_text.to_edge(RIGHT).shift(UP)
        
        # Animate
        self.play(
            Create(auditory),
            Create(visual),
            Create(line_a),
            Create(line_v),
            Write(scenario_text)
        )


class ConceptualDemo(Scene):
    def construct(self):
        # Title
        title = Text("Causal Inference Concept", font_size=32, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show the decision process
        self.show_decision_process()
        
    def show_decision_process(self):
        # Create decision tree
        question = Text("Are the signals from the same source?", font_size=24, color=WHITE)
        question.next_to(self.mobjects[0], DOWN, buff=1)
        self.play(Write(question))
        
        # Two branches
        yes_branch = Text("YES: Fuse the signals", font_size=20, color=GREEN)
        no_branch = Text("NO: Keep them separate", font_size=20, color=RED)
        
        yes_branch.next_to(question, DOWN, buff=0.8).shift(LEFT * 2)
        no_branch.next_to(question, DOWN, buff=0.8).shift(RIGHT * 2)
        
        self.play(Write(yes_branch), Write(no_branch))
        
        # Draw arrows
        arrow_yes = Arrow(question.get_bottom(), yes_branch.get_top(), color=GREEN)
        arrow_no = Arrow(question.get_bottom(), no_branch.get_top(), color=RED)
        
        self.play(Create(arrow_yes), Create(arrow_no))
        self.wait(2)
        
        # Show the probabilistic nature
        prob_text = Text("In reality: Use probability weights", font_size=22, color=YELLOW)
        prob_text.next_to(yes_branch, DOWN, buff=1).shift(RIGHT)
        self.play(Write(prob_text))
        
        # Final equation
        final_eq = Text("Estimate = P(same) × Fused + P(different) × Separate", 
                       font_size=18, color=YELLOW)
        final_eq.next_to(prob_text, DOWN, buff=0.5)
        self.play(Write(final_eq))
        self.wait(3)


class CausalInferenceSimulation(Scene):
    def construct(self):
        # Title
        title = Text("Causal Inference Simulation", font_size=32, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show parameters
        self.show_simulation_with_data()
        
    def show_simulation_with_data(self):
        # Create axes
        axes = Axes(
            x_range=[0, 3, 0.5],
            y_range=[0, 4, 1],
            x_length=10,
            y_length=5,
            axis_config={"color": BLUE},
        )
        axes.next_to(self.mobjects[0], DOWN, buff=0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("Duration (s)")
        y_label = axes.get_y_axis_label("Probability Density")
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Show different scenarios with actual data
        scenarios = [
            {"name": "No Conflict (P_common=0.9)", "S_a": 0.8, "conflict": 0.0, "p_c": 0.9, "color": GREEN},
            {"name": "Small Conflict (P_common=0.7)", "S_a": 0.8, "conflict": 0.4, "p_c": 0.7, "color": YELLOW},
            {"name": "Large Conflict (P_common=0.3)", "S_a": 0.8, "conflict": 0.8, "p_c": 0.3, "color": RED},
        ]
        
        for i, scenario in enumerate(scenarios):
            if i > 0:
                self.wait(2)
                # Clear previous elements but keep axes
                elements_to_remove = [mob for mob in self.mobjects if mob != self.mobjects[0] and mob != axes and mob != x_label and mob != y_label]
                if elements_to_remove:
                    self.play(*[FadeOut(mob) for mob in elements_to_remove])
            
            self.animate_simulation_scenario(axes, scenario)
            self.wait(2)
    
    def animate_simulation_scenario(self, axes, scenario):
        # Parameters
        S_a = scenario["S_a"]
        conflict = scenario["conflict"]
        S_v = S_a + conflict
        sigma_a = 0.2
        sigma_v = 0.2
        p_c = scenario["p_c"]
        
        # Generate simulation data
        n_samples = 1000
        np.random.seed(42)  # For reproducible results
        m_a = np.random.normal(S_a, sigma_a, n_samples)
        m_v = np.random.normal(S_v, sigma_v, n_samples)
        
        # Compute causal inference estimates
        estimates = self.compute_causal_inference(S_a, S_v, m_a, m_v, sigma_a, sigma_v, p_c)
        
        # Create histograms using rectangles
        self.create_histogram(axes, m_a, TEAL, alpha=0.6, label="Auditory", offset=0)
        self.create_histogram(axes, m_v, GREEN, alpha=0.6, label="Visual", offset=0.1)
        self.create_histogram(axes, estimates, scenario["color"], alpha=0.8, label="Estimate", offset=0.2)
        
        # Add vertical lines for true values and mean estimate
        line_a = axes.get_vertical_line(axes.c2p(S_a, 0), color=TEAL, stroke_width=3)
        line_v = axes.get_vertical_line(axes.c2p(S_v, 0), color=GREEN, stroke_width=3)
        line_est = axes.get_vertical_line(axes.c2p(np.mean(estimates), 0), 
                                        color=scenario["color"], stroke_width=3)
        
        # Add smooth curves over histograms
        x_range = np.linspace(0, 3, 100)
        curve_a = self.create_smooth_curve(axes, x_range, norm.pdf(x_range, S_a, sigma_a), TEAL)
        curve_v = self.create_smooth_curve(axes, x_range, norm.pdf(x_range, S_v, sigma_v), GREEN)
        curve_est = self.create_smooth_curve(axes, x_range, norm.pdf(x_range, np.mean(estimates), np.std(estimates)), scenario["color"])
        
        # Scenario info
        scenario_text = Text(scenario["name"], font_size=16, color=scenario["color"])
        scenario_text.to_edge(RIGHT).shift(UP * 2)
        
        # Parameter info
        param_text = Text(f"S_a={S_a:.1f}, Conflict={conflict:.1f}\nP(Common)={p_c:.1f}", 
                         font_size=14, color=WHITE)
        param_text.next_to(scenario_text, DOWN, buff=0.3)
        
        # Posterior probability
        mean_posterior = np.mean(self.compute_posterior_probability(S_a, S_v, m_a, m_v, sigma_a, sigma_v, p_c))
        posterior_text = Text(f"Avg P(Common|data)={mean_posterior:.2f}", 
                            font_size=14, color=YELLOW)
        posterior_text.next_to(param_text, DOWN, buff=0.2)
        
        # Animate everything
        self.play(
            Create(curve_a),
            Create(curve_v),
            Create(curve_est),
            Create(line_a),
            Create(line_v),
            Create(line_est),
            Write(scenario_text),
            Write(param_text),
            Write(posterior_text)
        )
    
    def create_histogram(self, axes, data, color, alpha=0.7, label="", offset=0):
        """Create a histogram using rectangles"""
        hist, bin_edges = np.histogram(data, bins=20, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        
        rectangles = VGroup()
        for i, height in enumerate(hist):
            if height > 0:  # Only create rectangles for non-zero bars
                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                rect_height = height * 0.8  # Scale for visibility
                rect = Rectangle(
                    width=bin_width * 0.8,
                    height=rect_height,
                    fill_color=color,
                    fill_opacity=alpha,
                    stroke_width=1,
                    stroke_color=color
                )
                rect.move_to(axes.c2p(bin_center, rect_height/2))
                rectangles.add(rect)
        
        if len(rectangles) > 0:
            self.play(Create(rectangles), run_time=0.5)
    
    def create_smooth_curve(self, axes, x_vals, y_vals, color):
        """Create a smooth curve from data points"""
        points = [axes.c2p(x, y) for x, y in zip(x_vals, y_vals)]
        curve = VMobject()
        curve.set_points_smoothly(points)
        curve.set_color(color)
        curve.set_stroke(width=3)
        return curve
    
    def compute_causal_inference(self, S_a, S_v, m_a, m_v, sigma_a, sigma_v, p_c):
        """Compute causal inference estimates (same as in notebook)"""
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
    
    def compute_posterior_probability(self, S_a, S_v, m_a, m_v, sigma_a, sigma_v, p_c):
        """Compute posterior probability of common cause"""
        var_sum = sigma_a**2 + sigma_v**2
        likelihood_c1 = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))
        likelihood_c2 = norm.pdf(m_a, S_a, sigma_a) * norm.pdf(m_v, S_v, sigma_v)
        
        posterior_c1 = (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))
        return posterior_c1


class InteractiveStyleDemo(Scene):
    def construct(self):
        # Title
        title = Text("Interactive-Style Parameter Demo", font_size=32, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show how parameters affect the outcome
        self.show_parameter_effects()
        
    def show_parameter_effects(self):
        # Create parameter display
        param_display = VGroup()
        
        # Current parameters
        current_params = Text("Current Parameters:", font_size=20, color=YELLOW)
        current_params.next_to(self.mobjects[0], DOWN, buff=0.5)
        param_display.add(current_params)
        
        # Parameter values
        param_values = [
            "S_a (Auditory duration) = 0.8s",
            "Conflict (Visual - Auditory) = 0.4s", 
            "σ_a (Auditory noise) = 0.2s",
            "σ_v (Visual noise) = 0.2s",
            "P(Common) prior = 0.7"
        ]
        
        for i, param in enumerate(param_values):
            text = Text(param, font_size=16, color=WHITE)
            if i == 0:
                text.next_to(current_params, DOWN, buff=0.3)
            else:
                text.next_to(param_display[-1], DOWN, buff=0.2)
            param_display.add(text)
        
        # Show all parameters
        for param in param_display:
            self.play(Write(param))
            self.wait(0.3)
        
        # Create simple result visualization
        self.wait(1)
        result_text = Text("Result: Moderate fusion (P≈0.6)", font_size=18, color=GREEN)
        result_text.next_to(param_display, DOWN, buff=0.5)
        self.play(Write(result_text))
        self.wait(2)
        
        # Show what happens with different conflicts
        self.show_conflict_comparison()
    
    def show_conflict_comparison(self):
        # Clear previous content except title
        self.play(*[FadeOut(mob) for mob in self.mobjects[1:]])
        
        # Comparison title
        comp_title = Text("Effect of Conflict Level", font_size=24, color=YELLOW)
        comp_title.next_to(self.mobjects[0], DOWN, buff=0.5)
        self.play(Write(comp_title))
        
        # Show three scenarios side by side
        scenarios = [
            {"conflict": 0.0, "fusion": 0.95, "color": GREEN, "label": "No Conflict"},
            {"conflict": 0.4, "fusion": 0.60, "color": YELLOW, "label": "Medium Conflict"},
            {"conflict": 0.8, "fusion": 0.25, "color": RED, "label": "High Conflict"}
        ]
        
        scenario_group = VGroup()
        for i, scenario in enumerate(scenarios):
            # Create scenario box
            scenario_box = VGroup()
            
            # Title
            title = Text(scenario["label"], font_size=16, color=scenario["color"])
            scenario_box.add(title)
            
            # Conflict value
            conflict_text = Text(f"Conflict: {scenario['conflict']:.1f}s", font_size=14, color=WHITE)
            conflict_text.next_to(title, DOWN, buff=0.2)
            scenario_box.add(conflict_text)
            
            # Fusion probability
            fusion_text = Text(f"P(Fusion): {scenario['fusion']:.2f}", font_size=14, color=scenario["color"])
            fusion_text.next_to(conflict_text, DOWN, buff=0.2)
            scenario_box.add(fusion_text)
            
            # Position scenarios
            if i == 0:
                scenario_box.next_to(comp_title, DOWN, buff=0.8).shift(LEFT * 3)
            elif i == 1:
                scenario_box.next_to(comp_title, DOWN, buff=0.8)
            else:
                scenario_box.next_to(comp_title, DOWN, buff=0.8).shift(RIGHT * 3)
            
            scenario_group.add(scenario_box)
        
        # Animate scenarios
        for scenario in scenario_group:
            self.play(Write(scenario))
            self.wait(0.5)
        
        self.wait(2)
        
        # Summary
        summary = Text("Higher conflict → Lower probability of fusion", 
                      font_size=18, color=YELLOW)
        summary.next_to(scenario_group, DOWN, buff=0.8)
        self.play(Write(summary))
        self.wait(3)
