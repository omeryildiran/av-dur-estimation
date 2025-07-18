"""
Manim Animation for Causal Inference in Auditory-Visual Duration Estimation

This animation demonstrates the mathematical formulation and simulation of 
causal inference in multisensory duration perception.

To run this animation:
1. Install dependencies: pip install -r requirements.txt
2. Run: manim -pql causal_inference_animation.py CausalInferenceDemo
"""

from manim import *
import numpy as np
from scipy.stats import norm, gaussian_kde

class CausalInferenceAnimation(Scene):
    def construct(self):
        # Title
        title = Text("Causal Inference in Auditory-Visual Duration Estimation", 
                    font_size=36, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(2)
        
        # Clear and show mathematical formulation
        self.play(FadeOut(title))
        self.show_mathematical_formulation()
        
        # Clear and show the simulation
        self.clear()
        self.show_simulation_process()
        
        # Clear and show parameter effects
        self.clear()
        self.show_parameter_effects()

    def show_mathematical_formulation(self):
        """Show the mathematical equations for causal inference"""
        
        # Section title
        math_title = Text("Mathematical Formulation", font_size=32, color=YELLOW)
        math_title.to_edge(UP)
        self.play(Write(math_title))
        self.wait(1)
        
        # Step 1: Measurements
        measurements_title = Text("Step 1: Noisy Measurements", font_size=24, color=GREEN)
        measurements_title.next_to(math_title, DOWN, buff=0.5)
        self.play(Write(measurements_title))
        
        # Measurement equations
        eq1 = MathTex(r"m_a \sim N(S_a, \sigma_{av,a}^2)")
        eq2 = MathTex(r"m_v \sim N(S_v, \sigma_{av,v}^2)")
        eq1.next_to(measurements_title, DOWN, buff=0.3)
        eq2.next_to(eq1, DOWN, buff=0.2)
        
        self.play(Write(eq1))
        self.play(Write(eq2))
        self.wait(2)
        
        # Step 2: Fusion under common cause
        fusion_title = Text("Step 2: Fusion (C=1)", font_size=24, color=GREEN)
        fusion_title.next_to(eq2, DOWN, buff=0.5)
        self.play(Write(fusion_title))
        
        # Fusion equations
        eq3 = MathTex(r"\hat{S}_{fused} = w_a m_a + w_v m_v")
        eq4 = MathTex(r"w_a = \frac{J_a}{J_a + J_v}, \quad J_a = \frac{1}{\sigma_{av,a}^2}")
        eq3.next_to(fusion_title, DOWN, buff=0.3)
        eq4.next_to(eq3, DOWN, buff=0.2)
        
        self.play(Write(eq3))
        self.play(Write(eq4))
        self.wait(2)
        
        # Step 3: Causal inference
        causal_title = Text("Step 3: Causal Inference", font_size=24, color=GREEN)
        causal_title.next_to(eq4, DOWN, buff=0.5)
        self.play(Write(causal_title))
        
        # Causal inference equations
        eq5 = MathTex(r"P(C=1|m_a, m_v) = \frac{L(C=1) \cdot P(C=1)}{L(C=1) \cdot P(C=1) + L(C=2) \cdot P(C=2)}")
        eq6 = MathTex(r"\hat{S}_{final} = P(C=1) \cdot \hat{S}_{fused} + P(C=2) \cdot m_a")
        eq5.next_to(causal_title, DOWN, buff=0.3)
        eq6.next_to(eq5, DOWN, buff=0.2)
        eq5.scale(0.8)
        
        self.play(Write(eq5))
        self.play(Write(eq6))
        self.wait(3)

    def show_simulation_process(self):
        """Show the simulation process with animated plots"""
        
        # Title
        sim_title = Text("Simulation Process", font_size=32, color=YELLOW)
        sim_title.to_edge(UP)
        self.play(Write(sim_title))
        
        # Parameters
        params_text = Text("Parameters:", font_size=20, color=WHITE)
        params_text.next_to(sim_title, DOWN, buff=0.5).to_edge(LEFT)
        self.play(Write(params_text))
        
        # Create axes for the plot
        axes = Axes(
            x_range=[0, 3, 0.5],
            y_range=[0, 4, 1],
            x_length=8,
            y_length=4,
            axis_config={"color": BLUE},
            tips=False
        )
        axes.next_to(params_text, DOWN, buff=0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("Duration", direction=DOWN)
        y_label = axes.get_y_axis_label("Probability Density", direction=LEFT)
        
        self.play(Create(axes))
        self.play(Write(x_label), Write(y_label))
        
        # Simulate different scenarios
        scenarios = [
            {"S_a": 0.5, "c": 0.0, "sigma_a": 0.2, "sigma_v": 0.2, "p_c": 0.8, "label": "No Conflict"},
            {"S_a": 0.5, "c": 0.5, "sigma_a": 0.2, "sigma_v": 0.2, "p_c": 0.8, "label": "Small Conflict"},
            {"S_a": 0.5, "c": 1.0, "sigma_a": 0.2, "sigma_v": 0.2, "p_c": 0.8, "label": "Large Conflict"},
            {"S_a": 0.5, "c": 0.5, "sigma_a": 0.2, "sigma_v": 0.2, "p_c": 0.2, "label": "Low Prior"},
        ]
        
        for i, scenario in enumerate(scenarios):
            self.animate_scenario(axes, scenario, i)
            self.wait(2)

    def animate_scenario(self, axes, scenario, scenario_num):
        """Animate a single scenario"""
        
        # Extract parameters
        S_a = scenario["S_a"]
        c = scenario["c"]
        sigma_a = scenario["sigma_a"]
        sigma_v = scenario["sigma_v"]
        p_c = scenario["p_c"]
        label = scenario["label"]
        
        # Generate data
        S_v = S_a + c
        n_samples = 1000
        
        # Generate measurements
        m_a = np.random.normal(S_a, sigma_a, n_samples)
        m_v = np.random.normal(S_v, sigma_v, n_samples)
        
        # Compute causal inference
        estimates = self.causal_inference_vectorized(S_a, S_v, m_a, m_v, sigma_a, sigma_v, p_c)
        
        # Create KDE for smooth curves
        x_range = np.linspace(0, 3, 200)
        kde_a = gaussian_kde(m_a)
        kde_v = gaussian_kde(m_v)
        kde_est = gaussian_kde(estimates)
        
        # Create curves
        curve_a = axes.plot(lambda x: kde_a(x)[0] if hasattr(kde_a(x), '__len__') else kde_a(x), 
                           x_range=[0, 3], color=TEAL)
        curve_v = axes.plot(lambda x: kde_v(x)[0] if hasattr(kde_v(x), '__len__') else kde_v(x), 
                           x_range=[0, 3], color=GREEN)
        curve_est = axes.plot(lambda x: kde_est(x)[0] if hasattr(kde_est(x), '__len__') else kde_est(x), 
                             x_range=[0, 3], color=RED)
        
        # Vertical lines for true values and estimates
        line_Sa = axes.get_vertical_line(axes.c2p(S_a, 0), color=TEAL, stroke_width=2)
        line_Sv = axes.get_vertical_line(axes.c2p(S_v, 0), color=GREEN, stroke_width=2)
        line_est = axes.get_vertical_line(axes.c2p(np.mean(estimates), 0), color=RED, stroke_width=2)
        
        # Scenario label
        scenario_text = Text(f"Scenario {scenario_num + 1}: {label}", font_size=18, color=YELLOW)
        scenario_text.to_edge(UP, buff=1.5)
        
        # Parameter display
        param_text = Text(f"S_a={S_a}, c={c}, P(C=1)={p_c}", font_size=14, color=WHITE)
        param_text.next_to(scenario_text, DOWN, buff=0.3)
        
        # Animate
        if scenario_num == 0:
            self.play(
                Write(scenario_text),
                Write(param_text),
                Create(curve_a),
                Create(curve_v),
                Create(curve_est),
                Create(line_Sa),
                Create(line_Sv),
                Create(line_est)
            )
        else:
            # Clear previous scenario
            self.clear()
            # Recreate axes
            axes = Axes(
                x_range=[0, 3, 0.5],
                y_range=[0, 4, 1],
                x_length=8,
                y_length=4,
                axis_config={"color": BLUE},
                tips=False
            )
            axes.move_to(ORIGIN)
            x_label = axes.get_x_axis_label("Duration", direction=DOWN)
            y_label = axes.get_y_axis_label("Probability Density", direction=LEFT)
            
            self.add(axes, x_label, y_label)
            
            self.play(
                Write(scenario_text),
                Write(param_text),
                Create(curve_a),
                Create(curve_v),
                Create(curve_est),
                Create(line_Sa),
                Create(line_Sv),
                Create(line_est)
            )

    def show_parameter_effects(self):
        """Show how different parameters affect the estimation"""
        
        # Title
        effects_title = Text("Parameter Effects on Causal Inference", font_size=28, color=YELLOW)
        effects_title.to_edge(UP)
        self.play(Write(effects_title))
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Effect of conflict magnitude
        conflicts = [0.0, 0.5, 1.0, 1.5]
        self.plot_conflict_effects(axes[0, 0], conflicts)
        
        # Effect of prior probability
        priors = [0.2, 0.5, 0.8, 0.95]
        self.plot_prior_effects(axes[0, 1], priors)
        
        # Effect of noise levels
        noise_levels = [0.1, 0.2, 0.4, 0.6]
        self.plot_noise_effects(axes[1, 0], noise_levels)
        
        # Effect of asymmetric noise
        noise_ratios = [(0.1, 0.3), (0.2, 0.2), (0.3, 0.1)]
        self.plot_asymmetric_noise_effects(axes[1, 1], noise_ratios)
        
        plt.tight_layout()
        
        # Convert matplotlib figure to Manim
        # (This would require additional setup to properly convert matplotlib to Manim)
        # For now, show text descriptions
        
        effect_descriptions = [
            "Conflict Effect: Larger conflicts → Lower fusion probability",
            "Prior Effect: Higher P(C=1) → More fusion",
            "Noise Effect: Higher noise → More uncertainty",
            "Asymmetric Noise: Affects reliability weighting"
        ]
        
        for i, desc in enumerate(effect_descriptions):
            text = Text(desc, font_size=18, color=WHITE)
            text.next_to(effects_title, DOWN, buff=0.5 + i * 0.7)
            self.play(Write(text))
            self.wait(1)

    def causal_inference_vectorized(self, S_a, S_v, m_a, m_v, sigma_av_a, sigma_av_v, p_c):
        """Vectorized causal inference computation"""
        # Fusion
        J_a = 1 / sigma_av_a**2
        J_v = 1 / sigma_av_v**2
        w_a = J_a / (J_a + J_v)
        w_v = 1 - w_a
        fused_S_av = w_a * m_a + w_v * m_v
        
        # Likelihoods
        var_sum = sigma_av_a**2 + sigma_av_v**2
        likelihood_c1 = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))
        likelihood_c2 = norm.pdf(m_a, loc=S_a, scale=sigma_av_a) * norm.pdf(m_v, loc=S_v, scale=sigma_av_v)
        
        # Posterior
        posterior_c1 = (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))
        
        # Final estimate
        final_estimate = posterior_c1 * fused_S_av + (1 - posterior_c1) * m_a
        return final_estimate

    def plot_conflict_effects(self, ax, conflicts):
        """Plot effect of conflict on causal inference"""
        for conflict in conflicts:
            # Simulation code here
            pass
    
    def plot_prior_effects(self, ax, priors):
        """Plot effect of prior probability on causal inference"""
        for prior in priors:
            # Simulation code here
            pass
    
    def plot_noise_effects(self, ax, noise_levels):
        """Plot effect of noise on causal inference"""
        for noise in noise_levels:
            # Simulation code here
            pass
    
    def plot_asymmetric_noise_effects(self, ax, noise_ratios):
        """Plot effect of asymmetric noise on causal inference"""
        for ratio in noise_ratios:
            # Simulation code here
            pass


class CausalInferenceDemo(Scene):
    """A simpler demo scene focusing on the core concept"""
    def construct(self):
        # Title
        title = Text("Causal Inference Demo", font_size=36, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # Show the basic idea
        concept_text = Text("Key Idea: Should auditory and visual signals be fused?", 
                           font_size=24, color=WHITE)
        concept_text.next_to(title, DOWN, buff=0.5)
        self.play(Write(concept_text))
        
        # Show two scenarios
        scenario1 = Text("Scenario 1: Common source → Fuse signals", 
                        font_size=20, color=GREEN)
        scenario2 = Text("Scenario 2: Different sources → Keep separate", 
                        font_size=20, color=RED)
        
        scenario1.next_to(concept_text, DOWN, buff=0.5)
        scenario2.next_to(scenario1, DOWN, buff=0.3)
        
        self.play(Write(scenario1))
        self.play(Write(scenario2))
        self.wait(2)
        
        # Show mathematical solution
        solution_text = Text("Solution: Bayesian causal inference", 
                           font_size=24, color=YELLOW)
        solution_text.next_to(scenario2, DOWN, buff=0.5)
        self.play(Write(solution_text))
        
        # Key equation
        key_eq = MathTex(r"\hat{S} = P(C=1) \cdot \hat{S}_{fused} + P(C=2) \cdot \hat{S}_{separate}")
        key_eq.next_to(solution_text, DOWN, buff=0.5)
        self.play(Write(key_eq))
        self.wait(3)


# Additional utility functions for the animation
def create_distribution_curve(axes, data, color, x_range=(0, 3)):
    """Create a smooth curve from data using KDE"""
    kde = gaussian_kde(data)
    x_vals = np.linspace(x_range[0], x_range[1], 200)
    y_vals = kde(x_vals)
    
    # Create points for the curve
    points = [axes.c2p(x, y) for x, y in zip(x_vals, y_vals)]
    curve = VMobject()
    curve.set_points_as_corners(points)
    curve.set_color(color)
    curve.set_stroke(width=2)
    return curve


if __name__ == "__main__":
    # To render the animation, run:
    # manim -pql causal_inference_animation.py CausalInferenceAnimation
    # or
    # manim -pql causal_inference_animation.py CausalInferenceDemo
    pass
