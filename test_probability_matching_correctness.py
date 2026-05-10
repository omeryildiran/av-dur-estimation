#!/usr/bin/env python3
"""Targeted checks for model decision-rule semantics."""

import numpy as np

from monteCarloClass import OmerMonteCarlo


def make_fitter():
    fitter = object.__new__(OmerMonteCarlo)
    fitter.modelName = "probabilityMatchingLogNorm"
    fitter.sharedLambda = True
    fitter.freeP_c = False
    fitter.sharedSigma_v = True
    fitter.data_t_min = 0.05
    fitter.data_t_max = 0.95
    return fitter


def test_causal_averaging_uses_posterior_c1_weight():
    fitter = make_fitter()
    fitter.posterior_C1 = lambda *args: np.array([1.0, 0.0, 0.25])

    m_a = np.array([1.0, 1.0, 1.0])
    m_v = np.array([3.0, 3.0, 3.0])
    estimates = fitter.causalInference_vectorized(
        m_a, m_v, sigma_a=1.0, sigma_v=1.0, p_c=0.5, t_min=0.0, t_max=4.0
    )

    expected = np.array([2.0, 1.0, 1.25])
    np.testing.assert_allclose(estimates, expected)


def test_probability_matching_uses_posterior_c1_probability():
    fitter = make_fitter()
    fitter.posterior_C1 = lambda *args: np.array([1.0, 0.0])

    m_a = np.array([1.0, 1.0])
    m_v = np.array([3.0, 3.0])
    estimates = fitter.probabilityMatching_vectorized(
        m_a, m_v, sigma_a=1.0, sigma_v=1.0, p_c=0.5, t_min=0.0, t_max=4.0
    )

    expected_fused = np.array([2.0, 1.0])
    np.testing.assert_allclose(estimates, expected_fused)


def test_selection_uses_most_probable_causal_structure():
    fitter = make_fitter()
    fitter.posterior_C1 = lambda *args: np.array([0.51, 0.50, 0.49])

    m_a = np.array([1.0, 1.0, 1.0])
    m_v = np.array([3.0, 3.0, 3.0])
    estimates = fitter.selection_vectorized(
        m_a, m_v, sigma_a=1.0, sigma_v=1.0, p_c=0.5, t_min=0.0, t_max=4.0
    )

    expected = np.array([2.0, 1.0, 1.0])
    np.testing.assert_allclose(estimates, expected)


def test_selection_choice_path_uses_log_space_measurements_and_bounds():
    fitter = make_fitter()
    fitter.modelName = "selection"
    fitter.nSimul = 4
    captured = []

    def fake_selection(m_a, m_v, sigma_a, sigma_v, p_c, t_min, t_max):
        captured.append((m_a.copy(), m_v.copy(), t_min, t_max))
        return m_a

    fitter.selection_vectorized = fake_selection
    true_stims = (0.5, 0.6, 0.7, 0.6)
    p_test = fitter.probTestLonger_vectorized_mc(
        true_stims, sigma_av_a=0.0, sigma_av_v=0.0,
        p_c=0.5, lambda_=0.0, t_min=0.1, t_max=1.0
    )

    assert p_test == 1.0
    assert len(captured) == 2
    std_m_a, std_m_v, std_t_min, std_t_max = captured[0]
    test_m_a, test_m_v, test_t_min, test_t_max = captured[1]

    np.testing.assert_allclose(std_m_a, np.log(0.5))
    np.testing.assert_allclose(std_m_v, np.log(0.7))
    np.testing.assert_allclose(test_m_a, np.log(0.6))
    np.testing.assert_allclose(test_m_v, np.log(0.6))
    np.testing.assert_allclose([std_t_min, test_t_min], np.log(0.1))
    np.testing.assert_allclose([std_t_max, test_t_max], np.log(1.0))


def test_forced_fusion_choice_path_uses_log_space_measurements():
    fitter = make_fitter()
    fitter.modelName = "fusionOnlyLogNorm"
    fitter.nSimul = 4
    captured = []

    def fake_fusion(m_a, m_v, sigma_a, sigma_v):
        captured.append((m_a.copy(), m_v.copy()))
        return m_a

    fitter.fusionAV_vectorized = fake_fusion
    true_stims = (0.5, 0.6, 0.7, 0.6)
    p_test = fitter.probTestLonger_vectorized_mc(
        true_stims, sigma_av_a=0.0, sigma_av_v=0.0,
        p_c=1.0, lambda_=0.0, t_min=0.1, t_max=1.0
    )

    assert p_test == 1.0
    assert len(captured) == 2
    std_m_a, std_m_v = captured[0]
    test_m_a, test_m_v = captured[1]

    np.testing.assert_allclose(std_m_a, np.log(0.5))
    np.testing.assert_allclose(std_m_v, np.log(0.7))
    np.testing.assert_allclose(test_m_a, np.log(0.6))
    np.testing.assert_allclose(test_m_v, np.log(0.6))


def test_switching_free_choice_path_uses_log_space_measurements():
    fitter = make_fitter()
    fitter.modelName = "switchingFree"
    fitter.nSimul = 4
    captured = []

    def fake_switching(m_a, m_v, p_switch):
        captured.append((m_a.copy(), m_v.copy(), p_switch))
        return m_a

    fitter.switching_free_vectorized = fake_switching
    true_stims = (0.5, 0.6, 0.7, 0.6)
    p_test = fitter.probTestLonger_vectorized_mc(
        true_stims, sigma_av_a=0.0, sigma_av_v=0.0,
        p_c=0.25, lambda_=0.0, t_min=0.1, t_max=1.0
    )

    assert p_test == 1.0
    assert len(captured) == 2
    std_m_a, std_m_v, std_p_switch = captured[0]
    test_m_a, test_m_v, test_p_switch = captured[1]

    np.testing.assert_allclose(std_m_a, np.log(0.5))
    np.testing.assert_allclose(std_m_v, np.log(0.7))
    np.testing.assert_allclose(test_m_a, np.log(0.6))
    np.testing.assert_allclose(test_m_v, np.log(0.6))
    assert std_p_switch == 0.25
    assert test_p_switch == 0.25


def test_switching_free_uses_visual_with_p_switch_probability():
    fitter = make_fitter()

    m_a = np.array([1.0, 1.0])
    m_v = np.array([3.0, 3.0])

    auditory_only = fitter.switching_free_vectorized(m_a, m_v, p_switch=0.0)
    visual_only = fitter.switching_free_vectorized(m_a, m_v, p_switch=1.0)

    np.testing.assert_allclose(auditory_only, m_a)
    np.testing.assert_allclose(visual_only, m_v)


def test_causal_model_parameter_extraction_shared_lambda():
    fitter = make_fitter()
    fitter.modelName = "lognorm"

    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    high_snr = fitter.getParamsCausal(params, SNR=0.1, conflict=0.0)
    low_snr = fitter.getParamsCausal(params, SNR=1.2, conflict=0.0)

    assert high_snr == (0.1, 0.2, 0.3, 0.4, 0.05, 0.95)
    assert low_snr == (0.1, 0.5, 0.3, 0.4, 0.05, 0.95)


def test_switching_free_parameter_extraction_shared_lambda():
    fitter = make_fitter()
    fitter.modelName = "switchingFree"

    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    high_snr = fitter.getParamsCausal(params, SNR=0.1, conflict=0.0)
    low_snr = fitter.getParamsCausal(params, SNR=1.2, conflict=0.0)

    assert high_snr == (0.1, 0.2, 0.3, 0.4, 0.05, 0.95)
    assert low_snr == (0.1, 0.5, 0.3, 0.6, 0.05, 0.95)


if __name__ == "__main__":
    test_causal_averaging_uses_posterior_c1_weight()
    test_probability_matching_uses_posterior_c1_probability()
    test_selection_uses_most_probable_causal_structure()
    test_selection_choice_path_uses_log_space_measurements_and_bounds()
    test_forced_fusion_choice_path_uses_log_space_measurements()
    test_switching_free_choice_path_uses_log_space_measurements()
    test_switching_free_uses_visual_with_p_switch_probability()
    test_causal_model_parameter_extraction_shared_lambda()
    test_switching_free_parameter_extraction_shared_lambda()
    print("model decision-rule correctness checks passed")
