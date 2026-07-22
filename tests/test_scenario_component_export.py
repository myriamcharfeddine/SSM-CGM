from dataclasses import asdict

import torch

from ssmcgm.data.aireadi import AireadiFeatureSpec, AireadiPreprocessor
from ssmcgm.models.aireadi_stream import AireadiStreamModel, AireadiStreamModelConfig
from ssmcgm.stream.decoder import ScenarioHorizonDecoder


def _decoder(decompose=True):
    torch.manual_seed(7)
    decoder = ScenarioHorizonDecoder(
        d_model=8,
        e_s_dim=8,
        n_time_features=4,
        n_scenario=3,
        horizon=4,
        output_size=3,
        hidden_size=12,
        dropout=0.0,
        scenario_decompose=decompose,
    ).eval()
    if decompose:
        with torch.no_grad():
            decoder.effect_heads[0].bias.fill_(0.75)
    return decoder


def _inputs(mask_value=1.0):
    torch.manual_seed(11)
    return (
        torch.randn(2, 8),
        torch.randn(2, 8),
        torch.randn(2, 4, 4),
        torch.randn(2, 4, 3),
        torch.full((2, 4, 3), mask_value),
    )


def _model():
    spec = AireadiFeatureSpec(
        dynamic_reals=["x"],
        time_reals=["clock"],
        static_reals=["age"],
        static_categoricals=[],
        scenario_reals=["activity"],
        scenario_groups={"activity": ["activity"]},
        subgroup_columns=[],
        horizon_steps=4,
        bin_minutes=5,
    )
    pre = AireadiPreprocessor(
        dynamic_reals=spec.dynamic_reals,
        time_reals=spec.time_reals,
        static_reals=spec.static_reals,
        static_categoricals=spec.static_categoricals,
        scenario_reals=spec.scenario_reals,
        continuous_stats={
            "x": {"median": 0.0, "mean": 0.0, "std": 1.0},
            "clock": {"median": 0.0, "mean": 0.0, "std": 1.0},
            "age": {"median": 0.0, "mean": 0.0, "std": 1.0},
            "activity": {"median": 0.0, "mean": 0.0, "std": 1.0},
        },
        static_category_maps={},
    )
    cfg = AireadiStreamModelConfig(
        hidden_size=8,
        mamba_depth=1,
        d_state=4,
        headdim=4,
        horizon_emb_dim=4,
        decoder_hidden_size=12,
        dropout=0.0,
        scenario_decompose=True,
    )
    model = AireadiStreamModel(spec, pre, cfg).eval()
    with torch.no_grad():
        model.decoder.effect_heads[0].bias.fill_(0.75)
    return model


def test_decoder_component_export_is_exact_and_default_is_unchanged():
    decoder = _decoder()
    inputs = _inputs()
    ordinary = decoder(*inputs)
    components = decoder(*inputs, return_components=True)
    assert torch.equal(ordinary, components["final"])
    assert torch.max(torch.abs(
        components["final"]
        - (components["base"] + components["scenario_effect"])
    )).item() < 1e-6
    assert components["scenario_availability"].shape == (2, 4, 1)
    assert components["base_latent"].shape == (2, 4, 12)
    assert components["scenario_latent"].shape == (2, 4, 12)


def test_decoder_component_export_supports_non_decomposed_models():
    decoder = _decoder(decompose=False)
    components = decoder(*_inputs(), return_components=True)
    assert torch.count_nonzero(components["scenario_effect"]).item() == 0
    assert torch.equal(components["final"], components["base"])


def test_model_hard_gate_is_opt_in_and_backward_compatible():
    model = _model()
    static = model.encode_static(torch.empty(1, 0), torch.zeros(1, 1))
    h_t = torch.randn(1, 8)
    time_features = torch.randn(1, 4, 1)
    scenario_values = torch.zeros(1, 4, 1)
    zero_mask = torch.zeros_like(scenario_values)

    ordinary = model.decode_horizon(
        h_t, static, time_features, scenario_values, zero_mask
    )
    components = model.decode_horizon(
        h_t, static, time_features, scenario_values, zero_mask,
        return_components=True,
    )
    assert torch.equal(ordinary, components["final"])
    assert torch.max(torch.abs(components["scenario_effect"])).item() > 0.1

    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    model.config.hard_gate_scenario_effect = True
    gated = model.decode_horizon(
        h_t, static, time_features, scenario_values, zero_mask,
        return_components=True,
    )
    assert torch.max(torch.abs(gated["final"] - gated["base"])).item() < 1e-7
    assert list(state_before) == list(model.state_dict())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state_before.items())

    real_mask = torch.ones_like(scenario_values)
    model.config.hard_gate_scenario_effect = False
    ungated_real = model.decode_horizon(
        h_t, static, time_features, scenario_values, real_mask
    )
    model.config.hard_gate_scenario_effect = True
    gated_real = model.decode_horizon(
        h_t, static, time_features, scenario_values, real_mask
    )
    assert torch.max(torch.abs(ungated_real - gated_real)).item() < 1e-7
    assert torch.max(torch.abs(gated_real - gated["final"])).item() > 1e-7


def test_existing_checkpoints_load_strictly_without_gate_activation():
    from pathlib import Path

    from scripts.evaluate_stream_aireadi import load_model_from_checkpoint

    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "outputs/exercise_detector_model/checkpoints/best_model_checkpoint.pt",
        root / "outputs/aireadi_stream_mamba_stateful_5epoch/checkpoints/best_model_checkpoint.pt",
    ]
    for path in paths:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu")
        model, _, _, loaded = load_model_from_checkpoint(path, "cpu")
        assert model.config.hard_gate_scenario_effect is False
        assert set(model.state_dict()) == set(checkpoint["model_state_dict"])
        assert set(loaded["model_state_dict"]) == set(checkpoint["model_state_dict"])


def test_old_model_config_metadata_loads_with_gate_disabled():
    old_metadata = asdict(AireadiStreamModelConfig())
    old_metadata.pop("hard_gate_scenario_effect")
    loaded = AireadiStreamModelConfig(**old_metadata)
    assert loaded.hard_gate_scenario_effect is False
