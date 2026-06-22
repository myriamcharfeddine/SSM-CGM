# Factual-Future Gain Diagnostic

Forecast-only is the deployable pathway. Factual-future is a retrospective diagnostic only.
Future glucose labels were never used as decoder inputs; the factual-future path reveals only non-target future scenario variables with observed masks.

Scenario channel status: **alive**.
Scored anchors: 4534 across 13 streams.
Overall mean G_ff: 0.255. Positive-gain anchors: 50.9%.

Positive G_ff event groups: all, meal_proxy_event, activity_event, sleep_transition_event, stress_event, large_glucose_change_event, hypo_risk_event, hyper_risk_event
Future variables with mask coverage: activity_stage_sedentary, activity_steps_per_min, heart_rate_mean, predmeal_flag, sleep_stage_awake, sleep_stage_deep, sleep_stage_light, sleep_stage_rem, stress_level_mean
predmeal_flag future mask coverage: 0.199

Counterfactual scenario editing is justified for supported proxy scenarios.

AI-READI lacks timed insulin dose, insulin-on-board, and carbohydrate quantity logs. med_insulin is static metadata only, not an editable action.
Proxy effects are model-implied scenario effects, not validated causal effects unless support and natural-experiment validation pass.

## Event Summary

| event | anchors | G_ff mean | positive gain % | MAE forecast-only | MAE factual-future |
|---|---:|---:|---:|---:|---:|
| all | 4534 | 0.255 | 50.9 | 9.072 | 9.053 |
| meal_proxy_event | 2433 | 0.422 | 50.8 | 11.237 | 11.207 |
| activity_event | 2036 | 0.457 | 49.7 | 11.057 | 11.032 |
| sleep_transition_event | 4506 | 0.218 | 50.9 | 9.056 | 9.039 |
| stress_event | 2085 | 0.517 | 50.2 | 11.366 | 11.334 |
| large_glucose_change_event | 1655 | 0.646 | 51.6 | 14.985 | 14.943 |
| hypo_risk_event | 28 | 4.422 | 75.0 | 15.706 | 15.420 |
| hyper_risk_event | 306 | 1.678 | 68.3 | 20.596 | 20.469 |

## Future Mask Coverage

| variable | mask nonzero fraction | anchor availability |
|---|---:|---:|
| activity_stage_sedentary | 1.000 | 1.000 |
| activity_steps_per_min | 1.000 | 1.000 |
| heart_rate_mean | 1.000 | 1.000 |
| predmeal_flag | 0.199 | 0.415 |
| sleep_stage_awake | 0.355 | 0.393 |
| sleep_stage_deep | 0.355 | 0.393 |
| sleep_stage_light | 0.355 | 0.393 |
| sleep_stage_rem | 0.355 | 0.393 |
| stress_level_mean | 0.908 | 0.997 |
