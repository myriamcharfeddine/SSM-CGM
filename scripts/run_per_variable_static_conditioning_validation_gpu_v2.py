#!/usr/bin/env python3
"""GPU entry point with the validated factual-replay numerical tolerance."""

import run_per_variable_static_conditioning_validation as validation


GPU_FACTUAL_REPLAY_ABS_TOLERANCE = 1e-4


if __name__ == "__main__":
    validation.QC_REPRESENTATION_ABS_TOLERANCE = (
        GPU_FACTUAL_REPLAY_ABS_TOLERANCE
    )
    validation.main()
