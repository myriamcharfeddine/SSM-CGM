#!/usr/bin/env python3
"""GPU entry point with the accepted factual replay numerical tolerance."""

import run_per_variable_static_conditioning_test as test_application


GPU_FACTUAL_REPLAY_ABS_TOLERANCE = 5e-4


if __name__ == "__main__":
    test_application.QC_REPRESENTATION_ABS_TOLERANCE = (
        GPU_FACTUAL_REPLAY_ABS_TOLERANCE
    )
    test_application.main()
