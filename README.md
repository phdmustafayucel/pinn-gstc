data/           - Stores boundary, initial, and measurement points
  • boundary_conditions.csv
  • initial_conditions.csv
  • collocation_points.npy

models/         - Neural network architectures that approximate u(x,t)
  • pinn.py
  • siren_pinn.py (optional advanced model)

physics/        - PDE equations written as residual functions
  • heat_equation.py
  • burgers_equation.py
  • navier_stokes.py (if needed)

losses/         - Loss functions combining physics + data constraints
  • pinn_loss.py
  • adaptive_weights.py (optional)

trainers/       - Training loop and optimization logic
  • trainer.py
  • callbacks.py (logging, early stopping)

configs/        - Experiment and hyperparameter settings
  • default.yaml
  • burgers_experiment.yaml

utils/          - Helper and reusable utilities
  • sampling.py (collocation point generation)
  • plotting.py (solution visualization)
  • metrics.py (L2 error, relative error)

experiments/    - Saved outputs from each training run
  • run_001/
  • run_002/
  • best_models/

notebooks/      - Interactive exploration and debugging
  • sandbox.ipynb
  • visualize_results.ipynb

main.py         - Script that loads config and starts training
  • builds model
  • loads data
  • launches trainer

requirements.txt- Python dependencies
  • torch
  • numpy
  • matplotlib
  • pyyaml

README.md       - Project explanation and usage instructions
  • project description
  • how to run training
  • example results
