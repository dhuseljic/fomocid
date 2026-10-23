"""Launch the desktop simulator for one experiment at a time.

From the repository root, run:
    PYTHONPATH=src python tutorials/simulate_single_experiment_gui.py
"""
from scattering_calculator.gui.single_experiment import main


if __name__ == '__main__':
    main()
