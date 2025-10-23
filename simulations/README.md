# Simulation Workflow

The simulation codes are written in **MATLAB**. The workflow is organized into sequential files named **`mainstep1` – `mainstep10`**.  

## File Descriptions

- **`mainprepare.m`**  Generates the synthetic data $\mathbf{Z}$ and $\mathbf{U}$.

- **`mainstep1.m`**  Performs regression with either fixed $\lambda$ or cross-validation.


## Usage

Run the files in the following order:

This will generate synthetic data, run the regressions, compute debiased estimators, and produce the final simulation results. The corresponding .sh file helps us to submit the job to the High Performance Computing (HPC) platform.

