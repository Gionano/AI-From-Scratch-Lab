# Cross-Language Ports

This folder contains small standalone ports of the numeric neural-network demo in:

- Julia
- MATLAB
- C++

Each port includes:

- Synthetic 2D circle dataset generation
- Forward pass with `y = f(Wx + b)`
- Hidden activation choices: `relu`, `sigmoid`
- Loss choices: `binary_cross_entropy`, `mean_squared_error`
- Gradient descent training

These ports were added from the Python reference implementation in this repository.
They were written to be easy to read and modify, especially for learning.

## Files

- `julia/own_ai_model.jl`
- `matlab/own_ai_model_demo.m`
- `cpp/own_ai_model.cpp`

## Notes

- The current environment did not have Julia, MATLAB, or a C++ compiler available, so these files were not executed here.
- The Python implementation remains the verified reference version.

## Suggested Run Commands

Julia:

```bash
julia ports/julia/own_ai_model.jl
```

MATLAB:

```matlab
run('ports/matlab/own_ai_model_demo.m')
```

C++:

```bash
g++ -std=c++17 -O2 ports/cpp/own_ai_model.cpp -o own_ai_model
./own_ai_model
```
