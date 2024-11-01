# Active Set Method for Constrained Least Squares

A C++ implementation of the Active Set Method for solving constrained least squares problems, based on:

- **Härkegård, O.** "Efficient active set algorithms for solving constrained least squares problems in aircraft control allocation," *Proceedings of the 41st IEEE Conference on Decision and Control, 2002*.
- **Smeur, E., Höppener, D., De Wagter, C.** "Prioritized control allocation for quadrotors subject to saturation," *International Micro Air Vehicle Conference and Flight Competition*, 2017.

## Features

- **Active Set Algorithm**: Efficiently handles bounds and constraints.
- **High Precision**: Uses double-precision calculations for stability.
- **Template-Based Design**: Flexible to accommodate different problem sizes.
- **Eigen Library**: Utilizes Eigen for efficient matrix operations.

## Requirements

- C++11 or later
- Eigen library
- CMake

## Quick Start

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/active-set-method.git
    cd active-set-method
    ```

2. **Build**:
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

3. **Run the Example**:
    ```bash
    ./ActiveSetSolverExample
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

⭐️ If you find this project useful, please consider giving it a star!
