# Practice 1 - Coding a Simple 1D Wave (Dam-Break) Solver

This is a classic application of the 1D Shallow Water Equations.
*   **The Scenario (Dam-Break Problem):**
    Imagine a long channel with a wall (dam) in the middle. On one side, the water is deep; on the other, it's shallow (or dry). At time t=0, the dam instantly vanishes. What happens?

*   **What you'd observe:**
    *   A "wave" of water (called a bore or a shock wave) will rush into the shallow/dry region.
    *   The water level behind this wave will be lower than the initial deep water but higher than the initial shallow water.
    *   A "rarefaction wave" (a smooth decrease in water level) will travel back into the initially deep region.

*   **How to code it (very high level):**
    1.  **Discretize:** Divide your 1D channel into a series of cells.
    2.  **Initialize:** Set the initial height $h$ and velocity $u$ (usually 0) in each cell based on the dam problem (e.g., $h_{\text{left}}$ for cells left of the dam, $h_{\text{right}}$ for cells right of the dam).
    3. **Boundary Conditions:** Apply boundary conditions as walls on left and right side affecting first and last cell.
    4.  **Time Loop:**
        *   For each cell, use the discretized (finite difference) versions of the SWEs to calculate how $h$ and $u$ will change over a small time step $\Delta t$.
        *   Update the $h$ and $u$ values in all cells.
        *   Repeat for many time steps.
    5.  **Visualize:** Plot $h$ (and maybe $u$) along the channel at different times to see the wave propagate.

*   **Why is this exercise valuable?**
    *   It gives you a concrete, hands-on experience with solving fluid dynamic equations, even simplified ones.
    *   You see how initial conditions evolve into dynamic behavior.
    *   It's a stepping stone to more complex solvers.

**Outcome Check for this Part:**
You should now:
*   Understand the core assumption of shallow water flow (horizontal scale >> vertical depth).
*   Know that this leads to simplified equations for height and horizontal velocity.
*   Appreciate that these simpler equations can model important real-world phenomena like tsunamis or dam breaks.
*   Have a conceptual idea of how one might numerically solve these equations for a dam-break problem.

---