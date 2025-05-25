# Practice 2 - Your First SPH Simulation & Data Generation

**Goal:** The primary goal here is not to become an SPH expert, but to:
1.  Run a pre-existing 2D SPH simulation.
2.  Understand its key input parameters and how they affect behavior.
3.  **Extract particle data (positions, velocities, accelerations over time) that will serve as your initial training dataset for the GNN in Chapter 2.**

*   **1. Choosing a Simulator (as per curriculum):**
    *   **SPlisHSPlasH (with Python bindings):** A powerful 3D SPH library. Might be overkill for initial 2D, but if Python bindings are good and 2D examples exist, it's an option. Focus on finding simple 2D setups.
    *   **PySPH:** A Python framework for SPH. This is generally an excellent choice as it's well-suited for scientific SPH and Python integrates well with ML tools. It has good documentation and examples.
    *   **Other Open-Source Options:** A simpler, educational 2D SPH code in Python (many on GitHub) can also be very effective for learning the basics and data extraction. The key is ease of use and data output.

*   **2. Setting Up and Running Basic Scenarios:**
    *   Install your chosen tool. (For PySPH: `pip install pysph`, may need Cython, NumPy).
    *   Find and run simple 2D examples. Classic SPH test cases:
        *   **2D Dam Break:** A column of water collapses under gravity.
        *   **Droplet Splash:** A droplet falls into a still body of water or onto a surface.
        *   **Generating Waves:** (If available) A moving boundary creating waves.

*   **3. Experimenting and Observing – Focus On:**
    *   **Input Parameters (How to change them in the code/config):**
        *   **Particle Count / Initial Spacing:** How does increasing particle number affect visual detail and simulation speed?
        *   **Stiffness ($k$ in Tait's EOS) / Target Sound Speed:** How does it affect the "bounciness" or compressibility? Too low might make it act like jelly; too high might need tiny $\Delta t$.
        *   **Viscosity Coefficient ($μ$):** Compare low viscosity (water-like, splashy) vs. high viscosity (honey-like, slow, damped).
        *   **Timestep ($\Delta t$):** Crucial for stability (CFL-like conditions apply to SPH too!). If you make $\Delta t$ too large, the simulation will likely "explode" (particles fly off to infinity).
        *   **Smoothing Length ($h$):** Often tied to particle spacing (e.g., `h = 1.2 * initial_particle_spacing`). Understand its role in interaction range.
    *   **Particle Behavior:** Watch how particles interact, form surfaces, respond to forces.
    *   **Data Extraction (THE MOST IMPORTANT PART FOR ML):**
        *   Your simulator *must* allow you to save the state of the particles over time.
        *   **Data needed per particle at each saved frame/timestep:**
            *   `Position (x, y)`
            *   `Velocity (vx, vy)`
            *   `Acceleration (ax, ay)` (If the simulator calculates and stores this explicitly, great! If not, you might compute it from changes in velocity between frames as a target for your GNN, or have the GNN predict it directly).
            *   `Particle Type` (e.g., fluid, boundary – useful for GNNs)
        *   **Format:** CSV files, NumPy arrays (`.npy`), HDF5 are common. Aim for something easy to load in Python.
        *   **Frequency:** You don't necessarily need to save every single SPH time step (which can be very small). Saving every Nth step to get a reasonable number of frames for your ML dataset is fine.
        *   **Goal:** Generate a small, manageable 2D dataset. "Small" might mean a few hundred to a couple of thousand particles, simulated for a few hundred saved frames. This will be enough for initial GNN experiments.

*   **4. Visualize Your SPH Simulation Output:**
    *   Most simulators have a basic built-in visualizer (PySPH uses Matplotlib or Mayavi).
    *   It's also highly recommended to write a simple Python script (e.g., using Matplotlib's `scatter` plot) to load your *saved data files* and plot particle positions for a specific frame. This verifies your data extraction is working correctly. You can extend this to animate the sequence of frames.

**Outcome Check for this Lesson (and effectively for Chapter 1):**
By the end of this, you should:
*   Have a solid conceptual understanding of how SPH works (particles, kernels, density, pressure, forces, time-stepping).
*   Have successfully run a 2D SPH simulation using a pre-existing tool.
*   Have experimented with key SPH parameters to see their effect.
*   **Most importantly: Have generated and saved a 2D particle trajectory dataset (positions, velocities, ideally accelerations and particle types) that is ready to be used as input for training your first Graph Neural Network in Chapter 2.**
*   Be mentally prepared to think about: "How can I represent these interacting particles and their properties as a graph?"

---
