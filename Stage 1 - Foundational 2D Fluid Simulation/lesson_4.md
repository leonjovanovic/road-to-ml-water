## Lesson 4: Simulating Fluids with Particles (SPH)

**Recap:** We've explored grid-based CFD methods. Now, we turn to a different paradigm: **particle-based methods**, with a focus on **Smoothed Particle Hydrodynamics (SPH)**. This is highly relevant because GNNs are naturally suited to learning from particle data, and SPH is a common way to generate such data, especially for visually dynamic fluids like those in games.

---

### Part 1: Why Particle Methods? And What is SPH?

**Why shift from grids to particles? (The Big Picture)**
While grid-based methods are powerful, they can have challenges with:
*   **Highly deforming free surfaces:** Think of water splashing, breaking into droplets, or merging. Tracking the exact boundary of the fluid on a fixed grid can be complex (requiring special techniques like Volume of Fluid or Level Set).
*   **Large deformations and fragmentation:** Simulating explosions or fluids that break apart.

**Particle methods offer a more natural (Lagrangian) way to handle these:**
*   The fluid itself is represented by a collection of discrete **particles**.
*   These particles carry fluid properties (mass, velocity, etc.) and move with the flow.
*   The fluid's boundary is simply defined by where the particles are – no explicit interface tracking is needed.

**Smoothed Particle Hydrodynamics (SPH): A Popular Particle Method**
*   **Core Idea:** In SPH, any fluid property (like density or pressure) at a point in space is estimated by **summing up the contributions of all nearby particles**. These contributions are weighted by a **smoothing kernel function (W)**, which gives more weight to closer particles.
*   **Analogy (Kernel Smoothing):** Imagine you want to know the average height of people in a small area of a crowd. You wouldn't just pick one person. Instead, you'd look at people within a certain radius, and maybe give more importance (weight) to those very close to your point of interest. The SPH kernel does something similar for fluid properties.

---

### Part 2: Key Concepts in SPH

Understanding these components is key to seeing how SPH simulates fluid motion.

*   **1. Particles as Fluid Carriers:**
    *   Each particle $i$ has attributes:
        *   $m_i$: Mass (usually constant for each particle throughout the simulation).
        *   $r_i$: Position vector $(x_i, y_i, z_i)$.
        *   $v_i$: Velocity vector $(\mathrm{vx}_i, \mathrm{vy}_i, \mathrm{vz}_i)$.
        *   It will also have or acquire values for density $\rho_i$, pressure $P_i$, etc.

*   **2. The Smoothing Kernel (W) and Smoothing Length (h):**
    *   **Smoothing Kernel $W(r, h)$:** A function that defines the "zone of influence" of a particle.
        *   $r$: The distance between the particle and the point of interest.
        *   $h$: The **smoothing length** (also called kernel radius or support radius). This is a crucial parameter that defines how far a particle's influence extends.
        *   **Properties of W:**
            *   It's usually bell-shaped or similarly peaked: highest at $r=0$ (at the particle itself).
            *   It smoothly decreases to zero at $r=h$. Particles further than $h$ have no influence.
            *   It's normalized (its integral over its support domain is 1), so it properly averages quantities.
        *   **Common Kernels:** Different mathematical forms are used for $W$ depending on what's being calculated (e.g., Poly6 kernel for density, Spiky kernel for pressure gradients, Viscosity kernel for viscosity). You don't need to memorize their exact formulas, but know they exist and are chosen for specific properties.

    *   **Smoothing Length $h$:**
        *   **Too small $h$:** Particles interact only with very close neighbors. Can lead to clumpy, unstable simulations if particles don't "see" enough neighbors.
        *   **Too large $h$:** Details are overly smoothed. Each particle interacts with many others, increasing computational cost per step.
        *   Typically, $h$ is set to be a small multiple of the initial average particle spacing.

*   **3. Density Estimation (The Starting Point for SPH Calculations):**
    *   The density $\rho_i$ at the location of particle $i$ is estimated by summing the mass contributions of all neighboring particles $j$ (including $i$ itself), weighted by the kernel:
        $\rho_i = \sum_j m_j \cdot W(|\mathbf{r}_i - \mathbf{r}_j|, h)$

        *   $|\mathbf{r}_i - \mathbf{r}_j|$
 is the distance between particle $i$ and particle $j$.
        *   The sum is over all particles $j$ within the smoothing length $h$ of particle $i$.
    *   **Intuition:** If many particles are packed closely around particle $i$,  $\rho_i$ will be high. If they are sparse, $\rho_i$ will be low.

*   **4. Equation of State (Pressure from Density):**
    *   Once density $\rho_i$ is known for each particle, pressure $P_i$ is typically calculated using an **equation of state (EOS)**. This is a thermodynamic relation that links pressure, density, and sometimes temperature.
    *   For simulating water-like fluids (which are nearly incompressible), a common EOS is **Tait's equation (or a similar stiff EOS):**
    
        $P_i = k \cdot \left[ \left( \frac{\rho_i}{\rho_0} \right)^\gamma - 1 \right]$

        Where:
        *   $ρ_0$: The **rest density** of the fluid (the target density it "wants" to be).
        *   $k$: A **stiffness parameter** (sometimes related to a "speed of sound" in the SPH fluid). Higher $k$ makes the fluid more resistant to compression (more incompressible). Too high $k$ might require very small time steps for stability.
        *   $γ$ (gamma): An exponent, often 7 for water-like simulations.
    *   **Intuition:** If $ρ_i > ρ_0$ (particle $i$ is in a compressed region), then $P_i$ becomes positive and large, indicating it wants to expand. If $ρ_i < ρ_0$ (dilute region), $P_i$ can be negative (tensile, though often clamped at zero for simple fluids).

*   **5. Calculating Forces for Motion:**
    The heart of SPH is calculating the forces that make particles move. The Navier-Stokes equations are approximated in SPH form. The main forces are:

    *   **a. Pressure Force:**
        *   **Goal:** To make particles move from high-pressure regions to low-pressure regions, effectively trying to make the density uniform (simulating incompressibility).
        *   **How (Conceptual):** The SPH formulation for the pressure gradient term $(-\nabla P)$ in Navier-Stokes involves summing contributions from neighboring particles. A common symmetric form for the pressure force on particle $i$ from particle $j$ is:

            $$F_{ij}^{\text{pressure}} = - m_i \cdot m_j \cdot \left( \frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} \right) \cdot \nabla W_{ij}$$

            $$F_{ij}^{\text{viscosity}} = \mu \cdot m_i \cdot m_j \cdot \left( \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_i \cdot \rho_j} \right) \cdot \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot 1}{\lvert\mathbf{r}_i - \mathbf{r}_j\rvert^2 + \varepsilon}$$

            $$F_{ij}^{\text{viscosity}} = \mu \cdot m_i \cdot m_j \cdot \left( \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_i \cdot \rho_j} \right) \cdot \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot {\nabla}}{\lvert\mathbf{r}_i - \mathbf{r}_j\rvert^2 + \varepsilon}$$

            $$F_{ij}^{\text{viscosity}} = \mu \cdot m_i \cdot m_j \cdot \left( \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_i \cdot \rho_j} \right) \cdot \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot {W}}{\lvert\mathbf{r}_i - \mathbf{r}_j\rvert^2 + \varepsilon}$$

            Where $\nabla W_{ij}$ is the gradient of the smoothing kernel $W\big(|\mathbf{r}_i - \mathbf{r}_j|, h\big)$ with respect to $r_i$. This gradient vector points from $j$ to $i$ and its magnitude depends on how rapidly the kernel changes with distance.
        *   The total pressure force on $i$ is $\sum_j F_{ij}^{\text{pressure}}$.
        *   **Intuition:** If particle $j$ is in a higher pressure state relative to its distance from $i$ (as captured by the pressures and kernel gradient), it will exert a repulsive force on $i$.

    *   **b. Viscosity Force:**
        *   **Goal:** To simulate the internal friction (viscosity $μ$) of the fluid. This should damp relative motion between particles and smooth out velocity differences.
        *   **How (Conceptual):** The SPH form for the viscous term $(\mu \nabla^2 \mathbf{u})$ in Navier-Stokes also involves summing contributions from neighbors. A common form for the viscosity force on particle $i$ from particle $j$ is:

            $$
            F_{ij}^{\text{viscosity}} = \mu \cdot m_i \cdot m_j \cdot \left( \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_i \cdot \rho_j} \right) \cdot \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot \nabla W_{ij}}{\lvert\mathbf{r}_i - \mathbf{r}_j\rvert^2 + \varepsilon}
            $$

            (This is one form; others exist, sometimes simpler, involving the Laplacian of the kernel or direct velocity differences). More simply, often expressed as:

            $$
            F_{ij}^{\text{viscosity}} = \sum_j m_j \cdot \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_j} \cdot \mu \cdot K_{\text{visc}} \cdot \nabla^2 W_{ij}
            $$
            where $K_{\text{visc}}$ depends on $h$.
        *   Essentially, if particle $j$ is moving significantly differently from $i$, a force is generated that tries to reduce this velocity difference.
        *   The total viscosity force on $i$ is $\sum_j F_{ij}^{\text{viscosity}}$.

    *   **c. External Forces:**
        *   **Gravity:** $F_i^{\text{gravity}} = m_i \cdot \mathbf{g}$ (where $g$ is the gravitational acceleration vector, e.g., $(0, -9.81, 0) \ \text{m/s}^2$). This is added to each particle.
        *   **Boundary Forces:** Handling interactions with solid boundaries (like container walls) is a critical aspect. Common methods:
            *   **Repulsive Force Particles:** Place fixed SPH particles at the boundary that exert strong short-range repulsion forces on fluid particles that get too close.
            *   **Ghost Particles:** Create mirror "ghost" particles inside the boundary to enforce conditions like no-slip.

*   **6. Time Integration – Advancing the Simulation:**
    1.  For each particle $i$, sum all forces: $F_i^{\text{total}} = F_i^{\text{pressure}} + F_i^{\text{viscosity}} + F_i^{\text{gravity}} + F_i^{\text{boundary}} + \ldots$
    2.  Calculate acceleration using Newton's second law: $a_i = \frac{F_i^{\text{total}}}{m_i}$.
    3.  Update particle velocities and positions over a small time step $\Delta t$. Common integration schemes:
        *   **Leapfrog Integration (often preferred for SPH):** Staggered time updates for position and velocity. Good for energy conservation.
            $v_i\left(t + \frac{\Delta t}{2}\right) = v_i\left(t - \frac{\Delta t}{2}\right) + a_i(t) \cdot \Delta t$
            $r_i(t + \Delta t) = r_i(t) + v_i\left(t + \frac{\Delta t}{2}\right) \cdot \Delta t$
        *   **Euler Integration (simpler, but less stable/accurate):**
            $v_i(t + \Delta t) = v_i(t) + a_i(t) \cdot \Delta t$
            $r_i(t + \Delta t) = r_i(t) + v_i(t) \cdot \Delta t$ (Explicit Euler)
            or $r_i(t + \Delta t) = r_i(t) + v_i(t + \Delta t) \cdot \Delta t$ (Semi-Implicit Euler, often better)

    This entire cycle (neighbor search, density, pressure, forces, integration) is repeated for every time step.

---

### Part 3: Practice - Your First SPH Simulation & Data Generation

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
