**Chapter 1: Fluid Theory Fundamentals**.

Imagine you want to create realistic water in a video game, predict how a new airplane wing will perform, or even understand how cream mixes into your coffee. All of  these involve **fluid dynamics**, the study of how fluids (liquids and gases) move. This first stage is all about understanding the basic rules and concepts that govern this movement.

Think of it like learning the rules of chess before you can play a game. These fundamentals are the building blocks for everything else.

Let's break down the first section:

---

## Lesson 1: Fluid Theory Fundamentals

### Part 1: Incompressible Flow & Navier-Stokes Equations

**Why do we need to understand this? (The Big Picture)**

Ever noticed how water in a glass basically keeps the same volume, even if you pour it into a differently shaped glass? Or how it's really hard to squeeze a full, sealed water bottle? This is the essence of "incompressible flow." Many fluids we interact with daily (like water, or air at low speeds like a gentle breeze) behave this way.

If we want to simulate these fluids – make a computer predict their behavior – we need a mathematical description of their motion. That's where the **Navier-Stokes equations** come in. They are the cornerstone of fluid dynamics, like Newton's laws (F=ma) are for solid objects, but for fluids.

**Real-life examples:**
*   Designing the hull of a ship to minimize drag.
*   Predicting how water will flow through pipes in a city's water system.
*   Creating realistic water splashes or smoke effects in movies and games.
*   Understanding blood flow in arteries.

---

**1. Incompressible Flow**

*   **What is it?**
    A fluid is considered **incompressible** if its **density** remains nearly constant as it moves. Density, you might remember, is mass per unit volume (how much "stuff" is packed into a certain space).
    *   Think of water: If you have 1 liter of water (about 1 kg), it will pretty much always occupy 1 liter of space, whether it's in a puddle, a river, or a pipe. Its density (1 kg/liter) doesn't change much under normal conditions.
    *   Contrast this with air in a bicycle pump: You can easily compress the air, forcing the same mass into a smaller volume, thus increasing its density. So, air *can* be compressible. However, for slow-moving air (like a gentle wind), the density changes are often so small that we can treat it as incompressible to simplify things.

*   **Why is this concept useful?**
    Assuming a fluid is incompressible greatly simplifies the mathematics used to describe its flow. This is a very common and useful assumption for liquids like water and for gases moving at low speeds (typically much less than the speed of sound).

*   **The Mathematical Implication (Continuity Equation for Incompressible Flow):**
    If the density is constant, it means that fluid cannot "pile up" or "thin out" anywhere. The amount of fluid flowing into any tiny imaginary box within the fluid must equal the amount flowing out. This idea is captured by the **continuity equation**.
    For an incompressible fluid, this simplifies to:
    $\nabla \cdot \mathbf{u} = 0$ (pronounced "nabla dot u equals zero" or "divergence of u equals zero")

    *   $u$: This represents the **velocity vector** of the fluid. Since we're in 2D (as per your course title), $u$ would have two components, say $(u_x, u_y)$, representing the speed in the x-direction and y-direction at any point.
    *   $\nabla \cdot $ (Nabla dot): This is the **divergence operator**. It measures how much a vector field (like our velocity field) is "spreading out" or "sourcing" from a point.
        *   Imagine water flowing out from a sprinkler head – that's positive divergence.
        *   Imagine water flowing down a drain – that's negative divergence (a "sink").
    *   $\nabla \cdot \mathbf{u} = 0$ means there are no sources or sinks within the fluid. The flow is just moving around, not being created or destroyed, and not compressing or expanding.

    **Real-life example for $\nabla \cdot \mathbf{u} = 0$:**
    Think of a garden hose. If you partially cover the nozzle with your thumb, the opening gets smaller. For the same amount of water to come out (it can't compress inside the hose), the water must speed up. The total *volume* of water passing through any cross-section of the hose per second remains the same (if incompressible). $\nabla \cdot \mathbf{u} = 0$ ensures this kind of mass balance.

---

**2. The Navier-Stokes Equations**

These are a set of equations that describe the motion of viscous (sticky, like honey, or even slightly sticky, like water) fluid substances. They arise from applying Newton's second law (Force = mass × acceleration, or F=ma) to a tiny parcel of fluid.

*   **What problem do they solve?**
    They help us predict how the velocity of a fluid changes over time and space, considering various forces acting on it.

*   **The Main Components (Conceptually for now):**
    The Navier-Stokes equations essentially say:
    `Rate of change of fluid momentum = Sum of forces acting on the fluid`

    Let's break down the "forces" part for a small piece of fluid:
    1.  **Pressure Forces:** Fluids push on their surroundings. If the pressure on one side of our tiny fluid parcel is higher than on the other, there's a net force that will make it move. (Think of squeezing one end of a toothpaste tube – the higher pressure pushes the toothpaste out).
    2.  **Viscous Forces:** This is like friction within the fluid. If one layer of fluid is moving faster than an adjacent layer, there's a "drag" between them. Honey is very viscous; water is much less so, but still has some viscosity. (Think of stirring thick honey vs. stirring water).
    3.  **External Forces (Body Forces):** These are forces that act on the entire volume of the fluid parcel, like gravity pulling water downwards.

    And the "rate of change of momentum" part includes:
    *   How the velocity at a fixed point changes over time (local acceleration).
    *   How momentum is carried along by the fluid flow itself (convective acceleration – think of how a fast-moving river carries a leaf along).

*   **The Equations (Simplified Idea for 2D Incompressible Flow):**
    We typically have one equation for each direction of motion. For 2D, this means one for the x-velocity (let's call it $u_x$ or just $u$) and one for the y-velocity ($u_y$ or just $v$).

    A very conceptual form for the x-momentum equation might look like:
    `Change in u-velocity = - (Pressure change in x-dir) + (Viscous effects on u) + (External forces in x-dir)`

    And similarly for the y-momentum equation:
    `Change in v-velocity = - (Pressure change in y-dir) + (Viscous effects on v) + (External forces in y-dir)`

    **Don't worry about the exact mathematical symbols yet.** The key is to understand that these equations balance acceleration with forces due to pressure differences, internal friction (viscosity), and external influences like gravity.

    Combined with the continuity equation ($\nabla \cdot \mathbf{u} = 0$) for incompressible flow, these equations form a complete system to describe the fluid's velocity and pressure fields.

*   **Real-life example of Navier-Stokes in action:**
    When you see smoke curling upwards from a candle:
    *   The heat from the candle warms the air, making it less dense (buoyancy, an external force effect).
    *   This buoyant air rises.
    *   Pressure differences develop as the air moves.
    *   Viscosity causes the smoke to mix and diffuse with the surrounding air, creating those intricate patterns.
    The Navier-Stokes equations (in their more complex form for compressible, heated flow) would describe this entire process. For simpler water flow, the incompressible version is often sufficient.

**Outcome Check for this Part:**
You should now have a basic understanding that:
*   "Incompressible" means density is constant, simplifying to $\nabla \cdot \mathbf{u} = 0$ (no net flow into or out of any point).
*   The Navier-Stokes equations are like F=ma for fluids, relating fluid acceleration to pressure forces, viscous (friction) forces, and external forces (like gravity).
*   Together, these equations govern how common fluids like water move.

---

### Part 2: Eulerian vs. Lagrangian Perspective

**Why do we need to understand this? (The Big Picture)**

When we study fluids, we need a way to "observe" and "track" their properties (like velocity, pressure, temperature). There are two main viewpoints, or "perspectives," for doing this: Eulerian and Lagrangian. Choosing the right perspective can make simulating or analyzing a fluid problem much easier.

**Real-life examples of needing a perspective:**
*   **Weather forecasting:** Do you measure wind speed at fixed weather stations (Eulerian) or by releasing weather balloons that float with the wind (Lagrangian)? Both are used!
*   **Tracking pollution in a river:** Do you set up sensors at fixed points along the riverbank (Eulerian) or release a dye/tracer and follow its path (Lagrangian)?

---

**1. Eulerian Perspective**

*   **What is it?**
    Imagine you're standing on a bridge overlooking a river, and you're focused on a specific spot in the water *under the bridge*. You observe the water flowing *past this fixed spot*. You'd measure the velocity, temperature, or pressure of whatever fluid parcel happens to be at that exact spot at any given moment.
    *   **Analogy:** A traffic camera fixed at an intersection. It records the speed of different cars passing through that specific point. It doesn't follow a single car.

*   **How it works in simulations:**
    You define a **fixed grid** (like a checkerboard) in your simulation space. At each point or cell in this grid, you store the fluid properties (e.g., velocity, pressure). As the fluid moves, the values at these fixed grid points change over time.
    *   This is often called a **grid-based method**.

*   **When is it used? Why?**
    *   **Good for:** Getting a "field" view of the fluid – seeing how pressure or velocity varies smoothly across a region.
    *   Often computationally more straightforward for problems where the fluid domain (the space it occupies) is fixed or doesn't change too drastically.
    *   Many numerical methods for solving Navier-Stokes (like the one you'll build in Part 2 of the course) are Eulerian.

*   **Potential Challenge:** If you're trying to track the boundary or interface between two different fluids (like oil and water), or the free surface of water, it can be tricky in a purely Eulerian framework, as the interface moves *through* your fixed grid cells.

---

**2. Lagrangian Perspective**

*   **What is it?**
    Imagine you throw a small rubber duck into the river. Instead of watching a fixed spot, you *follow the duck* as it bobs and weaves along with the current. You are measuring the properties (like velocity, or the temperature of the water immediately around the duck) of that *specific parcel of fluid* (represented by the duck) as it moves.
    *   **Analogy:** Using a GPS tracker on a specific car to monitor its journey and speed.

*   **How it works in simulations:**
    You represent the fluid as a collection of discrete **particles**. Each particle has its own properties (mass, velocity, etc.) and you track the individual motion of each particle over time.
    *   This is often called a **particle-based method** (like SPH, which is coming up).

*   **When is it used? Why?**
    *   **Good for:** Tracking complex boundaries, free surfaces, splashes, or when the fluid breaks apart into droplets. It's very intuitive for these scenarios because the particles *are* the fluid.
    *   Can be more natural for problems with large deformations or moving boundaries.

*   **Potential Challenge:** Calculating properties that depend on interactions between particles (like pressure or density in SPH) can be computationally intensive, as you need to find "neighboring" particles for each particle. Also, getting a smooth "field" view can require averaging over many particles.

---

**Contrasting Grid-Based (Eulerian) and Particle-Based (Lagrangian) Methods:**

| Feature         | Eulerian (Grid-Based)                   | Lagrangian (Particle-Based)                |
| :-------------- | :-------------------------------------- | :----------------------------------------- |
| **Viewpoint**   | Fixed points in space (field view)      | Follow individual fluid elements (parcels) |
| **Data Storage**| Values on a grid (velocity, pressure)   | Properties attached to particles           |
| **Strengths**   | Smooth fields, fixed domains, pressure  | Complex boundaries, splashes, interfaces   |
| **Challenges**  | Tracking moving interfaces, numerical diffusion | Calculating interaction forces, smooth fields, can be many particles |
| **Example**     | Weather station, most CFD solvers       | Weather balloon, SPH, smoke particles    |

**Why use each?**
*   You might use an **Eulerian** approach for simulating airflow around an airplane wing (where the domain is relatively fixed) or water flow in a pipe. The grid-based solver you will build in "2. Basic 2D Grid-Based Solver in Python" is Eulerian.
*   You might use a **Lagrangian** approach for simulating a dam breaking and water splashing everywhere, or for simulating smoke where you want to see the individual wisps. The SPH simulation in "3. Basic 2D SPH Simulation in Python" is Lagrangian.

Sometimes, hybrid methods are used that combine the strengths of both!

**Outcome Check for this Part:**
You should now be able to:
*   Explain the difference between watching fluid flow past a fixed point (Eulerian) and following a fluid parcel as it moves (Lagrangian).
*   Relate Eulerian to grid-based methods and Lagrangian to particle-based methods.
*   Give a simple reason why one might be preferred over the other for a given scenario.

---

### Part 3: Shallow Water Basics (Optional)

**Why do we need to understand this, even if it's optional? (The Big Picture)**

The full Navier-Stokes equations can be very complex to solve. Sometimes, we can make simplifying assumptions about the flow that lead to much simpler, yet still useful, equations. The **Shallow Water Equations (SWEs)** are a perfect example of this.
Understanding them gives you insight into:
1.  How physical approximations simplify mathematical models.
2.  A concrete, solvable example of fluid dynamics before tackling the full complexity.
3.  A model that's genuinely useful for many real-world phenomena.

**Real-life examples where SWEs are applicable:**
*   Predicting tsunami wave propagation across the ocean.
*   Modeling tides in estuaries and coastal regions.
*   Simulating river flooding.
*   Analyzing flow in wide, shallow channels or even large-scale atmospheric flows.

---

**1. What are "Shallow Water" Conditions?**

The key assumption is that the **horizontal scale of motion is much larger than the vertical depth of the fluid**.
*   Think of water sloshing in a very wide, shallow pan. The waves might travel many centimeters horizontally, but the water depth is only a centimeter or two.
*   For a tsunami, the wavelength can be hundreds of kilometers, while the ocean depth is "only" a few kilometers.

**Consequences of this assumption:**
*   **Vertical velocity is negligible:** The water mainly moves horizontally. Any vertical motion is small and quick compared to the horizontal flow.
*   **Pressure is hydrostatic:** This is a big one! It means the pressure at any point in the fluid only depends on the depth of the water above it (and atmospheric pressure). $Pressure = \rho \cdot g \cdot h$ (density * gravity * depth below surface). This simplifies the pressure term in the momentum equations significantly.
*   **Horizontal velocity is nearly uniform with depth:** The speed of the water doesn't change much as you go from the surface to the bottom. We can just talk about *the* horizontal velocity at a given (x,y) location.

---

**2. The Shallow Water Equations (SWEs)**

The SWEs are derived from the principles of conservation of mass and conservation of momentum, but with the shallow water assumptions applied. For 2D flow (movement in x and y directions), they typically consist of:

*   **A. Conservation of Mass (or Height Equation):**
    *   **Problem it solves:** How does the water height $h$ change over time?
    *   **How it solves it:** If more water flows into a region than out, the height $h$ in that region must increase. If more flows out than in, $h$ must decrease.
    *   **Conceptually (1D for simplicity, just x-direction):**
        $\frac{\partial h}{\partial t} = - \frac{\partial (h u)}{\partial x}$
        
        `Change in height over time = - (Change in (height * x-velocity) over x-distance)`.
        This equation essentially says that the rate at which the water level $h$ changes at a point is due to the net flow of water `(h*u)` into or out of that point.

*   **B. Conservation of Momentum (Velocity Equations):**
    *   **Problem it solves:** How do the horizontal velocities (let's say $u$ in x-direction, $v$ in y-direction) change over time?
    *   **How it solves it:** Fluid accelerates (velocity changes) due to:
        1.  **Pressure Gradients (which are now just Slopes in Water Height):** If the water surface is sloped, gravity effectively pulls water downhill. A steeper slope means faster acceleration.
        2.  **Convection:** The fluid carries its own momentum. If faster-moving water flows into a region, it can speed up the water already there.
        3.  **Bottom Friction (Optional):** Drag from the channel bed can slow the flow.
        4.  **Coriolis Force (Optional, for large-scale geophysical flows like oceans/atmosphere):** Due to Earth's rotation.
    *   **Conceptually (1D x-momentum for simplicity, ignoring friction/Coriolis):**
        $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = -g \frac{\partial h}{\partial x}$

        This says that the acceleration of the fluid ($\frac{\partial u}{\partial t}$ and the convective part) is driven by gravity acting on the slope of the water surface ($-g \frac{\partial h}{\partial x}$).

---

**3. Coding a Simple 1D Wave (Dam-Break) Solver**

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
    3.  **Time Loop:**
        *   For each cell, use the discretized (finite difference) versions of the SWEs to calculate how $h$ and $u$ will change over a small time step $\Delta t$.
        *   Update the $h$ and $u$ values in all cells.
        *   Repeat for many time steps.
    4.  **Visualize:** Plot $h$ (and maybe $u$) along the channel at different times to see the wave propagate.

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