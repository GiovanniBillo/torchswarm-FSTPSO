import numpy as np
import matplotlib.pyplot as plt
import random
import logging
from scipy.integrate import odeint

# -------------------------------
# Setup logger
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

random.seed(0)
np.random.seed(0)

solution = np.zeros((100, 2)) # placeholder
# ----------------------------------------------------
# Lotka–Volterra system — UNCHANGED, just logging
# ----------------------------------------------------
def lotka_volterra(state, t, params):
    r = params[0][0]
    a = params[0][1]
    b = params[1][0]
    z = params[1][1]

    X = state[0]
    Y = state[1]

    dxdt = r*X - a*X*Y
    dydt = b*X*Y - z*Y

    return [dxdt, dydt]



# ----------------------------------------------------
# Position update — unchanged, logging added
# ----------------------------------------------------
def update_position(position, velocity, boundaries):
    new_position = position + velocity
    log.debug(f"Updating position {position} + vel {velocity} -> {new_position}")

    for i, b in enumerate(boundaries):
        if new_position[i] < b[0]:
            log.debug(f"Position dim {i} hit lower bound {b[0]}")
            new_position[i] = b[0]
            velocity[i] = - np.random.random() * velocity[i]
        elif new_position[i] > b[1]:
            log.debug(f"Position dim {i} hit upper bound {b[1]}")
            new_position[i] = b[1]
            velocity[i] = - np.random.random() * velocity[i]

    return new_position


# ----------------------------------------------------
# Fitness function — unchanged, logging added
# ----------------------------------------------------
def fit(es_params, solution):
    log.debug(f"Evaluating fitness at params={es_params}")

    try:
        es_solution, info = odeint(lotka_volterra, initial_conditions, t,
                                   args=(es_params,), full_output=1)
        log.debug(f"ODE solver info: {info}")
    except Exception as e:
        log.error(f"ODE solver failed at params={es_params} -> {e}")
        return 1e12

    if es_solution.shape != solution.shape:
        log.error(f"Shape mismatch: expected {solution.shape}, got {es_solution.shape}")
        return 1e12

    error_predator = es_solution[0] - solution[0]
    error_prey = es_solution[1] - solution[1]

    total_error = np.sum(np.abs(error_predator + error_prey))

    log.debug(f"Total fitness error = {total_error}")
    return total_error


# ----------------------------------------------------
# Your MSE fitness object — unchanged, logging added
# ----------------------------------------------------
class MSE_fit():
    def __init__(self, target_traj):
        self.target_traj = target_traj

    def eval(self, individual):
        log.debug(f"MSE_fit evaluating individual={individual}")
        t = np.linspace(0, 100, 100)
        initial_conditions = self.target_traj[0]

        try:
            sol = odeint(lotka_volterra, initial_conditions, t, args=(individual,))
        except Exception as e:
            log.error(f"ODE solver failure inside MSE_fit, params={individual}: {e}")
            return 1e12

        mse = np.mean((sol - self.target_traj) ** 2)
        log.debug(f" -> MSE = {mse}")
        return mse


# ----------------------------------------------------
# Velocity update — unchanged, logging added
# ----------------------------------------------------
def update_velocity(position, velocity, global_best, local_best, max_velocities, w=0.7, c_soc=1.5, c_cog=1.5):
    n = len(velocity)
    print("velocity length:", n)
    r1 = np.random.random(n)
    r2 = np.random.random(n)
    print("global best shape:", global_best.shape)
    print("position shape:", position.shape)
    social_component = c_soc * r1 * (global_best - position)
    cognitive_component = c_cog * r2 * (local_best - position)
    inertia = w * velocity

    new_velocity = inertia + social_component + cognitive_component

    log.debug(f"Velocity update:\n"
              f"pos={position}\nvel={velocity}\n"
              f"pbest diff={local_best-position}\ngbest diff={global_best-position}\n"
              f"new vel={new_velocity}")

    # Clip velocity to bounds
    for i, v in enumerate(max_velocities):
        for j in range(max_velocities[i].shape[1]):
            if np.abs(new_velocity[i]) < v[0]: # this only needs to be done because we don't work on dimensions...
                log.debug(f"Velocity dim {i} hit min speed {v[0]}")
                new_velocity[i] = np.sign(new_velocity[i]) * v[0]
            elif np.abs(new_velocity[i]) > v[0]:
                log.debug(f"Velocity dim {i} hit max speed {v[1]}")
                new_velocity[i] = np.sign(new_velocity[i]) * v[1]

    return new_velocity


# ----------------------------------------------------
# PSO — unchanged, only logging added
# ----------------------------------------------------
def pso(swarm_size, boundaries, max_velocities, n_iter, fit, sol_shape):
    m = len(boundaries)

    # positions = [
    #     np.array([np.random.random() * (b[1] - b[0]) + b[0] for b in boundaries])
    #     for i in range(swarm_size)
    # ]

    # positions = [np.random.uniform((2, 2)) for i in range(swarm_size)]
    positions = []
    for _ in range(swarm_size):
        individual = np.zeros(sol_shape)
        for i in range(sol_shape[0]):
            for j in range(sol_shape[1]):
                individual[i,j] = random.uniform(0, 1)
            positions.append(individual)
    
    assert positions[0].shape == (2, 2), f"AssertionError: shape is not (2, 2)"  
    log.info(f"Initial positions:\n{positions}")
    
    velocities = []
    v0 = max_velocities[0][0]
    v1 = max_velocities[0][1]

    for _ in range(swarm_size):
        individual_velocity = np.zeros(sol_shape)
        for i in range(sol_shape[0]):
            for j in range(sol_shape[1]):
                individual_velocity[i,j] = random.uniform(v0, v1)
            velocities.append(individual_velocity)

    # velocities = [
    #     np.array([np.random.choice([-1,1]) * np.random.uniform(v[0], v[1])
    #               for v in max_velocities])
    #     for i in range(swarm_size)
    # ]

    log.info(f"Initial velocities:\n{velocities}")

    ## they have to be like this
    # real_params = np.array(
    #     [[0.1, 0.02],
    #     [0.01, 0.1]],
    # )

    local_best = positions.copy()
    print("positions:", positions)
    global_best = min(positions, key=fit)
    log.info(f"Initial global best: {global_best}")

    hist = [positions]

    # Main PSO loop
    for it in range(n_iter):
        log.info(f"--- Iteration {it+1}/{n_iter} ---")

        velocities = [
            update_velocity(p, v, global_best, lb, max_velocities)
            for p, v, lb in zip(positions, velocities, local_best)
        ]

        positions = [
            update_position(p, v, boundaries)
            for p, v in zip(positions, velocities)
        ]

        # Update personal bests
        for i in range(swarm_size):
            candidate = positions[i]
            if fit(candidate) < fit(local_best[i]):
                log.debug(f"Particle {i} improved local best")
                local_best[i] = candidate

        # Update global best
        candidate_best = min(positions, key=fit)
        if fit(candidate_best) < fit(global_best):
            log.info(f"New global best: {candidate_best}")
            global_best = candidate_best

        hist.append(positions)

    return global_best, hist

def myfit(es_params, solution=solution):
    print("ES PARAMS:", es_params)
    es_solution, info = odeint(lotka_volterra, initial_conditions, t, args=(es_params,), full_output=1)
    # print("Debug info: ", info)
    assert es_solution.shape == solution.shape, f"AssertionError: solution shape is {es_solution.shape} when it should be {solution.shape}" 
    error_predator = es_solution[0] - solution[0]
    error_prey = es_solution[1] - solution[1]
    return np.sum(np.abs(error_predator + error_prey))

# Example call

if __name__=="__main__":
    # Set the initial conditions and parameters
    initial_conditions = [30, 10]  # Initial populations of species X and Y

    real_params = np.array(
        [[0.1, 0.02],
        [0.01, 0.1]],
    )

    print("SHape of the real params:", real_params.shape)

    # Create a time vector
    t = np.linspace(0, 100, 100)

    # Solve the differential equations using odeint
    solution = odeint(lotka_volterra, initial_conditions, t, args=(real_params,))

    # Extract the populations of each species
    x, y = solution.T
    mse = MSE_fit(target_traj=solution).eval

    best, hist = pso(
        swarm_size=10,
        boundaries=[[0,1],[0,1],[0,1],[0,1]],
        max_velocities=[[0.001,1],[0.001,1],[0.001,1],[0.001,1]],
        n_iter=50,
        # fit=MSE_fit
        fit=myfit,
        sol_shape=real_params.shape
    )

global_best
