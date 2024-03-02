from data import EMPLOYMENT_COEFFICIENT, AREA, \
    SOCIAL_VALUE, CAT, SUBCATS, NUM_SUBCATS, ENVIRONMENTAL_VALUE, SALES_PER_UNIT_AREA, \
    alpha, Z_1, Z_2, Z_3, threshold
import math
import numpy as np

np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')

ECONOMIC_VALUE = SALES_PER_UNIT_AREA * (1 + alpha / EMPLOYMENT_COEFFICIENT)
ECONOMIC_VALUE = np.nan_to_num(ECONOMIC_VALUE) / 10


def intrinsic_value(cat, subcat, size):
    def economic_value(cat, subcat, size):
        eco_values = ECONOMIC_VALUE[cat, subcat]
        return eco_values * size

    def social_value(cat, subcat, size):
        return SOCIAL_VALUE[cat, subcat] * size

    def environmental_value(cat, subcat, size):
        w_1 = 0.5
        w_2 = 0.3
        sigma = 0.2
        nu = 0.1
        epsilon = 0.01

        gamma_vector = np.array(ENVIRONMENTAL_VALUE[cat, subcat]).T
        env_v = np.zeros_like(gamma_vector)
        env_mat = np.array([[1 - w_1, sigma, 0],
                            [w_1, 1 - sigma - nu, w_2],
                            [0, nu, 1 - w_2]])

        for i in range(threshold):
            env_v = np.dot(env_mat, env_v) + gamma_vector
            if np.all(env_v < epsilon):
                break
        Gamma_sum = np.sum(env_v, axis=0)
        E = np.log(Gamma_sum * size)
        return np.where(E > threshold, E * 3, 0)

    return (Z_1 * economic_value(cat, subcat, size) + Z_2 * social_value(cat, subcat, size)
            + Z_3 * environmental_value(cat, subcat, size))


def interaction_value(r, r0, model):
    def gaussian(x):
        return np.exp(-(x ** 2) / 2)

    def exponential_decay(x):
        return -np.exp(-x)

    def sigmoid(x):
        return np.exp(-(x / 2) ** 5)

    def commercial_commercial_interaction(x):
        return sigmoid(2 * x) + exponential_decay(x)

    def residential_industrial_interaction(x):
        return 10 * gaussian(2 * x) + exponential_decay(2 * x)

    result = np.zeros_like(r)
    r = r / r0
    result[model == 0] = 0
    result[model == 1] = commercial_commercial_interaction(r[model == 1])
    result[model == 2] = residential_industrial_interaction(r[model == 2])
    result[model == 3] = gaussian(r[model == 3])
    result[model == 4] = gaussian(r[model == 4])
    result[model == 5] = gaussian(r[model == 5])
    return result
