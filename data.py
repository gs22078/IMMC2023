import numpy as np

CAT = ['residential', 'commercial', 'industrial', 'agricultural', 'leisure', 'office']
SUBCATS = [['apartment', 'housing estate', 'schools', 'park'],
           ['general merchandise', 'restaurant', 'wholesale', 'hospital', 'service', 'security'],
           ['chemical manufacturing', 'electronics manufacturing', 'other manufacturing'],
           ['paddy', 'field', 'pasture'],
           ['skiing resort', 'swimming pool', 'golf course', 'spectator sports', 'museum', 'accommodation'],
           ['office']]

NUM_SUBCATS = np.array([len(SUBCATS[i]) for i in range(len(SUBCATS))])

# its unit is 10^6 won / m^2
sales_per_unit_area = {
    'residential': {
        'apartment': 1e-10,
        'housing estate': 1e-10,
        'schools': 1,
        'park': 1e-10,
    },
    'commercial': {
        'general merchandise': 240,
        'restaurant': 100,
        'wholesale': 318,
        'hospital': 150,
        'service': 160,
        'security': 1
    },
    'industrial': {
        'chemical manufacturing': 100,
        'electronics manufacturing': 15.76,
        'other manufacturing': 2.10
    },
    'agricultural': {
        'paddy': 0.0013,
        'field': 0.02,
        'pasture': 4e-4
    },
    'leisure': {
        'skiing resort': 10,
        'swimming pool': 4,
        'golf course': 0.01,
        'spectator sports': 4,
        'museum': 0.1,
        'accommodation': 0.5

    },
    'office': {
        'office': 1
    }
}
# if there's no employment, coefficient is 1e10
employment_coefficient = {
    'residential': {
        'apartment': 1e10,
        'housing estate': 1e10,
        'schools': 11.78,
        'park': 1e10
    },
    'commercial': {
        'general merchandise': 13.77,
        'restaurant': 11.35,
        'wholesale': 11.74,
        'hospital': 7.14,
        'service': 9.08,
        'security': 1e10
    },
    'industrial': {
        'chemical manufacturing': 2.09,
        'electronics manufacturing': 2.39,
        'other manufacturing': 2.12
    },
    'agricultural': {
        'paddy': 20.25,
        'field': 20.25,
        'pasture': 20.25
    },
    'leisure': {
        'skiing resort': 9.54,
        'swimming pool': 9.54,
        'golf course': 9.54,
        'spectator sports': 9.54,
        'museum': 9.54,
        'accommodation': 11.35
    },
    'office': {
        'office': 5.75
    }
}

AREA_DICT = {
    'residential': {
        'apartment': 2,
        'housing estate': 10,
        'schools': 1,
        'park': 5
    },
    'commercial': {
        'general merchandise': 2,
        'restaurant': 1,
        'wholesale': 1,
        'hospital': 4,
        'service': 2,
        'security': 2
    },
    'industrial': {
        'chemical manufacturing': 5,
        'electronics manufacturing': 5,
        'other manufacturing': 5
    },
    'agricultural': {
        'paddy': 50,
        'field': 30,
        'pasture': 10
    },
    'leisure': {
        'skiing resort': 350,
        'swimming pool': 300,
        'golf course': 320,
        'spectator sports': 5,
        'museum': 3,
        'accommodation': 3
    },
    'office': {
        'office': 3
    }
}

SOCIAL_VALUE_DICT = {
    'residential': {
        'apartment': 0,
        'housing estate': 0,
        'schools': 0,
        'park': 0.0327586,
    },
    'commercial': {
        'general merchandise': 0,
        'restaurant': 0,
        'wholesale': 0,
        'hospital': 0.0529,
        'service': 0,
        'security': 0
    },
    'industrial': {
        'chemical manufacturing': 0,
        'electronics manufacturing': 0,
        'other manufacturing': 0
    },
    'agricultural': {
        'paddy': 0,
        'field': 0,
        'pasture': 0
    },
    'leisure': {
        'skiing resort': 0,
        'swimming pool': 0,
        'golf course': 0,
        'spectator sports': 0,
        'museum': 1.73,
        'accommodation': 0
    },
    'office': {
        'office': 0
    }
}

ENVIRONMENTAL_VALUE_DICT = {
    'residential': {
        'apartment': [1, 1, 1],
        'housing estate': [1, 1, 1],
        'schools': [0, 0, 0],
        'park': [-3, 0, 0]
    },
    'commercial': {
        'general merchandise': [0, 0, 1],
        'restaurant': [0, 0, 1],
        'wholesale': [1, 0, 1],
        'hospital': [1, 0, 0],
        'service': [1, 0, 0],
        'security': [0, 0, 0]
    },
    'industrial': {
        'chemical manufacturing': [2, 3, 3],
        'electronics manufacturing': [2, 2, 2],
        'other manufacturing': [3, 2, 2]
    },
    'agricultural': {
        'paddy': [0, 1, 1],
        'field': [0, 1, 1],
        'pasture': [0, 0, 1]
    },
    'leisure': {
        'skiing resort': [0, 1, 1],
        'swimming pool': [1, 2, 0],
        'golf course': [1, 0, 1],
        'spectator sports': [0, 1, 0],
        'museum': [1, 0, 0],
        'accommodation': [1, 1, 1]
    },
    'office': {
        'office': [1, 1, 1]
    }
}

INTERACTION_MODEL = [[1, 1, 5, 2, 0, 3],
                     [1, 4, 3, 3, 1, 1],
                     [5, 3, 3, 2, 2, 3],
                     [2, 3, 2, 0, 2, 3],
                     [0, 1, 2, 2, 3, 0],
                     [3, 1, 3, 3, 0, 0]]

INTERACTION_COEFFICIENT = [[3, 8, 10, -7, 0, 7],
                           [8, 8, -8, 8, 6, 5],
                           [10, 8, 9, 10, 10, 6],
                           [7, 8, -10, 0, 10, 6],
                           [0, 6, -10, 10, 4, 0],
                           [7, 5, -6, -6, 0, 0]]

CHARACTERISTIC_DISTANCE = [[0.5, 10., 30., 5., 0., 30.],
                           [10., 3., 10., 5., 1., 0.5],
                           [30., 10., 5., 10., 15., 10.],
                           [5., 5., 10., 0., 5., 5.],
                           [0., 1., 15., 5., 10., 0.],
                           [30., 0.5, 10., 5., 0., 0.]]


def construct_array(data_dict):
    max_subcat_len = max([len(data_dict[cat]) for cat in CAT])
    array = [np.array(list(data_dict[cat].values())) for cat in CAT]
    if array[0].ndim == 1:
        array = [np.pad(arr, (0, max_subcat_len - len(arr))) for arr in array]
    else:
        array = [np.pad(arr, ((0, max_subcat_len - len(arr)), (0, 0))) for arr in array]
    return np.array(array)


# 1D array of all values in Direct_Economic_Multipliers_Dict
SALES_PER_UNIT_AREA = construct_array(sales_per_unit_area)
EMPLOYMENT_COEFFICIENT = construct_array(employment_coefficient)
SOCIAL_VALUE = construct_array(SOCIAL_VALUE_DICT)
AREA = construct_array(AREA_DICT)
ENVIRONMENTAL_VALUE = construct_array(ENVIRONMENTAL_VALUE_DICT)

# print(ENVIRONMENTAL_VALUE)

INTERACTION_MODEL = np.array(INTERACTION_MODEL)
INTERACTION_COEFFICIENT = np.array(INTERACTION_COEFFICIENT)
CHARACTERISTIC_DISTANCE = np.array(CHARACTERISTIC_DISTANCE)

LAND_DIMENSION = (10, 10)
POPULATION_SIZE = 100
MAX_GENERATION = 50
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.9
ELITISM_RATE = 0.1
ROULETTE_RATE = 0.4

alpha = 5  # alpha is the weight of employment in the economic value
threshold = 10  # threshold is the threshold of the environmental value
# Z_1, Z_2, Z_3 are the weights of economic, social, and environmental values
Z_1 = 1.0
Z_2 = 1.0
Z_3 = 1.0

# LaTeX stuff
preamble = r"""\documentclass[margin=3mm]{standalone}
\usepackage[dvipsnames]{xcolor} % For colors
\usepackage{tabularray}

% Define custom colors
\colorlet{residential}{blue!40}
\colorlet{commercial}{cyan!40}
\colorlet{industrial}{red!40}
\colorlet{agricultural}{Green!40}
\colorlet{leisure}{orange!40}
\colorlet{office}{violet!40}

"""
post = r"""
\begin{tblr}{colsep=3pt, rowsep=3pt, column{1}={4mm}, rows={4mm,l},
	cell{1}{1}={}{residential}, cell{2}{1}={}{commercial}, cell{3}{1}={}{industrial}, cell{4}{1}={}{agricultural}, cell{5}{1}={}{leisure}, cell{6}{1}={}{office}}
	 & residential \\
	 & commercial \\
	 & industrial \\
	 & agricultural \\
	 & leisure \\
	 & office \\
\end{tblr}
"""
COLORS_DICT = ['residential', 'commercial', 'industrial', 'agricultural', 'leisure', 'office']
