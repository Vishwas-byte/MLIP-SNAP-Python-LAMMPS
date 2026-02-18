print("weight 60000")
import os

def find_executable(executable):
    paths = os.environ["PATH"].split(os.pathsep)
    for path in paths:
        exe_path = os.path.join(path, executable)
        if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
            return exe_path
    return None

lmp_exe = find_executable("lmp_serial")
print(f"Path to lmp executable: {lmp_exe}")


# general imports
import numpy as np
np.random.BitGenerator = np.random.bit_generator.BitGenerator
import matplotlib.pyplot as plt
from monty.serialization import loadfn
from maml.utils import pool_from, convert_docs
print("hi")
# local environment descriptors imports
from maml.describers import BispectrumCoefficients
from sklearn.decomposition import PCA
print("hi")
# machine learning interatomic potentials imports
from maml.base import SKLModel
from maml.apps.pes import SNAPotential
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
print("hi")
# materials properties prediction imports
from pymatgen.core import Structure, Lattice
from maml.apps.pes import LatticeConstant, ElasticConstant, NudgedElasticBand, DefectFormation
print("hi")
# disable logging information
import logging
logging.disable(logging.CRITICAL)
print("hi")
import requests
from pymatgen.core.structure import Structure
import time


#Record the starting time to calculate the total runtime
start_time = time.time()
print("hi")
# Define the Google Drive file IDs for each file
file_ids = {
    'AIMD': '1K20u-wYm2Txk99RyNCwr1gjI0bFou5Z0',
    'Vacancy': '1L7hYDB4Rix7KpyxWut1Nm-87djIjHu9N',
    'Elastic': '1jkq_hFAKq5biktncMaI8-QClW7uraQSy',
    'Surface': '1BZ7tM1JtP23N-p6MFFqUDuPtC-2bEttj',
}

# Define the base URL for Google Drive downloads
base_url = 'https://drive.google.com/uc?id={}'

# Download each JSON file and save it
for file_name, file_id in file_ids.items():
    url = base_url.format(file_id)
    response = requests.get(url)
    with open(f'{file_name}.json', 'wb') as f:
        f.write(response.content)

# Load each JSON file into separate variables
Cu_aimd_data = loadfn('AIMD.json')
Cu_elastic_data = loadfn('Elastic.json')
Cu_surface_data = loadfn('Surface.json')
Cu_vacancy_data = loadfn('Vacancy.json')

# Print the number of entries in each loaded file
print(' # of AIMD data:', len(Cu_aimd_data))
print(' # of Elastic data:', len(Cu_elastic_data))
print(' # of Surface data:', len(Cu_surface_data))
print(' # of Vacancy data:', len(Cu_vacancy_data))

element_profile = {'Cu': {'r': 0.75, 'w': 1.0}}
per_atom_describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8, 
                                                 element_profile=element_profile, 
                                                 quadratic=False, 
                                                 pot_fit=False, 
                                                 include_stress=False)

Cu_aimd_structures = [d['structure'] for d in Cu_aimd_data]
per_atom_features = per_atom_describer.transform(Cu_aimd_structures)

print("Total # of atoms in Cu AIMD : 108 * 120 = {}\n".format(sum([len(struct) for struct in Cu_aimd_structures])), 
      "     # of features generated: {} (one feature per atom)\n".format(per_atom_features.shape[0]),
      "     # of dimensions: {} (for twojmax = 6)".format(per_atom_features.shape[1]))

print(per_atom_features)


element_profile = {'Cu': {'r': 0.75, 'w': 1.0}}
per_force_describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8, 
                                                 element_profile=element_profile, 
                                                 quadratic=False, 
                                                 pot_fit=True, 
                                                 include_stress=False)

Cu_aimd_structures = [d['structure'] for d in Cu_aimd_data]
per_force_features = per_force_describer.transform(Cu_aimd_structures)

print("Total # of features expected in Cu AIMD : (1 + 108 * 3) * 120 = {}\n".format(sum([(1 + 3 * len(struct)) for struct in Cu_aimd_structures])), 
      "     # of features generated: {} (1+3n features for n-atom structure)\n".format(per_force_features.shape[0]),
      "     # of dimensions: {}".format(per_force_features.shape[1]))

print(per_force_features)






import numpy as np
from sklearn.decomposition import PCA


# Obtain structures from each category
Cu_aimd_structures = [d['structure'] for d in Cu_aimd_data]
Cu_elastic_structures = [d['structure'] for d in Cu_elastic_data]
Cu_surface_structures = [d['structure'] for d in Cu_surface_data]
Cu_vacancy_structures = [d['structure'] for d in Cu_vacancy_data]

# Obtain the features from each category
Cu_aimd_features = per_atom_describer.transform(Cu_aimd_structures)
Cu_elastic_features = per_atom_describer.transform(Cu_elastic_structures)
Cu_surface_features = per_atom_describer.transform(Cu_surface_structures)
Cu_vacancy_features = per_atom_describer.transform(Cu_vacancy_structures)

# Function to check for inf or NaN values and print their indices
def check_inf_nan(features, name):
    features_array = features.values if hasattr(features, 'values') else features
    if np.isnan(features_array).any() or np.isinf(features_array).any():
        print(f"Problematic values found in {name}")
        nan_indices = np.argwhere(np.isnan(features_array))
        inf_indices = np.argwhere(np.isinf(features_array))
        print(f"NaN indices in {name}: {nan_indices}")
        print(f"Inf indices in {name}: {inf_indices}")
        raise ValueError(f"Features in {name} contain inf or NaN values")

# Check for inf or NaN values in each feature array
check_inf_nan(Cu_aimd_features, "Cu_aimd_features")
check_inf_nan(Cu_elastic_features, "Cu_elastic_features")
check_inf_nan(Cu_surface_features, "Cu_surface_features")
check_inf_nan(Cu_vacancy_features, "Cu_vacancy_features")

# Concatenate features from all categories
total_Cu_features = np.concatenate((Cu_aimd_features, Cu_elastic_features, 
                                    Cu_surface_features, Cu_vacancy_features), axis=0)

# Check for inf or NaN values in the concatenated array
check_inf_nan(total_Cu_features, "total_Cu_features")
'''
# Fit the PCA
pca = PCA(n_components=2)
pca.fit(total_Cu_features)

Cu_aimd_pcs = pca.transform(Cu_aimd_features)
Cu_elastic_pcs = pca.transform(Cu_elastic_features)
Cu_surface_pcs = pca.transform(Cu_surface_features)
Cu_vacancy_pcs = pca.transform(Cu_vacancy_features)

Cu_aimd_pc1 = Cu_aimd_pcs[:, 0]
Cu_aimd_pc2 = Cu_aimd_pcs[:, 1]

Cu_elastic_pc1 = Cu_elastic_pcs[:, 0]
Cu_elastic_pc2 = Cu_elastic_pcs[:, 1]

Cu_surface_pc1 = Cu_surface_pcs[:, 0]
Cu_surface_pc2 = Cu_surface_pcs[:, 1]

Cu_vacancy_pc1 = Cu_vacancy_pcs[:, 0]
Cu_vacancy_pc2 = Cu_vacancy_pcs[:, 1]

# Plotting the PCA results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(Cu_aimd_pc1, Cu_aimd_pc2, label='Cu_aimd', alpha=0.6)
plt.scatter(Cu_elastic_pc1, Cu_elastic_pc2, label='Cu_elastic', alpha=0.6)
plt.scatter(Cu_surface_pc1, Cu_surface_pc2, label='Cu_surface', alpha=0.6)
plt.scatter(Cu_vacancy_pc1, Cu_vacancy_pc2, label='Cu_vacancy', alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Cu Features')
plt.legend()
plt.show()

'''

#Preparing training dataset for the linear regression
Cu_data = Cu_aimd_data + Cu_elastic_data + Cu_surface_data + Cu_vacancy_data
Cu_train_structures = [d['structure'] for d in Cu_data]
Cu_train_energies = [d['outputs']['energy'] for d in Cu_data]
Cu_train_forces = [d['outputs']['forces'] for d in Cu_data]

print(" # of structures in Cu data: {}\n".format(len(Cu_train_structures)),
      "# of energies in Cu data: {}\n".format(len(Cu_train_energies)),
      "# of forces in Cu data: {}\n".format(len(Cu_train_forces)),
      "first item in energies: {}\n".format(Cu_train_energies[0]),
      "first item in forces: (n x 3 array)\n", np.array(Cu_train_forces[0]))




#Training features
Cu_features = per_force_describer.transform(Cu_train_structures)
print("# of features generated: {}".format(Cu_features.shape[0]))
print(Cu_features)


#Creating targets per atom
Cu_train_pool = pool_from(Cu_train_structures, Cu_train_energies, Cu_train_forces)
_, Cu_df = convert_docs(Cu_train_pool)
print("# of targets: ", len(Cu_df))
Cu_df


##### Simple Linear regression
y = Cu_df['y_orig'] / Cu_df['n']
x = Cu_features

simple_model = LinearRegression()
simple_model.fit(x, y)

##### Increase the weights of energies since the number of forces are overwhelming
weights = np.ones(len(Cu_df['dtype']), )
weights[Cu_df['dtype'] == 'energy'] = 160000
weights[Cu_df['dtype'] == 'force'] = 1

weighted_model = LinearRegression()
weighted_model.fit(x, y, sample_weight=weights)

print("# of parameters in simple linear model: {}\n".format(len(simple_model.coef_)), 
      "parameters in simple linear model: \n", simple_model.coef_, "\n")
print("# of parameters in weighted linear model: {}\n".format(len(weighted_model.coef_)), 
      "parameters in weighted linear model: \n", weighted_model.coef_)




###Energy and force prediction

energy_indices = np.argwhere(np.array(Cu_df["dtype"]) == "energy").ravel()
forces_indices = np.argwhere(np.array(Cu_df["dtype"]) == "force").ravel()

simple_predict_y = simple_model.predict(x)
weighted_predict_y = weighted_model.predict(x)

original_energy = y[energy_indices]
original_forces = y[forces_indices]
simple_predict_energy = simple_predict_y[energy_indices]
simple_predict_forces = simple_predict_y[forces_indices]
weighted_predict_energy = weighted_predict_y[energy_indices]
weighted_predict_forces = weighted_predict_y[forces_indices]

print(" Simple model energy MAE: {:.3f} meV/atom\n".format(mean_absolute_error(original_energy, simple_predict_energy) * 1000),
      "Simple model forces MAE: {:.3f} eV/Å\n".format(mean_absolute_error(original_forces, simple_predict_forces)),
      "Weighted model energy MAE: {:.3f} meV/atom\n".format(mean_absolute_error(original_energy, weighted_predict_energy) * 1000),
      "Weighted model forces MAE: {:.3f} eV/Å\n".format(mean_absolute_error(original_forces, weighted_predict_forces)),)



end_time = time.time()

total_runtime = end_time - start_time

print(f"Total runtime: {total_runtime} seconds")
