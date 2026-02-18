

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from matplotlib import ticker


# local environment descriptors imports
from maml.describers import BispectrumCoefficients
from sklearn.decomposition import PCA

# machine learning interatomic potentials imports
from maml.base import SKLModel
from maml.apps.pes import SNAPotential
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

# materials properties prediction imports
from pymatgen.core import Structure, Lattice
from maml.apps.pes import EnergyForceStress, SurfaceEnergy
from maml.apps.pes import LatticeConstant, ElasticConstant, NudgedElasticBand, DefectFormation
# disable logging information
import logging
logging.disable(logging.CRITICAL)
 
import requests
from pymatgen.core.structure import Structure
import time


# Record the starting time to calculate the total runtime
start_time = time.time()

# Define the Google Drive file IDs for each file
file_ids = {
    'AIMD': '1K20u-wYm2Txk99RyNCwr1gjI0bFou5Z0',
    'Vacancy': '1L7hYDB4Rix7KpyxWut1Nm-87djIjHu9N',
    'Elastic': '1jkq_hFAKq5biktncMaI8-QClW7uraQSy',
    'Surface': '1BZ7tM1JtP23N-p6MFFqUDuPtC-2bEttj',
    'training': '1gH6VUrqWNn5C1ajsz0TwADzaxxkPxrU9',
    'test': '1oVL5Ol1QKg44nazShpfvPwqbgTIdA2ux',
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
Cu_train_data = loadfn('training.json')
Cu_test_data =loadfn('test.json')

# Print the number of entries in each loaded file
print(' # of AIMD data:', len(Cu_aimd_data))
print(' # of Elastic data:', len(Cu_elastic_data))
print(' # of Surface data:', len(Cu_surface_data))
print(' # of Vacancy data:', len(Cu_vacancy_data))
print(' # of training data:', len(Cu_train_data))
print(' # of training data:', len(Cu_test_data))

# Three lists of structures, energies, and forces array.

Cu_data = Cu_aimd_data + Cu_aimd_data + Cu_elastic_data + Cu_surface_data + Cu_vacancy_data + Cu_aimd_data + Cu_aimd_data + Cu_elastic_data + Cu_surface_data + Cu_vacancy_data
Cu_train_structures = [d['structure'] for d in Cu_data]
Cu_train_energies = [d['outputs']['energy'] for d in Cu_data]
Cu_train_forces = [d['outputs']['forces'] for d in Cu_data]
print('hi')
# Set the external weights. Increase the weight of energy to 10000. Feel free to modify the weight.
Cu_train_pool = pool_from(Cu_train_structures, Cu_train_energies, Cu_train_forces)
_, Cu_df = convert_docs(Cu_train_pool)
weights = np.ones(len(Cu_df['dtype']), )
weights[Cu_df['dtype'] == 'energy'] = 160000
weights[Cu_df['dtype'] == 'force'] = 1
print('hi')
# Initialize the bispectrum coefficients describer and linear regression model
element_profile = {'Cu': {'r': 0.5, 'w': 1.0}}
describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8, 
                                   element_profile=element_profile, quadratic=False, 
                                   pot_fit=True, include_stress=False)
ml_model = LinearRegression()
skl_model = SKLModel(describer=describer, model=ml_model)
Cu_snap = SNAPotential(model=skl_model)

# Train the potential with lists of structures, energies, forces
Cu_snap.train(Cu_train_structures, Cu_train_energies, Cu_train_forces, sample_weight=weights)


df_orig, df_predict = Cu_snap.evaluate(test_structures=Cu_train_structures, 
                                       test_energies=Cu_train_energies,
                                       test_forces=Cu_train_forces)



energy_indices = np.argwhere(np.array(df_orig["dtype"]) == "energy").ravel()
forces_indices = np.argwhere(np.array(df_orig["dtype"]) == "force").ravel()
orig = df_orig['y_orig'] / df_orig['n']
predict = df_predict['y_orig'] / df_predict['n']

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

fig, ax1 = plt.subplots(figsize=(10, 10))
ax1.scatter(orig[energy_indices], predict[energy_indices], color='#1f77b4', s=80, alpha=0.8, 
            label="Energy MAE: {:.1f} meV/atom".format(mean_absolute_error(orig[energy_indices], predict[energy_indices]) * 1000))
ax1.set_xlim(-4.2, -3.4)
ax1.set_ylim(-4.2, -3.4)
ax1.set_xlabel("DFT energy (meV/atom)", fontsize=20)
ax1.set_ylabel("SNAP energy (meV/atom)", fontsize=20)
ax1.legend(fontsize=20)
from matplotlib import ticker

ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.show()


fig, ax2 = plt.subplots(figsize=(10,10))
ax2.scatter(orig[forces_indices], predict[forces_indices], color='#1f77b4', s=80, alpha=0.8,
            label="Forces MAE: {:.1f} eV/Å".format(mean_absolute_error(orig[forces_indices], predict[forces_indices])))
ax2.set_xlim(-11, 12)
ax2.set_ylim(-11, 12)
ax2.set_xlabel("DFT Forces (eV/Å)", fontsize=20)
ax2.set_ylabel("SNAP Forces (eV/Å)", fontsize=20)
ax2.legend(fontsize=20)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(4))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(4))

plt.show()

#material properties prediction


#lattice constant prediction
Cu_cell = Structure.from_spacegroup(sg='Fm-3m', species=['Cu'], lattice=Lattice.cubic(3.621), coords=[[0, 0, 0]])

Cu_lc_calculator = LatticeConstant(Cu_snap)
a, b, c = Cu_lc_calculator.calculate([Cu_cell])[0]
print('Cu', 'Lattice a: {:.3f} Å, Lattice b: {:.3f} Å, Lattice c: {:.3f} Å'.format(a, b, c))



#Vanacy energy calculator
Cu_vacancy_calcualtor = DefectFormation(ff_settings=Cu_snap, specie='Cu', lattice='fcc', alat=3.621)
formation_energy = Cu_vacancy_calcualtor.calculate()
print('Cu Vacancy formation energy: {:.2f} eV'.format(formation_energy))


#Elastic constants prediction
Cu_ec_calculator = ElasticConstant(ff_settings=Cu_snap)
Cu_C11, Cu_C12, Cu_C44, Cu_B = Cu_ec_calculator.calculate([Cu_cell])[0]
print('Cu', ' C11: {:.0f} GPa'.format(Cu_C11), 'C12: {:.0f} GPa'.format(Cu_C12), 'C44: {:.0f} GPa'.format(Cu_C44), 'Bvrh: {:.0f} GPa'.format(Cu_B))


#defect formation
Defect= DefectFormation(ff_settings=Cu_snap,specie='Cu', lattice='fcc', alat=3.621)
Defect_formation=Defect.calculate()
print('Deformation energy of Cu  : {:.2f} eV'.format(Defect_formation))


#surface energy
miller_indexes = [(1, 0, 0), (2, 2, 1),(2, 1, 0),(3, 1 ,1),(3, 2, 0)]
Cu_sur = SurfaceEnergy(ff_settings=Cu_snap,bulk_structure=Cu_cell,miller_indexes=miller_indexes)
Cu_100,Cu_221,Cu_210,Cu_311,Cu_320 = Cu_sur.calculate()
print('Surface energy for 100 :',Cu_100[-1],'J/m^2','\n','Surface energy for 210 :',Cu_210[-1],'J/m^2','\n','Surface energy for 221 :',Cu_221[-1],'J/m^2','\n','Surface energy for 311 :',Cu_311[-1],'J/m^2','\n','Surface energy for 320 :', Cu_320[-1],'J/m^2')


#Energy force stress calculator
efs=EnergyForceStress(ff_settings=Cu_snap)
energy_force_stress=efs.calculate([Cu_cell])[0]
e=energy_force_stress[0]
f=energy_force_stress[1]
s=energy_force_stress[2]
print('Cu', 'Energy:',e,'Force:',f, 'Stress:',s,sep='\n')




end_time = time.time()

total_runtime = end_time - start_time

print(f"Total runtime: {total_runtime} seconds")

