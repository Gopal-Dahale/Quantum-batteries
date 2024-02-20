import cudaq
import pickle
from time import time
from utils import get_ham_from_dict, rel_err
import matplotlib.pyplot as plt
import numpy as np

cudaq.set_target("nvidia")

ccsd_energy = -3688.046308050882

with open('CS_hams.pickle', 'rb') as handle:
	hams = pickle.load(handle)

hamiltonians = []
hf_states = []
for n_qubits, val in hams.items():
	ham_dict = hams[n_qubits]['ham']
	hamiltonians.append(get_ham_from_dict(ham_dict))

	hf_dict = hams[n_qubits]['hf']
	if hf_dict:
		hf_states.append(list(hf_dict.keys())[0])
	else:
		hf_states.append(None)


mean_durations = []
std_durations = []

mean_rel_errs = []
std_rel_errs = []

num_qubits = []

num_layers = 4
num_iterations = 10

for ham, hf in zip(hamiltonians[::-1], hf_states[::-1]):

	n_qubits = ham.get_qubit_count()
	num_qubits.append(n_qubits)
	print(f"\nnum qubits = {n_qubits}")

	temp_durations = []
	temp_rel_errs = []

	for _ in range(num_iterations):
		kernel, thetas = cudaq.make_kernel(list)
		qubits = kernel.qalloc(n_qubits)

		# Prepare the Hartree Fock State.
		if hf is not None:
			for i, q in enumerate(hf):
				if q == '1':
					kernel.x(qubits[i])

		# hardware efficient ansatz
		for l in range(num_layers):
			for q in range(n_qubits):
				kernel.ry(thetas[l*n_qubits + q], qubits[q])
			for q in range(n_qubits - 1):
				kernel.cx(qubits[q], qubits[q+1])

		for q in range(n_qubits):
			kernel.ry(thetas[num_layers*n_qubits + q], qubits[q])

		# Adds parameterized gates based on the UCCSD ansatz.
		optimizer = cudaq.optimizers.NelderMead()
		optimizer.max_iterations = 500

		parameter_count = (num_layers+1)*n_qubits
		start = time()
		energy, parameters = cudaq.vqe(kernel,
								ham,
								optimizer,
								parameter_count=parameter_count)
		end = time()
		temp_rel_errs.append(rel_err(ccsd_energy, energy))
		temp_durations.append(end-start)

		del kernel, thetas, qubits, optimizer

	mean_durations.append(np.mean(temp_durations))
	mean_rel_errs.append(np.mean(temp_rel_errs))

	std_durations.append(np.std(temp_durations))
	std_rel_errs.append(np.std(temp_rel_errs))

	print(f"minimized <H> = {round(energy,16)}")
	print(f"num params = {parameter_count}")
	print(f"rel_error = {mean_rel_errs[-1]} +- {std_rel_errs[-1]}")
	print(f"duration = {mean_durations[-1]} += {std_durations[-1]}")

fig,ax = plt.subplots(1,2,figsize=(12,5))
ax[0].errorbar(num_qubits, mean_rel_errs, std_rel_errs, marker='o')
ax[0].set_xlabel('# qubits')
ax[0].set_ylabel('Rel Error')
ax[0].set_xticks(num_qubits)
ax[0].set_yscale('log')

ax[1].errorbar(num_qubits, mean_durations, std_durations, marker='o', color='green')
ax[1].set_xlabel('# qubits')
ax[1].set_ylabel('durations')
ax[1].set_xticks(num_qubits)


plt.savefig('cs_vqe')
