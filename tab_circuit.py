import pennylane as qml
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

n_qubits= 17
circuit_depth = 4
register_spec=["c5", "n2","n3", "c2", "c3"]
weights= [1,]

def standard_subcircuit(weights, n_qubits, circuit_depth, qubit_offset):
    shift = 0
    count = 0
    for _ in range(circuit_depth):
        for k in range(n_qubits):
            qml.RY(weights[k + shift], wires=k)
            count += 1
        for k in range(n_qubits - 1):
            qubit_1 = k
            qubit_2 = k + 1
            qml.IsingYY(weights[k + shift + n_qubits], wires=[qubit_1 + qubit_offset, qubit_2 + qubit_offset])
            count += 1

        for k in range(n_qubits - 1):
            control_qubit = k
            target_qubit = k + 1
            qml.CRY(weights[k + shift + n_qubits], wires=[control_qubit + qubit_offset, target_qubit + qubit_offset])
            count += 1

def qnode_fn(weights):
    qubit_offset = 0
    weights_idx = 0
    numerical_qubits= int(register_spec[0][1:])
    # TODO  take the register spec and order in  first with  numeric  then catagorial and then sort the cat registers largest  to smallest.
    # Use this order to combine the circuit.
    # Use as many standard circuit layers as needed for the largest cat, and use numerial  register as control qubits for largest cat.
    # Then for each cat register  condition each single excitment gate on a control qubit in the next register  (n-1 control quibts needed per layer)
    for idx, entry in enumerate(register_spec):
        ## Check if string ends with "c"
        n_qubits_subcircuit = int(entry[1:])
        # if len(entry) != 2:
        #     raise ValueError(f"Invalid entry {entry} encountered in register_spec. Must be of the form 'nX' or 'cX',"
        #                      + f"where X stands for the number of qubits and c stands for categorical, and n for "
        #                      + f"numerical registers.")
        if entry[0] == "n":
            # TODO: Check actual numbers
            standard_subcircuit(
                weights[weights_idx:weights_idx + 36 * circuit_depth],
                n_qubits_subcircuit,
                circuit_depth=circuit_depth,
                qubit_offset=qubit_offset,
            )
            weights_idx += 36 * circuit_depth
        elif entry[0] == "c":
            # apply categorical subcicuit?
            # categorical_subcircuit(
            #     weights[weights_idx:weights_idx + n_qubits_subcircuit - 1],
            #     n_qubits_subcircuit,
            #     qubit_offset=qubit_offset,
            # )
            #weights_idx += n_qubits_subcircuit - 1

            # apply conditional single excitation gates on categorical variables conditioned on numerical register
            qml.RX(jnp.pi, wires=qubit_offset)

            # no of conditional single excitation or subcircuits can be derived from number of numerical qubits
            max_index_current_register = qubit_offset + n_qubits_subcircuit # calculate the limit till current register to avoid index error
            no_of_subcircuits = n_qubits_subcircuit - 1 # number of conditional single excitations required to entangle categorical register with the numerical register

            # number of conditional single excitations which will be left. standard circuits needs to be inserted
            left_over_circuits = max(0, no_of_subcircuits - numerical_qubits)

            while(qubit_offset < max_index_current_register and left_over_circuits >= 0 ):
                # calculate the gates and apply the gates on the pair
                for i in range( no_of_subcircuits - left_over_circuits):
                    qml.ctrl(qml.SingleExcitation, control= i)(weights[weights_idx:weights_idx + n_qubits_subcircuit - 1],wires=[qubit_offset, qubit_offset + 1])
                weights_idx += n_qubits_subcircuit - 1
                # apply standard circuit between interleaving qubits
                if left_over_circuits > 0:
                    standard_subcircuit(
                        weights[weights_idx:weights_idx + 36 * circuit_depth],
                        numerical_qubits,
                        circuit_depth=circuit_depth,
                        qubit_offset=0,
                    )
                    # update the
                no_of_subcircuits = left_over_circuits
                left_over_circuits = no_of_subcircuits - numerical_qubits
        else:
            raise ValueError(f"Invalid entry {entry} encountered in register_spec. Must start with 'c' or 'n'")
        qubit_offset = qubit_offset + n_qubits_subcircuit
    return qml.sample()


dummy_device = qml.device("default.qubit", wires=n_qubits, shots=1)
dummy_qnode = qml.QNode(qnode_fn, dummy_device, diff_method=None)
qml.draw_mpl(dummy_qnode, expansion_strategy="device")(weights)
plt.show()