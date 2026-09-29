import os
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


class QuantumCircuitManager:

    def __init__(self, backend):

        self.backend = backend

        self.num_qubits = min(32, backend.num_qubits)
        self.total_backend_qubits = backend.num_qubits

        self.hardware_utilization = (
            self.num_qubits /
            self.total_backend_qubits
        ) * 100

        self.logical_gate_count = 0

        self.circuit_count = 4

        self.logical_circuit = None
        self.measurement_circuit = None
        self.analysis_circuit = None

        self.output_folder = "qrng/output"

        os.makedirs(self.output_folder, exist_ok=True)


    def create_logical_qrng_circuit(self):

        print("\n" + "=" * 70)
        print("LOGICAL QRNG CIRCUIT")
        print("=" * 70)

        self.logical_circuit = QuantumCircuit(self.num_qubits)

        for qubit in range(self.num_qubits):
            self.logical_circuit.h(qubit)
        self.logical_gate_count = self.num_qubits

        print("\nLogical QRNG Circuit Created Successfully.\n")

        print(self.logical_circuit)

        return self.logical_circuit


    def create_measurement_circuit(self):

        print("\nCreating Measurement Circuit...")

        self.measurement_circuit = self.logical_circuit.copy()

        self.measurement_circuit.measure_all()

        print("Measurement Gates Added Successfully.\n")

        print(self.measurement_circuit)

        return self.measurement_circuit


    def create_theoretical_circuit(self):

        print("\nCreating Theoretical Analysis Circuit...")

        print("Using 1-Qubit Circuit for Bloch Sphere and State Analysis.")

        self.analysis_circuit = QuantumCircuit(1)

        self.analysis_circuit.h(0)

        print("Theoretical Circuit Created Successfully.")

        return self.analysis_circuit


    def logical_analysis(self):

        print("\n" + "=" * 70)
        print("LOGICAL CIRCUIT ANALYSIS")
        print("=" * 70)

        print(f"\nNumber of Qubits      : {self.measurement_circuit.num_qubits}")
        print(f"Classical Bits        : {self.measurement_circuit.num_clbits}")
        print(f"Circuit Depth         : {self.measurement_circuit.depth()}")
        print(f"Circuit Width         : {self.measurement_circuit.width()}")
        print(f"Circuit Size          : {self.measurement_circuit.size()}")

        print("\nGate Counts")
        print("-" * 50)

        counts = self.measurement_circuit.count_ops()

        total = 0

        for gate, value in counts.items():

            print(f"{gate:<20}{value}")

            total += value

        print("-" * 50)
        print(f"Total Gates           : {total}")


    def save_circuit_images(self):

        print("\nSaving Circuit Diagrams...")

        figure = self.logical_circuit.draw(output="mpl")

        figure.savefig(
            os.path.join(
                self.output_folder,
                "logical_qrng_circuit.png"
            )
        )

        plt.close()

        figure = self.measurement_circuit.draw(output="mpl")

        figure.savefig(
            os.path.join(
                self.output_folder,
                "measurement_qrng_circuit.png"
            )
        )

        plt.close()

        print("Logical Circuit Saved.")
        print("Measurement Circuit Saved.")


    def print_workflow(self):

        print("\n" + "=" * 70)
        print("QUANTUM CIRCUIT WORKFLOW")
        print("=" * 70)

        print()
        print(f"{self.num_qubits} Qubits")
        print("        │")
        print("        ▼")
        print(f"{self.num_qubits} Hadamard Gates")
        print("        │")
        print("        ▼")
        print("Superposition")
        print("        │")
        print("        ▼")
        print(f"{self.num_qubits} Measurements")
        print("        │")
        print("        ▼")
        print(f"{self.num_qubits} Quantum Random Bits / Shot")
        print()


    def summary(self):

        print("\n" + "=" * 70)
        print("QUANTUM CIRCUIT SUMMARY")
        print("=" * 70)

        print(f"\nLogical Qubits            : {self.num_qubits}")
        print(f"Random Bits / Shot        : {self.num_qubits}")
        print("Superposition Gates       : Hadamard")
        print("Measurement               : Computational Basis")
        print("Circuit Status            : Ready for IBM Transpilation")


    def run(self):

        self.create_logical_qrng_circuit()

        self.create_measurement_circuit()

        self.create_theoretical_circuit()

        self.logical_analysis()

        self.save_circuit_images()

        self.print_workflow()

        self.summary()

        return (
            self.analysis_circuit,
            self.measurement_circuit
        )