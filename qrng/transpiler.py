import os
import matplotlib.pyplot as plt

from qiskit import transpile


class QuantumTranspiler:

    def __init__(self, backend, circuit):

        self.backend = backend

        self.logical_circuit = circuit

        self.transpiled_circuit = None

        self.output_folder = "qrng/output"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )


    def transpile_circuit(self):

        print("\n" + "=" * 70)
        print("IBM QUANTUM TRANSPILATION")
        print("=" * 70)

        print("\nTranspiling Logical Circuit...\n")

        self.transpiled_circuit = transpile(

            self.logical_circuit,

            backend=self.backend,

            optimization_level=3

        )

        print("Circuit Successfully Transpiled.")

        return self.transpiled_circuit


    def transpilation_statistics(self):

        print("\n" + "=" * 70)
        print("TRANSPILATION STATISTICS")
        print("=" * 70)

        print(f"\nCircuit Depth             : {self.transpiled_circuit.depth()}")

        print(f"Circuit Width             : {self.transpiled_circuit.width()}")

        print(f"Circuit Size              : {self.transpiled_circuit.size()}")

        print(f"Number of Qubits          : {self.transpiled_circuit.num_qubits}")

        print(f"Number of Classical Bits  : {self.transpiled_circuit.num_clbits}")

        print()


    def native_gate_statistics(self):

        print("\n" + "=" * 70)
        print("NATIVE GATE ANALYSIS")
        print("=" * 70)

        counts = self.transpiled_circuit.count_ops()

        total = 0

        print()

        for gate, value in counts.items():

            print(f"{gate:<20}{value}")

            total += value

        print()

        print(f"Total Native Gates : {total}")


    def print_transpiled_circuit(self):

        print("\n" + "=" * 70)
        print("TRANSPILED CIRCUIT")
        print("=" * 70)

        print()

        print("Transpiled_circuit Generated Successfully.\n")


    def save_transpiled_circuit(self):

        print("\nSaving Transpiled Circuit Diagram...")

        figure = self.transpiled_circuit.draw(
            output="mpl"
        )

        figure.savefig(

            os.path.join(

                self.output_folder,

                "transpiled_circuit.png"

            )

        )

        plt.close()

        print("Saved : transpiled_circuit.png")


    def compare_circuits(self):

        print("\n" + "=" * 70)
        print("CIRCUIT COMPARISON")
        print("=" * 70)

        print(f"\nLogical Depth      : {self.logical_circuit.depth()}")

        print(f"Hardware Depth     : {self.transpiled_circuit.depth()}")

        print()

        print(f"Logical Size       : {self.logical_circuit.size()}")

        print(f"Hardware Size      : {self.transpiled_circuit.size()}")

        print()

        print("Logical Circuit")

        print("↓")

        print("IBM Transpiler")

        print("↓")

        print("Hardware Native Circuit")


    def summary(self):

        print("\n" + "=" * 70)
        print("TRANSPILATION COMPLETED")
        print("=" * 70)

        print("\nGenerated Files")

        print("----------------------------")

        print("transpiled_circuit.png")

        print()

        print("Circuit Ready For IBM Quantum Execution.")


    def run(self):

        self.transpile_circuit()

        self.transpilation_statistics()

        self.native_gate_statistics()

        self.print_transpiled_circuit()

        self.save_transpiled_circuit()

        self.compare_circuits()

        self.summary()

        return self.transpiled_circuit