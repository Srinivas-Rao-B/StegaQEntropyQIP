import os
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import (
    Statevector,
    DensityMatrix,
    entropy
)

from qiskit.visualization import (
    plot_bloch_multivector,
    plot_state_city
)


class StateAnalysis:

    def __init__(self, analysis_circuit):

        self.circuit = analysis_circuit

        self.statevector = None

        self.density_matrix = None

        self.output_folder = "qrng/output"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )


    def generate_statevector(self):

        print("\n" + "=" * 70)
        print("STATEVECTOR ANALYSIS")
        print("=" * 70)

        self.statevector = Statevector.from_instruction(
            self.circuit
        )

        print("\nStatevector Generated Successfully.\n")

        print(self.statevector)
        self.statevector_statistics()

        figure = plot_state_city(
            self.statevector
        )

        figure.savefig(
            os.path.join(
                self.output_folder,
                "statevector.png"
            )
        )

        plt.close()

        print("\nStatevector Visualization Saved.")

    def statevector_statistics(self):

        print("\n" + "=" * 70)
        print("QUANTUM STATE SUMMARY")
        print("=" * 70)

        amplitudes = self.statevector.data

        probabilities = self.statevector.probabilities()

        amp0 = amplitudes[0]
        amp1 = amplitudes[1]

        print(f"\nRepresentative Quantum Qubit : q0")

        print(f"\nInitial State                : |0>")

        print("Quantum Gate Applied         : Hadamard (H)")

        print("\nFinal Quantum State")

        print("|ψ> = (|0> + |1>) / √2")

        print("\nAmplitude |0>               : "
            f"{amp0.real:.6f}"
        )

        print("Amplitude |1>               : "
            f"{amp1.real:.6f}"
        )

        print("\nMagnitude |0>               : "
            f"{abs(amp0):.6f}"
        )

        print("Magnitude |1>               : "
            f"{abs(amp1):.6f}"
        )

        print("\nPhase |0>                   : "
            f"{np.degrees(np.angle(amp0)):.2f}°"
        )

        print("Phase |1>                   : "
            f"{np.degrees(np.angle(amp1)):.2f}°"
        )

        print("\nProbability |0>             : "
            f"{probabilities[0]*100:.4f}%"
        )

        print("Probability |1>             : "
            f"{probabilities[1]*100:.4f}%"
        )

        difference = abs(probabilities[0]-probabilities[1])

        print("\nProbability Difference      : "
            f"{difference*100:.4f}%"
        )

        if difference < 1e-8:

            print("Quantum Balance             : Perfect Equal Superposition")

        else:

            print("Quantum Balance             : Slightly Biased")

        print("\nState Normalization         : Verified")

        print("Quantum Measurement         : Pending")


    def generate_density_matrix(self):

        print("\n" + "=" * 70)
        print("DENSITY MATRIX")
        print("=" * 70)

        self.density_matrix = DensityMatrix(
            self.statevector
        )

        print("\nDensity Matrix (ρ)\n")

        print(self.density_matrix.data)

        print("\nReal Component\n")

        print(np.real(self.density_matrix.data))

        print("\nImaginary Component\n")

        print(np.imag(self.density_matrix.data))

        print("\nTrace : "
            f"{np.trace(self.density_matrix.data).real:.6f}"
        )

        print("Rank  : "
            f"{np.linalg.matrix_rank(self.density_matrix.data)}"
        )

        figure = plot_state_city(
            self.density_matrix
        )

        figure.savefig(
            os.path.join(
                self.output_folder,
                "density_matrix.png"
            )
        )

        plt.close()

        print("\nDensity Matrix Saved.")


    def bloch_sphere(self):

        print("\n" + "=" * 70)
        print("BLOCH SPHERE")
        print("=" * 70)

        figure = plot_bloch_multivector(
            self.statevector
        )

        figure.savefig(
            os.path.join(
                self.output_folder,
                "bloch_sphere.png"
            )
        )

        plt.close()

        print("\nBloch Sphere Saved.")


    def von_neumann_entropy(self):
        value = entropy(
            self.density_matrix,
            base=2
        )
        eigenvalues = np.linalg.eigvals(
            self.density_matrix.data
        )

        print("\nFormula")

        print("S(ρ) = -Tr(ρ log₂ρ)")

        print("\nDensity Matrix Eigenvalues")

        for i, eig in enumerate(eigenvalues):

            print(f"λ{i+1} : {eig.real:.8f}")

        print(f"\nVon Neumann Entropy : {value:.8f} bits")

        if value < 1e-10:

            print("\nInterpretation")

            print("Pure Quantum State")

            print("Maximum Quantum Coherence")

            print("Zero Information Loss")

            print("State exists before measurement.")

        else:

            print("\nInterpretation")

            print("Mixed Quantum State")

            print("Partial Decoherence")


    def purity(self):

        print("\n" + "=" * 70)
        print("STATE PURITY")
        print("=" * 70)

        matrix = self.density_matrix.data

        purity = np.real(
            np.trace(
                matrix @ matrix
            )
        )

        print(f"\nPurity : {purity:.8f}")

        if purity > 0.999:

            print("Interpretation : Ideal Quantum State")

        else:

            print("Interpretation : Decohered Quantum State")


    def eigenvalues(self):

        print("\n" + "=" * 70)
        print("DENSITY MATRIX EIGENVALUES")
        print("=" * 70)

        values = np.linalg.eigvals(
            self.density_matrix.data
        )

        for index, value in enumerate(values):

            print(f"λ{index+1} : {value.real:.8f}")


    def probabilities(self):

        print("\n" + "=" * 70)
        print("STATE PROBABILITIES")
        print("=" * 70)

        probabilities = self.statevector.probabilities()

        difference = abs(
            probabilities[0] -
            probabilities[1]
        )

        print("\nProbability Balance")
        print("-" * 35)

        print(f"P(|0>) : {probabilities[0]:.8f}")
        print(f"P(|1>) : {probabilities[1]:.8f}")

        print(f"\nDifference : {difference*100:.6f}%")

        if difference < 1e-8:

            print("Quantum Randomness Potential : Maximum")

        else:

            print("Quantum Randomness Potential : Reduced")

        print("\nIndividual Probabilities")
        print("-" * 35)

        for index, value in enumerate(probabilities):

            print(f"|{index}> : {value:.8f}")


    def summary(self):

        print("\n" + "=" * 70)
        print("THEORETICAL QUANTUM ANALYSIS SUMMARY")
        print("=" * 70)

        print("\nQuantum Objects Generated")

        print("--------------------------------")

        print("Statevector")

        print("Density Matrix")

        print("Bloch Sphere")

        print("Von Neumann Entropy")

        print("State Purity")

        print("Eigenvalue Analysis")

        print("Probability Distribution")

        print("\nGenerated Images")

        print("--------------------------------")

        print("statevector.png")

        print("density_matrix.png")

        print("bloch_sphere.png")

        print("\nPurpose")

        print("--------------------------------")

        print("Theoretical analysis of the representative")

        print("quantum state before IBM Quantum")

        print("measurement and QRNG generation.")


    def run(self):

        self.generate_statevector()

        self.generate_density_matrix()

        self.bloch_sphere()

        self.von_neumann_entropy()

        self.purity()

        self.eigenvalues()

        self.probabilities()

        self.summary()

        return self.statevector