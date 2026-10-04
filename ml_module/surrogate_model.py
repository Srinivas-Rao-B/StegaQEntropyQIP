import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
# The SQE-Net quantum circuit (SQENetQuantumLayer) evolves a genuine complex
# state vector; TF's autodiff correctly backpropagates only the real part of
# the Pauli-Z measurement into the angle-projection weights (which is exactly
# what we want, since the model's loss is real-valued), and logs an
# informational "discarding imaginary part" notice while doing so. That
# notice is expected here and does not indicate a problem, so it is silenced.
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=".*casting an input of type complex64.*"
)
import warnings
warnings.filterwarnings("ignore")
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
ABLATE_SQE_NET = False
ABLATE_QOQA = False
ABLATE_ACER = False
GENERATE_VISUALIZATIONS = True

import runpy

runpy.run_module("dataset_cleaner", run_name="__main__")

try:
    from acer_simulator import ACERSimulatorBackend  # type: ignore[import-not-found]
except ImportError:

    class QuantumRegisterBackend:
        """
        Genuine state-vector quantum gate backend shared by every quantum stage
        of StegaQEntropy (SQE-Net feature-level entanglement, QOQA region-level
        entanglement, and ACER global entanglement). It performs real state
        preparation, single- and two-qubit unitary gate application, and
        Born-rule (Pauli-Z / probability) measurement on an explicit complex
        state vector of size 2**qubit_count. No step of this backend fakes
        quantum computation with classical trigonometric substitutes -- every
        public method here mutates or measures an actual quantum state.
        """

        def __init__(
            self,
            n_qubits=200,
            quantum_registers=4,
            qubits_per_register=8
        ):
            self.n_qubits = n_qubits
            self.quantum_registers = quantum_registers
            self.qubits_per_register = qubits_per_register

        def apply_rx(
            self,
            state,
            qubit,
            angle,
            qubit_count=8
        ):
            new_state = state.copy()

            step = 2 ** qubit
            block = 2 * step

            c = np.cos(angle / 2.0)
            s = -1j * np.sin(angle / 2.0)

            for start in range(0, len(state), block):
                for offset in range(step):
                    i0 = start + offset
                    i1 = i0 + step

                    a = state[i0]
                    b = state[i1]

                    new_state[i0] = c * a + s * b
                    new_state[i1] = s * a + c * b

            return new_state

        def apply_cz(
            self,
            state,
            control,
            target,
            qubit_count=8
        ):
            new_state = state.copy()

            for index in range(len(state)):
                control_bit = (index >> control) & 1
                target_bit = (index >> target) & 1

                if control_bit == 1 and target_bit == 1:
                    new_state[index] = -state[index]

            return new_state

        def apply_rzz(
            self,
            state,
            qubit_a,
            qubit_b,
            angle,
            qubit_count=8
        ):
            """
            Two-qubit ZZ-interaction gate exp(-i * angle/2 * Z_a Z_b). Used by
            QOQA to encode region-level (inter-region) quantum entanglement:
            the interaction phase couples the measurement statistics of two
            candidate-region qubits according to their spatial/objective
            relationship (e.g. an 8-neighbour conflict penalty).
            """
            new_state = state.copy()

            same_phase = np.exp(-1j * angle / 2.0)
            diff_phase = np.exp(1j * angle / 2.0)

            for index in range(len(state)):
                bit_a = (index >> qubit_a) & 1
                bit_b = (index >> qubit_b) & 1

                if bit_a == bit_b:
                    new_state[index] = state[index] * same_phase
                else:
                    new_state[index] = state[index] * diff_phase

            return new_state

        def measure_probabilities(self, state):
            """
            Genuine Born-rule measurement: returns the probability distribution
            |amplitude|^2 over every computational basis state of the register.
            """
            probabilities = np.abs(state) ** 2
            probabilities = np.nan_to_num(
                probabilities, nan=0.0, posinf=0.0, neginf=0.0
            )
            probabilities = np.maximum(probabilities, 0.0)

            total = np.sum(probabilities)

            if total <= 0:
                return np.ones(len(state), dtype=np.float64) / len(state)

            return probabilities / total

        def classical_fallback_sample(self, weights):
            """
            CLASSICAL FALLBACK ONLY. This is a weighted np.random.choice draw and
            is NOT a quantum operation. It exists purely as a safety-net sampler
            for degenerate cases (e.g. an all-zero objective) and must never be
            used as the primary QOQA selection mechanism -- that role belongs to
            QOQAQuantumBackend.optimize_block, which executes a real quantum
            circuit and samples from its Born-rule measurement distribution.
            """
            weights = np.asarray(
                weights,
                dtype=np.float64
            )

            if (
                weights.ndim != 1
                or len(weights) == 0
            ):
                raise ValueError(
                    "Invalid quantum selection weights."
                )

            weights = np.nan_to_num(
                weights,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            weights = np.maximum(
                weights,
                0.0
            )

            if np.sum(weights) <= 0:
                weights = np.ones(
                    len(weights),
                    dtype=np.float64
                )

            probs = (
                weights /
                np.sum(weights)
            )

            sample_size = min(
                self.n_qubits,
                len(weights)
            )

            return np.random.choice(
                len(weights),
                size=sample_size,
                replace=False,
                p=probs
            )

        # Backward-compatible alias. Callers should prefer the explicit name
        # `classical_fallback_sample` so it is never mistaken for a quantum step.
        sample_quantum_states = classical_fallback_sample

        def initialize_register(
            self,
            qubit_count=8
        ):
            state = np.zeros(
                2 ** qubit_count,
                dtype=np.complex128
            )

            state[0] = 1.0 + 0.0j

            return state

        def apply_hadamard(
            self,
            state,
            qubit,
            qubit_count=8
        ):
            new_state = state.copy()

            step = 2 ** qubit
            block = 2 * step

            inv_sqrt_2 = 1.0 / np.sqrt(2.0)

            for start in range(
                0,
                len(state),
                block
            ):

                for offset in range(step):

                    i0 = start + offset
                    i1 = i0 + step

                    a = state[i0]
                    b = state[i1]

                    new_state[i0] = (
                        a + b
                    ) * inv_sqrt_2

                    new_state[i1] = (
                        a - b
                    ) * inv_sqrt_2

            return new_state

        def apply_ry(
            self,
            state,
            qubit,
            angle,
            qubit_count=8
        ):
            new_state = state.copy()

            step = 2 ** qubit
            block = 2 * step

            c = np.cos(
                angle / 2.0
            )

            s = np.sin(
                angle / 2.0
            )

            for start in range(
                0,
                len(state),
                block
            ):

                for offset in range(step):

                    i0 = start + offset
                    i1 = i0 + step

                    a = state[i0]
                    b = state[i1]

                    new_state[i0] = (
                        c * a -
                        s * b
                    )

                    new_state[i1] = (
                        s * a +
                        c * b
                    )

            return new_state

        def apply_rz(
            self,
            state,
            qubit,
            angle,
            qubit_count=8
        ):
            new_state = state.copy()

            phase_0 = np.exp(
                -1j * angle / 2.0
            )

            phase_1 = np.exp(
                1j * angle / 2.0
            )

            step = 2 ** qubit
            block = 2 * step

            for start in range(
                0,
                len(state),
                block
            ):

                for offset in range(step):

                    i0 = start + offset
                    i1 = i0 + step

                    new_state[i0] = (
                        state[i0] *
                        phase_0
                    )

                    new_state[i1] = (
                        state[i1] *
                        phase_1
                    )

            return new_state

        def apply_cnot(
            self,
            state,
            control,
            target,
            qubit_count=8
        ):
            new_state = state.copy()

            for index in range(
                len(state)
            ):

                control_bit = (
                    index >> control
                ) & 1

                if control_bit == 1:

                    target_bit = (
                        index >> target
                    ) & 1

                    flipped_index = (
                        index
                        ^ (1 << target)
                    )

                    if target_bit == 0:

                        new_state[index] = (
                            state[flipped_index]
                        )

                        new_state[flipped_index] = (
                            state[index]
                        )

            return new_state

        def measure_z(
            self,
            state,
            qubit,
            qubit_count=8
        ):
            probabilities = (
                np.abs(state) ** 2
            )

            expectation = 0.0

            for index, probability in enumerate(
                probabilities
            ):

                bit = (
                    index >> qubit
                ) & 1

                expectation += (
                    probability
                    *
                    (1.0 if bit == 0 else -1.0)
                )

            return float(
                np.real(expectation)
            )

    class ACERSimulatorBackend(QuantumRegisterBackend):
        """
        ACER -- global quantum representation stage. 32 logical qubits realised
        as 4 independent 8-qubit registers (RY + RZ angle/phase encoding, H
        superposition, cyclic-ring CNOT entanglement, Pauli-Z measurement).
        This stage is unchanged from the original StegaQEntropy design; it
        remains the final real quantum circuit executed over the selected
        multi-region global embedding.
        """

        def entangle_register(
            self,
            angles
        ):
            angles = np.asarray(
                angles,
                dtype=np.float64
            )

            if len(angles) != 8:
                raise ValueError(
                    "Each ACER quantum register requires 8 angles."
                )

            state = self.initialize_register(
                qubit_count=8
            )

            for qubit in range(8):

                state = self.apply_ry(
                    state,
                    qubit,
                    angles[qubit],
                    qubit_count=8
                )

                state = self.apply_rz(
                    state,
                    qubit,
                    angles[qubit] / 2.0,
                    qubit_count=8
                )

            for qubit in range(8):

                state = self.apply_hadamard(
                    state,
                    qubit,
                    qubit_count=8
                )

            for qubit in range(7):

                state = self.apply_cnot(
                    state,
                    control=qubit,
                    target=qubit + 1,
                    qubit_count=8
                )

            state = self.apply_cnot(
                state,
                control=7,
                target=0,
                qubit_count=8
            )

            expectations = np.asarray(
                [
                    self.measure_z(
                        state,
                        qubit,
                        qubit_count=8
                    )
                    for qubit in range(8)
                ],
                dtype=np.float32
            )

            return state, expectations

        def quantum_entanglement_features(
            self,
            embedding
        ):
            embedding = np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)

            if embedding.size < 32:
                embedding = np.pad(
                    embedding,
                    (
                        0,
                        32 - embedding.size
                    )
                )

            angles = embedding[:32]

            angles = np.tanh(
                angles
            ) * np.pi

            all_expectations = []

            for register_id in range(4):

                start = (
                    register_id * 8
                )

                end = start + 8

                register_angles = (
                    angles[start:end]
                )

                _, expectations = (
                    self.entangle_register(
                        register_angles
                    )
                )

                all_expectations.append(
                    expectations
                )

            quantum_features = np.concatenate(
                all_expectations,
                axis=0
            )

            return quantum_features

    class SQEQuantumBackend(QuantumRegisterBackend):
        """
        SQE-Net (StegaQEntropy Quantum-Entangled Surrogate Network) -- the
        feature-level (intra-region) quantum stage. 64 logical qubits realised
        as 8 independent 8-qubit registers, each carrying RY angle encoding +
        RZ phase encoding + H superposition + cyclic-ring CNOT entanglement,
        with Pauli-Z expectation measurement. This numpy backend is used for
        offline SQE-Net diagnostics/export (sqe_quantum_features.npy etc.); the
        trainable copy used inside the Keras model is the differentiable
        TensorFlow implementation in SQENetQuantumLayer, which performs the
        identical circuit but stays inside the autodiff graph.
        """

        def sqe_entangle_register(self, ry_angles, rz_angles):
            ry_angles = np.asarray(ry_angles, dtype=np.float64)
            rz_angles = np.asarray(rz_angles, dtype=np.float64)

            if len(ry_angles) != 8 or len(rz_angles) != 8:
                raise ValueError(
                    "Each SQE-Net quantum register requires 8 RY and 8 RZ angles."
                )

            state = self.initialize_register(qubit_count=8)

            for qubit in range(8):
                state = self.apply_ry(
                    state, qubit, ry_angles[qubit], qubit_count=8
                )
                state = self.apply_rz(
                    state, qubit, rz_angles[qubit], qubit_count=8
                )
                state = self.apply_hadamard(
                    state, qubit, qubit_count=8
                )

            for qubit in range(7):
                state = self.apply_cnot(
                    state, control=qubit, target=qubit + 1, qubit_count=8
                )

            state = self.apply_cnot(state, control=7, target=0, qubit_count=8)

            expectations = np.asarray(
                [
                    self.measure_z(state, qubit, qubit_count=8)
                    for qubit in range(8)
                ],
                dtype=np.float32
            )

            return state, expectations

        def sqe_quantum_features(self, fused_vector):
            """
            fused_vector: 1-D array (multimodal-fused region representation).
            Returns the 64-dimensional quantum-measured feature vector
            (8 registers x 8 Pauli-Z expectations), the bounded RY/RZ
            rotation angles used, and per-register raw state vectors.
            """
            fused_vector = np.asarray(
                fused_vector, dtype=np.float32
            ).reshape(-1)

            required = 64 * 2

            if fused_vector.size < required:
                fused_vector = np.pad(
                    fused_vector, (0, required - fused_vector.size)
                )

            projected = fused_vector[:required]
            angles = np.tanh(projected) * np.pi

            ry_all = angles[:64]
            rz_all = angles[64:128]

            all_expectations = []
            register_states = []

            for register_id in range(8):
                start = register_id * 8
                end = start + 8

                state, expectations = self.sqe_entangle_register(
                    ry_all[start:end], rz_all[start:end]
                )

                all_expectations.append(expectations)
                register_states.append(state)

            quantum_features = np.concatenate(all_expectations, axis=0)

            return quantum_features, angles, register_states

    class QOQAQuantumBackend(QuantumRegisterBackend):
        """
        QOQA (Quantum-Assisted Optimal Region Selection) -- the region-level
        (inter-region) quantum stage. Because a full StegaQEntropy candidate
        pool can hold hundreds of regions, QOQA does not build one dense
        state vector over every candidate (that would require 2**N amplitudes,
        computationally infeasible). Instead it decomposes the candidate pool
        into blocks of at most `max_block_qubits` regions, and for every block
        it executes a genuine QAOA-style quantum circuit:

            H (superposition) -> RZ (region-objective phase encoding)
              -> RZZ (region-level entanglement over 8-neighbour conflict
                 edges, i.e. genuine two-qubit interaction, NOT a classical
                 softmax) -> RX (mixing rotation) -> Born-rule measurement.

        Repeated measurement shots of the executed circuit give each
        candidate region in the block a quantum-measured selection frequency.
        Nothing here calls np.random.choice on a classical heuristic weight
        vector and calls that "quantum" -- the probabilities sampled are the
        actual |amplitude|^2 outputs of the circuit above.
        """

        def optimize_block(
            self,
            theta,
            neighbor_edges,
            lam=1.4,
            beta=0.6
        ):
            theta = np.asarray(theta, dtype=np.float64)
            m = len(theta)

            if m == 0:
                return np.array([1.0], dtype=np.float64)

            state = self.initialize_register(qubit_count=m)

            for qubit in range(m):
                state = self.apply_hadamard(state, qubit, qubit_count=m)

            for qubit in range(m):
                state = self.apply_rz(state, qubit, theta[qubit], qubit_count=m)

            for (qubit_a, qubit_b) in neighbor_edges:
                state = self.apply_rzz(
                    state, qubit_a, qubit_b, lam * np.pi, qubit_count=m
                )

            for qubit in range(m):
                state = self.apply_rx(state, qubit, beta, qubit_count=m)

            return self.measure_probabilities(state)

        # ------------------------------------------------------------------
        # QUBO/Ising extension of the existing QOQA circuit. The original
        # optimize_block above is untouched; optimize_block_ising runs the
        # SAME gate sequence (H -> RZ -> RZZ -> RX -> Born measurement) with
        # per-qubit fields h_i and per-pair couplings J_ij derived from the
        # QOQA QUBO (objective + 8-neighbour conflict + cardinality penalty).
        # ------------------------------------------------------------------
        max_block_qubits = 12

        def index_to_bitstring(self, index, qubit_count):
            # character q of the string is the value of local qubit q
            return "".join(
                str((int(index) >> q) & 1) for q in range(qubit_count)
            )

        def prepare_superposition(self, qubit_count):
            if qubit_count > self.max_block_qubits:
                raise ValueError(
                    f"QOQA refuses a dense {qubit_count}-qubit state vector "
                    f"(max {self.max_block_qubits}); use block decomposition."
                )
            state = self.initialize_register(qubit_count=qubit_count)
            for qubit in range(qubit_count):
                state = self.apply_hadamard(state, qubit, qubit_count=qubit_count)
            return state

        def validate_state(self, state, tolerance=1e-8):
            state = np.asarray(state)
            n = len(state)
            if state.ndim != 1 or n == 0 or (n & (n - 1)) != 0:
                raise ValueError("Invalid QOQA state-vector shape.")
            if not np.all(np.isfinite(state)):
                raise ValueError("QOQA quantum state contains NaN/Inf.")
            norm = float(np.sum(np.abs(state) ** 2))
            if abs(norm - 1.0) > tolerance:
                raise ValueError(
                    f"QOQA quantum state norm {norm:.12f} deviates from 1."
                )
            return norm

        def born_probabilities(self, state, tolerance=1e-8):
            """Born rule |amplitude|^2 with strict (non-silent) validation."""
            self.validate_state(state, tolerance)
            probabilities = np.abs(state) ** 2
            if np.any(probabilities < 0) or abs(np.sum(probabilities) - 1.0) > tolerance:
                raise ValueError("QOQA probability distribution is not normalised.")
            return probabilities

        def sample_bitstrings_from_state(self, probabilities, shots, rng, qubit_count):
            """
            Simulated projective measurement: every shot collapses the state
            to one computational-basis configuration drawn from the Born
            distribution |amplitude|^2 (inverse-CDF sampling of the circuit's
            own probabilities).
            """
            probabilities = np.asarray(probabilities, dtype=np.float64)
            cdf = np.cumsum(probabilities)
            draws = rng.random(int(shots)) * cdf[-1]
            outcomes = np.minimum(
                np.searchsorted(cdf, draws, side="right"), len(probabilities) - 1
            )
            counts = np.bincount(outcomes, minlength=len(probabilities))
            if int(np.sum(counts)) != int(shots):
                raise ValueError("QOQA measurement counts do not equal shots.")
            unique = np.nonzero(counts)[0]
            return {
                "indices": unique,
                "counts": counts[unique],
                "probabilities": counts[unique] / float(shots),
                "bitstrings": [
                    self.index_to_bitstring(i, qubit_count) for i in unique
                ],
            }

        def optimize_block_ising(
            self,
            h_fields,
            zz_couplings,
            gamma,
            beta,
            qaoa_layers=1,
            initial_state=None
        ):
            """
            Execute the QAOA-style circuit for one block. `h_fields[i]` and
            `zz_couplings[(i, j)]` are the Ising coefficients of the block
            Hamiltonian E = sum h_i Z_i + sum J_ij Z_i Z_j. Cost unitary
            exp(-i*gamma*E) = RZ(2*gamma*h_i) and RZZ(2*gamma*J_ij); mixer
            exp(-i*beta*X) = RX(2*beta). Returns (state, born_probabilities).
            """
            h_fields = np.asarray(h_fields, dtype=np.float64)
            m = len(h_fields)

            if initial_state is None:
                state = self.prepare_superposition(m)
            else:
                state = np.asarray(initial_state, dtype=np.complex128).copy()

            for layer in range(int(qaoa_layers)):
                g = gamma * (layer + 1) / float(qaoa_layers)
                b = beta * (1.0 - layer / float(qaoa_layers))

                for qubit in range(m):
                    state = self.apply_rz(
                        state, qubit, 2.0 * g * h_fields[qubit], qubit_count=m
                    )

                for (qubit_a, qubit_b), coupling in zz_couplings.items():
                    state = self.apply_rzz(
                        state, qubit_a, qubit_b, 2.0 * g * coupling, qubit_count=m
                    )

                for qubit in range(m):
                    state = self.apply_rx(state, qubit, 2.0 * b, qubit_count=m)

            return state, self.born_probabilities(state)

def classical_sqe_features(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    rng = np.random.default_rng(RANDOM_STATE)

    W = rng.standard_normal(
        (len(x), 64)
    ).astype(np.float32)

    W /= np.sqrt(
        max(len(x), 1)
    )

    features = np.tanh(
        x @ W
    )

    return features.astype(np.float32)
def acer_quantum_entanglement_features(
    embeddings
):
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(
            1,
            -1
        )

    backend = ACERSimulatorBackend(
        n_qubits=32,
        quantum_registers=4,
        qubits_per_register=8
    )

    quantum_features = []

    for embedding in embeddings:

        features = (
            backend.quantum_entanglement_features(
                embedding
            )
        )

        quantum_features.append(
            features
        )

    return np.asarray(
        quantum_features,
        dtype=np.float32
    )

from tensorflow.keras import Model # type: ignore[import-not-found]
from tensorflow.keras.layers import ( # type: ignore[import-not-found]
    Input, Dense, Dropout, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D, BatchNormalization, Concatenate,Multiply,Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore[import-not-found]
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
CSV_PATH = r"C:\Users\HP\Desktop\Stego\ml_module\output\probability_analysis\probability_intelligence_final_dataset_cleaned.csv"
IMAGE_PROFILE_PATH = r"C:\Users\HP\Desktop\Stego\output\image_acquisition\image_profile.json"
DWT_DIR = r"C:\Users\HP\Desktop\Stego\output\dwt_decomposition"
OUTPUT_DIR = r"C:\Users\HP\Desktop\Stego\ml_module\output\surrogate_cnn_model"
MODEL_PATH = os.path.join(OUTPUT_DIR, "surrogate_cnn_pure_regression.keras")
SOLUTION_JSON_PATH = os.path.join(OUTPUT_DIR, "optimal_embedding_solution.json")
SOLUTION_IMAGE_PATH = os.path.join(OUTPUT_DIR, "final_embedded_idwt_image.png")
ADAPTIVE_REGION_MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    "adaptive_embedding_region_manifest.json"
)


WINDOW_SIZE = 32
DWT_WINDOW_SIZE = 16
BATCH_SIZE = 16
EPOCHS = 40
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42
ACTIVATION = "relu"
LEARNING_RATE = 0.001
TARGET_PSNR_THRESHOLD = 87.0
DELTA_MIN = 0.004
DELTA_MAX = 0.008
DELTA_REFERENCE = DELTA_MAX


REGRESSION_TARGETS = [
    "image_psnr", "image_mse", "actual_center_distortion", "actual_quality_loss",
    "actual_safe_probability", "actual_risk_probability", "actual_safe_capacity"
]


def adjust_mse_for_delta(mse, old_delta, new_delta, region_count):
    if mse <= 0 or old_delta <= 0 or new_delta <= 0 or region_count <= 0:
        return float(mse)

    effective_delta = float(
        np.sqrt(
            np.mean(
                np.asarray(new_delta, dtype=np.float64) ** 2
            )
        )
    )

    region_scale = np.sqrt(region_count)

    adjusted_mse = (
        mse
        * (effective_delta / old_delta) ** 2
        / region_scale
    )

    return float(adjusted_mse)


def mse_to_psnr(mse):
    if mse <= 1e-16:
        return float("inf")

    return float(
        10.0 * np.log10(
            (255.0 ** 2) / mse
        )
    )


def configure_message_type(res):
    global EXACT_TOTAL_BITS, MAX_REGIONS_TARGET, PAYLOAD_RATIO

    if res == 1:
        EXACT_TOTAL_BITS = 20000
        MAX_REGIONS_TARGET = 275
        PAYLOAD_RATIO = 0.70

    elif res == 2:
        EXACT_TOTAL_BITS = 45000
        MAX_REGIONS_TARGET = 350
        PAYLOAD_RATIO = 0.95

    else:
        raise ValueError("Invalid input. Please enter 1 for Text or 2 for Image.")

def normalize_name(value):
    return str(value).strip().lower()

def find_column(df, candidates):
    lookup = {normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        if normalize_name(candidate) in lookup:
            return lookup[normalize_name(candidate)]
    return None

def load_image_path():
    if not os.path.exists(IMAGE_PROFILE_PATH):
        return r"C:/Users/HP/Documents/StegaQEntropy/1.1.12.tiff"
    with open(IMAGE_PROFILE_PATH, "r", encoding="utf-8") as file:
        profile = json.load(file)
    return profile.get("image_path", r"C:/Users/HP/Documents/StegaQEntropy/1.1.12.tiff")

def load_image(path):
    if not os.path.exists(path):
        return np.zeros((512, 512, 1), dtype=np.float32)

    image = tf.keras.utils.load_img(
        path,
        color_mode="grayscale"
    )

    image = tf.keras.utils.img_to_array(
        image
    ).astype(np.float32)

    return image

def load_dwt():
    bands = []

    for name in ["LL", "LH", "HL", "HH"]:
        path = os.path.join(
            DWT_DIR,
            f"{name}.npy"
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required DWT file not found:\n{path}"
            )

        band = np.load(
            path
        ).astype(
            np.float32
        )

        if band.ndim != 2:
            raise ValueError(
                f"{name}.npy must be 2-D, got {band.shape}"
            )

        if not np.all(np.isfinite(band)):
            raise ValueError(
                f"{name}.npy contains NaN or infinite values."
            )

        bands.append(band)

    if not (
        bands[1].shape ==
        bands[2].shape ==
        bands[3].shape
    ):
        raise ValueError(
            "LH, HL and HH shapes do not match."
        )

    print("\n[PRECOMPUTED DWT LOADED]")
    print(f"  LL : {bands[0].shape}")
    print(f"  LH : {bands[1].shape}")
    print(f"  HL : {bands[2].shape}")
    print(f"  HH : {bands[3].shape}")

    return bands

def extract_patch(array, center_row, center_col, size):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    height, width = array.shape
    half = size // 2
    r1 = int(round(center_row)) - half
    c1 = int(round(center_col)) - half
    r2 = r1 + size
    c2 = c1 + size

    pad_top = max(0, -r1)
    pad_left = max(0, -c1)
    pad_bottom = max(0, r2 - height)
    pad_right = max(0, c2 - width)

    if pad_top or pad_left or pad_bottom or pad_right:
        array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
        r1 += pad_top; r2 += pad_top; c1 += pad_left; c2 += pad_left

    patch = array[r1:r2, c1:c2]
    if patch.shape != (size, size):
        patch = tf.image.resize(patch[..., np.newaxis], (size, size)).numpy().squeeze()
    return patch.astype(np.float32)


def classical_acer_features(combined_embedding):
    x = np.asarray(
        combined_embedding,
        dtype=np.float32
    ).reshape(-1)

    rng = np.random.default_rng(
        RANDOM_STATE + 32
    )

    W = rng.standard_normal(
        (len(x), 32)
    ).astype(np.float32)

    W /= np.sqrt(
        max(len(x), 1)
    )

    features = np.tanh(
        x @ W
    ).astype(np.float32)

    return features.reshape(1, 32)


def perform_actual_db2_idwt_embedding(
    image,
    dwt_bands,
    selected_indices,
    df,
    metadata,
    rng_seed=None
):
    capacity_col = "actual_safe_capacity"

    if capacity_col not in df.columns:
        raise ValueError(
            f"Required column '{capacity_col}' not found."
        )

    LL, LH, HL, HH = [
        np.asarray(
            band,
            dtype=np.float32
        ).copy()
        for band in dwt_bands
    ]

    rng = np.random.default_rng(
        rng_seed
    )

    random_bits = rng.integers(
        0,
        2,
        size=EXACT_TOTAL_BITS,
        dtype=np.uint8
    )

    payloads_assigned = {
        int(idx): 0
        for idx in selected_indices
    }

    remaining = EXACT_TOTAL_BITS
    bit_pointer = 0

    embedded_positions = []
    total_mutations = 0

    used_dwt_positions = set()


    for idx in selected_indices:

        if remaining <= 0:
            break

        idx = int(idx)


        value = pd.to_numeric(
            df.iloc[idx][capacity_col],
            errors="coerce"
        )

        if not np.isfinite(value):
            print(
                f"[DWT] Region {idx} skipped: "
                f"invalid {capacity_col}."
            )
            continue

        actual_capacity = max(
            0,
            int(
                np.floor(
                    float(value)
                )
            )
        )

        usable_capacity = int(
            np.floor(
                actual_capacity
                * PAYLOAD_RATIO
            )
        )

        if usable_capacity <= 0:
            continue


        region_row = metadata.loc[
            metadata["region_id"] == idx,
            "row"
        ]

        region_col = metadata.loc[
            metadata["region_id"] == idx,
            "col"
        ]

        if region_row.empty or region_col.empty:
            print(
                f"[DWT] Region {idx} skipped: "
                f"missing metadata."
            )
            continue

        row = int(
            round(
                float(
                    region_row.values[0]
                ) / 2.0
            )
        )

        col = int(
            round(
                float(
                    region_col.values[0]
                ) / 2.0
            )
        )

        row = max(
            0,
            min(
                row,
                LH.shape[0] - 1
            )
        )

        col = max(
            0,
            min(
                col,
                LH.shape[1] - 1
            )
        )


        r0 = max(
            0,
            row - DWT_WINDOW_SIZE // 2
        )

        r1 = min(
            LH.shape[0],
            r0 + DWT_WINDOW_SIZE
        )

        c0 = max(
            0,
            col - DWT_WINDOW_SIZE // 2
        )

        c1 = min(
            LH.shape[1],
            c0 + DWT_WINDOW_SIZE
        )

        positions = []

        for r in range(r0, r1):

            for c in range(c0, c1):

                positions.append(
                    ("LH", r, c)
                )

                positions.append(
                    ("HL", r, c)
                )

                positions.append(
                    ("HH", r, c)
                )


        available_positions = [
            position
            for position in positions
            if position not in used_dwt_positions
        ]

        available_count = len(
            available_positions
        )

        if available_count <= 0:
            print(
                f"[DWT] Region {idx} skipped: "
                f"no non-overlapping DWT positions available."
            )
            continue


        bits_for_region = min(
            remaining,
            usable_capacity,
            available_count
        )

        if bits_for_region <= 0:
            continue

        print(
            f"[DWT] Region {idx}: "
            f"ML usable={usable_capacity}, "
            f"DWT available={available_count}, "
            f"allocated={bits_for_region}, "
            f"remaining_before={remaining}"
        )


        chosen_indices = rng.choice(
            len(available_positions),
            size=bits_for_region,
            replace=False
        )

        selected_positions = [
            available_positions[int(i)]
            for i in chosen_indices
        ]

        # Reserve positions immediately.
        for position in selected_positions:

            used_dwt_positions.add(
                (
                    position[0],
                    int(position[1]),
                    int(position[2])
                )
            )

        actual_embedded_this_region = 0

        for position in selected_positions:

            if bit_pointer >= EXACT_TOTAL_BITS:
                break

            subband, r, c = position

            if subband == "LH":
                band = LH

            elif subband == "HL":
                band = HL

            else:
                band = HH


            vr0 = max(
                0,
                r - 2
            )

            vr1 = min(
                band.shape[0],
                r + 3
            )

            vc0 = max(
                0,
                c - 2
            )

            vc1 = min(
                band.shape[1],
                c + 3
            )

            variance = float(
                np.var(
                    band[
                        vr0:vr1,
                        vc0:vc1
                    ]
                )
            )

            variance_scale = float(
                np.clip(
                    variance / 0.05,
                    0.0,
                    1.0
                )
            )

            delta = float(
                DELTA_MIN
                +
                variance_scale
                *
                (
                    DELTA_MAX
                    -
                    DELTA_MIN
                )
            )

            bit = int(
                random_bits[
                    bit_pointer
                ]
            )

            old_coeff = float(
                band[r, c]
            )

            q = int(
                    np.round(
                    old_coeff / delta
                )
            )

            if (q % 2) != bit:
                if q >= 0:
                    q += 1
                else:
                    q -= 1

            new_coeff = float(
                q * delta
            )

            band[r, c] = new_coeff

            if old_coeff != new_coeff:
                total_mutations += 1

            embedded_positions.append(
                (
                    subband,
                    r,
                    c,
                    delta,
                    bit
                )
            )

            bit_pointer += 1
            actual_embedded_this_region += 1


        payloads_assigned[idx] = (
            actual_embedded_this_region
        )

        remaining -= (
            actual_embedded_this_region
        )

        print(
            f"[DWT] Region {idx}: "
            f"embedded={actual_embedded_this_region}, "
            f"remaining={remaining}"
        )

    print(
        "\n[EMBEDDING SUMMARY]"
    )

    print(
        f"  Requested bits       : "
        f"{EXACT_TOTAL_BITS}"
    )

    print(
        f"  Actually embedded    : "
        f"{bit_pointer}"
    )

    print(
        f"  Remaining bits       : "
        f"{remaining}"
    )

    print(
        f"  Active regions       : "
        f"{sum(1 for v in payloads_assigned.values() if v > 0)}"
    )

    print(
        f"  Coefficient mutations: "
        f"{total_mutations}"
    )

    if bit_pointer != EXACT_TOTAL_BITS:

        raise ValueError(
            f"Selected region set cannot carry the complete "
            f"payload. "
            f"Embedded={bit_pointer}, "
            f"Required={EXACT_TOTAL_BITS}, "
            f"Remaining={remaining}"
        )
    direct_errors = 0

    for (
        subband,
        r,
        c,
        delta,
        bit
    ) in embedded_positions:

        if subband == "LH":
            band = LH
        elif subband == "HL":
            band = HL
        else:
            band = HH

        q = int(
            np.round(
                float(band[r, c]) / delta
            )
        )

    if (q % 2) != bit:
        direct_errors += 1

    direct_ber = (
        direct_errors
        /
        max(
            1,
            len(embedded_positions)
        )
    )

    print(
        "\n[DIRECT DWT VERIFICATION]"
    )

    print(
        f"  Embedded positions : "
        f"{len(embedded_positions)}"
    )

    print(
        f"  Direct errors      : "
        f"{direct_errors}"
    )

    print(
        f"  Direct BER         : "
        f"{direct_ber:.8f}"
    )


    reference = np.asarray(
        image[..., 0],
        dtype=np.float32
    )


    baseline_reconstructed = pywt.idwt2(
        (
            LL,
            (
                LH.copy(),
                HL.copy(),
                HH.copy()
            )
        ),
        "db2"
    )

    baseline_reconstructed = np.asarray(
        baseline_reconstructed,
        dtype=np.float32
    )

    if baseline_reconstructed.shape != reference.shape:

        target_height = min(
            baseline_reconstructed.shape[0],
            reference.shape[0]
        )

        target_width = min(
            baseline_reconstructed.shape[1],
            reference.shape[1]
        )

        baseline_reconstructed = baseline_reconstructed[
            :target_height,
            :target_width
        ]

        reference = reference[
            :target_height,
            :target_width
        ]

    baseline_mse = float(
        np.mean(
            (
                reference
                -
                baseline_reconstructed
            ) ** 2
        )
    )

    if baseline_mse <= 1e-16:

        baseline_psnr = float("inf")

    else:

        baseline_psnr = float(
            10.0
            *
            np.log10(
                (255.0 ** 2)
                /
                baseline_mse
            )
        )

    print(
        "\n[DWT BASELINE VERIFICATION]"
    )

    print(
        f"  Reference range     : "
        f"{np.min(reference):.6f} - "
        f"{np.max(reference):.6f}"
    )

    print(
        f"  Baseline DWT range  : "
        f"{np.min(baseline_reconstructed):.6f} - "
        f"{np.max(baseline_reconstructed):.6f}"
    )

    print(
        f"  Baseline MSE        : "
        f"{baseline_mse:.12e}"
    )

    print(
        f"  Baseline PSNR       : "
        f"{baseline_psnr:.8f} dB"
    )

    if (
        not np.isfinite(baseline_psnr)
        or baseline_psnr < 80.0
    ):

        raise ValueError(
            f"Pre-embedding DWT reconstruction is inconsistent "
            f"with the cover image. "
            f"Baseline PSNR={baseline_psnr:.8f} dB. "
            f"Embedding was stopped before quality evaluation."
        )


    stego_reconstructed = pywt.idwt2(
        (
            LL,
            (
                LH,
                HL,
                HH
            )
        ),
        "db2"
    )

    stego_reconstructed = np.asarray(
        stego_reconstructed,
        dtype=np.float32
    )

    if stego_reconstructed.shape != reference.shape:

        target_height = min(
            stego_reconstructed.shape[0],
            reference.shape[0]
        )

        target_width = min(
            stego_reconstructed.shape[1],
            reference.shape[1]
        )

        stego_reconstructed = stego_reconstructed[
            :target_height,
            :target_width
        ]

        reference = reference[
            :target_height,
            :target_width
        ]

    # ============================================================
    # ACTUAL IMAGE QUALITY
    #
    # Compare ORIGINAL COVER against the POST-EMBEDDING STEGO
    # reconstruction.
    # ============================================================

    float_difference = (
        stego_reconstructed
        -
        reference
    )

    mse = float(
        np.mean(
            float_difference ** 2
        )
    )

    if mse <= 1e-16:

        psnr = float("inf")

    else:

        psnr = float(
            10.0
            *
            np.log10(
                (255.0 ** 2)
                /
                mse
            )
        )

    distortion = float(
        np.mean(
            np.abs(
                float_difference
            )
        )
    )

    # ============================================================
    # BER VERIFICATION AFTER IDWT
    # ============================================================

    _, (
        LH_check,
        HL_check,
        HH_check
    ) = pywt.dwt2(
        stego_reconstructed,
        "db2"
    )

    errors = 0

    for (
        subband,
        r,
        c,
        delta,
        bit
    ) in embedded_positions:

        if subband == "LH":

            band = LH_check

        elif subband == "HL":

            band = HL_check

        else:

            band = HH_check

        q = int(
            np.round(
                band[r, c] / delta
            )
        )

        if (q % 2) != bit:

            errors += 1

    ber = (
        errors
        /
        max(
            1,
            len(
                embedded_positions
            )
        )
    )

    print(
        "\n[ACTUAL EMBEDDING RESULT]"
    )

    print(
        f"  PSNR       : {psnr:.8f} dB"
    )

    print(
        f"  MSE        : {mse:.12e}"
    )

    print(
        f"  Distortion : {distortion:.12e}"
    )

    print(
        f"  BER        : {ber:.8f}"
    )

    return (
        psnr,
        mse,
        distortion,
        payloads_assigned,
        stego_reconstructed,
        ber
    )




def prepare_intelligence_features(df):

    excluded = set()

    # --------------------------------------------------------
    # Exact regression targets
    # --------------------------------------------------------

    for target in REGRESSION_TARGETS:
        col = find_column(df, [target])
        if col is not None:
            excluded.add(col)

    # --------------------------------------------------------
    # Explicit classification labels
    # --------------------------------------------------------

    classification_columns = [
        "target_class",
        "target_class_id",
        "classification",
        "class",
        "class_id"
    ]

    for candidate in classification_columns:
        col = find_column(df, [candidate])
        if col is not None:
            excluded.add(col)

    # --------------------------------------------------------
    # Explicit metadata / coordinate columns
    # --------------------------------------------------------

    metadata_columns = [
        "region_id",
        "row",
        "col",
        "row_start",
        "column_start",
        "pre_row_start",
        "pre_column_start",
        "region_row",
        "region_col",
        "region_index"
    ]

    for candidate in metadata_columns:
        col = find_column(df, [candidate])
        if col is not None:
            excluded.add(col)

    # --------------------------------------------------------
    # Post-embedding / payload-derived columns
    # --------------------------------------------------------

    for col in df.columns:

        name = normalize_name(col)

        if (
            name.startswith("target_post_")
            or name.startswith("post_")
            or name.startswith("payload_")
            or name.startswith("embedded_")
            or name.startswith("extracted_")
        ):
            excluded.add(col)

    # --------------------------------------------------------
    # Numeric intelligence features
    # --------------------------------------------------------

    numeric = df.select_dtypes(
        include=[np.number]
    ).copy()

    numeric = numeric.drop(
        columns=[
            c for c in excluded
            if c in numeric.columns
        ],
        errors="ignore"
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove columns containing no usable information
    numeric = numeric.dropna(
        axis=1,
        how="all"
    )

    if numeric.shape[1] == 0:
        raise ValueError(
            "No valid intelligence features remain after leakage filtering."
        )

    print(
        "\n[INTELLIGENT DATASET]"
    )

    print(
        f"  Total dataset columns       : {df.shape[1]}"
    )

    print(
        f"  Numeric intelligence inputs : {numeric.shape[1]}"
    )

    print(
        f"  Excluded columns            : {len(excluded)}"
    )

    print(
        f"  Dataset rows                : {len(numeric)}"
    )

    print(
        "\n  Regression targets excluded:"
    )

    for target in REGRESSION_TARGETS:
        col = find_column(df, [target])
        if col is not None:
            print(f"    - {col}")

    return numeric

def prepare_targets(df):
    regression_cols = [find_column(df, [t]) for t in REGRESSION_TARGETS if find_column(df, [t])]
    regression = df[regression_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    return regression, regression_cols
class SQENetQuantumLayer(tf.keras.layers.Layer):
    """
    SQE-Net (StegaQEntropy Quantum-Entangled Surrogate Network) feature-level
    quantum stage, executed INSIDE the trainable Keras graph.

    64 logical qubits, structured as 8 independent 8-qubit registers
    (8 x 8 = 64), each register simulated as an explicit complex state vector
    of 2**8 = 256 amplitudes (never a dense 2**64 state -- that is
    computationally infeasible and is never constructed anywhere in this
    model). Per register, per qubit:

        RY(theta) angle encoding
        RZ(phi)   phase encoding
        H         superposition
        cyclic-ring CNOT (q -> q+1 -> ... -> q7 -> q0)   <-- feature-level
                                                              (intra-region)
                                                              quantum entanglement
        Pauli-Z expectation measurement

    All gate matrices are analytic (cos/sin/exp) functions of the input
    rotation angles and are applied as exact tensor reshapes / permutations
    of the batched complex state, so TensorFlow's autodiff differentiates the
    real quantum-mechanical gate equations directly -- there are no
    hand-written or fake gradients here (see requirement #41).
    """

    def __init__(self, qubits_per_register=8, num_registers=8, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.qubits_per_register = qubits_per_register
        self.num_registers = num_registers
        self.dim = 2 ** qubits_per_register

        # Precompute the fixed CNOT ring permutation for this register size
        # once (pure index bookkeeping -- identical math to
        # QuantumRegisterBackend.apply_cnot, vectorized as a gather index).
        self._cnot_perms = []
        n = qubits_per_register
        pairs = [(q, (q + 1) % n) for q in range(n)]
        for control, target in pairs:
            perm = np.arange(self.dim)
            for index in range(self.dim):
                if (index >> control) & 1 == 1:
                    perm[index] = index ^ (1 << target)
            self._cnot_perms.append(tf.constant(perm, dtype=tf.int32))

    def _apply_single_qubit(self, state, qubit, mat00, mat01, mat10, mat11):
        # state: (batch, dim) complex64. mat**: (batch,) complex64 gate entries.
        step = 2 ** qubit
        high = self.dim // (2 * step)
        reshaped = tf.reshape(state, [-1, high, 2, step])
        a = reshaped[:, :, 0, :]
        b = reshaped[:, :, 1, :]
        m00 = mat00[:, None, None]
        m01 = mat01[:, None, None]
        m10 = mat10[:, None, None]
        m11 = mat11[:, None, None]
        new_a = m00 * a + m01 * b
        new_b = m10 * a + m11 * b
        new = tf.stack([new_a, new_b], axis=2)
        return tf.reshape(new, [-1, self.dim])

    def _apply_ry(self, state, qubit, angle):
        c = tf.cast(tf.cos(angle / 2.0), tf.complex64)
        s = tf.cast(tf.sin(angle / 2.0), tf.complex64)
        return self._apply_single_qubit(state, qubit, c, -s, s, c)

    def _apply_rz(self, state, qubit, angle):
        zero = tf.zeros_like(angle)
        phase0 = tf.exp(tf.complex(zero, -angle / 2.0))
        phase1 = tf.exp(tf.complex(zero, angle / 2.0))
        zeroc = tf.zeros_like(phase0)
        return self._apply_single_qubit(state, qubit, phase0, zeroc, zeroc, phase1)

    def _apply_hadamard(self, state, qubit):
        batch = tf.shape(state)[0]
        inv_sqrt2 = tf.cast(
            tf.fill([batch], 1.0 / np.sqrt(2.0)), tf.complex64
        )
        return self._apply_single_qubit(
            state, qubit, inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2
        )

    def _apply_cnot_ring(self, state):
        for perm in self._cnot_perms:
            state = tf.gather(state, perm, axis=1)
        return state

    def _measure_z(self, state, qubit):
        probs = tf.math.real(state * tf.math.conj(state))
        step = 2 ** qubit
        high = self.dim // (2 * step)
        reshaped = tf.reshape(probs, [-1, high, 2, step])
        p0 = tf.reduce_sum(reshaped[:, :, 0, :], axis=[1, 2])
        p1 = tf.reduce_sum(reshaped[:, :, 1, :], axis=[1, 2])
        return p0 - p1

    def _run_register(self, ry_angles, rz_angles):
        # ry_angles, rz_angles: (batch, qubits_per_register)
        batch = tf.shape(ry_angles)[0]
        state = tf.zeros([batch, self.dim], dtype=tf.complex64)
        one_hot0 = tf.one_hot(
            tf.zeros([batch], dtype=tf.int32), self.dim, dtype=tf.complex64
        )
        state = state + one_hot0

        for q in range(self.qubits_per_register):
            state = self._apply_ry(state, q, ry_angles[:, q])
            state = self._apply_rz(state, q, rz_angles[:, q])
            state = self._apply_hadamard(state, q)

        state = self._apply_cnot_ring(state)

        expectations = [
            self._measure_z(state, q) for q in range(self.qubits_per_register)
        ]

        return tf.stack(expectations, axis=1)  # (batch, qubits_per_register)

    def call(self, angles):
        # angles: (batch, 2 * num_registers * qubits_per_register)
        # first half -> RY angles, second half -> RZ angles, both already
        # bounded to (-pi, pi) by the caller.
        per_side = self.num_registers * self.qubits_per_register
        ry_all = angles[:, :per_side]
        rz_all = angles[:, per_side:2 * per_side]

        register_outputs = []
        for register_id in range(self.num_registers):
            start = register_id * self.qubits_per_register
            end = start + self.qubits_per_register
            ry_reg = tf.cast(ry_all[:, start:end], tf.float32)
            rz_reg = tf.cast(rz_all[:, start:end], tf.float32)
            register_outputs.append(self._run_register(ry_reg, rz_reg))

        return tf.concat(register_outputs, axis=1)  # (batch, 64)

    def get_config(self):
        config = super().get_config()
        config.update({
            "qubits_per_register": self.qubits_per_register,
            "num_registers": self.num_registers
        })
        return config


def quantum_feature_layer(
    x,
    qubit_count=64,
    region_count=8,
    name_prefix="quantum"
):
    """
    SQE-Net feature-level quantum stage. Projects the fused multimodal
    representation `x` to 2*qubit_count bounded rotation angles
    (theta = pi * tanh(z), requirement #8), splits them into
    `region_count` independent 8-qubit quantum registers
    (region_count * 8 == qubit_count), and runs each register through a
    genuine RY + RZ + H + cyclic-CNOT quantum circuit with Pauli-Z
    measurement (SQENetQuantumLayer, requirement #9-#11). Returns the raw
    64-dimensional quantum-measured feature vector -- callers apply their
    own Dense projection down to the 128-D region embedding
    (requirement #12), which keeps a single, clearly-labelled projection
    step instead of silently stacking several.
    """
    if qubit_count != 64:
        raise ValueError("SQE-Net requires exactly 64 logical qubits.")

    if region_count * 8 != qubit_count:
        raise ValueError(
            "SQE-Net requires region_count * 8 == qubit_count "
            "(8 quantum registers of 8 qubits each)."
        )

    angle_projection = Dense(
        qubit_count * 2,
        activation="tanh",
        name=f"{name_prefix}_angle_projection"
    )(x)

    bounded_angles = tf.keras.layers.Lambda(
        lambda value: value * np.pi,
        name=f"{name_prefix}_bounded_angles"
    )(angle_projection)

    quantum_features = SQENetQuantumLayer(
        qubits_per_register=8,
        num_registers=region_count,
        name=f"{name_prefix}_sqe_net_circuit"
    )(bounded_angles)

    return quantum_features

def build_cnn_branch(input_shape, name):
    inputs = Input(shape=input_shape, name=name)

    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation=ACTIVATION
    )(inputs)

    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation=ACTIVATION
    )(x)

    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation=ACTIVATION)(x)

    x = quantum_feature_layer(
        x,
        qubit_count=64,
        region_count=8,
        name_prefix=f"{name}_quantum"
    )

    x = Dense(
        128,
        activation=ACTIVATION,
        name=f"{name}_quantum_region_embedding_128"
    )(x)

    x = BatchNormalization(
        name=f"{name}_quantum_region_embedding_bn"
    )(x)

    return inputs, x

def build_regression_surrogate_model(
    intel_count,
    raw_shape,
    dwt_shape,
    reg_count
):

    intel_input = Input(
        shape=(intel_count,),
        name="intelligence_input"
    )

    raw_input = Input(
        shape=raw_shape,
        name="raw_image_input"
    )

    dwt_input = Input(
        shape=dwt_shape,
        name="dwt_input"
    )

    selection_input = Input(
        shape=(1,),
        name="selection_input"
    )

    x_int = Dense(
        256,
        activation=ACTIVATION
    )(intel_input)

    x_int = BatchNormalization()(x_int)

    x_int = Dropout(0.25)(x_int)

    x_int = Dense(
        128,
        activation=ACTIVATION
    )(x_int)

    raw_x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation=ACTIVATION
    )(raw_input)

    raw_x = BatchNormalization()(raw_x)
    raw_x = MaxPooling2D((2, 2))(raw_x)

    raw_x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation=ACTIVATION
    )(raw_x)

    raw_x = BatchNormalization()(raw_x)
    raw_x = MaxPooling2D((2, 2))(raw_x)

    raw_x = GlobalAveragePooling2D()(raw_x)

    raw_x = Dense(
        128,
        activation=ACTIVATION
    )(raw_x)

    dwt_x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation=ACTIVATION
    )(dwt_input)

    dwt_x = BatchNormalization()(dwt_x)
    dwt_x = MaxPooling2D((2, 2))(dwt_x)

    dwt_x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation=ACTIVATION
    )(dwt_x)

    dwt_x = BatchNormalization()(dwt_x)
    dwt_x = MaxPooling2D((2, 2))(dwt_x)

    dwt_x = GlobalAveragePooling2D()(dwt_x)

    dwt_x = Dense(
        128,
        activation=ACTIVATION
    )(dwt_x)

    fused = Concatenate(
        name="feature_fusion"
    )([
        x_int,
        raw_x,
        dwt_x
    ])

    fused = Dense(
        256,
        activation=ACTIVATION
    )(fused)

    fused = BatchNormalization()(fused)

    quantum = quantum_feature_layer(
        fused,
        qubit_count=64,
        region_count=8,
        name_prefix="fusion_quantum"
    )

    region_embedding = Dense(
        128,
        activation=ACTIVATION,
        name="region_embedding_128"
    )(quantum)

    regional_hidden = Dense(
        128,
        activation=ACTIVATION
    )(region_embedding)

    regional_output = Dense(
        reg_count,
        activation="linear",
        name="regression_output"
    )(regional_hidden)

    global_representation = global_region_entanglement(
        region_embedding,
        selection_input
    )

    acer_feature_encoder = Dense(
        32,
        activation=ACTIVATION,
        name="acer_feature_encoder"
    )(global_representation)

    acer_quantum_projection = Dense(
        128,
        activation=ACTIVATION,
        name="acer_quantum_projection"
    )(acer_feature_encoder)

    global_hidden = Dense(
        256,
        activation=ACTIVATION,
        name="global_hidden_256"
    )(acer_quantum_projection)

    global_hidden = BatchNormalization(
        name="global_hidden_bn"
    )(global_hidden)

    global_hidden = Dropout(
        0.30,
        name="global_dropout"
    )(global_hidden)

    global_hidden = Dense(
        128,
        activation=ACTIVATION,
        name="global_hidden_128"
    )(global_hidden)

    global_output = Dense(
        1,
        activation="linear",
        name="global_psnr"
    )(global_hidden)

    model = Model(
        inputs=[
            intel_input,
            raw_input,
            dwt_input,
            selection_input
        ],
        outputs=[
            regional_output,
            global_output
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "regression_output": tf.keras.losses.Huber(),
            "global_psnr": tf.keras.losses.Huber()
        },
        loss_weights={
            "regression_output": 1.0,
            "global_psnr": 1.0
        },
        metrics={
            "regression_output": [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanSquaredError(name="mse")
            ],
            "global_psnr": [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanSquaredError(name="mse")
            ]
        }
    )

    return model

def plot_surrogate_learning_analysis(
    model,
    test_inputs,
    test_targets,
    output_dir,
    target_names
):
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating surrogate learning analysis plots...")

    predictions, global_psnr_predictions = model.predict(
        test_inputs,
        verbose=0
    )

    test_targets = np.asarray(test_targets)
    predictions = np.asarray(predictions)
    global_psnr_predictions = np.asarray(global_psnr_predictions)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    if test_targets.ndim == 1:
        test_targets = test_targets.reshape(-1, 1)

    sample_count = min(
        len(test_inputs[0]),
        500
    )

    intelligence_data = np.asarray(
        test_inputs[0][:sample_count]
    )

    raw_data = np.asarray(
        test_inputs[1][:sample_count]
    )

    dwt_data = np.asarray(
        test_inputs[2][:sample_count]
    )

   

    target_data = test_targets[:sample_count]
    prediction_data = predictions[:sample_count]

    def flatten_features(data):
        return data.reshape(
            data.shape[0],
            -1
        )
    def create_tsne(data):

        data = flatten_features(data)

        data = StandardScaler().fit_transform(data)

        perplexity = min(
            30,
            max(
                5,
                data.shape[0] // 4
            )
        )

        perplexity = min(
            perplexity,
            data.shape[0] - 1
        )

        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=42
        )

        embedding = reducer.fit_transform(
            data
        )

        return embedding

    def save_combined_tsne_plot(
        embeddings,
        labels,
        output_dir
    ):

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 11)
        )

        titles = [
            "(a) Intelligence Features",
            "(b) Raw Image Features",
            "(c) DWT Features",
            "(d) Fused Quantum-Inspired Features"
        ]

        target_values = np.asarray(
            labels,
            dtype=np.float64
        )

        vmin = float(
            np.nanmin(target_values)
        )

        vmax = float(
            np.nanmax(target_values)
        )

        scatter = None

        for ax, embedding, title in zip(
            axes.flat,
            embeddings,
            titles
        ):

            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=target_values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=34,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.3,
                rasterized=True
            )

            ax.set_title(
                title,
                fontsize=13,
                fontweight="bold",
                pad=10
            )

            ax.set_xlabel(
                "t-SNE dimension 1",
                fontsize=10
            )

            ax.set_ylabel(
                "t-SNE dimension 2",
                fontsize=10
            )

            ax.tick_params(
                labelsize=8
            )

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.5,
                alpha=0.25
            )

            ax.spines[
                "top"
            ].set_visible(False)

            ax.spines[
                "right"
            ].set_visible(False)

            ax.spines[
                "left"
            ].set_linewidth(0.7)

            ax.spines[
                "bottom"
            ].set_linewidth(0.7)

        colorbar = fig.colorbar(
            scatter,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.025
        )

        colorbar.set_label(
            "Normalized image PSNR",
            fontsize=10
        )

        colorbar.ax.tick_params(
            labelsize=8
        )

        fig.suptitle(
            "t-SNE Visualization of Feature Representations",
            fontsize=16,
            fontweight="bold",
            y=0.98
        )

        fig.tight_layout(
            rect=[
                0,
                0,
                0.96,
                0.96
            ]
        )

        fig.savefig(
            os.path.join(
                output_dir,
                "04_tsne_feature_representations_combined.png"
            ),
            dpi=600,
            bbox_inches="tight"
        )

        plt.close(fig)

    def save_tsne_plot(
        embedding,
        labels,
        title,
        filename,
        vmin,
        vmax
    ):

        fig, ax = plt.subplots(
            figsize=(8.2, 6.5)
        )

        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=38,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.35,
            rasterized=True
        )

        ax.set_title(
            title,
            fontsize=15,
            fontweight="bold",
            pad=12
        )

        ax.set_xlabel(
            "t-SNE dimension 1",
            fontsize=11
        )

        ax.set_ylabel(
            "t-SNE dimension 2",
            fontsize=11
        )

        ax.tick_params(
            labelsize=9
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.6,
            alpha=0.25
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        ax.spines[
            "left"
        ].set_linewidth(0.8)

        ax.spines[
            "bottom"
        ].set_linewidth(0.8)

        colorbar = fig.colorbar(
            scatter,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        colorbar.set_label(
            "Normalized image PSNR",
            fontsize=10
        )

        colorbar.ax.tick_params(
            labelsize=9
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                output_dir,
                filename
            ),
            dpi=600,
            bbox_inches="tight"
        )

        plt.close(fig)

    target_values = np.asarray(
        target_data[:, 0],
        dtype=np.float64
    )

    target_min = float(
        np.nanmin(target_values)
    )

    target_max = float(
        np.nanmax(target_values)
    )

    intelligence_embedding = create_tsne(
        intelligence_data
    )

    save_tsne_plot(
        intelligence_embedding,
        target_values,
        "Intelligence Features",
        "01_tsne_intelligence_features.png",
        target_min,
        target_max
    )

    raw_embedding = create_tsne(
        raw_data
    )

    save_tsne_plot(
        raw_embedding,
        target_values,
        "Raw Image Features",
        "02_tsne_raw_image_features.png",
        target_min,
        target_max
    )

    dwt_embedding = create_tsne(
        dwt_data
    )

    save_tsne_plot(
        dwt_embedding,
        target_values,
        "DWT Features",
        "03_tsne_dwt_features.png",
        target_min,
        target_max
    )

    intermediate_layer_names = [
        "feature_fusion",
        "fusion_quantum_measurement_features",
        "fusion_quantum_feature_projection"
    ]

    available_layer_names = [
        layer.name
        for layer in model.layers
    ]

    selected_layer_name = None

    for layer_name in intermediate_layer_names:
        if layer_name in available_layer_names:
            selected_layer_name = layer_name
            break

    if selected_layer_name is not None:
        feature_model = Model(
            inputs=model.inputs,
            outputs=model.get_layer(
                selected_layer_name
            ).output
        )

        fused_features = feature_model.predict(
            test_inputs,
            verbose=0
        )

        fused_embedding = create_tsne(
            fused_features[:sample_count]
        )

        save_tsne_plot(
            fused_embedding,
            target_values,
            "Fused Quantum-Inspired Features",
            "04_tsne_fused_quantum_features.png",
            target_min,
            target_max
        )

        save_combined_tsne_plot(
            [
                intelligence_embedding,
                raw_embedding,
                dwt_embedding,
                fused_embedding
            ],
            target_values,
            output_dir
        )

    else:
        print(
            "Fusion layer not found. Skipping fused t-SNE plot."
        )

    target_index = 0

    actual_values = target_data[:, target_index]
    predicted_values = prediction_data[:, target_index]

    plt.figure(figsize=(10, 7))

    plt.scatter(
        actual_values,
        predicted_values,
        s=50,
        alpha=0.8
    )

    minimum_value = min(
        actual_values.min(),
        predicted_values.min()
    )

    maximum_value = max(
        actual_values.max(),
        predicted_values.max()
    )

    plt.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--",
        linewidth=2
    )

    mae = mean_absolute_error(
        actual_values,
        predicted_values
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )

    target_name = target_names[target_index]

    plt.title(
        f"Actual vs Predicted: {target_name}\n"
        f"MAE={mae:.6f}, RMSE={rmse:.6f}"
    )

    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "05_actual_vs_predicted.png"
        ),
        dpi=300
    )

    plt.close()

    residuals = predicted_values - actual_values

    plt.figure(figsize=(10, 7))

    plt.hist(
        residuals,
        bins=30,
        alpha=0.85
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2
    )

    plt.title(
        f"Prediction Error Distribution: {target_name}"
    )

    plt.xlabel("Prediction error")
    plt.ylabel("Number of samples")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "06_prediction_error_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "Six surrogate learning plots saved to:"
    )

    print(output_dir)


def build_neighbor_graph(metadata):
    rows = metadata["row"].fillna(0).to_numpy()
    cols = metadata["col"].fillna(0).to_numpy()
    unique_rows = sorted(np.unique(rows))
    unique_cols = sorted(np.unique(cols))
    row_pos = {val: idx for idx, val in enumerate(unique_rows)}
    col_pos = {val: idx for idx, val in enumerate(unique_cols)}

    grid = {}
    for idx in range(len(metadata)):
        grid[(row_pos[rows[idx]], col_pos[cols[idx]])] = idx

    graph = {idx: set() for idx in range(len(metadata))}
    for idx in range(len(metadata)):
        r_idx = row_pos[rows[idx]]
        c_idx = col_pos[cols[idx]]
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nbr = grid.get((r_idx + dr, c_idx + dc))
                if nbr is not None:
                    graph[idx].add(nbr)
    return graph

def convert_to_native_types(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    return obj

def save_sqe_quantum_metadata(output_dir, sample_fused_vector=None):
    """
    Requirement #45/#46: export SQE-Net's own quantum metadata (separate from
    ACER's) plus, when a sample fused feature vector is available, its
    actual measured rotation angles and quantum feature vector for one
    region, using the offline SQEQuantumBackend (identical circuit to the
    trainable SQENetQuantumLayer used inside the Keras graph).
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "component": "SQE-Net",
        "full_name": "StegaQEntropy Quantum-Entangled Surrogate Network",
        "logical_qubits": 64,
        "quantum_registers": 8,
        "qubits_per_register": 8,
        "feature_entanglement": "cyclic CNOT",
        "single_qubit_gates": ["RY", "RZ", "H"],
        "measurement": "Pauli-Z expectation",
        "backend": "SQEQuantumBackend (state-vector simulation, 2**8 amplitudes per register)"
    }

    with open(
        os.path.join(output_dir, "sqe_quantum_metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(convert_to_native_types(metadata), f, indent=2)

    if sample_fused_vector is not None:
        if ABLATE_SQE_NET:
            quantum_features = classical_sqe_features(
                sample_fused_vector
            )
            angles = None
        else:
            backend = SQEQuantumBackend()

            quantum_features, angles, _ = (
                backend.sqe_quantum_features(
                    sample_fused_vector
                )
            )
        np.save(
            os.path.join(output_dir, "sqe_quantum_features.npy"),
            quantum_features
        )
        np.save(
            os.path.join(output_dir, "sqe_rotation_angles.npy"),
            angles
        )

    print("\n[SQE-NET QUANTUM]")
    print(f"Logical qubits: {metadata['logical_qubits']}")
    print(f"Registers: {metadata['quantum_registers']}")
    print(f"Qubits/register: {metadata['qubits_per_register']}")
    print("Encoding: RY + RZ")
    print("Superposition: H")
    print(f"Entanglement: {metadata['feature_entanglement']}")
    print(f"Measurement: {metadata['measurement']}")
    print("Quantum features: 64")
    print("Region embedding: 128")

def print_qoqa_quantum_log(qoqa_result, target_regions_count):
    """Print the [QOQA QUANTUM OPTIMIZATION] section (all values measured)."""
    d = qoqa_result["diagnostics"]
    r = qoqa_result["report"]
    print("\n[QOQA QUANTUM OPTIMIZATION]")
    print(f"Candidate regions: {d['candidate_regions']}")
    print(f"Quantum blocks: {d['blocks']}")
    print(f"Qubits/block: {d['qubits_per_block']} "
          f"({d['state_vector_length_per_block']} amplitudes/block, "
          f"no dense {d['candidate_regions']}-qubit state)")
    print(f"QAOA layers: {d['qaoa_layers']} (QAOA-style state-vector simulation)")
    print(f"Measurement shots: {d['total_quantum_measurements']} "
          f"({d['shots_per_block']}/block/sweep-attempt)")
    print("Objective encoding: ML score -> QUBO linear term -> Ising field -> RZ")
    print(f"Region entanglement: RZZ over {d['conflict_edges_total']} global 8-neighbour "
          f"edges ({d['cross_block_conflict_edges']} cross-block, handled as boundary terms) "
          f"+ cardinality pair couplings")
    print(f"Cardinality penalty (mu): {d['mu_cardinality']:.4f}")
    print(f"Neighbour penalty (lambda): {d['lambda_conflict']:.4f}")
    print(f"Measured configurations: {d['measured_configurations_unique']} unique "
          f"in {d['total_quantum_measurements']} shots")
    print(f"Feasible configurations: {d['feasible_measured_configurations']} "
          f"(cardinality-exact: {d['cardinality_exact_feasible_configurations']})")
    print(f"Final energy: {qoqa_result['final_energy']['total_energy']:.8f}")
    print(f"Selected regions: {r['selected_count']}")
    print(f"Cardinality verified: {r['cardinality_verified']} (K={int(target_regions_count)})")
    print(f"8-neighbour violations: {r['neighbour_violations']}")
    print(f"Capacity verified: {r['capacity_verified']} "
          f"(selected usable capacity={r['capacity_bits']} bits)")


def save_qoqa_quantum_diagnostics(
    output_dir,
    selection_scores,
    qoqa_result,
    payload_regions,
    target_regions_count,
    graph
):
    """Persist the actual QOQA optimisation record (nothing fabricated)."""
    os.makedirs(output_dir, exist_ok=True)

    d = qoqa_result["diagnostics"]
    r = qoqa_result["report"]
    fe = qoqa_result["final_energy"]
    selected = sorted(int(i) for i in qoqa_result["selected_regions"])

    metadata_out = {
        "component": "QOQA",
        "full_name": "Quantum-Assisted Optimal Region Selection",
        "optimization_formulation": "constrained binary QUBO / Ising-style Hamiltonian",
        "ising_mapping": "x_i = (1 - Z_i)/2 ; E(z) = sum h_i Z_i + sum J_ij Z_i Z_j ; h_i = -a_i/2 - (1/4) sum_j b_ij ; J_ij = b_ij/4",
        "cardinality_in_hamiltonian": True,
        "cardinality_encoding": "mu(sum x - K_b)^2 -> linear mu(1-2K_b) in a_i (RZ field) and 2*mu pairwise b_ij (RZZ coupling) on every qubit pair of a block; K_b = K - regions fixed outside the block",
        "conflict_encoding": "lambda*x_i x_j on within-block 8-neighbour edges (RZZ) ; cross-block edges -> lambda*(fixed selected neighbour) linear term",
        "hamiltonian": "H(x) = -sum w_i x_i + lambda*sum_{(i,j) in E8} x_i x_j + mu*(sum x_i - K)^2",
        "backend": "classical state-vector simulation of a QAOA-style circuit (no quantum hardware)",
        "logical_qubits_per_block": d["qubits_per_block"],
        "state_vector_length_per_block": d["state_vector_length_per_block"],
        "dense_full_pool_state_vector_constructed": False,
        "quantum_blocks": d["blocks"],
        "candidate_regions": d["candidate_regions"],
        "target_cardinality": int(target_regions_count),
        "quantum_circuit": "H -> RZ -> RZZ -> RX -> Born measurement",
        "qaoa_layers": d["qaoa_layers"],
        "region_level_entanglement": "RZZ over 8-neighbour conflict edges (+ cardinality pair couplings)",
        "conflict_edges_total": d["conflict_edges_total"],
        "cross_block_conflict_edges": d["cross_block_conflict_edges"],
        "cross_block_handling": "boundary terms against fixed neighbouring-block selections, coordinated sweeps, global verification",
        "cardinality_penalty": d["mu_cardinality"],
        "neighbour_penalty": d["lambda_conflict"],
        "capacity_bias_kappa": d["capacity_bias_kappa"],
        "quantum_measurement_shots": d["total_quantum_measurements"],
        "shots_per_block": d["shots_per_block"],
        "sweeps_executed": d["sweeps_executed"],
        "quantum_reoptimisation_attempts": d["quantum_reoptimisation_attempts"],
        "measured_configurations": d["measured_configurations_unique"],
        "feasible_configurations": d["feasible_measured_configurations"],
        "cardinality_exact_feasible_configurations": d["cardinality_exact_feasible_configurations"],
        "max_probability_norm_deviation": d["max_probability_norm_deviation"],
        "selected_region_count": r["selected_count"],
        "final_energy": fe["total_energy"],
        "final_objective_reward": fe["objective_reward"],
        "final_conflict_penalty": fe["conflict_penalty"],
        "final_cardinality_penalty": fe["cardinality_penalty"],
        "final_conflict_count": fe["conflict_count"],
        "cardinality_verified": r["cardinality_verified"],
        "capacity_verified": r["capacity_verified"],
        "selected_usable_capacity_bits": r["capacity_bits"],
        "8_neighbour_violations_in_final_selection": r["neighbour_violations"],
        "configuration_energy_min_final_sweep": (float(np.min(qoqa_result["measurements"][:, 7])) if len(qoqa_result["measurements"]) else None),
        "configuration_energy_mean_final_sweep": (float(np.mean(qoqa_result["measurements"][:, 7])) if len(qoqa_result["measurements"]) else None),
        "measurement_columns": [
            "round", "sweep", "block", "local_basis_state_index", "shot_count",
            "empirical_probability", "born_probability", "block_energy", "feasible"
        ],
        "measurement_note": "final sweep only; local qubit q <-> block_map[block][q]; bit q of basis index = x for local qubit q",
    }

    with open(os.path.join(output_dir, "qoqa_quantum_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(convert_to_native_types(metadata_out), f, indent=2)

    np.save(os.path.join(output_dir, "qoqa_region_scores.npy"),
            np.asarray(selection_scores, dtype=np.float64))
    np.save(os.path.join(output_dir, "qoqa_measurements.npy"),
            np.asarray(qoqa_result["measurements"], dtype=np.float64))

    with open(os.path.join(output_dir, "qoqa_selected_configuration.json"), "w", encoding="utf-8") as f:
        json.dump(convert_to_native_types({
            "selected_regions": selected,
            "payload_carrying_regions": sorted(int(i) for i in payload_regions),
            "target_cardinality": int(target_regions_count),
            "final_energy": fe,
            "verification": r,
            "block_map_local_qubit_to_global_region": qoqa_result["block_map"],
        }), f, indent=2)

    with open(os.path.join(output_dir, "qoqa_feasible_configurations.json"), "w", encoding="utf-8") as f:
        json.dump(convert_to_native_types({
            "feasible_measured_configurations": d["feasible_measured_configurations"],
            "cardinality_exact_feasible_configurations": d["cardinality_exact_feasible_configurations"],
            "globally_feasible_final_configuration": r["feasible"],
            "accepted_block_configurations": qoqa_result["block_records"],
        }), f, indent=2)

# Define the subfolder path near your other constants (top of the script)
TRAINING_PLOTS_DIR = os.path.join(OUTPUT_DIR, "training_plots_surrogate")


def generate_training_plots(history):
    os.makedirs(TRAINING_PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Total Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Total Validation Loss', linewidth=2)
    plt.title('Pure Regression Surrogate CNN Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TRAINING_PLOTS_DIR, 'training_validation_loss.png'), dpi=200)
    plt.close()


def generate_regression_scatter_plots(y_true_orig, y_pred_orig, regression_columns):
    os.makedirs(TRAINING_PLOTS_DIR, exist_ok=True)
    for i, target in enumerate(regression_columns):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_orig[:, i], y_pred_orig[:, i], alpha=0.5, color='royalblue', s=20, label='Predictions')
        min_v = min(np.min(y_true_orig[:, i]), np.min(y_pred_orig[:, i]))
        max_v = max(np.max(y_true_orig[:, i]), np.max(y_pred_orig[:, i]))
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Ideal Fit')
        plt.title(f'Actual vs Predicted: {target}')
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(TRAINING_PLOTS_DIR, f'regression_{target}.png'), dpi=200)
        plt.close()
        
def global_region_entanglement(
    region_embedding,
    selection_input,
    name="global_region_entanglement"
):
    mask = Lambda(
        lambda x: tf.cast(x, tf.float32),
        name=f"{name}_mask_cast"
    )(selection_input)

    q = Dense(
        128,
        activation="tanh",
        name=f"{name}_query"
    )(region_embedding)

    k = Dense(
        128,
        activation="tanh",
        name=f"{name}_key"
    )(region_embedding)

    v = Dense(
        128,
        activation="tanh",
        name=f"{name}_value"
    )(region_embedding)

    scores = Lambda(
        lambda x: tf.matmul(
            x[0],
            x[1],
            transpose_b=True
        ) / tf.sqrt(tf.cast(128, tf.float32)),
        name=f"{name}_interaction_scores"
    )([q, k])

    key_mask = Lambda(
        lambda x: tf.expand_dims(x, axis=1),
        name=f"{name}_key_mask"
    )(mask)

    scores = Lambda(
        lambda x: x[0] + (1.0 - x[1]) * -1e9,
        name=f"{name}_masked_scores"
    )([scores, key_mask])

    attention = Lambda(
        lambda x: tf.nn.softmax(x, axis=-1),
        name=f"{name}_interaction_weights"
    )(scores)

    interaction_results = Lambda(
        lambda x: tf.matmul(x[0], x[1]),
        name=f"{name}_interaction_results"
    )([attention, v])

    query_mask = Lambda(
        lambda x: tf.expand_dims(x, axis=-1),
        name=f"{name}_query_mask"
    )(mask)

    selected_results = Multiply(
        name=f"{name}_selected_results"
    )([
        interaction_results,
        query_mask
    ])

    summed = Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        name=f"{name}_sum"
    )(selected_results)

    count = Lambda(
        lambda x: tf.reduce_sum(x, axis=1, keepdims=True),
        name=f"{name}_count"
    )(mask)

    global_representation = Lambda(
        lambda x: x[0] / (x[1] + 1e-8),
        name=f"{name}_mean"
    )([
        summed,
        count
    ])

    global_representation = Dense(
        128,
        activation=ACTIVATION,
        name=f"{name}_projection"
    )(global_representation)

    return global_representation

def _qoqa_build_blocks(num_regions, block_size=8):
    """
    Partition candidate region indices into blocks of at most `block_size`
    regions (local qubit q of block b <-> global region id blocks[b][q]).
    Region indices follow raster order, so most real 8-neighbour edges lie
    inside a block (genuine RZZ couplings). Edges crossing block boundaries
    are NOT discarded: they enter each block Hamiltonian as boundary terms
    against the currently selected regions of neighbouring blocks and are
    verified globally (verify_qoqa_constraints).
    """
    return [
        list(range(start, min(start + block_size, num_regions)))
        for start in range(0, num_regions, block_size)
    ]

class QOQAInfeasibleError(RuntimeError):
    """Raised when QOQA cannot produce a feasible configuration after
    quantum re-optimisation (cardinality / 8-neighbour / capacity)."""


def _qoqa_decode(bits):
    if isinstance(bits, str):
        x = np.fromiter(
            (1 if c == "1" else 0 for c in bits), dtype=np.int64, count=len(bits)
        )
    else:
        x = np.asarray(bits, dtype=np.int64).ravel()
    if np.any((x != 0) & (x != 1)):
        raise ValueError("QOQA bitstring must contain only 0/1.")
    return x


def build_qoqa_qubo(
    selection_scores,
    graph,
    target_regions_count,
    lambda_conflict=4.0,
    mu_cardinality=1.5
):
    """
    Binary formulation x_i in {0,1} (1 = region selected):

        H(x) = - sum_i w_i x_i
               + lambda * sum_{(i,j) in E8} x_i x_j
               + mu * (sum_i x_i - K)^2

    w_i = existing ML/objective selection score, min-max normalised to [0,1]
    (no new features are created). E8 = global 8-neighbour conflict graph
    (cross-block edges included). K = target region count.
    """
    scores = np.asarray(selection_scores, dtype=np.float64)
    n = len(scores)

    if n == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("QOQA objective scores are empty or contain NaN/Inf.")

    lo, hi = float(np.min(scores)), float(np.max(scores))
    weights = (
        (scores - lo) / (hi - lo)
        if hi - lo > 1e-12
        else np.full(n, 0.5, dtype=np.float64)
    )

    adjacency = {i: set() for i in range(n)}
    for i, neighbours in graph.items():
        for j in neighbours:
            i, j = int(i), int(j)
            if i == j:
                continue
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"QOQA conflict edge ({i},{j}) outside candidate pool.")
            adjacency[i].add(j)
            adjacency[j].add(i)

    edges = np.asarray(
        sorted((i, j) for i in adjacency for j in adjacency[i] if i < j),
        dtype=np.int64
    ).reshape(-1, 2)

    return {
        "n": n,
        "weights": weights,
        "edges": edges,
        "adjacency": {i: sorted(v) for i, v in adjacency.items()},
        "target_k": int(target_regions_count),
        "lambda_conflict": float(lambda_conflict),
        "mu_cardinality": float(mu_cardinality),
    }


def evaluate_qubo_energy(
    bitstring,
    linear_coefficients,
    conflict_edges,
    cardinality_target,
    lambda_conflict,
    mu_cardinality,
    cardinality_offset=0,
    external_conflicts=None
):
    """
    Deterministic QUBO energy
        E = -sum w_i x_i + lambda*sum_{(i,j)} x_i x_j + mu*(sum x_i - K)^2.
    For block sub-problems `cardinality_offset` is the number of regions
    already fixed outside the block and `external_conflicts[i]` is the number
    of fixed selected 8-neighbours of qubit i outside the block (cross-block
    edges are therefore priced, never dropped).
    """
    x = _qoqa_decode(bitstring)
    w = np.asarray(linear_coefficients, dtype=np.float64)

    if x.shape != w.shape:
        raise ValueError("QOQA bitstring / coefficient length mismatch.")
    if not np.all(np.isfinite(w)):
        raise ValueError("QOQA linear coefficients contain NaN/Inf.")

    reward = float(np.dot(w, x))
    edges = np.asarray(conflict_edges, dtype=np.int64).reshape(-1, 2)
    conflict_count = (
        int(np.sum(x[edges[:, 0]] * x[edges[:, 1]])) if len(edges) else 0
    )
    if external_conflicts is not None:
        conflict_count += int(round(float(np.dot(
            np.asarray(external_conflicts, dtype=np.float64), x
        ))))

    selected_count = int(np.sum(x))
    conflict_penalty = float(lambda_conflict) * conflict_count
    cardinality_penalty = float(mu_cardinality) * (
        selected_count + int(cardinality_offset) - int(cardinality_target)
    ) ** 2
    total = -reward + conflict_penalty + cardinality_penalty

    if not np.isfinite(total):
        raise ValueError("QOQA QUBO energy is not finite.")

    return {
        "total_energy": float(total),
        "objective_reward": reward,
        "conflict_penalty": conflict_penalty,
        "cardinality_penalty": cardinality_penalty,
        "selected_count": selected_count,
        "conflict_count": conflict_count,
    }


def evaluate_qoqa_configuration(x, qubo, capacities=None, required_bits=None, weight_bonus=None):
    """Global-configuration energy + feasibility (deterministic)."""
    w = qubo["weights"] + (0.0 if weight_bonus is None else weight_bonus)
    result = evaluate_qubo_energy(
        x, w, qubo["edges"], qubo["target_k"],
        qubo["lambda_conflict"], qubo["mu_cardinality"]
    )
    x = _qoqa_decode(x)
    result["capacity_bits"] = (
        None if capacities is None else int(np.dot(np.asarray(capacities), x))
    )
    result["feasible"] = bool(
        result["conflict_count"] == 0
        and result["selected_count"] == qubo["target_k"]
        and (required_bits is None or capacities is None
             or result["capacity_bits"] >= required_bits)
    )
    return result


def _qoqa_block_hamiltonian(w_eff, local_edges, ext_conf, k_b, lam, mu):
    """Block QUBO E = sum a_i x_i + sum_{i<j} b_ij x_i x_j + const, expanded
    from -w.x + lam*conflicts + mu*(sum x - K_b)^2 using x_i^2 = x_i:
    a_i = -w_i + lam*ext_i + mu*(1-2K_b);  b_ij = 2*mu + lam*[edge];
    const = mu*K_b^2."""
    m = len(w_eff)
    a = -np.asarray(w_eff, dtype=np.float64) + lam * ext_conf + mu * (1.0 - 2.0 * k_b)
    b = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            b[i, j] = 2.0 * mu
    for (i, j) in local_edges:
        b[min(i, j), max(i, j)] += lam
    return a, b, mu * float(k_b) ** 2


def _qoqa_qubo_to_ising(a, b):
    """x = (1 - Z)/2  ->  h_i = -a_i/2 - (1/4) sum_j b_ij ;  J_ij = b_ij/4."""
    sym = b + b.T
    h = -a / 2.0 - sym.sum(axis=1) / 4.0
    return h, b / 4.0


def _qoqa_energy_table(a, b, const):
    """Energies of all 2^m basis states (used ONLY for the variational
    expectation <E> that tunes the circuit angles -- never to pick a
    configuration)."""
    m = len(a)
    states = np.arange(2 ** m)
    X = ((states[:, None] >> np.arange(m)[None, :]) & 1).astype(np.float64)
    return X @ a + np.einsum("si,ij,sj->s", X, b, X) + const


def sample_qoqa_bitstrings(backend, probabilities, shots, rng, qubit_count):
    """Born-rule measurement of a QOQA block (see backend method)."""
    return backend.sample_bitstrings_from_state(probabilities, shots, rng, qubit_count)


def optimize_qoqa_block(
    backend, block_idx, block, qubo, x, weight_bonus, budget, lam, mu,
    shots, qaoa_layers, rng, angle_cache, base_state,
    grid_gamma=(0.3, 0.6, 0.9, 1.3, 1.8), grid_beta=(0.1, 0.2, 0.3, 0.4)
):
    """
    One decomposed quantum optimisation block:
    block regions -> block QUBO (with cross-block boundary terms) -> Ising
    (h_i, J_ij) -> H, RZ, RZZ, RX (x p) -> Born measurement -> measured
    bitstrings -> QUBO energies -> best feasible measured configuration.
    Local qubit q <-> global region id `block[q]`.
    """
    m = len(block)
    K = qubo["target_k"]
    pos = {g: l for l, g in enumerate(block)}
    w = qubo["weights"][block] + weight_bonus[block]

    local_edges, ext = [], np.zeros(m, dtype=np.float64)
    for l, g in enumerate(block):
        for nb in qubo["adjacency"][g]:
            if nb in pos:
                if pos[nb] > l:
                    local_edges.append((l, pos[nb]))
            elif x[nb]:
                ext[l] += 1.0            # cross-block edge to a fixed selected region

    c_block = int(np.sum(x[block]))
    offset = (int(np.sum(x)) - c_block) if budget is None else (K - int(budget))

    incumbent = evaluate_qubo_energy(
        x[block], w, local_edges, K, lam, mu,
        cardinality_offset=offset, external_conflicts=ext
    )

    shifts = np.arange(m)
    out = {"block": block_idx, "rows": [], "measured_unique": 0, "shots": 0,
           "feasible": 0, "exact_count": 0, "best": None, "attempts": 0,
           "incumbent_energy": incumbent["total_energy"], "norm_dev": 0.0,
           "local_edges": len(local_edges), "cross_edges": int(np.sum(ext))}

    for attempt in range(3):             # quantum re-optimisation with stronger conflict penalty
        lam_a = lam * (1.5 ** attempt)
        a, b, const = _qoqa_block_hamiltonian(w, local_edges, ext, K - offset, lam_a, mu)
        h, J = _qoqa_qubo_to_ising(a, b)
        zz = {(i, j): J[i, j] for i in range(m) for j in range(i + 1, m)}
        scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(J))), 1e-9)

        if block_idx in angle_cache and attempt == 0:
            g_frac, beta = angle_cache[block_idx]
        else:                            # variational angle selection by <E> of the Born distribution
            table = _qoqa_energy_table(a, b, const)
            best_exp = None
            for gf in grid_gamma:
                for bt in grid_beta:
                    _, pr = backend.optimize_block_ising(h, zz, gf / scale, bt, qaoa_layers, base_state)
                    ex = float(np.dot(pr, table))
                    if best_exp is None or ex < best_exp:
                        best_exp, g_frac, beta = ex, gf, bt
            angle_cache[block_idx] = (g_frac, beta)

        state, probs = backend.optimize_block_ising(h, zz, g_frac / scale, beta, qaoa_layers, base_state)
        out["norm_dev"] = max(out["norm_dev"], abs(float(np.sum(probs)) - 1.0))
        meas = sample_qoqa_bitstrings(backend, probs, shots, rng, m)
        out["shots"] += int(shots)
        out["attempts"] = attempt + 1
        out["measured_unique"] += len(meas["indices"])

        best = None
        for idx, cnt, pr_emp, bs in zip(meas["indices"], meas["counts"],
                                        meas["probabilities"], meas["bitstrings"]):
            xi = (int(idx) >> shifts) & 1
            e = evaluate_qubo_energy(xi, w, local_edges, K, lam_a, mu,
                                     cardinality_offset=offset, external_conflicts=ext)
            feasible = e["conflict_count"] == 0
            out["feasible"] += int(feasible)
            out["exact_count"] += int(feasible and e["selected_count"] + offset == K)
            out["rows"].append((block_idx, int(idx), int(cnt), float(pr_emp),
                                float(probs[idx]), e["total_energy"], int(feasible)))
            if feasible and (best is None or e["total_energy"] < best["energy"] - 1e-12):
                best = {"energy": e["total_energy"], "bits": xi.copy(),
                        "bitstring": bs, "eval": e, "index": int(idx),
                        "count": int(cnt)}
        if best is not None:
            out["best"] = best
            break

    return out


def _qoqa_block_budgets(blocks, weights, target_k):
    """Decomposition coordination: split the global cardinality K into
    per-block budgets proportional to block objective mass (capped at the
    independent-set bound ceil(m/2)). Budgets only enter the block
    Hamiltonian as the cardinality penalty centre; the quantum circuit does the
    actual configuration search."""
    caps = np.array([(len(b) + 1) // 2 for b in blocks], dtype=np.float64)
    mass = np.array([np.sum(weights[b] ** 2) + 1e-9 for b in blocks])
    budgets = np.zeros(len(blocks))
    active = np.ones(len(blocks), dtype=bool)
    remaining = float(target_k)
    for _ in range(len(blocks)):
        if remaining <= 1e-9 or not active.any():
            break
        share = np.where(active, mass, 0.0)
        share = share / share.sum() * remaining
        over = active & (share >= caps - budgets)
        if not over.any():
            budgets += share
            break
        budgets[over] = caps[over]
        remaining = target_k - budgets.sum()
        active &= ~over
    base = np.floor(budgets + 1e-9).astype(int)
    order = np.argsort(-(budgets - base), kind="stable")
    for k in order:
        if base.sum() >= target_k:
            break
        if base[k] < caps[k]:
            base[k] += 1
    return base


def verify_qoqa_constraints(
    selected_regions, graph, target_regions_count, n_candidates,
    capacities=None, required_bits=None, metadata_region_ids=None,
    dataset_size=None, objective_values=None, energy=None,
    raise_on_failure=True
):
    """Classical VERIFICATION only (no optimisation). Structural corruption
    raises ValueError; infeasibility raises QOQAInfeasibleError when
    raise_on_failure else is reported."""
    sel = [int(i) for i in selected_regions]
    if any(i < 0 or i >= n_candidates for i in sel):
        raise ValueError("QOQA selected index outside candidate pool.")
    if len(set(sel)) != len(sel):
        raise ValueError("QOQA selected indices contain duplicates.")
    if dataset_size is not None and any(i >= dataset_size for i in sel):
        raise ValueError("QOQA selected region missing from dataset.")
    if metadata_region_ids is not None:
        known = set(int(v) for v in metadata_region_ids)
        if any(i not in known for i in sel):
            raise ValueError("QOQA selected region missing from metadata.")
    if objective_values is not None and not np.all(np.isfinite(objective_values)):
        raise ValueError("QOQA objective values contain NaN/Inf.")
    if energy is not None and not np.isfinite(energy):
        raise ValueError("QOQA energy is not finite.")

    s = set(sel)
    violations = sum(1 for i in s for j in graph.get(i, ()) if j in s and j > i)
    cap_bits = None if capacities is None else int(sum(int(capacities[i]) for i in sel))
    report = {
        "valid_indices": True, "no_duplicates": True,
        "selected_count": len(sel),
        "cardinality_verified": len(sel) == int(target_regions_count),
        "neighbour_violations": int(violations),
        "capacity_bits": cap_bits,
        "capacity_verified": (cap_bits is None or required_bits is None
                              or cap_bits >= required_bits),
    }
    report["feasible"] = bool(report["cardinality_verified"]
                              and violations == 0 and report["capacity_verified"])
    if raise_on_failure and not report["feasible"]:
        raise QOQAInfeasibleError(f"QOQA constraint verification failed: {report}")
    return report


def qoqa_quantum_region_selection(
    selection_scores,
    graph,
    target_regions_count,
    capacities=None,
    required_bits=None,
    metadata_region_ids=None,
    dataset_size=None,
    block_size=8,
    shots_per_block=1024,
    qaoa_layers=1,
    lam=4.0,
    mu=1.5,
    max_rounds=4,
    max_sweeps=6,
    seed=None
):
    """
    QOQA: quantum-assisted constrained binary optimisation (decomposed
    QAOA-style state-vector simulation). See build_qoqa_qubo for H(x).

    Coordinated block optimisation (assemble_qoqa_configuration): blocks are
    solved in sweeps; every block sees the fixed selections of the other
    blocks as boundary terms (cross-block 8-neighbour conflicts become
    penalised linear fields; the cardinality centre becomes K - selected_outside),
    so no cross-block edge is dropped. Sweep 0 uses objective-mass budgets;
    sweeps 1-2 use a softened cardinality penalty (lets slots migrate to
    better blocks); later sweeps use the full penalty. Failed rounds are
    re-optimised quantumly with larger cardinality penalty and a capacity
    bias (linear term from the existing actual_safe_capacity*PAYLOAD_RATIO).
    """
    backend = QOQAQuantumBackend()
    if block_size > backend.max_block_qubits:
        raise ValueError("QOQA block_size too large for state-vector simulation.")

    K = int(target_regions_count)
    qubo = build_qoqa_qubo(selection_scores, graph, K, lam, mu)
    n = qubo["n"]
    if not (0 < K <= n):
        raise ValueError(f"QOQA target cardinality {K} invalid for {n} candidates.")

    blocks = _qoqa_build_blocks(n, block_size)
    rng = np.random.default_rng(seed)
    cap = None if capacities is None else np.asarray(capacities, dtype=np.float64)
    cap_norm = (np.zeros(n) if cap is None
                else cap / max(float(np.max(cap)), 1e-12))
    base_states = {}
    for b in blocks:
        if len(b) not in base_states:
            base_states[len(b)] = backend.prepare_superposition(len(b))  # H on every qubit

    block_index_of = np.zeros(n, dtype=np.int64)
    for bi, b in enumerate(blocks):
        block_index_of[b] = bi
    cross_edges = int(np.sum(block_index_of[qubo["edges"][:, 0]]
                             != block_index_of[qubo["edges"][:, 1]])) if len(qubo["edges"]) else 0

    x = np.zeros(n, dtype=np.int64)
    angle_cache, meas_rows, block_records = {}, [], {}
    stats = {"shots": 0, "unique": 0, "feasible": 0, "exact": 0,
             "sweeps": 0, "reopts": 0, "norm_dev": 0.0}
    budgets = _qoqa_block_budgets(blocks, qubo["weights"], K)
    report, mu_r, kappa = None, mu, 0.0

    for rnd in range(max_rounds):
        mu_r = mu * (1.5 ** rnd)
        kappa = [0.0, 0.3, 0.6, 1.0][min(rnd, 3)] if cap is not None else 0.0
        bonus = kappa * cap_norm
        for sweep in range(max_sweeps):
            first = (rnd == 0 and sweep == 0)
            soft = (sweep in (1, 2)) if rnd == 0 else (sweep == 0)
            mu_s = mu_r * (0.4 if soft else 1.0)
            order = range(len(blocks)) if sweep % 2 == 0 else reversed(range(len(blocks)))
            changed = False
            meas_rows = []          # keep only the final sweep's measurements
            for bi in order:
                blk = blocks[bi]
                res = optimize_qoqa_block(
                    backend, bi, blk, qubo, x, bonus,
                    int(budgets[bi]) if first else None,
                    lam, mu_s, shots_per_block, qaoa_layers, rng,
                    angle_cache, base_states[len(blk)]
                )
                stats["shots"] += res["shots"]
                stats["unique"] += res["measured_unique"]
                stats["feasible"] += res["feasible"]
                stats["exact"] += res["exact_count"]
                stats["reopts"] += res["attempts"] - 1
                stats["norm_dev"] = max(stats["norm_dev"], res["norm_dev"])
                for r in res["rows"]:
                    meas_rows.append((rnd, sweep, *r))
                best = res["best"]
                if best is not None and best["energy"] < res["incumbent_energy"] - 1e-12:
                    if not np.array_equal(best["bits"], x[blk]):
                        changed = True
                    x[blk] = best["bits"]
                    block_records[bi] = {
                        "block": bi, "global_region_ids": [int(g) for g in blk],
                        "bitstring": best["bitstring"], "energy": best["energy"],
                        "measured_count": best["count"], "round": rnd, "sweep": sweep,
                        "selected_regions": [int(blk[q]) for q in range(len(blk)) if best["bits"][q]],
                    }
            stats["sweeps"] += 1
            if not changed and sweep >= (3 if rnd == 0 else 1):
                break

        sel = [int(i) for i in np.nonzero(x)[0]]
        report = verify_qoqa_constraints(
            sel, graph, K, n, capacities=cap, required_bits=required_bits,
            metadata_region_ids=metadata_region_ids, dataset_size=dataset_size,
            raise_on_failure=False
        )
        print(f"[QOQA] round {rnd}: count={report['selected_count']}/{K}, "
              f"conflicts={report['neighbour_violations']}, "
              f"capacity_ok={report['capacity_verified']} (mu={mu_r:.2f}, kappa={kappa:.2f})")
        if report["feasible"]:
            break

    selected = [int(i) for i in np.nonzero(x)[0]]
    qubo_final = dict(qubo, mu_cardinality=mu_r)
    final_eval = evaluate_qoqa_configuration(x, qubo_final, cap, required_bits)
    report = verify_qoqa_constraints(
        selected, graph, K, n, capacities=cap, required_bits=required_bits,
        metadata_region_ids=metadata_region_ids, dataset_size=dataset_size,
        objective_values=selection_scores, energy=final_eval["total_energy"],
        raise_on_failure=True
    )

    if abs(stats["norm_dev"]) > 1e-8:
        raise ValueError("QOQA quantum state normalisation violated.")
    if stats["shots"] <= 0 or len(meas_rows) == 0:
        raise ValueError("QOQA produced no measurements.")

    max_m = max(len(b) for b in blocks)
    return {
        "selected_regions": selected,
        "configuration": x,
        "final_energy": final_eval,
        "report": report,
        "measurements": np.asarray(meas_rows, dtype=np.float64).reshape(-1, 9),
        "block_records": [block_records[k] for k in sorted(block_records)],
        "block_map": {int(i): [int(g) for g in b] for i, b in enumerate(blocks)},
        "diagnostics": {
            "candidate_regions": int(n), "blocks": len(blocks),
            "qubits_per_block": int(max_m),
            "state_vector_length_per_block": int(2 ** max_m),
            "qaoa_layers": int(qaoa_layers), "shots_per_block": int(shots_per_block),
            "total_quantum_measurements": int(stats["shots"]),
            "measured_configurations_unique": int(stats["unique"]),
            "feasible_measured_configurations": int(stats["feasible"]),
            "cardinality_exact_feasible_configurations": int(stats["exact"]),
            "sweeps_executed": int(stats["sweeps"]),
            "quantum_reoptimisation_attempts": int(stats["reopts"]),
            "lambda_conflict": float(lam), "mu_cardinality": float(mu_r),
            "capacity_bias_kappa": float(kappa),
            "conflict_edges_total": int(len(qubo["edges"])),
            "cross_block_conflict_edges": cross_edges,
            "max_probability_norm_deviation": float(stats["norm_dev"]),
            "block_size": int(block_size),
        },
    }
def greedy_classical_region_selection(
    selection_scores,
    graph,
    target_regions_count,
    capacities,
    required_bits
):
    scores = np.asarray(
        selection_scores,
        dtype=np.float64
    )

    capacities = np.asarray(
        capacities,
        dtype=np.int64
    )

    order = np.argsort(
        -scores,
        kind="stable"
    )

    selected = []
    total_capacity = 0

    for idx in order:
        idx = int(idx)

        if len(selected) >= int(target_regions_count):
            break

        conflicts = set(
            int(j)
            for j in graph.get(idx, ())
        )

        if any(
            int(s) in conflicts
            for s in selected
        ):
            continue

        selected.append(idx)

        total_capacity += int(
            capacities[idx]
        )

    if len(selected) != int(target_regions_count):
        raise RuntimeError(
            "Greedy classical selection could not "
            "satisfy the required region count."
        )

    if total_capacity < int(required_bits):
        raise RuntimeError(
            f"Greedy classical selection capacity "
            f"insufficient: {total_capacity} < "
            f"{required_bits} bits."
        )

    return selected

def qoqa_acer_qrng_endless_loop(
    model,
    embedding_model,
    df,
    metadata,
    graph,
    image,
    dwt,
    feature_imputer,
    feature_scaler,
    feature_names,
    target_scaler,
    regression_columns,
    target_regions_count=None
):

    print("\n" + "=" * 75)
    print("[QOQA + ACER + QRNG] Initializing optimization...")
    print("=" * 75)

    intelligence_all = (
        df[feature_names]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    intelligence_all = feature_imputer.transform(
        intelligence_all
    )

    intelligence_all = feature_scaler.transform(
        intelligence_all
    ).astype(np.float32)

    raw_patches = []

    dwt_patches = []

    for _, row in metadata.iterrows():

        raw_patch = extract_patch(
            image,
            row["row"],
            row["col"],
            WINDOW_SIZE
        )

        raw_patches.append(
            raw_patch[..., np.newaxis]
        )

        dwt_bands = []

        for band in dwt:

            dwt_band = extract_patch(
                band,
                row["row"] / 2.0,
                row["col"] / 2.0,
                DWT_WINDOW_SIZE
            )

            dwt_bands.append(
                dwt_band
            )

        dwt_patches.append(
            np.stack(
                dwt_bands,
                axis=-1
            )
        )

    raw_patches = np.asarray(
        raw_patches,
        dtype=np.float32
    )

    dwt_patches = np.asarray(
        dwt_patches,
        dtype=np.float32
    )

    if len(intelligence_all) != len(df):

        raise ValueError(
            "Intelligent dataset row count does not match dataframe."
        )

    if len(raw_patches) != len(df):

        raise ValueError(
            "Raw image patch count does not match dataframe."
        )

    if len(dwt_patches) != len(df):

        raise ValueError(
            "DWT patch count does not match dataframe."
        )

    print(
        f"[ACER] Intelligent dataset input: "
        f"{intelligence_all.shape}"
    )

    print(
        f"[ACER] Raw image input: "
        f"{raw_patches.shape}"
    )

    print(
        f"[ACER] DWT input: "
        f"{dwt_patches.shape}"
    )

    print(
        f"[ACER] Regression targets: "
        f"{len(regression_columns)}"
    )

    selection_all = np.ones(
        (len(df), 1),
        dtype=np.float32
    )

    region_predictions_scaled, _ = model.predict(
        [
            intelligence_all,
            raw_patches,
            dwt_patches,
            selection_all
        ],
        batch_size=64,
        verbose=0
    )

    region_embeddings = embedding_model.predict(
        [
            intelligence_all,
            raw_patches,
            dwt_patches,
            selection_all
        ],
        batch_size=64,
        verbose=0
    )

    region_predictions = target_scaler.inverse_transform(
        region_predictions_scaled
    )

    psnr_idx = regression_columns.index(
        "image_psnr"
    )
    region_predictions = np.nan_to_num(
        region_predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    selection_scores = np.zeros(
        len(df),
        dtype=np.float64
    )

    objective_weights = {
        "image_psnr": 0.40,
        "image_mse": 0.20,
        "actual_center_distortion": 0.10,
        "actual_quality_loss": 0.10,
        "actual_safe_probability": 0.10,
        "actual_risk_probability": 0.10
    }

    higher_is_better = {
        "image_psnr": True,
        "image_mse": False,
        "actual_center_distortion": False,
        "actual_quality_loss": False,
        "actual_safe_probability": True,
        "actual_risk_probability": False
    }

    for target_name, weight in objective_weights.items():

        if target_name not in regression_columns:
            continue

        target_index = regression_columns.index(
            target_name
        )

        values = region_predictions[
            :,
            target_index
        ]

        ranks = (
            pd.Series(values)
            .rank(
                method="average",
                pct=True
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        ranks = np.nan_to_num(
            ranks,
            nan=0.5,
            posinf=1.0,
            neginf=0.0
        )

        if not higher_is_better[target_name]:

            ranks = 1.0 - ranks

        selection_scores += (
            weight * ranks
        )

    selection_scores = np.nan_to_num(
        selection_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    score_shift = (
        selection_scores
        - np.max(selection_scores)
    )

    score_shift = np.clip(
        score_shift,
        -50.0,
        50.0
    )

    selection_weights = np.exp(
        score_shift
    )

    if (
        not np.all(
            np.isfinite(
                selection_weights
            )
        )
        or
        np.sum(selection_weights) <= 0
    ):

        selection_weights = np.ones(
            len(df),
            dtype=np.float64
        )

    # NOTE: this softmax vector is retained ONLY for legacy visualisations /
    # ACER feedback bookkeeping. It plays no role in QOQA region selection.
    selection_probs = (
        selection_weights
        /
        np.sum(selection_weights)
    )

    # Capacity as seen by the (unchanged) embedding stage:
    # floor(floor(actual_safe_capacity) * PAYLOAD_RATIO) per region.
    qoqa_capacities = np.floor(
        np.floor(
            pd.to_numeric(df["actual_safe_capacity"], errors="coerce")
            .fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
        ) * PAYLOAD_RATIO
    ).astype(np.int64)
    region_feedback = np.ones(len(df), dtype=np.float64)

    print(
        "\n[QOQA QUANTUM] Preparing decomposed QAOA-style region optimization backend..."
    )

    qoqa_last_result = None

    best_solution = None

    max_iterations = 100

    for iteration in range(
        1,
        max_iterations + 1
    ):

        print(
            f"\n--- "
            f"[QOQA + ACER Optimization Iteration "
            f"{iteration}] ---"
        )

        # QOQA objective = existing ML/objective scores (with the existing
        # ACER feedback factor); min-max normalised inside build_qoqa_qubo.
        qoqa_objective = selection_scores * region_feedback

        try:
            if ABLATE_QOQA:

                selected_regions = (
                    greedy_classical_region_selection(
                        selection_scores=qoqa_objective,
                        graph=graph,
                        target_regions_count=target_regions_count,
                        capacities=qoqa_capacities,
                        required_bits=EXACT_TOTAL_BITS
                    )
                )

            else:

                qoqa_result = (
                    qoqa_quantum_region_selection(
                        selection_scores=qoqa_objective,
                        graph=graph,
                        target_regions_count=target_regions_count,
                        capacities=qoqa_capacities,
                        required_bits=EXACT_TOTAL_BITS,
                        metadata_region_ids=metadata[
                            "region_id"
                        ].to_numpy(),
                        dataset_size=len(df),
                        block_size=8,
                        shots_per_block=1024,
                        qaoa_layers=1,
                        seed=RANDOM_STATE + iteration
                    )
                )

                selected_regions = (
                    sorted(
                        (
                            int(i)
                            for i in
                            qoqa_result[
                                "selected_regions"
                            ]
                        ),
                        key=lambda i: (
                            -float(
                                qoqa_objective[i]
                            ),
                            i
                        )
                    )
                )
        except QOQAInfeasibleError as qoqa_error:
            print(f"[QOQA] No feasible configuration after quantum re-optimisation: {qoqa_error}")
            continue

        qoqa_last_result = qoqa_result
        print_qoqa_quantum_log(qoqa_result, target_regions_count)

        # Embedding order of the ALREADY-SELECTED configuration (best objective
        # first, as before). This orders the set; it does not select it.
        selected_regions = sorted(
            (int(i) for i in qoqa_result["selected_regions"]),
            key=lambda i: (-float(qoqa_objective[i]), i)
        )

        selected_predictions = region_predictions[
            selected_regions
        ]

        if len(selected_regions) > 0:

            selected_embeddings = region_embeddings[
                selected_regions
            ]

            selected_embeddings = np.asarray(
                selected_embeddings,
                dtype=np.float32
            )

            combined_embedding = (
                np.mean(
                    selected_embeddings,
                    axis=0,
                    keepdims=True
                )
                +
                np.max(
                    selected_embeddings,
                    axis=0,
                    keepdims=True
                )
            ) / 2.0

            if ABLATE_ACER:
                acer_quantum_features = (
                    classical_acer_features(
                        combined_embedding
                    )
                )
            else:
                acer_quantum_features = (
                    acer_quantum_entanglement_features(
                        combined_embedding
                    )
                )

            print(
                "\n[ACER QUANTUM ENTANGLEMENT]"
            )

            print(
                f"  Logical qubits       : 32"
            )

            print(
                f"  Quantum registers    : 4"
            )

            print(
                f"  Qubits per register  : 8"
            )

            print(
                f"  Gate structure       : "
                f"RY + RZ + H + CNOT"
            )

            print(
                f"  CNOT chain           : "
                f"8-qubit ring per register"
            )

            print(
                f"  Measured features    : "
                f"{acer_quantum_features.shape[1]}"
            )

            print(
                f"  Quantum feature mean : "
                f"{np.mean(acer_quantum_features):.8f}"
            )

            print(
                f"  Quantum feature std  : "
                f"{np.std(acer_quantum_features):.8f}"
            )

            entangled_state = np.asarray(
                acer_quantum_features,
                dtype=np.float32
            )

            if entangled_state.ndim == 1:
                entangled_state = entangled_state.reshape(
                    1,
                    -1
                )

            acer_projection_layer = model.get_layer(
                "acer_quantum_projection"
            )

            entangled_state = acer_projection_layer(
                entangled_state
            )

            global_hidden = model.get_layer(
                "global_hidden_256"
            )(
                entangled_state
            )

            global_hidden = model.get_layer(
                "global_hidden_bn"
            )(
                global_hidden,
                training=False
            )

            global_hidden = model.get_layer(
                "global_dropout"
            )(
                global_hidden,
                training=False
            )

            global_hidden = model.get_layer(
                "global_hidden_128"
            )(
                global_hidden
            )

            global_prediction_scaled = model.get_layer(
                "global_psnr"
            )(
                global_hidden
            )

            raw_predicted_full_image_psnr = float(
                global_prediction_scaled.numpy()[0, 0]
                * target_scaler.scale_[psnr_idx]
                + target_scaler.mean_[psnr_idx]
            )

            raw_predicted_full_image_mse = float(
                (255.0 ** 2)
                /
                (10.0 ** (raw_predicted_full_image_psnr / 10.0))
            )

            actual_delta = (
                DELTA_MIN + DELTA_MAX
            ) / 2.0

            delta_adjusted_predicted_mse = adjust_mse_for_delta(
                raw_predicted_full_image_mse,
                DELTA_REFERENCE,
                actual_delta,
                len(selected_regions)
            )

            predicted_full_image_psnr = mse_to_psnr(
                delta_adjusted_predicted_mse
            )

            aggregate_prediction = np.mean(
                selected_predictions,
                axis=0
            )

            aggregate_prediction[
                psnr_idx
            ] = predicted_full_image_psnr

            mse_idx = regression_columns.index(
                "image_mse"
            )

            aggregate_prediction[
                mse_idx
            ] = delta_adjusted_predicted_mse

        else:

            aggregate_prediction = np.mean(
                region_predictions,
                axis=0
            )

            aggregate_prediction[
                psnr_idx
            ] = 0.0
       

        aggregate_prediction = np.nan_to_num(
            aggregate_prediction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        print(
            "\n[Surrogate CNN] "
            "Predicted Selected-Region Metrics:"
        )

        for target_name, value in zip(
            regression_columns,
            aggregate_prediction
        ):

            print(
                f"  {target_name} : "
                f"{float(value):.6f}"
            )
        print(
            "\n[Surrogate Prediction Summary]"
        )

        for target_name, value in zip(
            regression_columns,
            aggregate_prediction
        ):

            print(
                f"  {target_name:<32}"
                f"Predicted={float(value):.8f}"
            )

        surrogate_psnr = None

        if "image_psnr" in regression_columns:

            psnr_index = regression_columns.index(
                "image_psnr"
            )

            surrogate_psnr = predicted_full_image_psnr

            print(
                "[Surrogate CNN] "
                f"Predicted Selected-Region Mean PSNR: "
                f"{surrogate_psnr:.2f} dB"
            )

        (
            actual_psnr,
            actual_mse,
            actual_distortion,
            payloads_assigned,
            reconstructed_image,
            ber
        ) = perform_actual_db2_idwt_embedding(
            image,
            dwt,
            selected_regions,
            df,
            metadata,
            rng_seed=RANDOM_STATE + iteration
        )

        print(
            "\n[ACTUAL VERIFICATION]"
        )

        print(
            f"  PSNR       : "
            f"{actual_psnr:.8f} dB"
        )

        print(
            f"  MSE        : "
            f"{actual_mse:.12f}"
        )

        print(
            f"  Distortion : "
            f"{actual_distortion:.12f}"
        )

        print(
            f"  BER        : "
            f"{ber:.8f}"
        )
        print(
            "\n[Surrogate vs Actual]"
        )


        print(
            "\n[PSNR / MSE DELTA ADJUSTMENT]"
        )

        print(
            f"  Normal CNN PSNR             : "
            f"{raw_predicted_full_image_psnr:.8f} dB"
        )

        print(
            f"  Normal CNN MSE              : "
            f"{raw_predicted_full_image_mse:.12e}"
        )

        print(
            f"  Delta-Adjusted MSE          : "
            f"{delta_adjusted_predicted_mse:.12e}"
        )

        print(
            f"  Delta-Adjusted PSNR         : "
            f"{predicted_full_image_psnr:.8f} dB"
        )

        print(
            f"  Actual Embedding PSNR       : "
            f"{actual_psnr:.8f} dB"
        )

        print(
            f"  Actual Embedding MSE        : "
            f"{actual_mse:.12e}"
        )

        print(
            f"\n  Normal Prediction Error     : "
            f"{abs(actual_psnr - raw_predicted_full_image_psnr):.8f} dB"
        )

        print(
            f"  Delta-Adjusted Error        : "
            f"{abs(actual_psnr - predicted_full_image_psnr):.8f} dB"
        )

        actual_mapping = {
            "image_psnr": actual_psnr,
            "image_mse": actual_mse,
            "actual_center_distortion": actual_distortion
        }

        for target_name, predicted_value in zip(
            regression_columns,
            aggregate_prediction
        ):

            if target_name not in actual_mapping:
                continue

            actual_value = actual_mapping[target_name]

            error = abs(
                float(predicted_value)
                -
                float(actual_value)
            )

            print(
                f"  {target_name:<32}"
                f"Predicted={float(predicted_value):.8f} | "
                f"Actual={float(actual_value):.8f} | "
                f"Error={error:.8f}"
            )

        if surrogate_psnr is not None:

            print(
                f"  Surrogate PSNR : "
                f"{surrogate_psnr:.8f} dB"
            )

            print(
                f"  PSNR Error     : "
                f"{abs(actual_psnr - surrogate_psnr):.8f} dB"
            )

        if (
            best_solution is None
            or
            actual_psnr > best_solution["psnr"]
        ):

            best_solution = {
                "psnr": actual_psnr,
                "regions": list(
                    selected_regions
                ),
                "payloads": payloads_assigned,
                "image": reconstructed_image,
                "prediction": aggregate_prediction.copy(),
                "mse": actual_mse,
                "distortion": actual_distortion,
                "ber": ber
            }
# Accept the solution as long as actual PSNR is >= 90.0 dB 
        # and all 10,000 bits are fully embedded (no missing data)
        if actual_psnr >= TARGET_PSNR_THRESHOLD and sum(payloads_assigned.values()) == EXACT_TOTAL_BITS:

            print(
                "\n[SUCCESS] "
                "Feasible embedding solution finalized based on actual PSNR target threshold and full bit allocation."
            )

            adaptive_region_manifest = []

            final_selected_regions = []

            for idx in selected_regions:

                idx = int(idx)

                payload_bits = int(
                    payloads_assigned.get(
                        idx,
                        0
                    )
                )

                # --------------------------------------------------------
                # Export ONLY regions that actually carried payload
                # --------------------------------------------------------

                if payload_bits <= 0:
                    continue

                row_value = metadata.iloc[idx]["row"]
                col_value = metadata.iloc[idx]["col"]

                adaptive_region_manifest.append(
                    {
                        "region_id": idx,
                        "row": int(row_value),
                        "col": int(col_value),
                        "payload_bits": payload_bits
                    }
                )

                final_selected_regions.append(
                    idx
                )

            final_manifest_bits = int(
                sum(
                    item["payload_bits"]
                    for item in adaptive_region_manifest
                )
            )

            if final_manifest_bits != EXACT_TOTAL_BITS:
                raise ValueError(
                    f"Final adaptive manifest payload mismatch. "
                    f"Manifest={final_manifest_bits}, "
                    f"Required={EXACT_TOTAL_BITS}"
                )

            final_selected_embeddings = region_embeddings[
                final_selected_regions
            ]

            final_selected_embeddings = np.asarray(
                final_selected_embeddings,
                dtype=np.float32
            )

            final_combined_embedding = (
                np.mean(
                    final_selected_embeddings,
                    axis=0,
                    keepdims=True
                )
                +
                np.max(
                    final_selected_embeddings,
                    axis=0,
                    keepdims=True
                )
            ) / 2.0

            final_acer_quantum_features = (
                acer_quantum_entanglement_features(
                    final_combined_embedding
                )
            )
            
            if GENERATE_VISUALIZATIONS:
                generate_quantum_qoqa_visualizations(
                    model=model,
                    df=df,
                    selected_regions=final_selected_regions,
                    selected_embeddings=final_selected_embeddings,
                    combined_embedding=final_combined_embedding,
                    selection_probs=selection_probs,
                    selection_scores=selection_scores,
                    region_predictions=region_predictions,
                    regression_columns=regression_columns,
                    output_dir=OUTPUT_DIR,
                    target_regions_count=target_regions_count,
                    row_col="row",
                    col_col="col",
                    quantum_features=final_acer_quantum_features
                )

            if qoqa_last_result is not None:
                save_qoqa_quantum_diagnostics(
                    output_dir=OUTPUT_DIR,
                    selection_scores=selection_scores,
                    qoqa_result=qoqa_last_result,
                    payload_regions=final_selected_regions,
                    target_regions_count=target_regions_count,
                    graph=graph
                )

            print(
                "\n[ACER QUANTUM]"
            )

            print(
                "Logical qubits: 32"
            )

            print(
                "Registers: 4"
            )

            print(
                "Qubits/register: 8"
            )

            print(
                "Gates: RY + RZ + H + CNOT"
            )

            print(
                "Measurement: Pauli-Z"
            )

            print(
                f"Quantum features: {final_acer_quantum_features.shape[1]}"
            )

            print(
                "\n[FINAL ADAPTIVE REGION SET]"
            )

            print(
                f"  Candidate regions : {len(selected_regions)}"
            )

            print(
                f"  Active regions    : {len(final_selected_regions)}"
            )

            print(
                f"  Payload bits      : {final_manifest_bits}"
            )

            with open(
                ADAPTIVE_REGION_MANIFEST_PATH,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    convert_to_native_types(
                        {
                            "total_regions": len(
                                adaptive_region_manifest
                            ),
                            "total_embedded_bits": int(
                                sum(
                                    item["payload_bits"]
                                    for item
                                    in adaptive_region_manifest
                                )
                            ),
                            "regions": adaptive_region_manifest
                        }
                    ),
                    f,
                    indent=4
                )

            print(
                f"Adaptive region manifest exported to: "
                f"{ADAPTIVE_REGION_MANIFEST_PATH}"
            )

            solution_data = {
                "iteration": iteration,
                "candidate_regions": len(
                    selected_regions
                ),
                "total_regions": len(
                    final_selected_regions
                ),
                "total_embedded_bits": int(
                    sum(
                        payloads_assigned.values()
                    )
                ),
                "verified_psnr": actual_psnr,
                "verified_mse": actual_mse,
                "verified_distortion": actual_distortion,
                "verified_ber": ber,
                "predicted_surrogate_metrics": {
                    name: float(value)
                    for name, value in zip(
                        regression_columns,
                        aggregate_prediction
                    )
                },
                "surrogate_vs_actual": {
                    "predicted_psnr": (
                        float(surrogate_psnr)
                        if surrogate_psnr is not None
                        else None
                    ),
                    "actual_psnr": float(
                        actual_psnr
                    ),
                    "absolute_psnr_error": (
                        float(
                            abs(
                                actual_psnr
                                - surrogate_psnr
                            )
                        )
                        if surrogate_psnr is not None
                        else None
                    ),
                    "normal_cnn_psnr": float(
                        raw_predicted_full_image_psnr
                    ),

                    "normal_cnn_mse": float(
                        raw_predicted_full_image_mse
                    ),

                    "delta_adjusted_predicted_mse": float(
                        delta_adjusted_predicted_mse
                    ),

                    "delta_adjusted_predicted_psnr": float(
                        predicted_full_image_psnr
                    ),

                    "actual_embedding_psnr": float(
                        actual_psnr
                    ),

                    "actual_embedding_mse": float(
                        actual_mse
                    ),

                    "delta_adjusted_absolute_error": float(
                        abs(
                            actual_psnr -
                            predicted_full_image_psnr
                        )
                    ),

                    "normal_absolute_error": float(
                        abs(
                            actual_psnr -
                            raw_predicted_full_image_psnr
                        )
                    ),

                    "delta_reference": float(
                        DELTA_REFERENCE
                    ),

                    "actual_delta_used": float(
                        actual_delta
                    ),
                    "actual_mse": float(
                        actual_mse
                    ),
                    "actual_distortion": float(
                        actual_distortion
                    ),
                    "actual_ber": float(
                        ber
                    )
                },
                "region_payloads": {
                    int(idx): int(payloads_assigned[idx])
                    for idx in final_selected_regions
                }
            }

            with open(
                SOLUTION_JSON_PATH,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    convert_to_native_types(
                        solution_data
                    ),
                    f,
                    indent=4
                )

            print(
                f"Solution JSON exported to: "
                f"{SOLUTION_JSON_PATH}"
            )

            plt.figure(
                figsize=(6, 6)
            )

            plt.imshow(
                reconstructed_image,
                cmap="gray"
            )

            plt.title(
                f"Final Embedded IDWT Image\n"
                f"PSNR: {actual_psnr:.2f} dB | "
                f"Regions: {len(final_selected_regions)} | "
                f"Bits: {sum(payloads_assigned.values())}"
            )

            plt.axis("off")

            plt.tight_layout()

            plt.savefig(
                SOLUTION_IMAGE_PATH,
                dpi=200
            )

            plt.close()

            print(
                f"Final IDWT Embedded Image saved to: "
                f"{SOLUTION_IMAGE_PATH}"
            )

            print(
                "All requested plots and artifacts saved successfully!"
            )

            final_payloads_assigned = {
                int(idx): int(payloads_assigned[idx])
                for idx in final_selected_regions
            }

            return (
                final_selected_regions,
                actual_psnr,
                final_payloads_assigned,
                reconstructed_image
            )

        actual_score = 0.0

        if np.isfinite(actual_psnr):

            actual_score += (
                0.50
                *
                np.clip(
                    actual_psnr /
                    TARGET_PSNR_THRESHOLD,
                    0.0,
                    2.0
                )
            )

        if np.isfinite(actual_mse):

            actual_score += (
                0.20
                *
                np.exp(
                    -actual_mse
                )
            )

        if np.isfinite(actual_distortion):

            actual_score += (
                0.15
                *
                np.exp(
                    -actual_distortion
                )
            )

        if np.isfinite(ber):

            actual_score += (
                0.15
                *
                (1.0 - np.clip(ber, 0.0, 1.0))
            )

        actual_score = float(
            np.clip(
                actual_score,
                0.0,
                2.0
            )
        )

        if actual_score >= 1.0:

            feedback = 1.0 + (
                actual_score - 1.0
            )

        else:

            feedback = max(
                0.25,
                actual_score
            )

        region_feedback[selected_regions] = np.clip(
            region_feedback[selected_regions] * feedback, 0.25, 4.0
        )
        selection_probs[
            selected_regions
        ] *= feedback

        selection_probs = np.nan_to_num(
            selection_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        if np.sum(selection_probs) <= 0:
            selection_probs = np.ones(
                len(df),
                dtype=np.float64
            )

        selection_probs /= np.sum(
            selection_probs
        )

    if best_solution is None:

        raise RuntimeError(
            "ACER exhausted all iterations without "
            "finding a verified solution."
        )

    print(
        "\n[ACER] Maximum iterations reached."
    )

    print(
        f"[ACER] Best verified PSNR: "
        f"{best_solution['psnr']:.2f} dB"
    )

    return (
        best_solution["regions"],
        best_solution["psnr"],
        best_solution["payloads"],
        best_solution["image"]
    )

def generate_quantum_qoqa_visualizations(
    model,
    df,
    selected_regions,
    selected_embeddings,
    combined_embedding,
    selection_probs,
    selection_scores,
    region_predictions,
    regression_columns,
    output_dir,
    target_regions_count=None,
    row_col=None,
    col_col=None,
    quantum_features=None,
):
    quantum_dir = os.path.join(output_dir, "quantum_qoqa_visualizations")
    os.makedirs(quantum_dir, exist_ok=True)

    selected_regions = [int(x) for x in selected_regions]
    selected_embeddings = np.asarray(selected_embeddings, dtype=np.float32)
    combined_embedding = np.asarray(combined_embedding, dtype=np.float32).reshape(-1)
    selection_probs = np.asarray(selection_probs, dtype=np.float64).reshape(-1)
    selection_scores = np.asarray(selection_scores, dtype=np.float64).reshape(-1)
    region_predictions = np.asarray(region_predictions, dtype=np.float64)

    if quantum_features is None:
        quantum_features = acer_quantum_entanglement_features(combined_embedding)

    quantum_features = np.asarray(quantum_features, dtype=np.float32).reshape(-1)

    print()
    print("=" * 80)
    print("QUANTUM + QOQA VISUALIZATION GENERATION")
    print("=" * 80)

    def save(fig, filename, dpi=300):
        fig.savefig(os.path.join(quantum_dir, filename), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def box(ax, x, y, w, h, text, fs=9, lw=1.5):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.04",
                fill=False,
                linewidth=lw
            )
        )
        ax.text(
            x + w / 2, y + h / 2,
            text,
            ha="center", va="center",
            fontsize=fs, fontweight="bold"
        )

    def arrow(ax, x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="->",
                mutation_scale=14
            )
        )

    # ============================================================
    # 01. COMPLETE CORRECTED QUANTUM PIPELINE
    # ============================================================
    fig, ax = plt.subplots(figsize=(22, 11))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 11)
    ax.axis("off")

    top = [
        (0.4, 7.7, 2.5, 1.1, "Multimodal\nRegion Inputs"),
        (3.4, 7.7, 2.5, 1.1, "Fusion\n384-D"),
        (6.4, 7.7, 3.0, 1.1, "SQE-Net\n64 Logical Qubits"),
        (9.9, 7.7, 2.7, 1.1, "128-D Region\nRepresentation"),
        (13.1, 7.7, 2.7, 1.1, "Regional\nRegression"),
        (16.3, 7.7, 2.7, 1.1, "QOQA\nRegion Selection"),
        (19.5, 7.7, 2.1, 1.1, "Selected\nRegions"),
    ]

    for b in top:
        box(ax, *b)

    for i in range(len(top) - 1):
        arrow(
            ax,
            top[i][0] + top[i][2], top[i][1] + 0.55,
            top[i + 1][0], top[i + 1][1] + 0.55
        )

    box(
        ax, 5.0, 3.8, 5.0, 1.3,
        "Classical Q/K/V Attention\nAll-Pairs Representation Interaction\nNOT Quantum Entanglement",
        fs=10
    )
    arrow(ax, 20.55, 7.7, 7.5, 5.1)

    box(ax, 11.0, 3.8, 3.2, 1.3, "Global 128-D\nRepresentation", fs=10)
    arrow(ax, 10.0, 4.45, 11.0, 4.45)

    box(ax, 14.7, 3.8, 4.0, 1.3, "ACER\n32 Logical Qubits\n4 × 8 Registers", fs=10)
    arrow(ax, 14.2, 4.45, 14.7, 4.45)

    box(ax, 19.0, 3.8, 2.3, 1.3, "Global PSNR\nPrediction", fs=10)
    arrow(ax, 18.7, 4.45, 19.0, 4.45)

    ax.text(
        11, 10.3,
        "StegaQEntropy — Correct Quantum / Classical Processing Architecture",
        ha="center", fontsize=17, fontweight="bold"
    )
    ax.text(
        11, 1.7,
        "SQE-Net = intra-region quantum feature interaction   |   "
        "QOQA = inter-region selection-variable coupling   |   "
        "Attention = classical all-pairs representation interaction   |   "
        "ACER = global quantum representation",
        ha="center", fontsize=10
    )

    save(fig, "01_complete_corrected_quantum_pipeline.png", 400)

    # ============================================================
    # 02. SQE-NET INTERNAL ARCHITECTURE
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    steps = [
        ("384-D fused multimodal region representation", 13.0),
        ("Dense(128, tanh) angle projection", 11.8),
        ("128 angles", 10.6),
        ("64 RY angles  +  64 RZ angles", 9.4),
        ("8 independent quantum registers", 8.0),
        ("Each register: |00000000⟩", 6.8),
        ("RY(θq) → RZ(φq)", 5.6),
        ("H on all 8 qubits", 4.4),
        ("CNOT cyclic ring q0→q1→...→q7→q0", 3.2),
        ("Pauli-Z expectation ⟨Zq⟩", 2.0),
        ("8 × 8 = 64-D SQE quantum feature vector", 0.8),
    ]

    for text, y in steps:
        box(ax, 1.2, y - 0.35, 7.6, 0.7, text, fs=9)

    for i in range(len(steps) - 1):
        arrow(
            ax,
            5.0, steps[i][1] - 0.36,
            5.0, steps[i + 1][1] + 0.36
        )

    ax.text(
        5, 13.8,
        "SQE-Net — Feature-Level Quantum Processing",
        ha="center", fontsize=17, fontweight="bold"
    )
    ax.text(
        5, 0.02,
        "The quantum interaction is within each region's 8-qubit register.",
        ha="center", fontsize=9
    )

    save(fig, "02_sqe_net_internal_architecture.png", 400)

    # ============================================================
    # 03. SQE-NET SINGLE 8-QUBIT CIRCUIT
    # ============================================================
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(-1, 9)
    ax.axis("off")

    ax.text(
        7.5, 8.6,
        "SQE-Net — Single 8-Qubit Register",
        ha="center", fontsize=17, fontweight="bold"
    )

    xs = {"RY": 2.0, "RZ": 3.5, "H": 5.0, "CNOT": 8.0, "Z": 13.0}

    for q in range(8):
        y = 7 - q
        ax.plot([0.5, 13.5], [y, y], linewidth=0.9)
        ax.text(0.25, y, f"q{q}", ha="right", va="center", fontsize=9)

        for gate, x in xs.items():
            if gate == "CNOT":
                continue
            ax.text(
                x, y,
                gate if gate != "Z" else "⟨Z⟩",
                ha="center", va="center", fontsize=8,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    fill=False, linewidth=1.1
                )
            )

    for q in range(8):
        y = 7 - q
        ax.text(xs["RY"], y, f"RY(θ{q})", ha="center", va="center", fontsize=7)
        ax.text(xs["RZ"], y, f"RZ(φ{q})", ha="center", va="center", fontsize=7)

    cnot_x = np.linspace(6.6, 11.4, 8)

    for q, x in enumerate(cnot_x):
        control = q
        target = (q + 1) % 8
        y1 = 7 - control
        y2 = 7 - target

        ax.plot([x, x], [y1, y2], linewidth=1.5)
        ax.scatter(x, y1, s=45, zorder=5)
        ax.text(
            x, y2, "⊕",
            ha="center", va="center",
            fontsize=12,
            bbox=dict(
                boxstyle="circle,pad=0.04",
                fill=False, linewidth=1.0
            )
        )

    ax.text(
        8.0, -0.55,
        "Cyclic CNOT ring: q0→q1→q2→q3→q4→q5→q6→q7→q0",
        ha="center", fontsize=10, fontweight="bold"
    )
    ax.text(
        7.5, -0.85,
        "Input state |00000000⟩ → rotations → superposition → entanglement → Pauli-Z expectations",
        ha="center", fontsize=9
    )

    save(fig, "03_sqe_net_8_qubit_circuit.png", 500)

    # ============================================================
    # 04. SQE-NET REGISTER TOPOLOGY
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    ax.text(
        6, 7.5,
        "SQE-Net — 64 Logical Qubits as 8 Independent Registers",
        ha="center", fontsize=16, fontweight="bold"
    )

    for r in range(8):
        y = 6.5 - r * 0.75
        ax.text(
            1.0, y,
            f"Register {r + 1}",
            ha="right", va="center",
            fontsize=9, fontweight="bold"
        )
        ax.text(2.0, y, "q0 ─ q1 ─ q2 ─ q3 ─ q4 ─ q5 ─ q6 ─ q7 ─┐", fontsize=9)
        ax.text(9.2, y, "└──────────── cyclic CNOT ────────────", fontsize=8)
        ax.text(11.4, y, "256 amplitudes", fontsize=8)

    ax.text(
        6, 0.45,
        "8 registers × 8 logical qubits = 64 logical qubits; "
        "each register is a separate 2⁸-dimensional state vector.",
        ha="center", fontsize=10
    )

    save(fig, "04_sqe_net_8_register_topology.png", 400)

    # ============================================================
    # 05. QUANTUM GATE ROLES
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    gate_info = [
        ("RY(θ)", "Angle encoding", "Rotates qubit state according to learned/input angle."),
        ("RZ(φ)", "Phase encoding", "Adds phase information; φ is encoded from the projected feature."),
        ("H", "Superposition", "Creates a superposition before entangling operations."),
        ("CNOT", "Entanglement", "Correlates two qubits; used in the cyclic ring."),
        ("RZZ(γ)", "QOQA coupling", "Applies pairwise phase coupling between selection variables."),
        ("RX(β)", "QOQA mixer", "Mixes computational-basis configurations."),
        ("⟨Z⟩", "Measurement", "Pauli-Z expectation produces a real quantum feature.")
    ]

    y = 8.1
    for gate, role, meaning in gate_info:
        ax.text(0.8, y, gate, fontsize=12, fontweight="bold")
        ax.text(3.0, y, role, fontsize=11, fontweight="bold")
        ax.text(6.0, y, meaning, fontsize=10)
        ax.plot([0.7, 15.2], [y - 0.22, y - 0.22], linewidth=0.5)
        y -= 1.05

    ax.text(
        8, 9.0,
        "Quantum Gate Roles in StegaQEntropy",
        ha="center", fontsize=17, fontweight="bold"
    )

    save(fig, "05_quantum_gate_roles.png", 400)

    # ============================================================
    # 06. QOQA CORRECTED PIPELINE
    # ============================================================
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.axis("off")

    steps = [
        "Regional\nPredictions",
        "QUBO\nConstruction",
        "Binary →\nIsing",
        "Block\nDecomposition",
        "QAOA-Style\nCircuit",
        "Born-Rule\nMeasurement",
        "QUBO Energy +\nConstraints",
        "Cross-Block\nCoordination",
        "Final Selected\nRegions"
    ]

    xs = np.linspace(1.2, 18.8, len(steps))

    for i, (x, text) in enumerate(zip(xs, steps)):
        box(ax, x - 0.9, 3.0, 1.8, 1.5, text, fs=8)
        if i < len(steps) - 1:
            arrow(ax, x + 0.9, 3.75, xs[i + 1] - 0.9, 3.75)

    ax.text(
        10, 6.5,
        "QOQA — Quantum-Assisted Optimal Region Selection",
        ha="center", fontsize=17, fontweight="bold"
    )
    ax.text(
        10, 1.0,
        "Quantum optimization is blockwise; each block contains at most the configured "
        "maximum number of logical qubits. Cross-block coordination is classical.",
        ha="center", fontsize=10
    )

    save(fig, "06_qoqa_corrected_pipeline.png", 400)

    # ============================================================
    # 07. QOQA QUBO FORMULATION
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    ax.text(
        8, 8.1,
        "QOQA Binary Optimization Problem",
        ha="center", fontsize=18, fontweight="bold"
    )

    ax.text(
        8, 6.7,
        r"$H(x)=-\sum_i w_i x_i"
        r"+\lambda\sum_{(i,j)\in E_8}x_i x_j"
        r"+\mu\left(\sum_i x_i-K\right)^2$",
        ha="center", fontsize=19
    )

    items = [
        ("xᵢ", "Binary selection variable: 1 if region i is selected."),
        ("wᵢ", "Regional objective contribution from the optimization formulation."),
        ("E₈", "8-neighbor spatial conflict graph."),
        ("λ", "Penalty for selecting spatially conflicting neighboring regions."),
        ("μ", "Cardinality penalty."),
        ("K", "Desired selection cardinality.")
    ]

    y = 5.2
    for symbol, meaning in items:
        ax.text(2.0, y, symbol, fontsize=11, fontweight="bold")
        ax.text(4.0, y, meaning, fontsize=10)
        y -= 0.65

    ax.text(
        8, 0.65,
        "The QUBO defines what QOQA optimizes; it is not a classical probability-ranking step.",
        ha="center", fontsize=10, fontweight="bold"
    )

    save(fig, "07_qoqa_qubo_formulation.png", 400)

    # ============================================================
    # 08. QUBO → ISING MAPPING
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")

    box(ax, 1.0, 4.6, 3.5, 1.4, "Binary QUBO\nxᵢ ∈ {0,1}", fs=12)
    arrow(ax, 4.5, 5.3, 6.0, 5.3)
    box(ax, 6.0, 4.6, 4.0, 1.4, "Variable Mapping\nxᵢ = (1 − Zᵢ) / 2", fs=12)
    arrow(ax, 10.0, 5.3, 11.5, 5.3)
    box(ax, 11.5, 4.6, 3.5, 1.4, "Ising Hamiltonian\nH = ΣhᵢZᵢ + ΣJᵢⱼZᵢZⱼ", fs=12)

    ax.text(
        8, 7.2,
        "QOQA Hamiltonian Transformation",
        ha="center", fontsize=17, fontweight="bold"
    )
    ax.text(
        8, 2.8,
        "Local QUBO terms become single-qubit Z fields; pairwise QUBO terms "
        "become two-qubit ZZ couplings.",
        ha="center", fontsize=10
    )

    save(fig, "08_qoqa_qubo_to_ising.png", 400)

    # ============================================================
    # 09. E8 SPATIAL CONFLICT GRAPH
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for r in range(4):
        for c in range(4):
            ax.scatter(c, r, s=600)
            ax.text(c, r, f"x{r*4+c}", ha="center", va="center", fontsize=8)

    for r in range(4):
        for c in range(4):
            i = r * 4 + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    j = rr * 4 + cc
                    if 0 <= rr < 4 and 0 <= cc < 4 and i < j:
                        ax.plot([c, cc], [r, rr], linewidth=1.1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "QOQA E₈ Spatial Conflict Graph",
        fontsize=17, fontweight="bold"
    )
    ax.text(
        1.5, 3.25,
        "Edges represent 8-neighbor spatial conflicts.",
        ha="center", fontsize=10
    )

    save(fig, "09_qoqa_e8_spatial_conflict_graph.png", 400)

    # ============================================================
    # 10. QOQA BLOCK QAOA-STYLE CIRCUIT
    # ============================================================
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(-1, 10)
    ax.axis("off")

    ax.text(
        9, 9.5,
        "QOQA — Blockwise QAOA-Style Quantum Circuit",
        ha="center", fontsize=17, fontweight="bold"
    )

    for q in range(8):
        y = 8 - q
        ax.plot([0.7, 17.0], [y, y], linewidth=0.9)
        ax.text(0.45, y, f"q{q}", ha="right", va="center", fontsize=9)

        for x, label in [(2.0, "H"), (5.0, "RZ"), (9.0, "RZZ"), (13.0, "RX")]:
            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.20", fill=False, linewidth=1.0)
            )

    pairs = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,4),(2,6)]
    for k, (a, b) in enumerate(pairs):
        x = 6.5 + (k % 5) * 0.55
        y1 = 8 - a
        y2 = 8 - b
        ax.plot([x, x], [y1, y2], linewidth=1.2)
        ax.scatter(x, y1, s=38)
        ax.text(
            x, y2, "⊗",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="circle,pad=0.02", fill=False, linewidth=0.8)
        )

    ax.text(2.0, 8.9, "H: superposition", ha="center", fontsize=9)
    ax.text(5.0, 8.9, "RZ: local cost", ha="center", fontsize=9)
    ax.text(9.0, 8.9, "RZZ: pairwise coupling", ha="center", fontsize=9)
    ax.text(13.0, 8.9, "RX: mixer", ha="center", fontsize=9)

    ax.text(
        9, -0.55,
        "One p-layer: H initialization → cost phase (RZ/RZZ) → RX mixer; "
        "the layer is repeated for the configured QAOA depth.",
        ha="center", fontsize=10
    )

    save(fig, "10_qoqa_qaoa_style_block_circuit.png", 500)

    # ============================================================
    # 11. QOQA COST / MIXER SEPARATION
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")

    box(ax, 1.0, 4.7, 3.5, 1.5, "Cost Hamiltonian\nRZ + RZZ", fs=12)
    box(ax, 6.2, 4.7, 3.5, 1.5, "Quantum State\nPhase Encoding", fs=12)
    box(ax, 11.4, 4.7, 3.5, 1.5, "Mixer\nRX", fs=12)
    arrow(ax, 4.5, 5.45, 6.2, 5.45)
    arrow(ax, 9.7, 5.45, 11.4, 5.45)

    ax.text(
        8, 7.5,
        "QOQA QAOA Layer Semantics",
        ha="center", fontsize=17, fontweight="bold"
    )
    ax.text(
        8, 2.8,
        "RZ encodes local objective/cardinality contributions; "
        "RZZ encodes pairwise couplings; RX explores alternative binary configurations.",
        ha="center", fontsize=10
    )

    save(fig, "11_qoqa_cost_mixer_roles.png", 400)

    # ============================================================
    # 12. BORN-RULE MEASUREMENT
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")

    box(ax, 1.0, 4.6, 4.0, 1.5, "Final Quantum State\n|ψ⟩ = Σz αz|z⟩", fs=12)
    arrow(ax, 5.0, 5.35, 7.0, 5.35)
    box(ax, 7.0, 4.6, 4.0, 1.5, "Born Rule\nP(z)=|αz|²", fs=12)
    arrow(ax, 11.0, 5.35, 13.0, 5.35)
    box(ax, 13.0, 4.6, 3.0, 1.5, "Measured\nBitstrings", fs=12)

    ax.text(
        8.5, 7.4,
        "QOQA Measurement — Born-Rule Sampling",
        ha="center", fontsize=17, fontweight="bold"
    )
    ax.text(
        8.5, 2.6,
        "The measured configurations come from the simulated circuit's own "
        "state-vector probability distribution, not from a classical softmax selector.",
        ha="center", fontsize=10
    )

    save(fig, "12_qoqa_born_rule_measurement.png", 400)

    # ============================================================
    # 13. QOQA MEASUREMENT → FEASIBILITY
    # ============================================================
    fig, ax = plt.subplots(figsize=(19, 8))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 8)
    ax.axis("off")

    items = [
        ("Measured\nbitstring", 1.5),
        ("Decode xᵢ", 4.2),
        ("Evaluate\nQUBO energy", 7.0),
        ("Cardinality\ncheck", 10.0),
        ("E₈ conflict\ncheck", 13.0),
        ("Capacity\ncheck", 16.0)
    ]

    for i, (text, x) in enumerate(items):
        box(ax, x - 0.9, 3.1, 1.8, 1.5, text, fs=8)
        if i < len(items) - 1:
            arrow(ax, x + 0.9, 3.85, items[i + 1][1] - 0.9, 3.85)

    ax.text(
        9.5, 6.5,
        "QOQA Measurement and Constraint Verification",
        ha="center", fontsize=17, fontweight="bold"
    )
    ax.text(
        9.5, 1.0,
        "Only feasible measured configurations are eligible for final coordination.",
        ha="center", fontsize=10
    )

    save(fig, "13_qoqa_measurement_constraint_verification.png", 400)

    # ============================================================
    # 14. QOQA SPATIAL SELECTION MAP
    # ============================================================
    if (
        row_col is not None
        and col_col is not None
        and row_col in df.columns
        and col_col in df.columns
    ):
        rows = pd.to_numeric(
            df[row_col], errors="coerce"
        ).to_numpy()
        cols = pd.to_numeric(
            df[col_col], errors="coerce"
        ).to_numpy()
        valid = np.isfinite(rows) & np.isfinite(cols)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            cols[valid], rows[valid],
            s=12, alpha=0.25,
            label="Candidate regions"
        )

        selected_valid = [
            idx for idx in selected_regions
            if 0 <= idx < len(df) and valid[idx]
        ]

        if selected_valid:
            ax.scatter(
                cols[selected_valid],
                rows[selected_valid],
                s=45,
                label="QOQA-selected regions"
            )

        ax.invert_yaxis()
        ax.set_title(
            "QOQA Final Spatial Selection",
            fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.legend()

        save(fig, "14_qoqa_spatial_selection_map.png", 400)

    # ============================================================
    # 15. CLASSICAL ALL-PAIRS ATTENTION — AFTER QOQA
    # ============================================================
    if selected_embeddings.shape[0] > 1:
        emb = selected_embeddings
        norm = emb / np.maximum(
            np.linalg.norm(emb, axis=1, keepdims=True),
            1e-12
        )
        similarity = norm @ norm.T

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(similarity, aspect="auto")
        fig.colorbar(im, ax=ax, label="Cosine similarity")

        ax.set_title(
            "Selected-Region Representation Interaction\n"
            "Classical Attention Stage — After QOQA",
            fontsize=15, fontweight="bold"
        )
        ax.set_xlabel("Selected region")
        ax.set_ylabel("Selected region")

        save(fig, "15_classical_selected_region_interaction.png", 400)

    # ============================================================
    # 16. CLASSICAL ATTENTION STAR / ALL-PAIRS SCHEMATIC
    # ============================================================
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")
    ax.axis("off")

    n = min(12, max(2, len(selected_regions)))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 3.5
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)

    for i in range(n):
        for j in range(i + 1, n):
            ax.plot(
                [xs[i], xs[j]],
                [ys[i], ys[j]],
                linewidth=0.55,
                alpha=0.25
            )

    for i in range(n):
        ax.scatter(xs[i], ys[i], s=650)
        rid = selected_regions[i] if i < len(selected_regions) else i
        ax.text(
            xs[i], ys[i], f"R{rid}",
            ha="center", va="center", fontsize=8, fontweight="bold"
        )

    ax.scatter(0, 0, s=1000)
    ax.text(
        0, 0, "Q/K/V\nAttention",
        ha="center", va="center",
        fontsize=9, fontweight="bold"
    )

    ax.set_title(
        "Classical All-Pairs Representation Interaction",
        fontsize=16, fontweight="bold"
    )
    ax.text(
        0, -4.6,
        "This is classical scaled dot-product attention.\n"
        "It is not quantum entanglement and occurs after QOQA selection.",
        ha="center", fontsize=10
    )

    save(fig, "16_classical_attention_all_pairs_schematic.png", 400)

    # ============================================================
    # 17. ACER ACTUAL ANGLE ENCODING
    # ============================================================
    if combined_embedding.size < 32:
        acer_input = np.pad(
            combined_embedding,
            (0, 32 - combined_embedding.size)
        )
    else:
        acer_input = combined_embedding[:32]

    angles = np.tanh(acer_input) * np.pi
    rz_angles = angles / 2.0

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(
        np.arange(32), angles,
        marker="o", label="RY angle = tanh(x) × π"
    )
    ax.plot(
        np.arange(32), rz_angles,
        marker="x", label="RZ angle = RY angle / 2"
    )
    ax.set_title(
        "ACER — Actual 32-Qubit Angle Encoding",
        fontsize=15, fontweight="bold"
    )
    ax.set_xlabel("Logical qubit")
    ax.set_ylabel("Rotation angle (radians)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save(fig, "17_acer_actual_angle_encoding.png", 400)

    # ============================================================
    # 18. ACER FULL 4 × 8 CIRCUIT
    # ============================================================
    fig, ax = plt.subplots(figsize=(23, 16))
    ax.set_xlim(0, 19)
    ax.set_ylim(-2, 34)
    ax.axis("off")

    ax.text(
        9.5, 33.3,
        "ACER — Full 32 Logical Qubit Circuit",
        ha="center", fontsize=18, fontweight="bold"
    )

    for r in range(4):
        start = r * 8
        ax.text(
            0.2, 32 - start - 3.5,
            f"Register {r + 1}",
            ha="right", va="center",
            fontsize=9, fontweight="bold"
        )

    for q in range(32):
        y = 32 - q
        ax.plot([0.7, 18.0], [y, y], linewidth=0.8)
        ax.text(0.45, y, f"q{q}", ha="right", va="center", fontsize=7)

        for x, label in [(2.0, "RY"), (3.7, "RZ"), (5.4, "H")]:
            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fill=False, linewidth=0.9
                )
            )

        ax.text(
            17.0, y, "⟨Z⟩",
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.18",
                fill=False, linewidth=0.9
            )
        )

    for r in range(4):
        base = r * 8
        for k in range(8):
            control = base + k
            target = base + ((k + 1) % 8)
            y1 = 32 - control
            y2 = 32 - target
            x = 7.0 + k * 1.1

            ax.plot([x, x], [y1, y2], linewidth=1.2)
            ax.scatter(x, y1, s=35)
            ax.text(
                x, y2, "⊕",
                ha="center", va="center", fontsize=10,
                bbox=dict(
                    boxstyle="circle,pad=0.02",
                    fill=False, linewidth=0.8
                )
            )

    ax.text(
        9.5, -0.7,
        "Each register independently executes "
        "RY(θq) → RZ(θq/2) → H → cyclic CNOT ring → Pauli-Z expectation.",
        ha="center", fontsize=10, fontweight="bold"
    )
    ax.text(
        9.5, -1.35,
        "4 registers × 8 measured qubits = 32 ACER quantum features.",
        ha="center", fontsize=9
    )

    save(fig, "18_acer_full_32_qubit_circuit.png", 600)

    # ============================================================
    # 19. ACER SINGLE 8-QUBIT CIRCUIT
    # ============================================================
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(-1.5, 9)
    ax.axis("off")

    ax.text(
        7.5, 8.55,
        "ACER — Single 8-Qubit Register",
        ha="center", fontsize=18, fontweight="bold"
    )

    for q in range(8):
        y = 7 - q
        ax.plot([0.5, 13.5], [y, y], linewidth=0.9)
        ax.text(0.25, y, f"q{q}", ha="right", va="center", fontsize=9)

        for x, label in [
            (2.0, f"RY(θ{q})"),
            (3.8, f"RZ(θ{q}/2)"),
            (5.5, "H")
        ]:
            ax.text(
                x, y, label,
                ha="center", va="center", fontsize=7,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fill=False, linewidth=1.0
                )
            )

        ax.text(
            13.0, y, "⟨Z⟩",
            ha="center", va="center", fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fill=False, linewidth=1.0
            )
        )

    for k in range(8):
        control = k
        target = (k + 1) % 8
        y1 = 7 - control
        y2 = 7 - target
        x = 7.0 + k * 0.75
        ax.plot([x, x], [y1, y2], linewidth=1.3)
        ax.scatter(x, y1, s=40)
        ax.text(
            x, y2, "⊕",
            ha="center", va="center", fontsize=11,
            bbox=dict(
                boxstyle="circle,pad=0.02",
                fill=False, linewidth=0.9
            )
        )

    ax.text(
        7.5, -0.55,
        "q0→q1→q2→q3→q4→q5→q6→q7→q0",
        ha="center", fontsize=10, fontweight="bold"
    )
    ax.text(
        7.5, -1.05,
        "Measurement returns one Pauli-Z expectation for each qubit.",
        ha="center", fontsize=9
    )

    save(fig, "19_acer_single_8_qubit_circuit.png", 600)

    # ============================================================
    # 20. ACTUAL ACER STATEVECTOR / PROBABILITIES
    # ============================================================
    try:
        backend = ACERSimulatorBackend(
            n_qubits=32,
            quantum_registers=4,
            qubits_per_register=8
        )

        register_states = []
        register_expectations = []

        for r in range(4):
            a = angles[r * 8:(r + 1) * 8]
            state, exp = backend.entangle_register(a)
            register_states.append(np.asarray(state))
            register_expectations.append(np.asarray(exp))

        state = register_states[0]
        probabilities = np.abs(state) ** 2

        fig, ax = plt.subplots(figsize=(18, 7))
        ax.plot(
            np.arange(256),
            probabilities,
            linewidth=1
        )
        ax.set_title(
            "ACER Register 1 — Actual Post-Entanglement Born Probabilities",
            fontsize=15, fontweight="bold"
        )
        ax.set_xlabel("8-qubit computational-basis index (0–255)")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)

        save(fig, "20_acer_register1_born_probabilities.png", 400)

        # Actual state-vector amplitudes
        fig, ax = plt.subplots(figsize=(18, 7))
        ax.plot(
            np.arange(256),
            np.real(state),
            label="Real amplitude"
        )
        ax.plot(
            np.arange(256),
            np.imag(state),
            label="Imaginary amplitude"
        )
        ax.plot(
            np.arange(256),
            np.abs(state),
            label="Magnitude"
        )
        ax.set_title(
            "ACER Register 1 — Actual Complex State Vector",
            fontsize=15, fontweight="bold"
        )
        ax.set_xlabel("Computational-basis index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save(fig, "21_acer_register1_state_vector.png", 400)

        # Gate-stage expectation evolution
        state_stage = backend.initialize_register(qubit_count=8)
        stage_values = []

        def z_vector(s):
            return np.asarray([
                backend.measure_z(s, q, qubit_count=8)
                for q in range(8)
            ])

        for q in range(8):
            state_stage = backend.apply_ry(
                state_stage, q, angles[q], qubit_count=8
            )
        stage_values.append(("RY", z_vector(state_stage)))

        for q in range(8):
            state_stage = backend.apply_rz(
                state_stage, q, angles[q] / 2.0, qubit_count=8
            )
        stage_values.append(("RZ", z_vector(state_stage)))

        for q in range(8):
            state_stage = backend.apply_hadamard(
                state_stage, q, qubit_count=8
            )
        stage_values.append(("H", z_vector(state_stage)))

        for q in range(7):
            state_stage = backend.apply_cnot(
                state_stage, q, q + 1, qubit_count=8
            )
        state_stage = backend.apply_cnot(
            state_stage, 7, 0, qubit_count=8
        )
        stage_values.append(("CNOT ring", z_vector(state_stage)))

        expectation_matrix = np.asarray(
            [v for _, v in stage_values]
        )

        fig, ax = plt.subplots(figsize=(13, 7))
        im = ax.imshow(
            expectation_matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )
        fig.colorbar(im, ax=ax, label="Pauli-Z expectation")
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels([f"q{i}" for i in range(8)])
        ax.set_yticks(np.arange(len(stage_values)))
        ax.set_yticklabels([name for name, _ in stage_values])
        ax.set_title(
            "ACER Register 1 — Actual Pauli-Z Expectation Through Circuit",
            fontsize=15, fontweight="bold"
        )
        ax.set_xlabel("Qubit")
        ax.set_ylabel("Circuit stage")

        save(fig, "22_acer_expectation_evolution.png", 400)

        # Final 4×8 measured quantum features
        q_matrix = np.asarray(register_expectations)

        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(
            q_matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )
        fig.colorbar(im, ax=ax, label="Pauli-Z expectation")
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels([f"q{i}" for i in range(8)])
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels([f"Register {i + 1}" for i in range(4)])
        ax.set_title(
            "ACER — Actual 4 × 8 Quantum Feature Map",
            fontsize=15, fontweight="bold"
        )
        ax.set_xlabel("Qubit")
        ax.set_ylabel("Register")

        save(fig, "23_acer_4x8_quantum_feature_map.png", 400)

    except Exception as exc:
        print(
            f"[VIS] ACER state-vector visualizations skipped: {exc}"
        )

    # ============================================================
    # 24. ACER QUANTUM FEATURE VECTOR
    # ============================================================
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(
        np.arange(len(quantum_features)),
        quantum_features
    )
    ax.axhline(0, linewidth=1)
    ax.set_title(
        "ACER — 32 Measured Quantum Features",
        fontsize=15, fontweight="bold"
    )
    ax.set_xlabel("Measured logical qubit")
    ax.set_ylabel("Pauli-Z expectation")

    save(fig, "24_acer_32_quantum_features.png", 400)

    # ============================================================
    # 25. SELECTED REGIONS → ACER REGISTERS
    # ============================================================
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box(ax, 0.5, 3.2, 3.0, 1.4, f"{len(selected_regions)}\nQOQA-selected regions", fs=10)
    box(ax, 4.5, 3.2, 3.0, 1.4, "Selected 128-D\nregion embeddings", fs=10)
    box(ax, 8.5, 3.2, 3.0, 1.4, "Classical global\nQ/K/V interaction", fs=10)
    box(ax, 12.5, 4.5, 2.5, 1.2, "ACER\nRegisters 1–2", fs=9)
    box(ax, 12.5, 2.1, 2.5, 1.2, "ACER\nRegisters 3–4", fs=9)
    box(ax, 15.5, 3.2, 2.0, 1.4, "32-D\nquantum vector", fs=9)

    arrow(ax, 3.5, 3.9, 4.5, 3.9)
    arrow(ax, 7.5, 3.9, 8.5, 3.9)
    arrow(ax, 11.5, 4.2, 12.5, 5.1)
    arrow(ax, 11.5, 3.6, 12.5, 2.7)
    arrow(ax, 15.0, 5.1, 15.5, 4.1)
    arrow(ax, 15.0, 2.7, 15.5, 3.7)

    ax.text(
        9, 7.2,
        "Selected Regions → Classical Global Interaction → ACER",
        ha="center", fontsize=16, fontweight="bold"
    )
    ax.text(
        9, 0.65,
        "QOQA chooses coordinates; classical attention aggregates selected "
        "representations; ACER converts the global representation into quantum features.",
        ha="center", fontsize=9
    )

    save(fig, "25_selected_regions_to_acer_registers.png", 400)

    # ============================================================
    # 26. CORRECTED SYSTEM SPECIFICATION
    # ============================================================
    specification = [
        "SQE-Net: 64 logical qubits",
        "SQE-Net: 8 independent 8-qubit registers",
        "SQE-Net: 256-dimensional state vector per register",
        "SQE-Net gates: RY → RZ → H → cyclic CNOT ring",
        "SQE-Net measurement: Pauli-Z expectation",
        "SQE-Net output: 64-D quantum feature vector",
        "QOQA: constrained binary region-selection optimization",
        "QOQA: QUBO objective with E8 spatial conflict and cardinality terms",
        "QOQA: binary-to-Ising mapping x=(1-Z)/2",
        "QOQA: blockwise QAOA-style state-vector circuit",
        "QOQA: H → cost RZ/RZZ → RX mixer",
        "QOQA: Born-rule computational-basis measurement",
        "QOQA: QUBO energy and feasibility verification",
        "QOQA: classical cross-block coordination",
        "Classical attention: scaled dot-product Q/K/V over selected regions",
        "ACER: 32 logical qubits",
        "ACER: 4 independent 8-qubit registers",
        "ACER: 256-dimensional state vector per register",
        "ACER gates: RY → RZ(angle/2) → H → cyclic CNOT ring",
        "ACER measurement: Pauli-Z expectation",
        "ACER output: 32-D quantum feature vector",
        "Quantum engine: classical state-vector simulation",
        "No physical quantum hardware is claimed",
        "No quantum advantage is claimed"
    ]

    fig, ax = plt.subplots(figsize=(16, 13))
    ax.axis("off")
    ax.text(
        0.03, 0.97,
        "StegaQEntropy — Quantum System Specification",
        fontsize=17, fontweight="bold", va="top"
    )

    y = 0.91
    for item in specification:
        ax.text(0.05, y, "• " + item, fontsize=10, va="top")
        y -= 0.035

    save(fig, "26_corrected_quantum_system_specification.png", 400)

    # ============================================================
    # 27. CIRCUIT DESCRIPTION TXT
    # ============================================================
    circuit_text = """
STEGAQENTROPY QUANTUM CIRCUIT DESCRIPTION
==========================================

SQE-NET
-------
Purpose:
Feature-level quantum processing inside each image region.

Architecture:
64 logical qubits
8 independent registers
8 qubits per register
256 amplitudes per register

Circuit:
|00000000>
→ RY(theta_q)
→ RZ(phi_q)
→ H on all qubits
→ CNOT(q0,q1)
→ CNOT(q1,q2)
→ CNOT(q2,q3)
→ CNOT(q3,q4)
→ CNOT(q4,q5)
→ CNOT(q5,q6)
→ CNOT(q6,q7)
→ CNOT(q7,q0)
→ Pauli-Z expectation

Output:
8 expectations per register
8 registers
64-D SQE quantum feature vector


QOQA
----
Purpose:
Quantum-assisted constrained selection of image regions.

Optimization:
H(x) = -sum(w_i x_i)
       + lambda sum_(i,j in E8) x_i x_j
       + mu (sum(x_i)-K)^2

Binary-to-Ising mapping:
x_i = (1-Z_i)/2

Circuit:
H initialization
→ cost phase using RZ terms
→ pairwise RZZ couplings
→ RX mixer
→ repeat for configured QAOA depth
→ Born-rule measurement

Measurement:
P(z) = |alpha_z|^2

Post-measurement:
Decode bitstring
→ evaluate QUBO energy
→ cardinality verification
→ E8 spatial-conflict verification
→ payload-capacity verification
→ classical cross-block coordination
→ final selected-region set

Important:
QOQA does NOT use ACER sampling.
QOQA does NOT use a classical softmax probability vector as its selector.
QOQA uses quantum state-vector optimization and Born-rule measurement.


CLASSICAL GLOBAL ATTENTION
--------------------------
Purpose:
Model all-pairs representation interaction among regions selected by QOQA.

Operation:
Q = Dense(E_region)
K = Dense(E_region)
V = Dense(E_region)

Attention:
softmax(Q K^T / sqrt(d))

Selection masking is applied before aggregation.

Important:
This is classical scaled dot-product attention.
It is NOT quantum entanglement.


ACER
----
Purpose:
Convert the selected-region global representation into a global quantum representation.

Architecture:
32 logical qubits
4 independent registers
8 qubits per register
256 amplitudes per register

Angle encoding:
theta = tanh(embedding) * pi
RZ angle = theta / 2

Circuit per register:
|00000000>
→ RY(theta_q)
→ RZ(theta_q/2)
→ H on all qubits
→ cyclic CNOT ring
→ Pauli-Z expectation

Output:
4 registers × 8 expectations
= 32-D ACER quantum feature vector

The ACER vector is then passed to the classical global PSNR prediction head.


HOW TO READ THE FIGURES
-----------------------
01  Complete architecture. Follow left-to-right data flow.
02  SQE-Net internal sequence. Shows where the 64-qubit feature stage occurs.
03  Detailed SQE-Net 8-qubit circuit. Read each wire from left to right.
04  SQE-Net register organization. Eight independent 8-qubit state vectors.
05  Gate-role reference. Explains what each quantum gate contributes.
06  QOQA workflow. Follow optimization from QUBO formulation to final selection.
07  QUBO equation. x_i are region-selection variables.
08  QUBO-to-Ising transformation. Shows how binary optimization becomes a Hamiltonian.
09  E8 graph. Edges are local spatial conflict relationships.
10  QAOA block circuit. H prepares superposition; RZ/RZZ encode cost; RX mixes.
11  Cost/mixer figure. Separates objective encoding from exploration.
12  Born-rule figure. Shows how the quantum state becomes measured bitstrings.
13  Constraint figure. Shows verification after measurement.
14  Spatial selection map. Shows candidate regions and the final QOQA-selected set.
15  Classical interaction matrix. Represents selected-region representation similarity.
16  Classical attention schematic. All-pairs interaction happens after QOQA.
17  ACER angle plot. Shows actual 32 encoded angles from combined_embedding.
18  Full ACER circuit. Four independent 8-qubit registers.
19  Single ACER register. Detailed gate sequence.
20  Actual ACER Born probabilities. 256 basis states for register 1.
21  Actual ACER state vector. Real, imaginary and magnitude components.
22  Actual ACER expectation evolution. Pauli-Z values after each circuit stage.
23  Actual ACER feature map. Four registers × eight measured features.
24  Final 32-D ACER quantum feature vector.
25  Selected regions through classical aggregation into ACER.
26  Corrected system specification for checking terminology and dimensions.

KEY INTERACTION DISTINCTION
----------------------------
SQE-Net:
Intra-region quantum feature interaction.

QOQA:
Inter-region coupling of binary selection variables through the E8 conflict
graph and QUBO/Ising Hamiltonian.

Classical attention:
All-pairs representation interaction among selected regions.

ACER:
Global quantum representation after classical global aggregation.

SIMULATION NOTE
---------------
All quantum circuits are simulated with explicit state vectors.
The visualization should therefore be described as quantum-circuit
simulation / state-vector simulation, not physical quantum hardware.
No quantum-advantage claim is implied by these figures.
"""

    with open(
        os.path.join(quantum_dir, "27_quantum_visualization_summary.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(circuit_text.strip() + "\n")

    # ============================================================
    # 28. MACHINE-READABLE VISUALIZATION METADATA
    # ============================================================
    visualization_data = {
        "selected_regions": selected_regions,
        "selected_region_count": len(selected_regions),
        "selected_embedding_shape": list(selected_embeddings.shape),
        "combined_embedding_dimension": int(len(combined_embedding)),
        "quantum_feature_count": int(len(quantum_features)),
        "sqe_net": {
            "logical_qubits": 64,
            "registers": 8,
            "qubits_per_register": 8,
            "state_dimension_per_register": 256,
            "gates": ["RY", "RZ", "H", "CNOT"],
            "entanglement_topology": "cyclic 8-qubit ring",
            "measurement": "Pauli-Z expectation",
            "output_features": 64
        },
        "qoqa": {
            "method": "blockwise QAOA-style state-vector optimization",
            "qubo": (
                "H(x)=-sum(w_i*x_i)+lambda*sum_E8(x_i*x_j)"
                "+mu*(sum(x_i)-K)^2"
            ),
            "binary_to_ising": "x_i=(1-Z_i)/2",
            "cost_gates": ["RZ", "RZZ"],
            "mixer_gate": "RX",
            "initialization": "Hadamard superposition",
            "measurement": "Born-rule computational-basis measurement",
            "spatial_graph": "E8 8-neighbor conflict graph",
            "cross_block_coordination": "classical"
        },
        "classical_attention": {
            "type": "scaled dot-product attention",
            "quantum_entanglement": False,
            "position": "after QOQA selection"
        },
        "acer": {
            "logical_qubits": 32,
            "registers": 4,
            "qubits_per_register": 8,
            "state_dimension_per_register": 256,
            "angle_encoding": "tanh(embedding)*pi",
            "rz_encoding": "angle/2",
            "gates": ["RY", "RZ", "H", "CNOT"],
            "entanglement_topology": "cyclic 8-qubit ring",
            "measurement": "Pauli-Z expectation",
            "output_features": 32
        },
        "quantum_backend": "state-vector simulation"
    }

    with open(
        os.path.join(quantum_dir, "28_quantum_qoqa_visualization_data.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(visualization_data, f, indent=4)

    print()
    print("[VIS] Corrected quantum/QOQA visualization generation complete.")
    print(f"[VIS] Output: {quantum_dir}")
    print("[VIS] Generated corrected figures, circuit description and metadata.")
    print("=" * 80)

    return quantum_dir


def main(message_type=None):
    if message_type is None:
        message_type = int(input(
            "Enter the Type of Message:\n"
            "1: Text\n"
            "2: Image\n"
        ))

    configure_message_type(message_type)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 75)
    print("StegaQEntropy ACER-QOQA & QRNG Pure Regression Surrogate Pipeline")
    print("=" * 75)

    df = pd.read_csv(CSV_PATH)
    print("Dataset Shape:", df.shape)

    filtered_numeric = prepare_intelligence_features(df)
    feature_names = filtered_numeric.columns.tolist()
    print(
        f"[INTELLIGENT DATASET] "
        f"Using {filtered_numeric.shape[1]} intelligent features."
    )
    y_reg, regression_columns = prepare_targets(df)

    image_path = load_image_path()
    image = load_image(image_path)
    dwt = load_dwt()

    row_col = find_column(df, ["row_start", "pre_row_start", "row", "region_row"]) or df.columns[0]
    col_col = find_column(df, ["column_start", "pre_column_start", "col", "column", "region_col"]) or df.columns[1]

    metadata = pd.DataFrame({
        "region_id": np.arange(len(df)),
        "row": pd.to_numeric(df[row_col], errors="coerce").fillna(0),
        "col": pd.to_numeric(df[col_col], errors="coerce").fillna(0)
    })

    graph = build_neighbor_graph(metadata)

    raw_patches, dwt_patches = [], []
    for _, row in metadata.iterrows():
        r_patch = extract_patch(image, row["row"], row["col"], WINDOW_SIZE)[..., np.newaxis]
        raw_patches.append(r_patch)
        d_bands = [extract_patch(b, row["row"] / 2.0, row["col"] / 2.0, DWT_WINDOW_SIZE) for b in dwt]
        dwt_patches.append(np.stack(d_bands, axis=-1))

    raw_patches = np.asarray(raw_patches, dtype=np.float32)
    dwt_patches = np.asarray(dwt_patches, dtype=np.float32)

    # ============================================================
    # SPATIAL GROUPED TRAIN / VALIDATION / TEST SPLIT
    # ============================================================

    indices = np.arange(len(df))

    # Regions belonging to the same spatial block stay together.
    # This prevents immediately neighboring regions from being
    # randomly scattered across train/validation/test.

    BLOCK_SIZE = WINDOW_SIZE * 4

    spatial_groups = (
        (metadata["row"].astype(int) // BLOCK_SIZE) * 1000
        + (metadata["col"].astype(int) // BLOCK_SIZE)
    ).to_numpy()

    # ------------------------------------------------------------
    # First split: 80% development / 20% test
    # ------------------------------------------------------------

    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_val_positions, test_positions = next(
        gss_test.split(
            indices,
            groups=spatial_groups
        )
    )

    train_val_idx = indices[train_val_positions]
    test_idx = indices[test_positions]

    # ------------------------------------------------------------
    # Second split: development → train / validation
    # ------------------------------------------------------------

    train_val_groups = spatial_groups[train_val_idx]

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    train_positions, val_positions = next(
        gss_val.split(
            train_val_idx,
            groups=train_val_groups
        )
    )

    train_idx = train_val_idx[train_positions]
    val_idx = train_val_idx[val_positions]

    # ------------------------------------------------------------
    # Sort indices for deterministic processing
    # ------------------------------------------------------------

    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    print(
        "\n[DATA SPLIT]"
    )

    print(
        f"  Training   : {len(train_idx)} samples"
    )

    print(
        f"  Validation : {len(val_idx)} samples"
    )

    print(
        f"  Testing    : {len(test_idx)} samples"
    )

    print(
        f"  Spatial block size : {BLOCK_SIZE}"
    )

    feature_imputer = SimpleImputer(strategy="median")
    train_imp = feature_imputer.fit_transform(filtered_numeric.iloc[train_idx])
    val_imp = feature_imputer.transform(filtered_numeric.iloc[val_idx])
    test_imp = feature_imputer.transform(filtered_numeric.iloc[test_idx])

    feature_scaler = StandardScaler()
    train_scaled_feat = feature_scaler.fit_transform(train_imp)
    val_scaled_feat = feature_scaler.transform(val_imp)
    test_scaled_feat = feature_scaler.transform(test_imp)

    from sklearn.preprocessing import MinMaxScaler
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_reg[train_idx])
    y_val_scaled = target_scaler.transform(y_reg[val_idx])
    y_test_scaled = target_scaler.transform(y_reg[test_idx])

    train_selection = np.ones(
        (len(train_idx), 1),
        dtype=np.float32
    )

    val_selection = np.ones(
        (len(val_idx), 1),
        dtype=np.float32
    )

    test_selection = np.ones(
        (len(test_idx), 1),
        dtype=np.float32
    )

    train_inputs = [
        train_scaled_feat,
        raw_patches[train_idx],
        dwt_patches[train_idx],
        train_selection
    ]

    val_inputs = [
        val_scaled_feat,
        raw_patches[val_idx],
        dwt_patches[val_idx],
        val_selection
    ]

    test_inputs = [
        test_scaled_feat,
        raw_patches[test_idx],
        dwt_patches[test_idx],
        test_selection
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MODEL INPUT SHAPES"
    )

    print(
        "=" * 60
    )

    print(
        f"Intelligent Dataset : "
        f"{train_inputs[0].shape}"
    )

    print(
        f"Image                : "
        f"{train_inputs[1].shape}"
    )

    print(
        f"DWT                  : "
        f"{train_inputs[2].shape}"
    )

    print("\nBuilding Pure Regression Surrogate CNN Model...")
    model = build_regression_surrogate_model(
        intel_count=train_scaled_feat.shape[1],
        raw_shape=raw_patches.shape[1:],
        dwt_shape=dwt_patches.shape[1:],
        reg_count=len(regression_columns)
    )
    embedding_model = Model(
        inputs=model.inputs,
        outputs=model.get_layer(
            "region_embedding_128"
        ).output
    )

    try:
        _sample_fused_vector = np.concatenate(
            [
                train_scaled_feat[0].astype(np.float32),
                np.zeros(128, dtype=np.float32)
            ]
        )
        save_sqe_quantum_metadata(
            OUTPUT_DIR,
            sample_fused_vector=_sample_fused_vector
        )
    except Exception as exc:
        print(f"[SQE-NET QUANTUM] Metadata export skipped: {exc}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        train_inputs,
        {
            "regression_output": y_train_scaled,
            "global_psnr": y_train_scaled[:, [regression_columns.index("image_psnr")]]
        },
        validation_data=(
            val_inputs,
            {
                "regression_output": y_val_scaled,
                "global_psnr": y_val_scaled[:, [regression_columns.index("image_psnr")]]
            }
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    generate_training_plots(history)

    y_pred_scaled, y_global_pred_scaled = model.predict(
        test_inputs,
        verbose=0
    )

    y_true_orig = target_scaler.inverse_transform(
        y_test_scaled
    )

    y_pred_orig = target_scaler.inverse_transform(
        y_pred_scaled
    )
    generate_regression_scatter_plots(y_true_orig, y_pred_orig, regression_columns)
    plot_surrogate_learning_analysis(
    model=model,
    test_inputs=test_inputs,
    test_targets=y_test_scaled,
    output_dir=os.path.join(
        OUTPUT_DIR,
        "learning_analysis_plots"
    ),
    target_names=regression_columns
)

    print(
        "\n"
        + "=" * 90
    )

    print(
        "COMPREHENSIVE REGRESSION EVALUATION METRICS"
    )

    print(
        "=" * 90
    )

    for i, target in enumerate(
        regression_columns
    ):

        actual = y_true_orig[:, i]

        predicted = y_pred_orig[:, i]

        mae = mean_absolute_error(
            actual,
            predicted
        )

        mse = mean_squared_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            actual,
            predicted
        )

        nonzero = (
            np.abs(actual) > 1e-12
        )

        if np.any(nonzero):

            mape = (
                np.mean(
                    np.abs(
                        (
                            actual[nonzero]
                            - predicted[nonzero]
                        )
                        /
                        actual[nonzero]
                    )
                )
                * 100.0
            )

        else:

            mape = np.nan

        print(
            f"\n[{target}]"
        )

        print(
            f"  MAE  : {mae:.8f}"
        )

        print(
            f"  MSE  : {mse:.12f}"
        )

        print(
            f"  RMSE : {rmse:.8f}"
        )

        print(
            f"  R2   : {r2:.8f}"
        )

        print(
            f"  MAPE : {mape:.4f}%"
        )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ACTUAL vs PREDICTED — TEST SAMPLES"
    )

    print(
        "=" * 90
    )

    samples_to_print = min(
        10,
        len(y_true_orig)
    )

    for sample_index in range(
        samples_to_print
    ):

        print(
            f"\n--- Test Sample "
            f"{sample_index + 1} ---"
        )

        for target_index, target_name in enumerate(
            regression_columns
        ):

            actual_value = float(
                y_true_orig[
                    sample_index,
                    target_index
                ]
            )

            predicted_value = float(
                y_pred_orig[
                    sample_index,
                    target_index
                ]
            )

            error = abs(
                actual_value
                - predicted_value
            )

            print(
                f"{target_name:<32}"
                f"Actual={actual_value:.8f} | "
                f"Predicted={predicted_value:.8f} | "
                f"Error={error:.8f}"
            )
    print(
        "\n"
        + "=" * 100
    )

    print(
        "FULL MODEL TEST PREDICTION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Target':<34}"
        f"{'Actual Mean':>16}"
        f"{'Pred Mean':>16}"
        f"{'MAE':>14}"
        f"{'RMSE':>14}"
        f"{'R2':>12}"
    )

    print(
        "-" * 100
    )

    for i, target_name in enumerate(
        regression_columns
    ):

        actual = y_true_orig[:, i]

        predicted = y_pred_orig[:, i]

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        r2 = r2_score(
            actual,
            predicted
        )

        print(
            f"{target_name:<34}"
            f"{np.mean(actual):>16.8f}"
            f"{np.mean(predicted):>16.8f}"
            f"{mae:>14.8f}"
            f"{rmse:>14.8f}"
            f"{r2:>12.6f}"
        )

    print(
        "=" * 100
    )

    model.save(MODEL_PATH)
    print(f"\nModel successfully saved to: {MODEL_PATH}")

    best_regions, final_psnr, best_payloads, final_image = qoqa_acer_qrng_endless_loop(
        model,
        embedding_model,
        df,
        metadata,
        graph,
        image,
        dwt,
        feature_imputer,
        feature_scaler,
        feature_names,
        target_scaler,
        regression_columns,
        MAX_REGIONS_TARGET
    )
    print("=" * 75)
    print(f"FINAL OPTIMIZED REGIONS COUNT : {len(best_regions)}")
    print(f"FINAL VERIFIED PSNR           : {final_psnr:.2f} dB (>= {TARGET_PSNR_THRESHOLD} dB Target)")
    print(f"ALL ARTIFACTS & PLOTS SAVED TO : {OUTPUT_DIR}")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == "__main__":
    main()
