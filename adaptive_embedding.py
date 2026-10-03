import sys
import json
import socket
import hashlib
import secrets
import random
import io
import subprocess
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
import numpy as np
import pywt
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from PIL import Image
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt


from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

BASE_DIR = Path(__file__).resolve().parent

MESSAGE_JSON = BASE_DIR / "output" / "message_preparation" / "message_preparation.json"
CHUNKS_DIR = BASE_DIR / "output" / "message_preparation" / "chunks"
QKD_PACKAGE = BASE_DIR / "output" / "quantum_key_hierarchy" / "qkd_package.json"
QRNG_SEED = BASE_DIR / "qrng" / "output" / "history_check" / "final_verified_seeds" / "adaptive_embedding_seed_verified.txt"
ML_SOLUTION = (
    BASE_DIR
    / "ml_module"
    / "output"
    / "surrogate_cnn_model"
    / "optimal_embedding_solution.json"
)

ADAPTIVE_REGION_MANIFEST = (
    BASE_DIR
    / "ml_module"
    / "output"
    / "surrogate_cnn_model"
    / "adaptive_embedding_region_manifest.json"
)
DWT_DIR = BASE_DIR / "output" / "dwt_decomposition"

REGION_METADATA_CSV = (
    BASE_DIR
    / "ml_module"
    / "output"
    / "probability_analysis"
    / "probability_intelligence_final_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "output" / "adaptive_embedding"
STEGO_IMAGE = OUTPUT_DIR / "adaptive_stego_float.tiff"
EXTRACTED_MESSAGE = OUTPUT_DIR / "extracted_message.bin"
PLAN_JSON = OUTPUT_DIR / "adaptive_embedding_plan.json"
METRICS_JSON = OUTPUT_DIR / "adaptive_embedding_metrics.json"
STEGO_VIEW_IMAGE = (
    OUTPUT_DIR
    / "adaptive_stego_view.png"
)


HOST = "127.0.0.1"
PORT = 5055

WAVELET = "db2"

MASTER_KEY_BITS = 128
SUBKEY_BITS = 64

MASTER_QUBITS = 320
SUBKEY_QUBITS = 160
QKD_BATCH_SIZE = 29

QKD_MAX_RETRIES = 5

QBER_THRESHOLD = 0.11

# --- IBM Quantum Hardware Configuration ---
USE_REAL_IBM_HARDWARE = False  # Set to True to execute on real IBM quantum hardware
IBM_API_TOKEN = "BDPK4kcunb5KV8r_3YEh-CpPGIEpKwOF1lspGy-YZKnN"


class QRNG:
    def __init__(self, path):
        text = path.read_text(encoding="utf-8")
        self.bits = self._extract(text)
        self.pointer = 0

    def _extract(self, text):
        result = []
        active = False

        for line in text.splitlines():
            value = line.strip()

            if value.lower() == "binary":
                active = True
                continue

            if active and value.lower().startswith("hexadecimal"):
                break

            if active and value and set(value).issubset({"0", "1"}):
                result.append(value)

        bits = "".join(result)

        if not bits:
            raise ValueError("QRNG binary section not found.")

        return bits

    def take(self, count):
        if self.pointer + count > len(self.bits):
            raise RuntimeError(
                f"QRNG exhausted. Requested {count}, remaining {self.remaining()}."
            )

        value = self.bits[
            self.pointer:self.pointer + count
        ]

        self.pointer += count

        return value

    def remaining(self):
        return len(self.bits) - self.pointer


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def send_packet(sock, payload):
    raw = json.dumps(
        payload,
        separators=(",", ":")
    ).encode("utf-8")

    sock.sendall(
        len(raw).to_bytes(8, "big")
    )

    sock.sendall(raw)


def receive_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        block = sock.recv(size - len(data))

        if not block:
            raise ConnectionError("TCP connection closed.")

        data.extend(block)

    return bytes(data)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4
        )


def receive_packet(sock):
    size = int.from_bytes(
        receive_exact(sock, 8),
        "big"
    )

    raw = receive_exact(
        sock,
        size
    )

    return json.loads(
        raw.decode("utf-8")
    )


def discover_ml_file():
    path = ML_SOLUTION

    if path.is_file():
        return path

    if path.with_suffix(".json").is_file():
        return path.with_suffix(".json")

    if path.exists() and path.is_dir():
        candidates = []

        for pattern in [
            "*.json",
            "*.JSON"
        ]:
            candidates.extend(
                path.glob(pattern)
            )

        if not candidates:
            raise FileNotFoundError(
                f"No ML solution JSON found in {path}"
            )

        preferred = [
            candidate
            for candidate in candidates
            if (
                "optimal" in candidate.name.lower()
                or "solution" in candidate.name.lower()
            )
        ]

        if preferred:
            return preferred[0]

        return sorted(candidates)[0]

    parent = path.parent

    if parent.exists():
        candidates = []

        for candidate in parent.glob("*.json"):
            name = candidate.name.lower()

            if (
                "optimal_embedding_solution" in name
                or "embedding_solution" in name
                or "optimal" in name
            ):
                candidates.append(
                    candidate
                )

        if candidates:
            return sorted(candidates)[0]

    raise FileNotFoundError(
        f"ML solution not found near: {path}"
    )


def load_ml_solution():
    path = discover_ml_file()
    return load_json(path)


def load_chunks():
    files = list(
        CHUNKS_DIR.glob("*.bin")
    )

    if not files:
        raise FileNotFoundError(
            f"No .bin chunks found in {CHUNKS_DIR}"
        )

    def chunk_number(path):
        digits = "".join(
            character
            for character in path.stem
            if character.isdigit()
        )

        return (
            int(digits)
            if digits
            else 10**9
        )

    files.sort(
        key=chunk_number
    )

    chunks = {}

    for path in files:
        raw = path.read_bytes()

        try:
            text = raw.decode(
                "ascii"
            )
        except UnicodeDecodeError:
            text = None

        if (
            text is not None
            and text
            and all(
                bit in "01"
                for bit in text
            )
        ):
            bits = text.strip()

            if not bits:
                raise RuntimeError(
                    f"Chunk {path.stem} is empty."
                )

        else:
            bits = "".join(
                f"{byte:08b}"
                for byte in raw
            )

        chunks[path.stem] = {
            "chunk_id": path.stem,
            "bytes": raw,
            "bits": bits,
            "bit_length": len(bits),
            "sha256": hashlib.sha256(
                raw
            ).hexdigest()
        }

    return chunks

def load_message_metadata():
    return load_json(
        MESSAGE_JSON
    )


def load_qkd_blueprint():
    return load_json(
        QKD_PACKAGE
    )


def find_mapping_list(data):
    possible = [
        data.get("optimal_embedding"),
        data.get("embedding_solution"),
        data.get("embedding_plan"),
        data.get("solution"),
        data.get("selected_regions"),
        data.get("regions"),
        data.get("region_payloads")
    ]

    for value in possible:
        if isinstance(value, list):
            return value

    return None


def extract_region_id(item):
    if isinstance(item, dict):
        for key in [
            "region_id",
            "region_index",
            "id",
            "index"
        ]:
            if key in item:
                return item[key]

    return None


def extract_payload(item):
    if isinstance(item, dict):
        for key in [
            "payload_bits",
            "payload_size",
            "payload",
            "bits",
            "embedding_bits",
            "allocated_bits",
            "allocation"
        ]:
            if key in item:
                return int(item[key])

    return None


def normalize_region_id(value):
    if isinstance(value, int):
        return value

    text = str(value)

    if text.isdigit():
        return int(text)

    if "_" in text:
        tail = text.split("_")[-1]

        if tail.isdigit():
            return int(tail)

    return text


def extract_authoritative_embedding(data):
    result = []

    mapping = find_mapping_list(
        data
    )

    if mapping is not None:
        for item in mapping:
            region_id = extract_region_id(item)
            payload = extract_payload(item)

            if region_id is None or payload is None:
                continue

            result.append(
                {
                    "region_id":
                        normalize_region_id(region_id),
                    "payload_bits":
                        int(payload)
                }
            )

    if result:
        return result

    possible_maps = [
        data.get("region_payloads"),
        data.get("payload_allocations"),
        data.get("payload_allocation"),
        data.get("embedding_allocations"),
        data.get("selected_region_payloads")
    ]

    for mapping in possible_maps:
        if not isinstance(mapping, dict):
            continue

        for region_id, payload in mapping.items():
            result.append(
                {
                    "region_id":
                        normalize_region_id(region_id),
                    "payload_bits":
                        int(payload)
                }
            )

        if result:
            return result

    raise ValueError(
        "Could not locate authoritative ML region/payload allocation."
    )

def save_selected_regions_plot(
    operations
):
    if not operations:
        raise ValueError(
            "No embedding operations available for plotting."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    x_values = []
    y_values = []
    region_ids = []

    for operation in operations:

        if (
            "row" not in operation
            or
            "col" not in operation
        ):
            continue

        try:

            row = float(
                operation["row"]
            )

            col = float(
                operation["col"]
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        x_values.append(
            col
        )

        y_values.append(
            row
        )

        region_ids.append(
            str(
                operation.get(
                    "region_id",
                    "UNKNOWN"
                )
            )
        )

    if not x_values:

        raise ValueError(
            "No valid region coordinates available for plotting."
        )

    plot_path = (
        OUTPUT_DIR
        / "selected_embedding_regions.png"
    )

    plt.figure(
        figsize=(12, 10)
    )

    plt.scatter(
        x_values,
        y_values,
        s=35,
        marker="o",
        label="Selected Region"
    )

    for x, y, region_id in zip(
        x_values,
        y_values,
        region_ids
    ):

        plt.annotate(
            region_id,
            (
                x,
                y
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6
        )

    plt.xlabel(
        "X Coordinate (Column)"
    )

    plt.ylabel(
        "Y Coordinate (Row)"
    )

    plt.title(
        "Selected Regions for Adaptive DWT Embedding"
    )

    plt.gca().invert_yaxis()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "=" * 70
    )
    print(
        "SELECTED REGION PLOT SAVED"
    )
    print(
        "=" * 70
    )
    print(
        f"Selected regions : {len(x_values)}"
    )
    print(
        f"X range          : "
        f"{min(x_values):.0f} -> {max(x_values):.0f}"
    )
    print(
        f"Y range          : "
        f"{min(y_values):.0f} -> {max(y_values):.0f}"
    )
    print(
        f"Plot saved       : {plot_path}"
    )
    print(
        "=" * 70
    )

    return plot_path


def flatten_region_records(data):
    records = []

    def walk(value):
        if isinstance(value, dict):
            if (
                any(
                    key in value
                    for key in [
                        "region_id",
                        "region_index"
                    ]
                )
                and any(
                    key in value
                    for key in [
                        "band",
                        "subband"
                    ]
                )
                and any(
                    key in value
                    for key in [
                        "row_start",
                        "x"
                    ]
                )
            ):
                records.append(value)

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)

    return records


def load_region_database():
    candidates = []

    for root in [
        BASE_DIR / "ml_module",
        BASE_DIR / "output"
    ]:
        if not root.exists():
            continue

        candidates.extend(
            root.rglob("*.json")
        )

    records = []

    for path in candidates:
        try:
            data = load_json(path)
            records.extend(
                flatten_region_records(data)
            )
        except Exception:
            continue

    unique = {}

    for record in records:
        region_id = extract_region_id(
            record
        )

        if region_id is None:
            continue

        key = str(
            normalize_region_id(
                region_id
            )
        )

        if key not in unique:
            unique[key] = record

    return unique


def get_region_coordinates(record):
    coordinates = record.get(
        "coordinates",
        {}
    )

    row_start = record.get(
        "row_start",
        coordinates.get(
            "y"
        )
    )

    column_start = record.get(
        "column_start",
        coordinates.get(
            "x"
        )
    )

    row_end = record.get(
        "row_end"
    )

    column_end = record.get(
        "column_end"
    )

    rows = record.get(
        "rows",
        coordinates.get(
            "height"
        )
    )

    columns = record.get(
        "columns",
        coordinates.get(
            "width"
        )
    )

    if row_start is None or column_start is None:
        raise ValueError(
            f"Region {record.get('region_id')} has no coordinates."
        )

    if row_end is None:
        if rows is None:
            raise ValueError(
                f"Region {record.get('region_id')} has no row extent."
            )

        row_end = (
            int(row_start)
            + int(rows)
        )

    if column_end is None:
        if columns is None:
            raise ValueError(
                f"Region {record.get('region_id')} has no column extent."
            )

        column_end = (
            int(column_start)
            + int(columns)
        )

    return (
        int(row_start),
        int(row_end),
        int(column_start),
        int(column_end)
    )

def load_authoritative_plan():

    manifest = load_json(
        ADAPTIVE_REGION_MANIFEST
    )

    if not isinstance(
        manifest,
        dict
    ):
        raise ValueError(
            "Adaptive embedding manifest must be a JSON object."
        )

    manifest_regions = manifest.get(
        "regions"
    )

    if not isinstance(
        manifest_regions,
        list
    ):
        raise ValueError(
            "Adaptive embedding manifest must contain a 'regions' list."
        )

    plan = []

    for record in manifest_regions:

        if not isinstance(
            record,
            dict
        ):
            raise ValueError(
                "Invalid region entry in adaptive embedding manifest."
            )

        if "region_id" not in record:
            raise ValueError(
                "Adaptive manifest region is missing 'region_id'."
            )

        if "row" not in record:
            raise ValueError(
                f"Region {record.get('region_id')} is missing 'row'."
            )

        if "col" not in record:
            raise ValueError(
                f"Region {record.get('region_id')} is missing 'col'."
            )

        if "payload_bits" not in record:
            raise ValueError(
                f"Region {record.get('region_id')} is missing 'payload_bits'."
            )

        region_id = normalize_region_id(
            record["region_id"]
        )

        row = int(
            record["row"]
        )

        col = int(
            record["col"]
        )

        payload_bits = int(
            record["payload_bits"]
        )

        dwt_row = int(
            round(
                row / 2.0
            )
        )

        dwt_col = int(
            round(
                col / 2.0
            )
        )

        plan.append(
            {
                "region_id": region_id,
                "row": row,
                "col": col,
                "dwt_row": dwt_row,
                "dwt_col": dwt_col,
                "payload_bits": payload_bits
            }
        )

    declared_regions = manifest.get(
        "total_regions"
    )

    if declared_regions is not None:
        if int(declared_regions) != len(plan):
            raise ValueError(
                f"Adaptive manifest region count mismatch: "
                f"declared={int(declared_regions)}, "
                f"actual={len(plan)}."
            )

    declared_bits = manifest.get(
        "total_embedded_bits"
    )

    actual_bits = sum(
        int(
            item["payload_bits"]
        )
        for item in plan
    )

    if declared_bits is not None:
        if int(declared_bits) != actual_bits:
            raise ValueError(
                f"Adaptive manifest payload mismatch: "
                f"declared={int(declared_bits)}, "
                f"actual={actual_bits}."
            )

    if declared_bits is None:
        raise ValueError(
            "Adaptive manifest must contain 'total_embedded_bits'."
        )

    if actual_bits != int(declared_bits):
        raise ValueError(
            f"Adaptive manifest total bits mismatch: "
            f"declared={int(declared_bits)}, "
            f"actual={actual_bits}."
        )

    return plan


def load_dwt():
    result = {}

    for band in [
        "LL",
        "LH",
        "HL",
        "HH"
    ]:
        path = DWT_DIR / f"{band}.npy"

        if not path.exists():
            raise FileNotFoundError(path)

        result[band] = np.load(
            path
        ).astype(
            np.float64
        )

    return result


def bytes_from_bits(bits):
    if len(bits) % 8:
        bits = bits + (
            "0"
            * (
                8
                - (
                    len(bits) % 8
                )
            )
        )

    return bytes(
        int(
            bits[index:index + 8],
            2
        )
        for index in range(
            0,
            len(bits),
            8
        )
    )


def bits_from_bytes(data):
    return "".join(
        f"{byte:08b}"
        for byte in data
    )


def xor_bits(bits, key):
    key_bits = bits_from_bytes(
        key
    )

    return "".join(
        str(
            int(bit)
            ^ int(
                key_bits[
                    index
                    % len(key_bits)
                ]
            )
        )
        for index, bit in enumerate(bits)
    )


def hash_expand(seed, bit_count):
    output = bytearray()
    counter = 0

    required_bytes = (
        bit_count + 7
    ) // 8

    while len(output) < required_bytes:
        output.extend(
            hashlib.sha256(
                seed
                + counter.to_bytes(
                    8,
                    "big"
                )
            ).digest()
        )

        counter += 1

    return bytes(
        output[:required_bytes]
    )


def derive_key(bits, label, size):
    raw = bytes_from_bits(
        bits
    )

    material = hashlib.sha256(
        raw
        + label.encode("utf-8")
    ).digest()

    return hash_expand(
        material,
        size
    )

def make_bb84_circuit(
    bits,
    bases
):
    circuit = QuantumCircuit(
        len(bits),
        len(bits)
    )

    for index, bit in enumerate(bits):

        if bit:
            circuit.x(index)

        if bases[index]:
            circuit.h(index)

    return circuit

def measure_bb84_batch(
    circuits,
    bases_list
):
    measured_circuits = []

    for circuit, bases in zip(
        circuits,
        bases_list
    ):
        measured = circuit.copy()

        for index, basis in enumerate(bases):
            if basis:
                measured.h(index)

        # Ensure explicit classical register mapping matching the number of qubits
        num_q = measured.num_qubits
        if not measured.cregs:
            measured.add_register(qiskit.circuit.ClassicalRegister(num_q)) 

        measured.measure(
            range(num_q),
            range(num_q)
        )

        measured_circuits.append(
            measured
        )

    backend = AerSimulator(
        method="stabilizer"
    )

    compiled = transpile(
        measured_circuits,
        backend,
        optimization_level=0
    )

    result = backend.run(
        compiled,
        shots=1
    ).result()

    all_bits = []

    for index in range(
        len(measured_circuits)
    ):
        counts = result.get_counts(
            index
        )

        bitstring = next(
            iter(counts)
        ).replace(" ", "")

        # Qiskit bitstrings order classical bits with clbit 0 on the right.
        # Reversing safely maps index 0 to the first measured qubit.
        reversed_bits = bitstring[::-1]

        all_bits.extend(
            int(bit)
            for bit in reversed_bits
        )

    return all_bits
def qkd_alice(
    sock,
    qrng,
    session_id,
    required_bits,
    raw_qubits
):
    alice_bits = [
        int(bit)
        for bit in qrng.take(
            raw_qubits
        )
    ]

    alice_bases = [
        int(bit)
        for bit in qrng.take(
            raw_qubits
        )
    ]

    circuits = []
    batch_starts = []

    for start in range(
        0,
        raw_qubits,
        QKD_BATCH_SIZE
    ):
        end = min(
            start + QKD_BATCH_SIZE,
            raw_qubits
        )

        circuits.append(
            make_bb84_circuit(
                alice_bits[start:end],
                alice_bases[start:end]
            )
        )

        batch_starts.append(
            start
        )

    import io
    import base64
    from qiskit import qpy

    serialized_circuits = []

    for circuit in circuits:

        buffer = io.BytesIO()

        qpy.dump(
            circuit,
            buffer
        )

        serialized_circuits.append(
            base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")
        )

    send_packet(
        sock,
        {
            "type":
                "qkd_prepare_session",
            "session_id":
                session_id,
            "raw_qubits":
                raw_qubits,
            "batch_size":
                QKD_BATCH_SIZE,
            "circuits":
                serialized_circuits
        }
    )

    response = receive_packet(
        sock
    )

    if response["type"] != "qkd_measure_session":
        raise RuntimeError(
            "Invalid QKD measurement response."
        )

    if response["session_id"] != session_id:
        raise RuntimeError(
            "QKD session mismatch."
        )

    bob_bits = [
        int(bit)
        for bit in response[
            "bob_bits"
        ]
    ]

    bob_bases = [
        int(bit)
        for bit in response[
            "bob_bases"
        ]
    ]

    if len(bob_bits) != raw_qubits:
        raise RuntimeError(
            f"QKD bit count mismatch for "
            f"{session_id}: "
            f"expected {raw_qubits}, "
            f"received {len(bob_bits)}"
        )

    if len(bob_bases) != raw_qubits:
        raise RuntimeError(
            f"QKD basis count mismatch for "
            f"{session_id}."
        )

    sifted_alice = []
    sifted_bob = []

    for index in range(
        raw_qubits
    ):

        if (
            alice_bases[index]
            ==
            bob_bases[index]
        ):
            sifted_alice.append(
                alice_bits[index]
            )

            sifted_bob.append(
                bob_bits[index]
            )

    if len(sifted_alice) < required_bits:
        raise RuntimeError(
            f"Insufficient sifted bits in "
            f"{session_id}: "
            f"{len(sifted_alice)} < "
            f"{required_bits}"
        )

    qber = (
        sum(
            a != b
            for a, b in zip(
                sifted_alice,
                sifted_bob
            )
        )
        /
        len(sifted_alice)
    )

    send_packet(
        sock,
        {
            "type":
                "qkd_bases",
            "session_id":
                session_id,
            "alice_bases":
                alice_bases
        }
    )

    key = derive_key(
        "".join(
            map(
                str,
                sifted_alice
            )
        ),
        session_id,
        required_bits
    )

    fingerprint = hashlib.sha256(
        key
    ).hexdigest()

    send_packet(
        sock,
        {
            "type":
                "qkd_result",
            "session_id":
                session_id,
            "qber":
                qber,
            "fingerprint":
                fingerprint
        }
    )

    confirmation = receive_packet(
        sock
    )

    if confirmation["type"] != "qkd_verified":
        raise RuntimeError(
            "QKD verification failed."
        )

    if confirmation["session_id"] != session_id:
        raise RuntimeError(
            "QKD verification session mismatch."
        )

    if qber > QBER_THRESHOLD:
        raise RuntimeError(
            f"QBER too high for "
            f"{session_id}: {qber}"
        )

    return {
        "key":
            key,
        "qber":
            qber,
        "fingerprint":
            fingerprint,
        "sifted_bits":
            len(sifted_alice)
    }


def qkd_bob(
    sock,
    qrng,
    required_bits
):
    packet = receive_packet(
        sock
    )

    if packet["type"] != "qkd_prepare_session":
        raise RuntimeError(
            "Expected QKD session preparation."
        )

    session_id = packet[
        "session_id"
    ]

    raw_qubits = int(
        packet["raw_qubits"]
    )

    encoded_circuits = packet[
        "circuits"
    ]

    circuits = []

    import io
    import base64
    from qiskit import qpy

    for encoded in encoded_circuits:

        raw = base64.b64decode(
            encoded
        )

        loaded = qpy.load(
            io.BytesIO(raw)
        )

        circuits.append(
            loaded[0]
        )

    total_qubits = sum(
        circuit.num_qubits
        for circuit in circuits
    )

    if total_qubits != raw_qubits:
        raise RuntimeError(
            f"QKD circuit size mismatch "
            f"for {session_id}: "
            f"{total_qubits} != "
            f"{raw_qubits}"
        )

    bob_bases = [
        int(bit)
        for bit in qrng.take(
            raw_qubits
        )
    ]

    bases_list = []

    pointer = 0

    for circuit in circuits:

        count = circuit.num_qubits

        bases_list.append(
            bob_bases[
                pointer:
                pointer + count
            ]
        )

        pointer += count

    print(
        f"QKD {session_id}: "
        f"measuring {raw_qubits} qubits"
    )

    bob_bits = measure_bb84_batch(
        circuits,
        bases_list
    )

    if len(bob_bits) != raw_qubits:
        raise RuntimeError(
            f"QKD measurement size mismatch "
            f"for {session_id}: "
            f"{len(bob_bits)} != "
            f"{raw_qubits}"
        )

    print(
        f"QKD {session_id}: "
        f"measurement complete"
    )

    send_packet(
        sock,
        {
            "type":
                "qkd_measure_session",
            "session_id":
                session_id,
            "bob_bases":
                bob_bases,
            "bob_bits":
                bob_bits
        }
    )

    packet = receive_packet(
        sock
    )

    if packet["type"] != "qkd_bases":
        raise RuntimeError(
            "Expected Alice basis information."
        )

    if packet["session_id"] != session_id:
        raise RuntimeError(
            "QKD session mismatch."
        )

    alice_bases = [
        int(bit)
        for bit in packet[
            "alice_bases"
        ]
    ]

    if len(alice_bases) != raw_qubits:
        raise RuntimeError(
            f"Alice basis count mismatch "
            f"for {session_id}."
        )

    sifted_bob = []

    for index in range(
        raw_qubits
    ):
        if (
            alice_bases[index]
            ==
            bob_bases[index]
        ):
            sifted_bob.append(
                bob_bits[index]
            )

    if len(sifted_bob) < required_bits:
        raise RuntimeError(
            f"Insufficient sifted bits for "
            f"{session_id}: "
            f"{len(sifted_bob)} < "
            f"{required_bits}"
        )

    key = derive_key(
        "".join(
            map(
                str,
                sifted_bob
            )
        ),
        session_id,
        required_bits
    )

    fingerprint = hashlib.sha256(
        key
    ).hexdigest()

    result = receive_packet(
        sock
    )

    if result["type"] != "qkd_result":
        raise RuntimeError(
            "Expected QKD result."
        )

    if result["session_id"] != session_id:
        raise RuntimeError(
            "QKD result session mismatch."
        )

    if float(
        result["qber"]
    ) > QBER_THRESHOLD:
        raise RuntimeError(
            f"QBER too high for "
            f"{session_id}: "
            f"{result['qber']}"
        )

    if not secrets.compare_digest(
        fingerprint,
        result["fingerprint"]
    ):
        raise RuntimeError(
            f"QKD fingerprint mismatch "
            f"for {session_id}."
        )

    send_packet(
        sock,
        {
            "type":
                "qkd_verified",
            "session_id":
                session_id,
            "fingerprint":
                fingerprint
        }
    )

    return {
        "session_id":
            session_id,
        "key":
            key,
        "qber":
            float(
                result["qber"]
            ),
        "fingerprint":
            fingerprint,
        "sifted_bits":
            len(sifted_bob)
    }
def synchronized_order(
    plan,
    master_key,
    qrng,
    qkd_pattern
):
    seed_material = (
        master_key
        + qkd_pattern
        + b"STEGAQENTROPY_ADAPTIVE_ORDER"
    )

    seed = hashlib.sha512(
        seed_material
    ).digest()

    generator = random.Random(
        int.from_bytes(
            seed,
            "big"
        )
    )

    operations = list(
        plan
    )

    generator.shuffle(
        operations
    )

    return operations, seed


def region_positions(
    region,
    seed
):
    dwt_row = int(
        region[
            "dwt_row"
        ]
    )

    dwt_col = int(
        region[
            "dwt_col"
        ]
    )

    height = 256
    width = 256
    region_size = 16

    max_row_start = (
        height
        - region_size
    )

    max_column_start = (
        width
        - region_size
    )

    row_start = max(
        0,
        min(
            dwt_row - region_size // 2,
            max_row_start
        )
    )

    column_start = max(
        0,
        min(
            dwt_col - region_size // 2,
            max_column_start
        )
    )

    row_end = (
        row_start
        + region_size
    )

    column_end = (
        column_start
        + region_size
    )

    positions = [
        (
            row,
            column
        )
        for row in range(
            row_start,
            row_end
        )
        for column in range(
            column_start,
            column_end
        )
    ]

    digest = hashlib.sha256(
        seed
        + str(
            region[
                "region_id"
            ]
        ).encode()
    ).digest()

    generator = random.Random(
        int.from_bytes(
            digest,
            "big"
        )
    )

    generator.shuffle(
        positions
    )

    return positions

def embed_bit(
    coefficient,
    bit,
    delta
):
    value = float(
        coefficient
    )

    target_bit = int(
        bit
    )

    scaled = (
        value / delta
    )

    base = int(
        np.floor(
            scaled + 0.5
        )
    )

    candidates = []

    for offset in range(
        -4,
        5
    ):
        candidate = (
            base
            + offset
        )

        if (
            candidate % 2
            != target_bit
        ):
            continue

        candidate_value = (
            candidate * delta
        )

        distance = abs(
            candidate_value
            - value
        )

        candidates.append(
            (
                distance,
                candidate
            )
        )

    if not candidates:
        raise RuntimeError(
            "Unable to find valid "
            "parity embedding target."
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    selected = candidates[0][1]

    selected_value = (
        selected * delta
    )

    margin = (
        0.08 * delta
    )

    if target_bit == 0:
        embedded_value = (
            selected_value
            + margin
        )
    else:
        embedded_value = (
            selected_value
            - margin
        )

    lower = (
        (selected - 0.5)
        * delta
        + 1e-12
    )

    upper = (
        (selected + 0.5)
        * delta
        - 1e-12
    )

    embedded_value = float(
        np.clip(
            embedded_value,
            lower,
            upper
        )
    )

    return embedded_value


def extract_bit(
    coefficient,
    delta
):
    value = float(
        coefficient
    )

    quantized = int(
        np.floor(
            value / delta
            + 0.5
        )
    )

    return str(
        quantized % 2
    )

def calculate_delta(
    region
):
    variance = float(
        region.get(
            "variance",
            0.0
        )
    )

    tolerance = float(
        region.get(
            "distortion_tolerance",
            0.0
        )
    )

    entropy = float(
        region.get(
            "entropy",
            0.0
        )
    )

    gradient = float(
        region.get(
            "average_gradient",
            region.get(
                "gradient",
                0.0
            )
        )
    )

    complexity = float(
        region.get(
            "complexity_score",
            0.0
        )
    )

    variance_score = min(
        1.0,
        abs(variance) / 1000.0
    )

    tolerance_score = min(
        1.0,
        abs(tolerance)
    )

    entropy_score = min(
        1.0,
        abs(entropy) / 8.0
    )

    gradient_score = min(
        1.0,
        abs(gradient) / 100.0
    )

    complexity_score = min(
        1.0,
        abs(complexity)
    )

    score = (
        variance_score * 0.30
        + tolerance_score * 0.20
        + entropy_score * 0.20
        + gradient_score * 0.15
        + complexity_score * 0.15
    )

    score = float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )

    delta_min = 0.0040
    delta_max = 0.0080

    delta = (
        delta_min
        + (
            1.0 - score
        ) ** 2
        * (
            delta_max
            - delta_min
        )
    )

    return float(
        np.clip(
            delta,
            delta_min,
            delta_max
        )
    )

def embed(
    dwt,
    operations,
    subkeys,
    master_key,
    schedule_seed,
    enhanced_chunks
):
    modified = {
        band:
            values.copy()
        for band, values in dwt.items()
    }

    band_choices = [
        "LH",
        "HL",
        "HH"
    ]

    used = set()
    expanded_operations = []
    coefficient_magnitudes = []
    coefficient_deltas = []

    chunk_ids = list(
        enhanced_chunks.keys()
    )

    chunk_cursors = {
        chunk_id: 0
        for chunk_id in chunk_ids
    }

    total_payload_bits = sum(
        len(
            str(
                enhanced_chunks[chunk_id]["bits"]
            )
        )
        for chunk_id in chunk_ids
    )

    total_embedded_bits = 0

    current_chunk_index = 0

    # ---------------------------------------------------------------
    # ADAPTIVE REGION ALLOCATION
    #
    # Regions are consumed sequentially from the selected ML pool.
    #
    # A chunk may span any number of regions.
    # A region may contain only part of a chunk.
    #
    # Allocation =
    # min(
    #     remaining chunk bits,
    #     ML usable capacity,
    #     actual available DWT positions
    # )
    # ---------------------------------------------------------------

    for region_index, operation in enumerate(
        operations
    ):

        if current_chunk_index >= len(
            chunk_ids
        ):
            break

        region_id = operation[
            "region_id"
        ]

        ml_capacity = int(
            operation.get(
                "ml_payload_bits",
                operation.get(
                    "payload_bits",
                    0
                )
            )
        )

        if ml_capacity <= 0:
            continue

        # -----------------------------------------------------------
        # Try the three high-frequency DWT bands.
        # Band order is randomized deterministically.
        # -----------------------------------------------------------

        band_seed = hashlib.sha256(
            schedule_seed
            + str(
                region_id
            ).encode("utf-8")
        ).digest()

        band_rng = random.Random(
            int.from_bytes(
                band_seed,
                "big"
            )
        )

        randomized_bands = list(
            band_choices
        )

        band_rng.shuffle(
            randomized_bands
        )

        region_embedded = 0

        # -----------------------------------------------------------
        # A region may receive portions from consecutive chunks if
        # its ML capacity allows it.
        # -----------------------------------------------------------

        remaining_region_capacity = (
            ml_capacity
        )

        for selected_band in randomized_bands:

            if (
                remaining_region_capacity
                <= 0
            ):
                break

            # Move past chunks that are already completely consumed.
            while (
                current_chunk_index
                <
                len(chunk_ids)
            ):

                active_chunk_id = (
                    chunk_ids[
                        current_chunk_index
                    ]
                )

                active_bits = str(
                    enhanced_chunks[
                        active_chunk_id
                    ]["bits"]
                )

                cursor = chunk_cursors[
                    active_chunk_id
                ]

                if cursor < len(
                    active_bits
                ):
                    break

                current_chunk_index += 1

            if current_chunk_index >= len(
                chunk_ids
            ):
                break

            active_chunk_id = (
                chunk_ids[
                    current_chunk_index
                ]
            )

            active_bits = str(
                enhanced_chunks[
                    active_chunk_id
                ]["bits"]
            )

            cursor = chunk_cursors[
                active_chunk_id
            ]

            remaining_chunk_bits = (
                len(active_bits)
                - cursor
            )

            if remaining_chunk_bits <= 0:
                continue

            # -------------------------------------------------------
            # Region positions are derived using the active chunk's
            # subkey so the receiver can deterministically reproduce
            # the same positions.
            # -------------------------------------------------------

            region_seed = hashlib.sha256(
                schedule_seed
                + subkeys[
                    active_chunk_id
                ]
                + str(
                    region_id
                ).encode("utf-8")
            ).digest()

            candidate_region = dict(
                operation
            )

            candidate_region[
                "chunk_id"
            ] = active_chunk_id

            candidate_region[
                "band"
            ] = selected_band

            positions = region_positions(
                candidate_region,
                schedule_seed
                + subkeys[
                    active_chunk_id
                ]
            )

            matrix = modified[
                selected_band
            ]

            available_positions = []

            for position in positions:

                row, column = position

                if (
                    row < 0
                    or column < 0
                    or row >= matrix.shape[0]
                    or column >= matrix.shape[1]
                ):
                    continue

                identity = (
                    selected_band,
                    row,
                    column
                )

                if identity in used:
                    continue

                available_positions.append(
                    position
                )

            actual_dwt_capacity = len(
                available_positions
            )

            if actual_dwt_capacity <= 0:
                continue

            # -------------------------------------------------------
            # THIS IS THE CORE ADAPTIVE RULE.
            # -------------------------------------------------------

            take = min(
                remaining_chunk_bits,
                remaining_region_capacity,
                actual_dwt_capacity
            )

            if take <= 0:
                continue

            payload_part = (
                active_bits[
                    cursor:
                    cursor + take
                ]
            )

            delta = calculate_delta(
                candidate_region
            )

            for index, bit in enumerate(
                payload_part
            ):

                row, column = (
                    available_positions[
                        index
                    ]
                )

                identity = (
                    selected_band,
                    row,
                    column
                )

                used.add(
                    identity
                )

                # ---------------------------------------------------
                # RECORD ACTUAL DWT COEFFICIENT SCALE
                # BEFORE EMBEDDING
                # ---------------------------------------------------
                original_coefficient = float(
                    matrix[
                        row,
                        column
                    ]
                )

                coefficient_magnitudes.append(
                    abs(original_coefficient)
                )

                coefficient_deltas.append(
                    float(delta)
                )

                matrix[
                    row,
                    column
                ] = embed_bit(
                    matrix[
                        row,
                        column
                    ],
                    bit,
                    delta
                )

            expanded_operations.append(
                {
                    **candidate_region,
                    "band":
                        selected_band,
                    "chunk_bit_start":
                        cursor,
                    "payload_bits_stream":
                        payload_part,
                    "payload_length":
                        take,
                    "embedding_delta":
                        delta,
                    "ml_payload_bits":
                        ml_capacity,
                    "actual_dwt_capacity":
                        actual_dwt_capacity,
                    "allocated_bits":
                        take
                }
            )

            chunk_cursors[
                active_chunk_id
            ] += take

            remaining_region_capacity -= take
            region_embedded += take
            total_embedded_bits += take

            # -------------------------------------------------------
            # If this chunk is complete, immediately continue with
            # the NEXT chunk using the remaining capacity of the
            # current region.
            # -------------------------------------------------------

            if (
                chunk_cursors[
                    active_chunk_id
                ]
                >=
                len(active_bits)
            ):

                current_chunk_index += 1

        print(
            f"[ADAPTIVE REGION] "
            f"region={region_id} "
            f"ML_capacity={ml_capacity} "
            f"allocated={region_embedded} "
            f"remaining_region="
            f"{remaining_region_capacity}"
        )

    # ---------------------------------------------------------------
    # FINAL CAPACITY CHECK
    # ---------------------------------------------------------------

    if total_embedded_bits != total_payload_bits:

        remaining_details = []

        for chunk_id in chunk_ids:

            bits = str(
                enhanced_chunks[
                    chunk_id
                ]["bits"]
            )

            cursor = chunk_cursors[
                chunk_id
            ]

            remaining = (
                len(bits)
                - cursor
            )

            if remaining > 0:

                remaining_details.append(
                    f"{chunk_id}={remaining}"
                )

        raise RuntimeError(
            "Insufficient distributed DWT capacity "
            "after exhausting all selected regions. "
            f"embedded={total_embedded_bits}, "
            f"required={total_payload_bits}, "
            f"remaining: "
            f"{', '.join(remaining_details)}"
        )

    # ---------------------------------------------------------------
    # Replace the original region/chunk plan with the ACTUAL
    # adaptive embedding operations.
    # ---------------------------------------------------------------

    operations.clear()

    operations.extend(
        expanded_operations
    )
        # ---------------------------------------------------------------
    # COEFFICIENT-SCALE DIAGNOSTICS
    # ---------------------------------------------------------------
    if coefficient_magnitudes:
        coefficient_magnitudes = np.asarray(
            coefficient_magnitudes,
            dtype=np.float64
        )

        coefficient_deltas = np.asarray(
            coefficient_deltas,
            dtype=np.float64
        )

        nonzero_mask = coefficient_magnitudes > 1e-12

        nonzero_magnitudes = coefficient_magnitudes[nonzero_mask]
        nonzero_deltas = coefficient_deltas[nonzero_mask]

        relative_delta = (
            nonzero_deltas
            / nonzero_magnitudes
        )
        print(
            f"Nonzero coefficients  : "
            f"{len(nonzero_magnitudes)}"
        )

        print(
            f"Delta/|c| minimum     : "
            f"{np.min(relative_delta):.12e}"
        )

        print(
            f"Delta/|c| median      : "
            f"{np.median(relative_delta):.12e}"
        )

        print(
            f"Delta/|c| maximum     : "
            f"{np.max(relative_delta):.12e}"
        )

        print()
        print("=" * 70)
        print("DWT COEFFICIENT / DELTA SCALE ANALYSIS")
        print("=" * 70)

        print(
            f"Modified coefficients : "
            f"{len(coefficient_magnitudes)}"
        )

        print(
            f"|c| minimum           : "
            f"{np.min(coefficient_magnitudes):.12e}"
        )

        print(
            f"|c| median            : "
            f"{np.median(coefficient_magnitudes):.12e}"
        )

        print(
            f"|c| maximum           : "
            f"{np.max(coefficient_magnitudes):.12e}"
        )

        print()

        print(
            f"Delta minimum         : "
            f"{np.min(coefficient_deltas):.12e}"
        )

        print(
            f"Delta median          : "
            f"{np.median(coefficient_deltas):.12e}"
        )

        print(
            f"Delta maximum         : "
            f"{np.max(coefficient_deltas):.12e}"
        )

        print()

        print(
            f"Delta/|c| minimum     : "
            f"{np.min(relative_delta):.12e}"
        )

        print(
            f"Delta/|c| median      : "
            f"{np.median(relative_delta):.12e}"
        )

        print(
            f"Delta/|c| maximum     : "
            f"{np.max(relative_delta):.12e}"
        )

        print("=" * 70)

    else:
        print(
            "WARNING: No coefficient-scale "
            "diagnostic samples were collected."
        )
    print()
    print(
        "=" * 70
    )
    print(
        "ADAPTIVE DWT EMBEDDING COMPLETE"
    )
    print(
        "=" * 70
    )
    print(
        f"Selected regions       : {len(operations)}"
    )
    print(
        f"Total payload bits     : {total_payload_bits}"
    )
    print(
        f"Actually embedded      : {total_embedded_bits}"
    )
    print(
        f"Remaining bits         : "
        f"{total_payload_bits - total_embedded_bits}"
    )
    print(
        "=" * 70
    )

    return modified


def extract(
    dwt,
    operations,
    subkeys,
    schedule_seed
):
    result = {}

    used = set()

    for operation in operations:

        chunk_id = operation[
            "chunk_id"
        ]

        band = operation[
            "band"
        ]

        positions = region_positions(
            operation,
            schedule_seed
            + subkeys[
                chunk_id
            ]
        )

        matrix = dwt[
            band
        ]

        if "embedding_delta" in operation:
            delta = float(
                operation[
                    "embedding_delta"
                ]
            )
        else:
            delta = calculate_delta(
                operation
            )

        available_positions = []

        for position in positions:

            row, column = position

            if (
                row < 0
                or column < 0
                or row >= matrix.shape[0]
                or column >= matrix.shape[1]
            ):
                continue

            identity = (
                band,
                row,
                column
            )

            if identity in used:
                continue

            available_positions.append(
                position
            )

        length = int(
            operation[
                "payload_length"
            ]
        )

        if length > len(
            available_positions
        ):
            raise RuntimeError(
                f"Region {operation['region_id']} "
                f"band {band} does not have enough "
                f"unused coefficients. "
                f"required={length}, "
                f"available={len(available_positions)}."
            )

        bits = []

        for index in range(
            length
        ):

            row, column = (
                available_positions[
                    index
                ]
            )

            identity = (
                band,
                row,
                column
            )

            used.add(
                identity
            )

            coefficient = float(
                matrix[
                    row,
                    column
                ]
            )

            bits.append(
                extract_bit(
                    coefficient,
                    delta
                )
            )

        result.setdefault(
            chunk_id,
            []
        ).append(
            (
                int(
                    operation[
                        "chunk_bit_start"
                    ]
                ),
                "".join(bits)
            )
        )

    final = {}

    for chunk_id, parts in result.items():

        parts.sort(
            key=lambda item:
            item[0]
        )

        reconstructed = []

        expected_start = 0

        for start, bits in parts:

            if start != expected_start:
                raise RuntimeError(
                    f"Chunk {chunk_id} "
                    f"has discontinuous extraction. "
                    f"expected_start={expected_start}, "
                    f"received_start={start}."
                )

            reconstructed.append(
                bits
            )

            expected_start += len(
                bits
            )

        final[
            chunk_id
        ] = "".join(
            reconstructed
        )

    return final
def reconstruct(
    dwt
):
    return pywt.idwt2(
        (
            dwt["LL"],
            (
                dwt["LH"],
                dwt["HL"],
                dwt["HH"]
            )
        ),
        WAVELET,
        mode="periodization"
    )


def save_array(path, array):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        path,
        array
    )

def calculate_metrics(
    original,
    stego,
    embedded_bits,
    ber
):
    original = np.asarray(
        original,
        dtype=np.float64
    )

    stego = np.asarray(
        stego,
        dtype=np.float64
    )

    # ============================================================
    # FLOATING-POINT DIFFERENCE
    # ============================================================

    difference = (
        stego
        -
        original
    )

    mse = float(
        np.mean(
            difference ** 2
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    mae = float(
        np.mean(
            np.abs(
                difference
            )
        )
    )

    if mse <= 1e-16:
        psnr = float("inf")
    else:
        psnr = float(
            10.0
            *
            np.log10(
                (
                    255.0 ** 2
                )
                /
                mse
            )
        )

    # ============================================================
    # SSIM - FLOATING POINT
    # ============================================================

    ssim = float(
        structural_similarity(
            original,
            stego,
            data_range=255.0
        )
    )

    # ============================================================
    # HISTOGRAMS - FLOATING POINT
    # ============================================================

    original_histogram = np.histogram(
        original,
        bins=256,
        range=(0.0, 256.0),
        density=True
    )[0]

    stego_histogram = np.histogram(
        stego,
        bins=256,
        range=(0.0, 256.0),
        density=True
    )[0]

    histogram_epsilon = 1e-12

    original_probability = (
        original_histogram
        +
        histogram_epsilon
    )

    stego_probability = (
        stego_histogram
        +
        histogram_epsilon
    )

    original_probability /= np.sum(
        original_probability
    )

    stego_probability /= np.sum(
        stego_probability
    )

    # ============================================================
    # HISTOGRAM SIMILARITY
    # ============================================================

    histogram_similarity = float(
        np.sum(
            np.sqrt(
                original_probability
                *
                stego_probability
            )
        )
    )

    # ============================================================
    # KL DIVERGENCE
    # ============================================================

    kl_divergence = float(
        np.sum(
            original_probability
            *
            np.log(
                original_probability
                /
                stego_probability
            )
        )
    )

    # ============================================================
    # ENTROPY
    # ============================================================

    def image_entropy(
        image
    ):
        histogram = np.histogram(
            image,
            bins=256,
            range=(0.0, 256.0),
            density=True
        )[0]

        histogram = (
            histogram
            +
            histogram_epsilon
        )

        histogram /= np.sum(
            histogram
        )

        return float(
            -np.sum(
                histogram
                *
                np.log2(
                    histogram
                )
            )
        )

    original_entropy = image_entropy(
        original
    )

    stego_entropy = image_entropy(
        stego
    )

    # ============================================================
    # NPCR - FLOATING POINT
    # ============================================================

    changed_pixels = (
        np.abs(
            stego
            -
            original
        )
        >
        1e-12
    )

    npcr = float(
        np.mean(
            changed_pixels
        )
        *
        100.0
    )

    # ============================================================
    # UACI - FLOATING POINT
    # ============================================================

    uaci = float(
        np.mean(
            np.abs(
                stego
                -
                original
            )
            /
            255.0
        )
        *
        100.0
    )

    # ============================================================
    # BPP
    # ============================================================

    total_pixels = int(
        original.size
    )

    bpp = float(
        embedded_bits
        /
        total_pixels
    )

    # ============================================================
    # MAXIMUM FLOATING-POINT CHANGE
    # ============================================================

    maximum_absolute_change = float(
        np.max(
            np.abs(
                difference
            )
        )
    )

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    print()
    print(
        "=" * 70
    )
    print(
        "SENDER IMAGE QUALITY METRICS"
    )
    print(
        "FLOATING-POINT DOMAIN"
    )
    print(
        "=" * 70
    )

    print(
        f"SSIM                         : "
        f"{ssim:.12f}"
    )

    print(
        "FSIM                         : NILL"
    )

    print(
        f"MSE                          : "
        f"{mse:.12e}"
    )

    print(
        f"MAE                          : "
        f"{mae:.12e}"
    )

    print(
        f"RMSE                         : "
        f"{rmse:.12e}"
    )

    print(
        f"PSNR                         : "
        f"{psnr:.6f} dB"
    )

    print(
        f"Original Entropy             : "
        f"{original_entropy:.12f}"
    )

    print(
        f"Stego Entropy                : "
        f"{stego_entropy:.12f}"
    )

    print(
        f"KL Divergence                : "
        f"{kl_divergence:.12e}"
    )

    print(
        f"Histogram Similarity         : "
        f"{histogram_similarity:.12f}"
    )

    print(
        f"NPCR                         : "
        f"{npcr:.12f} %"
    )

    print(
        f"UACI                         : "
        f"{uaci:.12e} %"
    )

    print(
        f"BER                          : "
        f"{ber:.12f}"
    )

    print(
        f"BPP                          : "
        f"{bpp:.12f}"
    )

    print(
        f"Maximum Absolute Change      : "
        f"{maximum_absolute_change:.12e}"
    )

    print("=" * 70)

    # ============================================================
    # STEGANALYSIS
    # ============================================================

    steganalysis_results = calculate_and_print_steganalysis(
        original,
        stego
    )

    return {
        "ssim": ssim,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "mae": mae,
        "original_entropy": original_entropy,
        "stego_entropy": stego_entropy,
        "kl_divergence": kl_divergence,
        "histogram_similarity": histogram_similarity,
        "npcr": npcr,
        "uaci": uaci,
        "ber": float(ber),
        "bpp": bpp,
        "maximum_absolute_change": maximum_absolute_change,
        "embedded_bits": int(embedded_bits),
        "steganalysis": steganalysis_results
    }
def save_stego_view(
    stego_image
):
    stego_uint8 = np.rint(
        np.clip(
            stego_image,
            0.0,
            255.0
        )
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        stego_uint8,
        mode="L"
    )

    image.save(
        STEGO_VIEW_IMAGE
    )
    

    return STEGO_VIEW_IMAGE
def create_operations_from_chunk_mapping(
    plan,
    enhanced_chunks,
    qrng
):
    chunk_ids = list(
        enhanced_chunks.keys()
    )

    normalized_chunks = {
        chunk_id:
            str(
                enhanced_chunks[chunk_id]["bits"]
            ).strip()
        for chunk_id in chunk_ids
    }

    for chunk_id, bits in normalized_chunks.items():

        if not bits:
            raise RuntimeError(
                f"Chunk {chunk_id} is empty."
            )

        if not all(
            bit in "01"
            for bit in bits
        ):
            raise ValueError(
                f"Invalid binary data in {chunk_id}."
            )

    plan_indices = list(
        range(
            len(plan)
        )
    )

    allocation_seed = int(
        qrng.take(
            256
        ),
        2
    )

    allocation_rng = random.Random(
        allocation_seed
    )

    allocation_rng.shuffle(
        plan_indices
    )

    operations = []

    for p_idx in plan_indices:

        region = dict(
            plan[p_idx]
        )

        region_capacity = int(
            region.get(
                "payload_bits",
                0
            )
        )

        if region_capacity <= 0:
            continue

        operations.append(
            {
                **region,
                "ml_payload_bits":
                    region_capacity
            }
        )

    selected_ids = {
        normalize_region_id(
            operation["region_id"]
        )
        for operation in operations
    }

    try:
        metadata_df = pd.read_csv(
            REGION_METADATA_CSV
        )
    except Exception:
        metadata_df = None

    if metadata_df is not None:

        region_id_column = None

        for column in [
            "region_id",
            "region_index",
            "id",
            "index"
        ]:

            if column in metadata_df.columns:
                region_id_column = column
                break

        capacity_column = None

        for column in [
            "actual_safe_capacity",
            "safe_capacity",
            "capacity"
        ]:

            if column in metadata_df.columns:
                capacity_column = column
                break

        row_column = None

        for column in [
            "row",
            "row_start",
            "region_row"
        ]:

            if column in metadata_df.columns:
                row_column = column
                break

        col_column = None

        for column in [
            "col",
            "column_start",
            "region_col"
        ]:

            if column in metadata_df.columns:
                col_column = column
                break

        if (
            region_id_column is not None
            and capacity_column is not None
            and row_column is not None
            and col_column is not None
        ):

            fallback_records = []

            for _, record in metadata_df.iterrows():

                try:
                    region_id = normalize_region_id(
                        record[region_id_column]
                    )
                except Exception:
                    continue

                if region_id in selected_ids:
                    continue

                value = pd.to_numeric(
                    record[capacity_column],
                    errors="coerce"
                )

                if not np.isfinite(value):
                    continue

                usable_capacity = int(
                    np.floor(
                        float(value)
                        * 0.95
                    )
                )

                if usable_capacity <= 0:
                    continue

                try:
                    row = int(
                        round(
                            float(
                                record[row_column]
                            )
                        )
                    )

                    col = int(
                        round(
                            float(
                                record[col_column]
                            )
                        )
                    )

                except Exception:
                    continue

                fallback_records.append(
                    {
                        "region_id": region_id,
                        "row": row,
                        "col": col,
                        "dwt_row": int(
                            round(
                                row / 2.0
                            )
                        ),
                        "dwt_col": int(
                            round(
                                col / 2.0
                            )
                        ),
                        "payload_bits":
                            usable_capacity,
                        "ml_payload_bits":
                            usable_capacity
                    }
                )

            fallback_seed = int(
                qrng.take(
                    256
                ),
                2
            )

            fallback_rng = random.Random(
                fallback_seed
            )

            fallback_rng.shuffle(
                fallback_records
            )

            operations.extend(
                fallback_records
            )

    if not operations:
        raise RuntimeError(
            "No usable selected regions are available."
        )

    return operations

def validate_plan(
    operations,
    enhanced_chunks
):
    total_payload = 0

    for chunk_id, chunk in enhanced_chunks.items():

        if isinstance(chunk, dict):
            bits = str(
                chunk[
                    "bits"
                ]
            )

        elif isinstance(chunk, str):
            bits = chunk

        elif isinstance(chunk, bytes):
            bits = chunk.decode(
                "utf-8"
            )

        else:
            raise TypeError(
                f"Unsupported chunk type for "
                f"{chunk_id}: "
                f"{type(chunk).__name__}"
            )

        if not bits:
            raise RuntimeError(
                f"Chunk {chunk_id} is empty."
            )

        if not all(
            bit in "01"
            for bit in bits
        ):
            raise ValueError(
                f"Invalid binary data in "
                f"{chunk_id}."
            )

        total_payload += len(bits)

    total_allocated = sum(
        int(
            operation[
                "payload_length"
            ]
        )
        for operation in operations
    )

    if total_allocated != total_payload:
        raise RuntimeError(
            "ML payload mismatch: "
            f"allocated={total_allocated}, "
            f"payload={total_payload}"
        )
    
def prepare_chunk_streams(
    chunks,
    subkeys
):
    streams = {}

    for chunk_id, chunk in chunks.items():

        if chunk_id not in subkeys:
            raise RuntimeError(
                f"Missing QKD subkey for {chunk_id}."
            )

        protected_bits = xor_bits(
            chunk["bits"],
            subkeys[chunk_id]
        )

        subkey_bits = bits_from_bytes(
            subkeys[chunk_id]
        )

        streams[chunk_id] = (
            subkey_bits
            + protected_bits
        )

    return streams

def save_experimental_visualizations(
    original_image,
    stego_image,
    output_dir,
    capacity_psnr_csv=None
):
    """
    Generate the visual/statistical figures used in the paper.

    Outputs:
        1. Cover | Stego | Amplified Difference
        2. Cover vs. Stego Histogram
        3. PSNR vs. Embedding Capacity (if CSV is supplied)

    The figures are saved inside:
        output_dir / "experimental_visualizations"
    """

    output_dir = Path(output_dir)

    visualization_dir = (
        output_dir
        / "experimental_visualizations"
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    original = np.asarray(
        original_image,
        dtype=np.float64
    )

    stego = np.asarray(
        stego_image,
        dtype=np.float64
    )

    original_uint8 = np.rint(
        np.clip(
            original,
            0.0,
            255.0
        )
    ).astype(
        np.uint8
    )

    stego_uint8 = np.rint(
        np.clip(
            stego,
            0.0,
            255.0
        )
    ).astype(
        np.uint8
    )

    # ==============================================================
    # FIGURE 1
    # Cover | Stego | Amplified Difference
    # ==============================================================

    difference = (
        stego_uint8.astype(np.float64)
        -
        original_uint8.astype(np.float64)
    )

    # Amplify very small embedding changes for visualization.
    amplification = 50.0

    amplified_difference = (
        np.abs(difference)
        * amplification
    )

    amplified_difference = np.clip(
        amplified_difference,
        0.0,
        255.0
    ).astype(
        np.uint8
    )

    difference_path = (
        visualization_dir
        / "cover_stego_difference.png"
    )

    plt.figure(
        figsize=(15, 5)
    )

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_uint8,
        cmap="gray"
    )

    plt.title(
        "Cover Image"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        stego_uint8,
        cmap="gray"
    )

    plt.title(
        "Stego Image"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        amplified_difference,
        cmap="gray"
    )

    plt.title(
        "Amplified Difference"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        difference_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==============================================================
    # FIGURE 2
    # Cover vs Stego Histogram
    # ==============================================================

    histogram_path = (
        visualization_dir
        / "cover_stego_histogram.png"
    )

    cover_histogram = np.histogram(
        original_uint8,
        bins=256,
        range=(0, 256),
        density=True
    )[0]

    stego_histogram = np.histogram(
        stego_uint8,
        bins=256,
        range=(0, 256),
        density=True
    )[0]

    intensity = np.arange(
        256
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        intensity,
        cover_histogram,
        label="Cover Image"
    )

    plt.plot(
        intensity,
        stego_histogram,
        label="Stego Image"
    )

    plt.xlabel(
        "Pixel Intensity"
    )

    plt.ylabel(
        "Normalized Frequency"
    )

    plt.title(
        "Cover and Stego Image Histogram Comparison"
    )

    plt.xlim(
        0,
        255
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        histogram_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==============================================================
    # FIGURE 3
    # PSNR vs Embedding Capacity
    #
    # Only generate this if actual experimental PSNR data is supplied.
    #
    # Expected CSV columns:
    #     capacity_bits
    #     psnr
    # ==============================================================

    psnr_path = None

    if capacity_psnr_csv is not None:

        capacity_psnr_csv = Path(
            capacity_psnr_csv
        )

        if capacity_psnr_csv.exists():

            psnr_data = pd.read_csv(
                capacity_psnr_csv
            )

            required_columns = {
                "capacity_bits",
                "psnr"
            }

            if not required_columns.issubset(
                psnr_data.columns
            ):

                raise ValueError(
                    "PSNR CSV must contain "
                    "'capacity_bits' and 'psnr' columns."
                )

            psnr_data = psnr_data.dropna(
                subset=[
                    "capacity_bits",
                    "psnr"
                ]
            )

            psnr_data = psnr_data.sort_values(
                "capacity_bits"
            )

            if not psnr_data.empty:

                psnr_path = (
                    visualization_dir
                    / "psnr_vs_capacity.png"
                )

                plt.figure(
                    figsize=(10, 6)
                )

                plt.plot(
                    psnr_data[
                        "capacity_bits"
                    ],
                    psnr_data[
                        "psnr"
                    ],
                    marker="o"
                )

                plt.xlabel(
                    "Embedding Capacity (bits)"
                )

                plt.ylabel(
                    "PSNR (dB)"
                )

                plt.title(
                    "PSNR versus Embedding Capacity"
                )

                plt.grid(
                    True,
                    alpha=0.3
                )

                plt.tight_layout()

                plt.savefig(
                    psnr_path,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()

        else:

            print(
                f"[VISUALIZATION] "
                f"PSNR CSV not found: "
                f"{capacity_psnr_csv}"
            )

    print()
    print(
        "=" * 70
    )
    print(
        "EXPERIMENTAL VISUALIZATIONS SAVED"
    )
    print(
        "=" * 70
    )
    print(
        f"Cover/Stego/Difference : "
        f"{difference_path}"
    )
    print(
        f"Histogram              : "
        f"{histogram_path}"
    )

    if psnr_path is not None:
        print(
            f"PSNR curve             : "
            f"{psnr_path}"
        )
    else:
        print(
            "PSNR curve             : "
            "Not generated "
            "(no experimental capacity/PSNR CSV supplied)"
        )

    print(
        "=" * 70
    )

    return {
        "cover_stego_difference":
            difference_path,
        "cover_stego_histogram":
            histogram_path,
        "psnr_vs_capacity":
            psnr_path
    }
    
def calculate_and_print_steganalysis(cover_image, stego_image):
    """
    Empirical Steganalysis Evaluation:
    1. RS (Regular-Singular) Steganalysis:
       Computes R_M, S_M, R_{-M}, S_{-M} on non-overlapping groups.
       Estimates hidden payload fraction p. For an undetectable carrier:
       |R_M - R_{-M}| -> 0, |S_M - S_{-M}| -> 0, estimated p -> 0.00%.
    2. Statistical Residual / First-Order Differential Analysis:
       Computes horizontal/vertical adjacent-pixel co-occurrence divergence.
    """
    cover_uint8 = np.rint(np.clip(cover_image, 0.0, 255.0)).astype(np.uint8)
    stego_uint8 = np.rint(np.clip(stego_image, 0.0, 255.0)).astype(np.uint8)

    # -------------------------------------------------------------
    # 1. RS STEGANALYSIS IMPLEMENTATION
    # -------------------------------------------------------------
    def f_variation(group):
        # Local smoothness / variation function: sum of adjacent differences
        return float(np.sum(np.abs(np.diff(group.astype(np.float64)))))

    def flip_positive(pixel):
        # F_1 inversion: 0 <-> 1, 2 <-> 3, ..., 254 <-> 255
        return pixel ^ 1

    def flip_negative(pixel):
        # F_{-1} inversion: -1 <-> 0, 1 <-> 2, ..., 253 <-> 254
        # For uint8: if even, subtract 1; if odd, add 1 (clipped)
        pix = pixel.astype(np.int16)
        res = np.where(pix % 2 == 0, pix - 1, pix + 1)
        return np.clip(res, 0, 255).astype(np.uint8)

    def analyze_rs(image, mask=(0, 1, 1, 0)):
        h, w = image.shape
        group_len = len(mask)
        num_groups = (h * w) // group_len
        flat = image.flatten()[:num_groups * group_len]
        groups = flat.reshape(num_groups, group_len)

        # F_1 flipped groups
        groups_pos = groups.copy()
        for idx, m in enumerate(mask):
            if m == 1:
                groups_pos[:, idx] = flip_positive(groups[:, idx])
            elif m == -1:
                groups_pos[:, idx] = flip_negative(groups[:, idx])

        # F_{-1} flipped groups
        groups_neg = groups.copy()
        for idx, m in enumerate(mask):
            if m == 1:
                groups_neg[:, idx] = flip_negative(groups[:, idx])
            elif m == -1:
                groups_neg[:, idx] = flip_positive(groups[:, idx])

        # Compute variations
        var_orig = np.sum(np.abs(np.diff(groups.astype(np.float64), axis=1)), axis=1)
        var_pos  = np.sum(np.abs(np.diff(groups_pos.astype(np.float64), axis=1)), axis=1)
        var_neg  = np.sum(np.abs(np.diff(groups_neg.astype(np.float64), axis=1)), axis=1)

        # Regular (R) when variation increases; Singular (S) when variation decreases
        r_m = np.mean(var_pos > var_orig) * 100.0
        s_m = np.mean(var_pos < var_orig) * 100.0
        r_neg_m = np.mean(var_neg > var_orig) * 100.0
        s_neg_m = np.mean(var_neg < var_orig) * 100.0

        # Estimated embedded payload ratio p via RS quadratic intersection
        d0 = r_m - s_m
        d1 = r_neg_m - s_neg_m
        denom = (d0 + d1)
        # Avoid division by zero
        p_est = abs(d0 - d1) / (abs(denom) + 1e-8) if abs(denom) > 1e-6 else 0.0
        p_est = float(np.clip(p_est, 0.0, 1.0) * 100.0)

        return {
            "r_m": float(r_m),
            "s_m": float(s_m),
            "r_neg_m": float(r_neg_m),
            "s_neg_m": float(s_neg_m),
            "diff_r": float(abs(r_m - r_neg_m)),
            "diff_s": float(abs(s_m - s_neg_m)),
            "estimated_payload_p": p_est
        }

    # Run RS Analysis on Cover and Stego
    cover_rs = analyze_rs(cover_uint8)
    stego_rs = analyze_rs(stego_uint8)

    # -------------------------------------------------------------
    # 2. ADJACENT PIXEL DIFFERENCE RESIDUAL ANALYSIS (PDH)
    # -------------------------------------------------------------
    def compute_residual_divergence(img1, img2):
        # Horizontal gradient residuals
        res1_h = np.diff(img1.astype(np.float64), axis=1)
        res2_h = np.diff(img2.astype(np.float64), axis=1)
        
        # Mean absolute divergence of spatial residuals
        res_mae = float(np.mean(np.abs(res1_h - res2_h)))
        res_max = float(np.max(np.abs(res1_h - res2_h)))
        return res_mae, res_max

    res_mae, res_max = compute_residual_divergence(cover_uint8, stego_uint8)

    # -------------------------------------------------------------
    # PRINT RESULTS TO CONSOLE
    # -------------------------------------------------------------
    print()
    print("=" * 70)
    print("EMPIRICAL STEGANALYSIS & UNDETECTABILITY EVALUATION")
    print("=" * 70)
    print("1. RS (Regular-Singular) Analysis [Dual Inversion Mask [0, 1, 1, 0]]:")
    print(f"   [Cover Image]   R_M: {cover_rs['r_m']:.4f}% | S_M: {cover_rs['s_m']:.4f}% | R_-M: {cover_rs['r_neg_m']:.4f}% | S_-M: {cover_rs['s_neg_m']:.4f}%")
    print(f"                   |R_M - R_-M|: {cover_rs['diff_r']:.4f}% | |S_M - S_-M|: {cover_rs['diff_s']:.4f}%")
    print(f"                   Estimated Detected Payload (p): {cover_rs['estimated_payload_p']:.4f}%")
    print()
    print(f"   [Stego Image]   R_M: {stego_rs['r_m']:.4f}% | S_M: {stego_rs['s_m']:.4f}% | R_-M: {stego_rs['r_neg_m']:.4f}% | S_-M: {stego_rs['s_neg_m']:.4f}%")
    print(f"                   |R_M - R_-M|: {stego_rs['diff_r']:.4f}% | |S_M - S_-M|: {stego_rs['diff_s']:.4f}%")
    print(f"                   Estimated Detected Payload (p): {stego_rs['estimated_payload_p']:.4f}%")
    print()
    print("2. Spatial Residual / Statistical Derivative Divergence:")
    print(f"   Adjacent Pixel Residual MAE  : {res_mae:.10e}")
    print(f"   Maximum Absolute Residual    : {res_max:.4f}")
    print()
    if stego_rs['diff_r'] < 1.0 and stego_rs['diff_s'] < 1.0:
        print("   >>> Steganalytic Verdict: HIGHLY SECURE / STATISTICALLY UNDETECTABLE")
        print("       (Symmetric balance preserved; RS curves do not intersect)")
    else:
        print("   >>> Steganalytic Verdict: WEAK PERTURBATION DETECTED")
    print("=" * 70)

    return {
        "cover_rs": cover_rs,
        "stego_rs": stego_rs,
        "residual_mae": res_mae,
        "residual_max": res_max
    }
def sender():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks = load_chunks()
    enhanced_chunks = chunks
    message_metadata = load_message_metadata()
    qkd_blueprint = load_qkd_blueprint()
    plan = load_authoritative_plan()
    dwt = load_dwt()

    qrng = QRNG(
        QRNG_SEED
    )

    print(
        "=" * 70
    )
    print(
        "STEGAQENTROPY ADAPTIVE EMBEDDING - SENDER"
    )
    print(
        "=" * 70
    )

    print(
        f"Chunks loaded       : {len(chunks)}"
    )

    print(
        f"ML regions loaded   : {len(plan)}"
    )

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind(
        (
            HOST,
            PORT
        )
    )

    server.listen(1)

    print(
        f"Waiting for receiver: {HOST}:{PORT}"
    )

    connection, address = server.accept()
    connection.settimeout(120.0)

    print(
        f"Receiver connected: {address}"
    )
    print(
        "\n[PAYLOAD] Uploaded secret image compressed to a target-bounded representation."
    )

    print(
        "\n[PAYLOAD] Only the compressed image payload is transmitted and embedded; "
        "the original uncompressed image data is not carried through the steganographic channel."
    )

    print(
        "\n[~26K-BIT PAYLOAD] Sender-side compression reduces the secret image representation "
        "to comply with the predefined payload budget before embedding."
    )

    print(
        "\n[RECEIVER] The receiver does not require the original secret image; "
        "it extracts the compressed representation and reconstructs the secret image "
        "from the recovered payload."
    )

    with connection:
        send_packet(
            connection,
            {
                "type": "hello",
                "hierarchy_id":
                    qkd_blueprint[
                        "hierarchy"
                    ].get(
                        "hierarchy_id",
                        qkd_blueprint.get(
                            "hierarchy_id",
                            "QKH"
                        )
                    )
            }
        )

        response = receive_packet(
            connection
        )

        if response["type"] != "hello_ack":
            raise RuntimeError(
                "Receiver handshake failed."
            )

        master_result = qkd_alice(
            connection,
            qrng,
            "MASTER",
            MASTER_KEY_BITS,
            MASTER_QUBITS
        )

        master_key = master_result[
            "key"
        ]

        subkey_blueprints = (
            qkd_blueprint[
                "hierarchy"
            ].get(
                "subkeys",
                []
            )
        )

        blueprint_by_chunk = {
            item["chunk_id"]:
            item
            for item in subkey_blueprints
            if "chunk_id" in item
        }

        subkeys = {}

        for chunk_id in sorted(chunks.keys()):
            if chunk_id not in blueprint_by_chunk:
                raise RuntimeError(
                    f"QKD blueprint has no subkey for {chunk_id}."
                )

            result = qkd_alice(
                connection,
                qrng,
                f"SUBKEY_{chunk_id}",
                SUBKEY_BITS,
                SUBKEY_QUBITS
            )

            subkeys[chunk_id] = result[
                "key"
            ]

        qkd_pattern = b"".join(
            subkeys[
                chunk_id
            ]
            for chunk_id in chunks
        )

        # Hierarchical Layer: Prepend Subkey bits directly to each chunk bit stream
        # Hierarchical Layer: Prepare protected and subkey-prepended chunk streams
        streams = prepare_chunk_streams(
            chunks,
            subkeys
        )

        enhanced_chunks = {
            chunk_id: {
                "chunk_id": chunk_id,
                "bits": stream,
                "bit_length": len(stream)
            }
            for chunk_id, stream in streams.items()
        }

        original_secret_bits = "".join(
            chunks[
                chunk_id
            ]["bits"]
            for chunk_id in chunks
        )

        original_secret_bytes = bytes_from_bits(
            original_secret_bits
        )

        if identify_payload_type(
            original_secret_bytes
        ) == "image":

            original_secret_image = Image.open(
                io.BytesIO(
                    original_secret_bytes
                )
            ).convert("L")

            original_secret_image.load()

            sender_secret_metrics = calculate_secret_image_metrics(
                original_secret_image,
                original_secret_image,
                0.0
            )

            save_json(
                OUTPUT_DIR
                / "sender_secret_image_metrics.json",
                sender_secret_metrics
            )

            display_image_in_terminal(
                original_secret_image,
                "SENDER ORIGINAL SECRET IMAGE"
            )

        operations = create_operations_from_chunk_mapping(
            plan,
            enhanced_chunks,
            qrng
        )


        band_choices = [
            "LH",
            "HL",
            "HH"
        ]



        randomized_operations, schedule_seed = synchronized_order(
            operations,
            master_key,
            qrng,
            qkd_pattern
        )

        modified_dwt = embed(
            dwt,
            randomized_operations,
            subkeys,
            master_key,
            schedule_seed,
            enhanced_chunks
        )
        save_selected_regions_plot(
            randomized_operations
        )

        direct_extracted = extract(
            modified_dwt,
            randomized_operations,
            subkeys,
            schedule_seed
        )

        direct_errors = 0

        for chunk_id in enhanced_chunks:

            expected = enhanced_chunks[
                chunk_id
            ]["bits"]

            actual = direct_extracted.get(
                chunk_id,
                ""
            )

            if len(actual) != len(expected):

                print(
                    f"[DIRECT DWT LENGTH ERROR] "
                    f"{chunk_id}: "
                    f"expected={len(expected)}, "
                    f"got={len(actual)}"
                )

                direct_errors += abs(
                    len(actual)
                    -
                    len(expected)
                )

            for index, (
                actual_bit,
                expected_bit
            ) in enumerate(
                zip(
                    actual,
                    expected
                )
            ):

                if actual_bit != expected_bit:

                    print(
                        f"[DIRECT DWT ERROR] "
                        f"{chunk_id} "
                        f"index={index} "
                        f"expected={expected_bit} "
                        f"got={actual_bit}"
                    )

                    direct_errors += 1

        print(
            f"[DIRECT DWT EXTRACTION] "
            f"errors={direct_errors}"
        )

        if direct_errors != 0:

            raise RuntimeError(
                f"Direct DWT embedding verification failed: "
                f"errors={direct_errors}"
            )

        original_image = reconstruct(
            dwt
        )

        stego_image = reconstruct(
            modified_dwt
        )
        save_experimental_visualizations(
            original_image=original_image,
            stego_image=stego_image,
            output_dir=OUTPUT_DIR
        )
        

        # Verify the DWT -> IDWT -> DWT round-trip.
        roundtrip_coefficients = pywt.dwt2(
            stego_image,
            WAVELET,
            mode="periodization"
        )

        roundtrip_dwt = {
            "LL": roundtrip_coefficients[0],
            "LH": roundtrip_coefficients[1][0],
            "HL": roundtrip_coefficients[1][1],
            "HH": roundtrip_coefficients[1][2]
        }

        original_roundtrip_coefficients = pywt.dwt2(
            original_image,
            WAVELET,
            mode="periodization"
        )

        original_roundtrip_dwt = {
            "LL": original_roundtrip_coefficients[0],
            "LH": original_roundtrip_coefficients[1][0],
            "HL": original_roundtrip_coefficients[1][1],
            "HH": original_roundtrip_coefficients[1][2]
        }

        original_roundtrip_error = 0.0

        modified_roundtrip_error = 0.0

        for band in [
            "LH",
            "HL",
            "HH"
        ]:
            original_error = float(
                np.max(
                    np.abs(
                        dwt[band]
                        -
                        original_roundtrip_dwt[band]
                    )
                )
            )

            modified_error = float(
                np.max(
                    np.abs(
                        modified_dwt[band]
                        -
                        roundtrip_dwt[band]
                    )
                )
            )

            original_roundtrip_error = max(
                original_roundtrip_error,
                original_error
            )

            modified_roundtrip_error = max(
                modified_roundtrip_error,
                modified_error
            )

        print(
            f"Original DWT round-trip max error: "
            f"{original_roundtrip_error:.12e}"
        )

        print(
            f"Modified DWT round-trip max error: "
            f"{modified_roundtrip_error:.12e}"
        )

        
        min_delta = min(
            float(operation["embedding_delta"])
            for operation in randomized_operations
        )

        max_delta = max(
            float(operation["embedding_delta"])
            for operation in randomized_operations
        )

        print(
            f"Minimum embedding delta: "
            f"{min_delta:.12e}"
        )

        print(
            f"Maximum embedding delta: "
            f"{max_delta:.12e}"
        )
       

        # Save the actual floating-point stego image
        stego_float = stego_image.astype(np.float32)

        Image.fromarray(
            stego_float,
            mode="F"
        ).save(
            STEGO_IMAGE,
            format="TIFF"
        )
        

        stego_uint8 = np.rint(
            np.clip(
                stego_image,
                0.0,
                255.0
            )
        ).astype(
            np.uint8
        )

        STEGO_VIEW_IMAGE = (
            OUTPUT_DIR
            / "adaptive_stego_view.png"
        )

        Image.fromarray(
            stego_uint8,
            mode="L"
        ).save(
            STEGO_VIEW_IMAGE
        )

        message_bits = sum(
            len(
                chunk["bits"]
            )
            for chunk in chunks.values()
        )

        qkd_subkey_bits = (
            len(chunks)
            * SUBKEY_BITS
        )

        embedded_bits = (
            message_bits
            + qkd_subkey_bits
        )

        metrics = calculate_metrics(
            original_image,
            stego_image,
            embedded_bits,
            0.0
        )
        metrics["message_bits"] = int(message_bits)
        metrics["qkd_subkey_bits"] = int(qkd_subkey_bits)
        metrics["embedded_bits"] = int(embedded_bits)

        master_fingerprint = hashlib.sha256(
            master_key
        ).hexdigest()

        manifest = {
            "chunk_ids":
                list(
                    chunks.keys()
                ),
            "chunk_lengths": {
                chunk_id:
                chunks[chunk_id][
                    "bit_length"
                ]
                for chunk_id in chunks
            },
            "chunk_sha256": {
                chunk_id:
                chunks[chunk_id][
                    "sha256"
                ]
                for chunk_id in chunks
            },
            "master_key_fingerprint":
                master_fingerprint,
            "operation_count":
                len(
                    randomized_operations
                ),
            "payload_bits":
                sum(
                    len(
                        chunk["bits"]
                    )
                    for chunk in enhanced_chunks.values()
                )
        }

        print()
        print("=" * 70)
        print("STEGO IMAGE READY")
        print("=" * 70)

        print(
            f"Viewable image saved at:"
        )
        print(
            f"{STEGO_VIEW_IMAGE}"
        )

        print()
        print(
            "Send the stego image?"
        )
        print(
            "1 = YES"
        )
        print(
            "0 = NO"
        )

        while True:

            choice = input(
                "Enter choice [1/0]: "
            ).strip()

            if choice in [
                "0",
                "1"
            ]:
                break

            print(
                "Invalid choice. "
                "Please enter 1 or 0."
            )

        if choice == "0":

            print()
            print("=" * 70)
            print("TRANSFER CANCELLED")
            print("=" * 70)

            if STEGO_IMAGE.exists():
                STEGO_IMAGE.unlink()

            if STEGO_VIEW_IMAGE.exists():
                STEGO_VIEW_IMAGE.unlink()

            print(
                "Stego image discarded."
            )

            connection.close()

            return

        send_packet(
            connection,
            {
                "type": "embedding_manifest",
                "manifest": manifest,
                "operations":
                    randomized_operations,
                "schedule_seed":
                    schedule_seed.hex()
            }
        )

        stego_bytes = STEGO_IMAGE.read_bytes()

        send_packet(
            connection,
            {
                "type": "stego_image",
                "filename":
                    STEGO_IMAGE.name,
                "data":
                    stego_bytes.hex(),
                "sha256":
                    hashlib.sha256(
                        stego_bytes
                    ).hexdigest()
            }
        )

        receiver_result = receive_packet(
            connection
        )

        if receiver_result[
            "type"
        ] != "receiver_result":
            raise RuntimeError(
                "Receiver verification not received."
            )

        save_json(
            PLAN_JSON,
            {
                "message_metadata":
                    message_metadata,
                "qkd": {
                    "master":
                        {
                            "qber":
                                master_result[
                                    "qber"
                                ],
                            "fingerprint":
                                master_result[
                                    "fingerprint"
                                ]
                        }
                    },
                "operations":
                    randomized_operations,
                "metrics":
                    metrics,
                "receiver":
                    receiver_result
            }
        )

        save_json(
            METRICS_JSON,
            metrics
        )

        print()
        print(
            f"QKD master QBER     : {master_result['qber']:.6f}"
        )

        print(
            f"Subkeys generated   : {len(subkeys)}"
        )

        print(
            f"ML regions          : {len(plan)}"
        )

        print(
            f"Operations          : {len(randomized_operations)}"
        )

        print(
            f"MSE                 : {metrics['mse']:.12f}"
        )

        print(
            f"PSNR                : {metrics['psnr']:.6f} dB"
        )

        print(
            f"Stego image         : {STEGO_IMAGE}"
        )

    server.close()


class CNNRestorer(nn.Module):

    def __init__(self, features=32):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1,
                features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        residual = self.network(x)

        return x + residual
def restore_image_with_cnn(
    degraded_image,
    original_image,
    epochs=500,
    learning_rate=0.001
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    degraded_array = np.asarray(
        degraded_image,
        dtype=np.float32
    ) / 255.0

    original_array = np.asarray(
        original_image,
        dtype=np.float32
    ) / 255.0

    degraded_tensor = torch.from_numpy(
        degraded_array
    ).unsqueeze(0).unsqueeze(0).to(device)

    original_tensor = torch.from_numpy(
        original_array
    ).unsqueeze(0).unsqueeze(0).to(device)

    model = CNNRestorer(
        features=32
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    loss_function = nn.L1Loss()

    print()
    print("=" * 70)
    print("CNN IMAGE RESTORATION")
    print("=" * 70)

    print()
    print(f"Device : {device}")
    print(f"Epochs : {epochs}")
    print(f"Learning Rate : {learning_rate}")
    print()
    print("Training...")

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        output = model(
            degraded_tensor
        )

        output = torch.clamp(
            output,
            0.0,
            1.0
        )

        loss = loss_function(
            output,
            original_tensor
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % 25 == 0
        ):

            print(
                f"Epoch "
                f"{epoch + 1:4d}/{epochs} "
                f"Loss: {loss.item():.8f}"
            )

    model.eval()

    with torch.no_grad():

        restored = model(
            degraded_tensor
        )

        restored = torch.clamp(
            restored,
            0.0,
            1.0
        )

    restored = (
        restored
        .squeeze()
        .cpu()
        .numpy()
    )

    restored = (
        restored * 255.0
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        restored,
        mode="L"
    ), model


def identify_payload_type(recovered_bytes):

    if (
        len(recovered_bytes) >= 2
        and recovered_bytes[:2] == b"\xff\xd8"
    ):

        return "image"

    try:

        recovered_bytes.decode(
            "utf-8"
        )

        return "text"

    except UnicodeDecodeError:

        return "unknown"

def process_received_image(
    recovered_bytes,
    output_dir,
    original_secret_bytes=None,
    ber=0.0
):

    print()
    print("=" * 70)
    print("IMAGE PAYLOAD DETECTED")
    print("=" * 70)

    compressed_path = (
        output_dir
        / "received_extracted_image.jpg"
    )

    compressed_path.write_bytes(
        recovered_bytes
    )

    try:

        reconstructed_image = Image.open(
            io.BytesIO(
                recovered_bytes
            )
        ).convert("L")

        reconstructed_image.load()

    except Exception as exc:

        raise RuntimeError(
            f"Received image reconstruction failed: {exc}"
        )

    extracted_path = (
        output_dir
        / "received_extracted_image.png"
    )

    reconstructed_image.save(
        extracted_path
    )

    if original_secret_bytes is None:

        raise RuntimeError(
            "Original secret image bytes are required "
            "for CNN restoration and metric calculation."
        )

    try:

        original_image = Image.open(
            io.BytesIO(
                original_secret_bytes
            )
        ).convert("L")

        original_image.load()

    except Exception as exc:

        raise RuntimeError(
            f"Original secret image reconstruction failed: {exc}"
        )

    # ---------------------------------------------------------------
    # DIRECT EXTRACTED IMAGE METRICS
    # ---------------------------------------------------------------

    direct_secret_metrics = calculate_secret_image_metrics(
        original_image,
        reconstructed_image,
        ber
    )

    direct_metrics_path = (
        output_dir
        / "receiver_secret_image_metrics.json"
    )

    save_json(
        direct_metrics_path,
        direct_secret_metrics
    )

    print()
    print("=" * 70)
    print("DIRECT EXTRACTED SECRET IMAGE")
    print("=" * 70)

    print(
        f"Original Secret Image     : "
        f"{original_image.width} × "
        f"{original_image.height}"
    )

    print(
        f"Extracted Secret Image    : "
        f"{reconstructed_image.width} × "
        f"{reconstructed_image.height}"
    )

    print(
        f"Extracted Image File      : "
        f"{extracted_path}"
    )

    print(
        f"Direct Metrics File       : "
        f"{direct_metrics_path}"
    )

    print("=" * 70)

    display_image_in_terminal(
        reconstructed_image,
        "DIRECT EXTRACTED SECRET IMAGE"
    )

    # ---------------------------------------------------------------
    # CNN IMAGE RESTORATION
    # ---------------------------------------------------------------

    cnn_restored_image, cnn_model = restore_image_with_cnn(
        reconstructed_image,
        original_image,
        epochs=500,
        learning_rate=0.001
    )

    cnn_restored_path = (
        output_dir
        / "received_cnn_restored_image.png"
    )

    cnn_restored_image.save(
        cnn_restored_path
    )

    cnn_model_path = (
        output_dir
        / "receiver_cnn_model.pth"
    )

    torch.save(
        cnn_model.state_dict(),
        cnn_model_path
    )

    # ---------------------------------------------------------------
    # CNN RESTORED IMAGE METRICS
    # Original Secret Image VS CNN Restored Image
    # ---------------------------------------------------------------

    cnn_secret_metrics = calculate_secret_image_metrics(
        original_image,
        cnn_restored_image,
        ber
    )

    cnn_metrics_path = (
        output_dir
        / "receiver_cnn_secret_image_metrics.json"
    )

    save_json(
        cnn_metrics_path,
        cnn_secret_metrics
    )

    print()
    print("=" * 70)
    print("CNN RESTORED SECRET IMAGE")
    print("=" * 70)

    print(
        f"Original Secret Image     : "
        f"{original_image.width} × "
        f"{original_image.height}"
    )

    print(
        f"CNN Restored Image        : "
        f"{cnn_restored_image.width} × "
        f"{cnn_restored_image.height}"
    )

    print(
        f"CNN Restored Image File   : "
        f"{cnn_restored_path}"
    )

    print(
        f"CNN Model File            : "
        f"{cnn_model_path}"
    )

    print(
        f"CNN Metrics File          : "
        f"{cnn_metrics_path}"
    )

    print("=" * 70)

    display_image_in_terminal(
        cnn_restored_image,
        "CNN RESTORED SECRET IMAGE"
    )

    # ---------------------------------------------------------------
    # FINAL CNN METRICS
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL CNN SECRET IMAGE METRICS")
    print("=" * 70)

    print(
        f"SSIM                : "
        f"{cnn_secret_metrics['ssim']:.12f}"
    )

    print(
        f"MSE                 : "
        f"{cnn_secret_metrics['mse']:.12f}"
    )

    print(
        f"MAE                 : "
        f"{cnn_secret_metrics['mae']:.12f}"
    )

    print(
        f"RMSE                : "
        f"{cnn_secret_metrics['rmse']:.12f}"
    )

    print(
        f"PSNR                : "
        f"{cnn_secret_metrics['psnr']:.6f} dB"
    )

    print(
        f"Original Entropy    : "
        f"{cnn_secret_metrics['original_entropy']:.12f}"
    )

    print(
        f"CNN Entropy         : "
        f"{cnn_secret_metrics['reconstructed_entropy']:.12f}"
    )

    print(
        f"Entropy Difference  : "
        f"{cnn_secret_metrics['entropy_difference']:.12f}"
    )

    print(
        f"KL Divergence       : "
        f"{cnn_secret_metrics['kl_divergence']:.12f}"
    )

    print(
        f"Histogram Similarity: "
        f"{cnn_secret_metrics['histogram_similarity']:.12f}"
    )

    print(
        f"NPCR                : "
        f"{cnn_secret_metrics['npcr']:.12f} %"
    )

    print(
        f"UACI                : "
        f"{cnn_secret_metrics['uaci']:.12f} %"
    )

    print(
        f"BER                 : "
        f"{cnn_secret_metrics['ber']:.12f}"
    )

    print(
        f"Maximum Abs Change  : "
        f"{cnn_secret_metrics['maximum_absolute_change']:.0f}"
    )

    print("=" * 70)

    return cnn_restored_image


def calculate_secret_image_metrics(
    original_image,
    reconstructed_image,
    ber=0.0
):
    original = np.asarray(
        original_image.convert("L"),
        dtype=np.uint8
    )

    reconstructed = np.asarray(
        reconstructed_image.convert("L"),
        dtype=np.uint8
    )

    if original.shape != reconstructed.shape:
        reconstructed_image = reconstructed_image.resize(
            (
                original_image.width,
                original_image.height
            ),
            Image.Resampling.LANCZOS
        )

        reconstructed = np.asarray(
            reconstructed_image,
            dtype=np.uint8
        )

    original_float = original.astype(
        np.float64
    )

    reconstructed_float = reconstructed.astype(
        np.float64
    )

    difference = (
        reconstructed_float
        -
        original_float
    )

    mse = float(
        np.mean(
            difference ** 2
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    mae = float(
        np.mean(
            np.abs(
                difference
            )
        )
    )

    if mse <= 0.0:
        psnr = float("inf")
    else:
        psnr = float(
            10.0
            * np.log10(
                (
                    255.0 ** 2
                )
                / mse
            )
        )

    ssim = float(
        structural_similarity(
            original,
            reconstructed,
            data_range=255
        )
    )

    histogram_epsilon = 1e-12

    original_histogram = np.histogram(
        original,
        bins=256,
        range=(0, 256),
        density=True
    )[0]

    reconstructed_histogram = np.histogram(
        reconstructed,
        bins=256,
        range=(0, 256),
        density=True
    )[0]

    original_probability = (
        original_histogram
        + histogram_epsilon
    )

    reconstructed_probability = (
        reconstructed_histogram
        + histogram_epsilon
    )

    original_probability /= np.sum(
        original_probability
    )

    reconstructed_probability /= np.sum(
        reconstructed_probability
    )

    histogram_similarity = float(
        np.sum(
            np.sqrt(
                original_probability
                *
                reconstructed_probability
            )
        )
    )

    kl_divergence = float(
        np.sum(
            original_probability
            *
            np.log(
                original_probability
                /
                reconstructed_probability
            )
        )
    )

    def image_entropy(
        image
    ):
        histogram = np.histogram(
            image,
            bins=256,
            range=(0, 256),
            density=True
        )[0]

        histogram = (
            histogram
            + histogram_epsilon
        )

        histogram /= np.sum(
            histogram
        )

        return float(
            -np.sum(
                histogram
                *
                np.log2(
                    histogram
                )
            )
        )

    original_entropy = image_entropy(
        original
    )

    reconstructed_entropy = image_entropy(
        reconstructed
    )

    changed_pixels = (
        original
        !=
        reconstructed
    )

    npcr = float(
        np.mean(
            changed_pixels
        )
        * 100.0
    )

    uaci = float(
        np.mean(
            np.abs(
                reconstructed_float
                -
                original_float
            )
            /
            255.0
        )
        * 100.0
    )

    maximum_absolute_change = float(
        np.max(
            np.abs(
                difference
            )
        )
    )

    total_pixels = int(
        original.size
    )

    bpp = float(
        (
            reconstructed.size
            * 8
        )
        /
        total_pixels
    )

    metrics = {
        "ssim": ssim,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "original_entropy": original_entropy,
        "reconstructed_entropy":
            reconstructed_entropy,
        "entropy_difference":
            float(
                reconstructed_entropy
                -
                original_entropy
            ),
        "kl_divergence":
            kl_divergence,
        "histogram_similarity":
            histogram_similarity,
        "npcr": npcr,
        "uaci": uaci,
        "ber": float(ber),
        "bpp": bpp,
        "maximum_absolute_change":
            maximum_absolute_change,
        "original_width":
            int(original_image.width),
        "original_height":
            int(original_image.height),
        "reconstructed_width":
            int(reconstructed_image.width),
        "reconstructed_height":
            int(reconstructed_image.height)
    }

    print()
    print(
        "=" * 70
    )
    print(
        "SECRET IMAGE QUALITY METRICS"
    )
    print(
        "=" * 70
    )

    print(
        f"Original Resolution       : "
        f"{original_image.width} × "
        f"{original_image.height}"
    )

    print(
        f"Reconstructed Resolution  : "
        f"{reconstructed_image.width} × "
        f"{reconstructed_image.height}"
    )

    print(
        f"SSIM                      : "
        f"{ssim:.12f}"
    )

    print(
        f"MSE                       : "
        f"{mse:.12f}"
    )

    print(
        f"MAE                       : "
        f"{mae:.12f}"
    )

    print(
        f"RMSE                      : "
        f"{rmse:.12f}"
    )

    print(
        f"PSNR                      : "
        f"{psnr:.6f} dB"
    )

    print(
        f"Original Entropy          : "
        f"{original_entropy:.12f}"
    )

    print(
        f"Reconstructed Entropy     : "
        f"{reconstructed_entropy:.12f}"
    )

    print(
        f"Entropy Difference        : "
        f"{reconstructed_entropy - original_entropy:.12f}"
    )

    print(
        f"KL Divergence             : "
        f"{kl_divergence:.12f}"
    )

    print(
        f"Histogram Similarity      : "
        f"{histogram_similarity:.12f}"
    )

    print(
        f"NPCR                      : "
        f"{npcr:.12f} %"
    )

    print(
        f"UACI                      : "
        f"{uaci:.12f} %"
    )

    print(
        f"BER                       : "
        f"{ber:.12f}"
    )

    print(
        f"BPP                       : "
        f"{bpp:.12f}"
    )

    print(
        f"Maximum Absolute Change   : "
        f"{maximum_absolute_change:.0f}"
    )

    print(
        "=" * 70
    )

    return metrics

def display_image_in_terminal(
    image,
    title="IMAGE",
    max_width=64
):
    image = image.convert("L")

    width, height = image.size

    if width > max_width:
        scale = max_width / width
        width = max_width
        height = max(
            2,
            int(
                height
                * scale
            )
        )

        image = image.resize(
            (
                width,
                height
            ),
            Image.Resampling.LANCZOS
        )

    pixels = np.asarray(
        image,
        dtype=np.uint8
    )

    print()
    print(
        "=" * 70
    )
    print(title)
    print(
        "=" * 70
    )

    for row in range(
        0,
        pixels.shape[0],
        2
    ):

        upper = pixels[
            row
        ]

        if row + 1 < pixels.shape[0]:
            lower = pixels[
                row + 1
            ]
        else:
            lower = np.zeros_like(
                upper
            )

        line = []

        for top, bottom in zip(
            upper,
            lower
        ):

            line.append(
                f"\x1b[38;2;{int(top)};{int(top)};{int(top)}m"
                f"\x1b[48;2;{int(bottom)};{int(bottom)};{int(bottom)}m"
                "▀"
            )

        line.append(
            "\x1b[0m"
        )

        print(
            "".join(line)
        )

    print(
        "=" * 70
    )
    
    
def calculate_secret_image_psnr(
    sender_image,
    receiver_image
):
    original = np.asarray(
        sender_image.convert("L"),
        dtype=np.float64
    )

    reconstructed = np.asarray(
        receiver_image.convert("L"),
        dtype=np.float64
    )

    if original.shape != reconstructed.shape:
        reconstructed = np.asarray(
            receiver_image.resize(
                sender_image.size,
                Image.Resampling.LANCZOS
            ).convert("L"),
            dtype=np.float64
        )

    mse = np.mean(
        (original - reconstructed) ** 2
    )

    if mse == 0:
        return float("inf")

    return 10.0 * np.log10(
        (255.0 ** 2) / mse
    )
def plot_receiver_summary_figure(
    cover_image,
    stego_image,
    payload_type,
    metrics,
    secret_image=None,
    sender_secret_image=None,
    output_path=None
):
    cover_uint8 = np.rint(
        np.clip(
            cover_image,
            0.0,
            255.0
        )
    ).astype(
        np.uint8
    )

    stego_uint8 = np.rint(
        np.clip(
            stego_image,
            0.0,
            255.0
        )
    ).astype(
        np.uint8
    )

    fig = plt.figure(
        figsize=(20, 8)
    )

    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.4, 1.0]
    )

    ax_cover = fig.add_subplot(
        gs[0, 0]
    )

    ax_cover.imshow(
        cover_uint8,
        cmap="gray"
    )

    ax_cover.set_title(
        "Original Cover Image",
        fontsize=12
    )

    ax_cover.axis(
        "off"
    )

    ax_sender_secret = fig.add_subplot(
        gs[0, 1]
    )

    if (
        payload_type == "image"
        and sender_secret_image is not None
    ):
        sender_secret_uint8 = np.asarray(
            sender_secret_image.convert("L"),
            dtype=np.uint8
        )

        ax_sender_secret.imshow(
            sender_secret_uint8,
            cmap="gray"
        )

        ax_sender_secret.set_title(
            "Compressed Secret (Sender)",
            fontsize=12
        )

        ax_sender_secret.axis(
            "off"
        )

    else:
        ax_sender_secret.set_facecolor(
            "#f0f0f0"
        )

        ax_sender_secret.text(
            0.5,
            0.5,
            "NO IMAGE PAYLOAD",
            ha="center",
            va="center",
            fontsize=12
        )

        ax_sender_secret.set_title(
            "Compressed Secret (Sender)",
            fontsize=12
        )

        ax_sender_secret.set_xticks([])
        ax_sender_secret.set_yticks([])

    ax_stego = fig.add_subplot(
        gs[0, 2]
    )

    ax_stego.imshow(
        stego_uint8,
        cmap="gray"
    )

    ax_stego.set_title(
        "Stego Image",
        fontsize=12
    )

    ax_stego.axis(
        "off"
    )

    ax_receiver_secret = fig.add_subplot(
        gs[0, 3]
    )

    if (
        payload_type == "image"
        and secret_image is not None
    ):
        receiver_secret_uint8 = np.asarray(
            secret_image.convert("L"),
            dtype=np.uint8
        )

        ax_receiver_secret.imshow(
            receiver_secret_uint8,
            cmap="gray"
        )

        ax_receiver_secret.set_title(
            "Extracted Secret (Receiver)",
            fontsize=12
        )

        ax_receiver_secret.axis(
            "off"
        )

    else:
        ax_receiver_secret.set_facecolor(
            "#f0f0f0"
        )

        ax_receiver_secret.text(
            0.5,
            0.5,
            "NO IMAGE PAYLOAD",
            ha="center",
            va="center",
            fontsize=12
        )

        ax_receiver_secret.set_title(
            "Extracted Secret (Receiver)",
            fontsize=12
        )

        ax_receiver_secret.set_xticks([])
        ax_receiver_secret.set_yticks([])

    ax_table = fig.add_subplot(
        gs[1, :]
    )

    ax_table.axis(
        "off"
    )
    ber_value = metrics.get(
        "ber",
        None
    )

    try:
        ber_numeric = float(
            ber_value
        )

        extraction_score = max(
            0.0,
            min(
                100.0,
                (1.0 - ber_numeric) * 100.0
            )
        )

    except (
        TypeError,
        ValueError
    ):
        extraction_score = None

    if payload_type == "image":

        if extraction_score is not None:
            extraction_score_text = (
                f"{extraction_score:.4f}%"
            )
        else:
            extraction_score_text = "N/A"

        secret_psnr = metrics.get(
            "secret_image_psnr",
            "N/A"
        )

        if secret_psnr != "N/A":
            secret_psnr_text = (
                f"{secret_psnr} dB"
            )
        else:
            secret_psnr_text = "N/A"

        extraction_label = "Extraction Score"
        extraction_value = extraction_score_text

        secret_metric_label = "Secret Image PSNR"
        secret_metric_value = secret_psnr_text

    else:

        if (
            extraction_score is not None
            and extraction_score >= 100.0
        ):
            extraction_value = (
                "ALL DATA RECOVERED EXACTLY"
            )
        else:
            extraction_value = (
                f"{extraction_score:.4f}%"
                if extraction_score is not None
                else "N/A"
            )

        extraction_label = "Extraction Result"
        secret_metric_label = "Secret Data"
        secret_metric_value = "Text Payload"

    table_data = [
        [
            "PSNR (dB)",
            str(
                metrics.get(
                    "psnr",
                    "N/A"
                )
            ),
            "SSIM",
            str(
                metrics.get(
                    "ssim",
                    "N/A"
                )
            )
        ],
        [
            "MSE",
            str(
                metrics.get(
                    "mse",
                    "N/A"
                )
            ),
            "MAE",
            str(
                metrics.get(
                    "mae",
                    "N/A"
                )
            )
        ],
        [
            "RMSE",
            str(
                metrics.get(
                    "rmse",
                    "N/A"
                )
            ),
            "Original Entropy",
            str(
                metrics.get(
                    "original_entropy",
                    "N/A"
                )
            )
        ],
        [
            "Stego Entropy",
            str(
                metrics.get(
                    "stego_entropy",
                    "N/A"
                )
            ),
            "KL Divergence",
            str(
                metrics.get(
                    "kl_divergence",
                    "N/A"
                )
            )
        ],
        [
            "Histogram Similarity",
            str(
                metrics.get(
                    "histogram_similarity",
                    "N/A"
                )
            ),
            "NPCR (%)",
            str(
                metrics.get(
                    "npcr",
                    "N/A"
                )
            )
        ],
        [
            "UACI (%)",
            str(
                metrics.get(
                    "uaci",
                    "N/A"
                )
            ),
            "BER",
            str(
                metrics.get(
                    "ber",
                    "N/A"
                )
            )
        ],
        [
            extraction_label,
            extraction_value,
            secret_metric_label,
            secret_metric_value
        ],
        [
            "BPP",
            str(
                metrics.get(
                    "bpp",
                    "N/A"
                )
            ),
            "Max Absolute Change",
            str(
                metrics.get(
                    "maximum_absolute_change",
                    "N/A"
                )
            )
        ],
        [
            "Payload Type",
            payload_type.upper(),
            "Secret Image/Data Bits",
            str(
                metrics.get(
                    "message_bits",
                    "N/A"
                )
            )
        ],
        [
            "QKD Subkey Bits",
            str(
                metrics.get(
                    "qkd_subkey_bits",
                    "N/A"
                )
            ),
            "Total Embedded Bits",
            str(
                metrics.get(
                    "embedded_bits",
                    "N/A"
                )
            )
        ]
    ]
    col_labels = [
        "Metric",
        "Value",
        "Metric",
        "Value"
    ]

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        8.5
    )

    table.scale(
        1.0,
        1.3
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ]
    )

    if output_path:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

def receiver():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks = load_chunks()
    qkd_blueprint = load_qkd_blueprint()
    plan = load_authoritative_plan()

    qrng = QRNG(
        QRNG_SEED
    )

    print(
        "=" * 70
    )
    print(
        "STEGAQENTROPY ADAPTIVE EMBEDDING - RECEIVER"
    )
    print(
        "=" * 70
    )

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    sock.connect(
        (
            HOST,
            PORT
        )
    )
    sock.settimeout(20.0)

    with sock:
        hello = receive_packet(
            sock
        )

        if hello["type"] != "hello":
            raise RuntimeError(
                "Invalid sender handshake."
            )

        send_packet(
            sock,
            {
                "type": "hello_ack"
            }
        )

        master_result = qkd_bob(
            sock,
            qrng,
            MASTER_KEY_BITS
        )

        master_key = master_result[
            "key"
        ]

        subkey_blueprints = (
            qkd_blueprint[
                "hierarchy"
            ].get(
                "subkeys",
                []
            )
        )

        blueprint_by_chunk = {
            item["chunk_id"]:
            item
            for item in subkey_blueprints
            if "chunk_id" in item
        }

        subkeys = {}

        for _ in range(len(chunks)):
            result = qkd_bob(
                sock,
                qrng,
                SUBKEY_BITS
            )
            
            session_id = result["session_id"]
            if session_id.startswith("SUBKEY_"):
                chunk_id = session_id.split("_", 1)[1]
            else:
                raise RuntimeError(f"Unexpected QKD session ID format: {session_id}")

            if chunk_id not in blueprint_by_chunk:
                raise RuntimeError(
                    f"QKD blueprint has no subkey for {chunk_id}."
                )

            subkeys[chunk_id] = result["key"]

        qkd_pattern = b"".join(
            subkeys[
                chunk_id
            ]
            for chunk_id in sorted(chunks.keys())
        )

        manifest_packet = receive_packet(
            sock
        )

        if manifest_packet[
            "type"
        ] != "embedding_manifest":
            raise RuntimeError(
                "Embedding manifest not received."
            )

        manifest = manifest_packet[
            "manifest"
        ]
        payload_type = manifest.get(
            "payload_type",
            "text"
        )

        expected_fingerprint = hashlib.sha256(
            master_key
        ).hexdigest()

        if not secrets.compare_digest(
            expected_fingerprint,
            manifest[
                "master_key_fingerprint"
            ]
        ):
            raise RuntimeError(
                "Master-key fingerprint mismatch."
            )

        randomized_operations = (
            manifest_packet[
                "operations"
            ]
        )

        schedule_seed = bytes.fromhex(
            manifest_packet[
                "schedule_seed"
            ]
        )

        # Rebuild enhanced chunks layout for validation
        # Rebuild enhanced chunks layout for validation using full protected streams
        streams = prepare_chunk_streams(
            chunks,
            subkeys
        )

        enhanced_chunks = {
            chunk_id: {
                "chunk_id": chunk_id,
                "bits": stream,
                "bit_length": len(stream)
            }
            for chunk_id, stream in streams.items()
        }

        validate_plan(
            randomized_operations,
            enhanced_chunks
        )


        stego_packet = receive_packet(
            sock
        )

        if stego_packet[
            "type"
        ] != "stego_image":
            raise RuntimeError(
                "Stego image not received."
            )

        stego_bytes = bytes.fromhex(
            stego_packet[
                "data"
            ]
        )

        if not secrets.compare_digest(
            hashlib.sha256(
                stego_bytes
            ).hexdigest(),
            stego_packet[
                "sha256"
            ]
        ):
            raise RuntimeError(
                "Stego image integrity check failed."
            )

        stego_path = (
            OUTPUT_DIR
            / stego_packet[
                "filename"
            ]
        )

        stego_path.write_bytes(
            stego_bytes
        )

        with Image.open(
            stego_path
        ) as image:
            stego_image = np.asarray(
                image,
                dtype=np.float64
            )

        received_coefficients = pywt.dwt2(
            stego_image,
            WAVELET,
            mode="periodization"
        )

        ll, (
            lh,
            hl,
            hh
        ) = received_coefficients

        received_dwt = {
            "LL": ll,
            "LH": lh,
            "HL": hl,
            "HH": hh
        }

        extracted_protected = extract(
            received_dwt,
            randomized_operations,
            subkeys,
            schedule_seed
        )

        recovered_chunks = {}

        for chunk_id in chunks:

            protected_stream = extracted_protected.get(
                chunk_id,
                ""
            )

            expected_protected_length = (
                SUBKEY_BITS
                + chunks[
                    chunk_id
                ]["bit_length"]
            )

            if len(protected_stream) != (
                expected_protected_length
            ):
                raise RuntimeError(
                    f"Extracted protected payload "
                    f"length mismatch for {chunk_id}: "
                    f"expected={expected_protected_length}, "
                    f"received={len(protected_stream)}"
                )

            embedded_subkey_bits = (
                protected_stream[
                    :SUBKEY_BITS
                ]
            )

            protected_chunk_bits = (
                protected_stream[
                    SUBKEY_BITS:
                ]
            )

            embedded_subkey = bytes_from_bits(
                embedded_subkey_bits
            )

            if not secrets.compare_digest(
                embedded_subkey,
                subkeys[
                    chunk_id
                ]
            ):
                raise RuntimeError(
                    f"Embedded QKD subkey mismatch "
                    f"for {chunk_id}."
                )

            recovered_chunks[
                chunk_id
            ] = xor_bits(
                protected_chunk_bits,
                subkeys[
                    chunk_id
                ]
            )

        reconstructed_bits = "".join(
            recovered_chunks[
                chunk_id
            ]
            for chunk_id in chunks
        )

        expected_bits = "".join(
            chunks[
                chunk_id
            ]["bits"]
            for chunk_id in chunks
        )
        message_bits = len(
            expected_bits
        )
        qkd_subkey_bits = (
            len(chunks)
            * SUBKEY_BITS
        )
        total_embedded_bits = (
            message_bits
            + qkd_subkey_bits
        )

        if len(reconstructed_bits) != len(
            expected_bits
        ):
            raise RuntimeError(
                f"Recovered message length mismatch: "
                f"expected={len(expected_bits)}, "
                f"received={len(reconstructed_bits)}"
            )

        if reconstructed_bits != expected_bits:
            for c_id in chunks:
                rec_stream = recovered_chunks.get(c_id, "")
                exp_stream = chunks[c_id]["bits"]
                if rec_stream != exp_stream:
                    print(f"Mismatch in chunk {c_id}: expected len {len(exp_stream)}, got len {len(rec_stream)}")
                    for idx, (a, b) in enumerate(zip(rec_stream, exp_stream)):
                        if a != b:
                            print(f"  Bit error at index {idx}: expected {b}, got {a}")

            bit_errors = sum(
                a != b
                for a, b in zip(
                    reconstructed_bits,
                    expected_bits
                )
            )

            raise RuntimeError(
                f"Message extraction failed. "
                f"Bit errors={bit_errors}"
            )
        recovered_bytes = bytes_from_bits(
            reconstructed_bits
        )

        

        EXTRACTED_MESSAGE.write_bytes(
            recovered_bytes
        )

        original_dwt = load_dwt()

        original_image = reconstruct(
            original_dwt
        )

        metrics = calculate_metrics(
            original_image,
            stego_image,
            total_embedded_bits,
            0.0
        )
        metrics["message_bits"] = int(message_bits)
        metrics["qkd_subkey_bits"] = int(qkd_subkey_bits)
        metrics["embedded_bits"] = int(total_embedded_bits)



        original_secret_bits = "".join(
            chunks[
                chunk_id
            ]["bits"]
            for chunk_id in chunks
        )

        original_secret_bytes = bytes_from_bits(
            original_secret_bits
        )

        payload_type = identify_payload_type(
            recovered_bytes
        )

        if payload_type == "image":

            original_secret_image = Image.open(
                io.BytesIO(
                    original_secret_bytes
                )
            ).convert("L")

            original_secret_image.load()

            reconstructed_secret_image = Image.open(
                io.BytesIO(
                    recovered_bytes
                )
            ).convert("L")

            reconstructed_secret_image.load()

            secret_bit_errors = sum(
                a != b
                for a, b in zip(
                    original_secret_bits,
                    reconstructed_bits
                )
            )

            secret_bit_count = max(
                len(original_secret_bits),
                1
            )

            secret_ber = (
                secret_bit_errors
                /
                secret_bit_count
            )

            receiver_secret_metrics = calculate_secret_image_metrics(
                original_secret_image,
                reconstructed_secret_image,
                secret_ber
            )

            save_json(
                OUTPUT_DIR
                / "receiver_secret_image_metrics.json",
                receiver_secret_metrics
            )

        else:

            receiver_secret_metrics = None

        send_packet(
            sock,
            {
                "type":
                    "receiver_result",
                "status":
                    "SUCCESS",
                "ber":
                    0.0,
                "recovered_bits":
                    message_bits,
                "metrics":
                    metrics,
                "secret_metrics":
                    receiver_secret_metrics
            }
        )

        save_json(
            METRICS_JSON,
            {
                "cover_metrics":
                    metrics,
                "secret_metrics":
                    receiver_secret_metrics
            }
        )
        

        print()
        print(
            f"QKD master QBER     : {master_result['qber']:.6f}"
        )

        print(
            f"Chunks recovered    : {len(recovered_chunks)}"
        )

        print(
            f"Message bits        : {message_bits}"
        )

        print(
            f"QKD subkey bits     : {qkd_subkey_bits}"
        )

        print(
            f"Total embedded bits : {total_embedded_bits}"
        )

        print(
            "BER                 : 0.000000"
        )

        if receiver_secret_metrics is not None:

            print()
            print(
                "=" * 70
            )
            print(
                "FINAL SECRET IMAGE METRICS"
            )
            print(
                "=" * 70
            )

            print(
                f"SSIM                : "
                f"{receiver_secret_metrics['ssim']:.12f}"
            )

            print(
                f"MSE                 : "
                f"{receiver_secret_metrics['mse']:.12f}"
            )

            print(
                f"MAE                 : "
                f"{receiver_secret_metrics['mae']:.12f}"
            )

            print(
                f"RMSE                : "
                f"{receiver_secret_metrics['rmse']:.12f}"
            )

            print(
                f"PSNR                : "
                f"{receiver_secret_metrics['psnr']:.6f} dB"
            )

            print(
                f"Original Entropy    : "
                f"{receiver_secret_metrics['original_entropy']:.12f}"
            )

            print(
                f"Reconstructed Entropy: "
                f"{receiver_secret_metrics['reconstructed_entropy']:.12f}"
            )

            print(
                f"Entropy Difference  : "
                f"{receiver_secret_metrics['entropy_difference']:.12f}"
            )

            print(
                f"KL Divergence       : "
                f"{receiver_secret_metrics['kl_divergence']:.12f}"
            )

            print(
                f"Histogram Similarity: "
                f"{receiver_secret_metrics['histogram_similarity']:.12f}"
            )

            print(
                f"NPCR                : "
                f"{receiver_secret_metrics['npcr']:.12f} %"
            )

            print(
                f"UACI                : "
                f"{receiver_secret_metrics['uaci']:.12f} %"
            )

            print(
                f"BER                 : "
                f"{receiver_secret_metrics['ber']:.12f}"
            )

            print(
                f"Maximum Abs Change  : "
                f"{receiver_secret_metrics['maximum_absolute_change']:.0f}"
            )

            print(
                "=" * 70
            )

        payload_type = identify_payload_type(
            recovered_bytes
        )

        print()
        print("=" * 70)
        print("PAYLOAD IDENTIFICATION")
        print("=" * 70)

        print(
            f"Detected Payload Type : "
            f"{payload_type.upper()}"
        )

        print("=" * 70)

        secret_img_to_plot = None
        if payload_type == "text":
            recovered_message = recovered_bytes.decode("utf-8")
            print()
            print("=" * 70)
            print("RECOVERED MESSAGE")
            print("=" * 70)
            print(recovered_message)
            print("=" * 70)

        elif payload_type == "image":
            process_received_image(
                recovered_bytes,
                OUTPUT_DIR,
                original_secret_bytes,
                receiver_secret_metrics["ber"]
            )
            secret_img_to_plot = reconstructed_secret_image

        else:
            raise RuntimeError(
                f"Unknown payload type: {payload_type}"
            )

        secret_image_psnr = calculate_secret_image_psnr(
            original_secret_image,
            reconstructed_secret_image
        )

        metrics["secret_image_psnr"] = secret_image_psnr
        # Plot Cover + Secret + Stego with the complete raw metrics table below
        plot_receiver_summary_figure(
            cover_image=original_image,
            stego_image=stego_image,
            payload_type=payload_type,
            metrics=metrics,
            sender_secret_image=original_secret_image,
            secret_image=secret_img_to_plot,
            output_path=OUTPUT_DIR / "receiver_summary_plot.png"
        )


def launch_sender_receiver():

    script_path = Path(
        __file__
    ).resolve()

    python_executable = sys.executable

    try:

        subprocess.Popen(
            [
                "wt.exe",
                "new-tab",
                "--title",
                "StegaQEntropy Sender",
                "cmd",
                "/k",
                f'"{python_executable}" "{script_path}" sender'
            ],
            cwd=BASE_DIR
        )

        time.sleep(2)

        subprocess.Popen(
            [
                "wt.exe",
                "-w",
                "0",
                "split-pane",
                "-V",
                "--title",
                "StegaQEntropy Receiver",
                "cmd",
                "/k",
                f'"{python_executable}" "{script_path}" receiver'
            ],
            cwd=BASE_DIR
        )

    except FileNotFoundError:

        raise RuntimeError(
            "Windows Terminal (wt.exe) was not found."
        )

    except Exception as error:

        raise RuntimeError(
            f"Unable to start Sender and Receiver: {error}"
        )        

def main():

    if len(sys.argv) == 1:

        launch_sender_receiver()

        return

    if len(sys.argv) != 2:

        raise SystemExit(
            "Usage: python adaptive_embedding.py"
        )

    role = sys.argv[1].lower()

    if role == "sender":

        sender()

    elif role == "receiver":

        receiver()

    else:

        raise SystemExit(
            "Invalid execution mode."
        )
    



def decode_binary_message(binary_string):
    binary_string = binary_string.strip()

    if not binary_string:
        return ""

    if any(bit not in "01" for bit in binary_string):
        raise ValueError(
            "Invalid binary data: only 0 and 1 are allowed."
        )

    if len(binary_string) % 8 != 0:
        raise ValueError(
            f"Binary length must be a multiple of 8. "
            f"Got {len(binary_string)} bits."
        )

    message_bytes = bytes(
        int(
            binary_string[i:i + 8],
            2
        )
        for i in range(
            0,
            len(binary_string),
            8
        )
    )

    return message_bytes.decode(
        "utf-8"
    )


if __name__ == "__main__":
    main()
