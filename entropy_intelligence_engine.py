import os
import cv2
import json
import math
import hashlib
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from skimage.feature import graycomatrix
from skimage.feature import graycoprops

import numpy as np

from scipy.stats import entropy
from scipy.stats import skew
from scipy.stats import kurtosis


class EntropyIntelligenceEngine:

    def __init__(self):

        self.project_root = Path(".").resolve()

        self.output_root = self.project_root / "output"

        self.module_root = self.output_root / "entropy_intelligence"

        self.module_root.mkdir(
            parents=True,
            exist_ok=True
        )

        self.timestamp = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        self.module_name = "Entropy Intelligence Engine"

        self.version = "1.0"

        self.image_profile = {}

        self.capacity_profile = {}

        self.message_profile = {}

        self.chunk_profile = {}

        self.chunk_statistics = {}

        self.chunk_entropy = {}

        self.quantum_hierarchy = {}

        self.chunk_key_mapping = {}

        self.key_configuration = {}

        self.global_profile = {}

        self.image_intelligence = {}

        self.region_analysis = {}

        self.candidate_regions = {}

        self.adaptive_regions = {}

        self.fitness_weights = {}

        self.region_database = {}

        self.image_statistics = {}

        self.visualization_maps = {}

        self.seed_information = {}

        self.manifest = {}

        self.readiness = {}

        self.report = []

        self.qrng_generators = {}

        self.ll = None

        self.lh = None

        self.hl = None

        self.hh = None

        self.pointer_file = (
            self.module_root /
            "pointer_state.json"
        )

        self.priority_weights = {
            "compatibility": 0.50,
            "confidence": 30.0,
            "diversity": 20.0
        }

        self.required_files = {

            "image_profile":
            "output/image_acquisition/image_profile.json",

            "capacity_profile":
            "output/capacity_profile.txt",

            "message_preparation":
            "output/message_preparation/message_preparation.json",

            "chunk_profile":
            "output/message_preparation/chunk_profile.txt",

            "chunk_statistics":
            "output/message_preparation/chunk_statistics.txt",

            "chunk_entropy":
            "output/message_preparation/chunk_entropy.txt",

            "hierarchy":
            "output/quantum_key_hierarchy/hierarchy.json",

            "chunk_key_mapping":
            "output/quantum_key_hierarchy/chunk_key_mapping.json",

            "key_configuration":
            "output/quantum_key_hierarchy/key_configuration.json",

            "global_profile":
            "output/dwt_decomposition/global_image_profile.json",

            "image_intelligence":
            "output/dwt_decomposition/image_intelligence.json",

            "candidate_regions":
            "output/dwt_decomposition/candidate_regions.json",

            "adaptive_regions":
            "output/dwt_decomposition/adaptive_regions.json",

            "region_analysis":
            "output/dwt_decomposition/region_analysis.json",

            "fitness_weights":
            "output/dwt_decomposition/adaptive_fitness_weights.json",

            "LL":
            "output/dwt_decomposition/LL.npy",

            "LH":
            "output/dwt_decomposition/LH.npy",

            "HL":
            "output/dwt_decomposition/HL.npy",

            "HH":
            "output/dwt_decomposition/HH.npy"

        }

        self.seed_files = {

            "entropy":
            "qrng/output/history_check/final_verified_seeds/entropy_seed_verified.txt",

            "region":
            "qrng/output/history_check/final_verified_seeds/region_seed_verified.txt",

            "position":
            "qrng/output/history_check/final_verified_seeds/position_seed_verified.txt",

            "coefficient":
            "qrng/output/history_check/final_verified_seeds/coefficient_seed_verified.txt",

            "permutation":
            "qrng/output/history_check/final_verified_seeds/permutation_seed_verified.txt",

            "noise":
            "qrng/output/history_check/final_verified_seeds/noise_seed_verified.txt",

            "gradient":
            "qrng/output/history_check/final_verified_seeds/gradient_seed_verified.txt",

            "threshold":
            "qrng/output/history_check/final_verified_seeds/threshold_seed_verified.txt",

            "payload":
            "qrng/output/history_check/final_verified_seeds/payload_seed_verified.txt"

        }

        self.output_files = {

            "manifest":
            self.module_root / "manifest.json",

            "seed_information":
            self.module_root / "seed_information.json",

            "readiness":
            self.module_root / "readiness.json",

            "region_statistics":
            self.module_root / "region_statistics.json",

            "image_statistics":
            self.module_root / "image_statistics.json",

            "entropy_database":
            self.module_root / "entropy_database.json",

            "report":
            self.module_root / "report.txt"

        }

    def print_banner(self):

        print("\n" + "=" * 100)
        print("                    ENTROPY INTELLIGENCE ENGINE")
        print("               CENTRAL INTELLIGENCE OF STEGAQENTROPY")
        print("=" * 100)

    def save_json(self, path, data):

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )

    def save_visualization_maps(self):

        for name, matrix in self.visualization_maps.items():

            self.save_heatmap_png(
                matrix,
                self.module_root / f"{name}.png"
            )

        self.log("Visualization Images Saved")

    def save_heatmap_png(self, matrix, filename, colormap=cv2.COLORMAP_JET):
       
        if matrix is None:
            return

        matrix = np.asarray(matrix, dtype=np.float32)

        if matrix.size == 0:
            return

        matrix = np.nan_to_num(matrix)

        normalized = cv2.normalize(
            matrix,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            normalized,
            colormap
        )

        cv2.imwrite(filename, heatmap)

    def load_json(self, path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    def save_report(self):

        with open(
            self.output_files["report"],
            "w",
            encoding="utf-8"
        ) as file:

            file.write("\n".join(self.report))

    def log(self, text):

        print(text)

        self.report.append(text)

    def validate_dependencies(self):

        self.log("\nChecking Module Dependencies...\n")

        missing = []

        for name, path in self.required_files.items():

            if os.path.exists(path):

                self.log(f"[OK] {path}")

            else:

                self.log(f"[MISSING] {path}")

                missing.append(path)

        for name, path in self.seed_files.items():

            if os.path.exists(path):

                self.log(f"[OK] {path}")

            else:

                self.log(f"[MISSING] {path}")

                missing.append(path)

        if len(missing):

            raise FileNotFoundError(
                "\n".join(missing)
            )

        self.log("\nDependency Validation Successful.\n")


    def load_text_file(self, path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()


    def load_numpy_arrays(self):

        self.log("\nLoading DWT Coefficient Matrices...\n")

        self.ll = np.load(
            self.required_files["LL"]
        )

        self.lh = np.load(
            self.required_files["LH"]
        )

        self.hl = np.load(
            self.required_files["HL"]
        )

        self.hh = np.load(
            self.required_files["HH"]
        )

        self.log(f"LL Shape : {self.ll.shape}")

        self.log(f"LH Shape : {self.lh.shape}")

        self.log(f"HL Shape : {self.hl.shape}")

        self.log(f"HH Shape : {self.hh.shape}")


    def load_previous_modules(self):

        self.log("\nLoading Previous Module Outputs...\n")

        self.image_profile = self.load_json(
            self.required_files["image_profile"]
        )

        self.capacity_profile = self.load_text_file(
            self.required_files["capacity_profile"]
        )

        self.message_profile = self.load_json(
            self.required_files["message_preparation"]
        )

        self.chunk_profile = self.load_text_file(
            self.required_files["chunk_profile"]
        )

        self.chunk_statistics = self.load_text_file(
            self.required_files["chunk_statistics"]
        )

        self.chunk_entropy = self.load_text_file(
            self.required_files["chunk_entropy"]
        )

        self.quantum_hierarchy = self.load_json(
            self.required_files["hierarchy"]
        )

        self.chunk_key_mapping = self.load_json(
            self.required_files["chunk_key_mapping"]
        )

        self.key_configuration = self.load_json(
            self.required_files["key_configuration"]
        )

        self.global_profile = self.load_json(
            self.required_files["global_profile"]
        )

        self.image_intelligence = self.load_json(
            self.required_files["image_intelligence"]
        )

        self.candidate_regions = self.load_json(
            self.required_files["candidate_regions"]
        )

        self.adaptive_regions = self.load_json(
            self.required_files["adaptive_regions"]
        )

        self.region_analysis = self.load_json(
            self.required_files["region_analysis"]
        )

        self.fitness_weights = self.load_json(
            self.required_files["fitness_weights"]
        )

        self.load_numpy_arrays()

        self.log("\nAll Previous Module Outputs Loaded.\n")


    def extract_binary_seed(self, path):

        binary = ""

        capture = False

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if line.strip() == "Binary":
                    capture = True
                    continue

                if line.strip() == "Hexadecimal":
                    break

                if capture:
                    binary += "".join(
                        c for c in line
                        if c in "01"
                    )

        return binary


    def create_seed_generator(

            self,

            generator_name,

            binary

    ):

        digest = hashlib.sha256(

            binary.encode()

        ).hexdigest()

        integer_seed = int(

            digest,

            16

        )

        return integer_seed, {

            "binary": binary,
            "seed_integer": integer_seed,

            "pointer": self.pointer_state.get(

                generator_name,

                0

            )

        }


    def load_qrng_seeds(self):

        self.log("\nLoading QRNG Intelligence Seeds...\n")

        for name, path in self.seed_files.items():

            binary = self.extract_binary_seed(path)

            integer_seed, generator = self.create_seed_generator(

                name,

                binary

            )

            self.qrng_generators[name] = generator

            self.seed_information[name] = {

                "file": path,

                "binary_bits": len(binary),

                "seed_integer": integer_seed,

                "sha256": hashlib.sha256(
                    binary.encode()
                ).hexdigest()

            }

            self.log(f"{name.upper()} Seed Loaded")

        self.save_json(

            self.output_files["seed_information"],

            self.seed_information

        )

        self.log("\nQRNG Seed Initialization Completed.\n")

    def get_qrng_bits(

            self,

            generator_name,

            count

    ):

        generator = self.qrng_generators[generator_name]

        binary = generator["binary"]

        pointer = generator["pointer"]

        bits = ""

        while len(bits) < count:

            remaining = len(binary) - pointer

            if remaining >= count - len(bits):

                bits += binary[
                    pointer:
                    pointer + (count - len(bits))
                ]

                pointer += count - len(bits)

            else:

                bits += binary[pointer:]

                pointer = 0

        generator["pointer"] = pointer

        self.pointer_state[generator_name] = pointer

        return bits

    def initialize_manifest(self):

        self.manifest = {

            "module": self.module_name,

            "version": self.version,

            "timestamp": self.timestamp,

            "input_files": self.required_files,

            "qrng_seeds": self.seed_files,

            "generated_files": {

                key: str(value)

                for key, value in self.output_files.items()

            }

        }

        self.save_json(

            self.output_files["manifest"],

            self.manifest

        )

        self.log("Manifest Generated")


    def initialize_readiness(self):

        self.readiness = {

            "dependency_validation": True,

            "previous_modules_loaded": True,

            "qrng_initialized": True,

            "manifest_generated": True,

            "region_database": False,

            "image_statistics": False,

            "region_statistics": False,

            "entropy_database": False,

            "relationship_analysis": False,

            "detectability_analysis": False,

            "graph_generation": False,

            "heatmap_generation": False,

            "final_report": False

        }

        self.save_json(

            self.output_files["readiness"],

            self.readiness

        )

        self.log("Readiness File Generated")


    def get_random_integer(

            self,

            generator_name,

            minimum,

            maximum

    ):

        bits = self.get_qrng_bits(

            generator_name,

            32

        )

        value = int(bits,2)

        return minimum + (

            value %

            (

                maximum -

                minimum +

                1

            )

        )


    def random_permutation(

            self,

            generator_name,

            values

    ):

        values = list(values)

        for i in range(

            len(values)-1,

            0,

            -1

        ):

            j = self.get_random_integer(

                generator_name,

                0,

                i

            )

            values[i], values[j] = (

                values[j],

                values[i]

            )

        return values


    def random_choice(

            self,

            generator_name,

            values

    ):

        index = self.get_random_integer(

            generator_name,

            0,

            len(values)-1

        )

        return values[index]


    def random_float(

            self,

            generator_name

    ):

        bits = self.get_qrng_bits(

            generator_name,

            53

        )

        return int(bits,2) / (

            2**53

        )


    def image_dimensions(self):

        return {

            "rows": self.ll.shape[0],

            "columns": self.ll.shape[1]

        }


    def coefficient_information(self):

        total = (

            self.ll.size +

            self.lh.size +

            self.hl.size +

            self.hh.size

        )

        return {

            "total_coefficients": int(total),

            "ll": int(self.ll.size),

            "lh": int(self.lh.size),

            "hl": int(self.hl.size),

            "hh": int(self.hh.size)

        }


    def initialize_engine(self):

        self.print_banner()

        self.validate_dependencies()

        self.load_previous_modules()

        self.load_pointer_state()

        self.load_qrng_seeds()

        self.initialize_manifest()

        self.initialize_readiness()

        self.log("")

        self.log("=" * 100)

        self.log("Entropy Intelligence Engine Initialized")

        self.log("=" * 100)

        info = self.image_dimensions()

        coeff = self.coefficient_information()

        self.log(

            f"Rows                : {info['rows']}"

        )

        self.log(

            f"Columns             : {info['columns']}"

        )

        self.log(

            f"Total Coefficients  : {coeff['total_coefficients']}"

        )

        self.log(

            f"QRNG Generators     : {len(self.qrng_generators)}"

        )

        self.log("")

        self.save_report()

    def create_region_database(self):

        self.log("\nCreating Region Intelligence Database...\n")

        regions = self.region_analysis

        if isinstance(regions, dict):

            if "regions" in regions:

                regions = regions["regions"]

            elif "region_analysis" in regions:

                regions = regions["region_analysis"]

            else:

                regions = list(regions.values())

        self.region_database = {}

        for index, region in enumerate(regions, start=1):

            region_id = region.get(

                "region_id",

                f"REGION_{index:05d}"

            )

            band = region.get(

                "band",

                "UNKNOWN"

            )

            coordinates = {

                "x": region.get("column_start"),

                "y": region.get("row_start"),

                "width": region.get("columns"),

                "height": region.get("rows")

            }

            self.region_database[region_id] = {

                "region_id": region_id,

                "serial_number": index,

                "band": band,

                "coordinates": coordinates,

                "original_information": region,

                "statistics": {},

                "neighbors": [],

                "correlation": {},

                "risk": {},

                "classification": {},

                "embedding": {},

                "randomization": {},

                "graph": {}

            }

        self.log(

            f"Total Regions Loaded : {len(self.region_database)}"

        )


    def create_global_database(self):

        self.image_statistics = {

            "timestamp": self.timestamp,

            "module": self.module_name,

            "image_profile": self.image_profile,

            "global_profile": self.global_profile,

            "image_intelligence": self.image_intelligence,

            "coefficient_information": self.coefficient_information(),

            "dimensions": self.image_dimensions(),

            "total_regions": len(self.region_database),

            "bands": {

                "LL": int(self.ll.size),

                "LH": int(self.lh.size),

                "HL": int(self.hl.size),

                "HH": int(self.hh.size)

            }

        }


    def save_initial_database(self):

        self.save_json(

            self.output_files["image_statistics"],

            self.image_statistics

        )

        self.save_json(

            self.output_files["entropy_database"],

            self.region_database

        )

        self.readiness["region_database"] = True

        self.save_json(

            self.output_files["readiness"],

            self.readiness

        )


    def initialize_part1(self):

        self.initialize_engine()

        self.create_region_database()

        self.create_global_database()

        self.save_initial_database()

        self.log("\n" + "=" * 100)

        self.log("PART 1 COMPLETED")

        self.log("=" * 100)

        self.log(f"Regions Initialized        : {len(self.region_database)}")

        self.log("Image Database Created     : YES")

        self.log("Region Database Created    : YES")

        self.log("QRNG Ready                 : YES")

        self.log("Manifest Ready             : YES")

        self.log("Readiness Updated          : YES")

        self.save_report()

    def get_band_matrix(self, band):

        band = str(band).upper()

        if band == "LL":
            return self.ll

        if band == "LH":
            return self.lh

        if band == "HL":
            return self.hl

        if band == "HH":
            return self.hh

        return None


    def extract_region_matrix(self, region):

        band = region["band"]

        matrix = self.get_band_matrix(band)

        coordinates = region["coordinates"]

        x = int(coordinates["x"])
        y = int(coordinates["y"])
        w = int(coordinates["width"])
        h = int(coordinates["height"])

        return matrix[
            y:y + h,
            x:x + w
        ]

    def calculate_shannon_entropy(self, values):

        values = np.asarray(
            values,
            dtype=np.float64
        ).flatten()

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            return 0.0

        minimum = float(np.min(values))
        maximum = float(np.max(values))

        if maximum - minimum <= 1e-12:
            return 0.0

        hist, _ = np.histogram(
            values,
            bins=256,
            range=(minimum, maximum)
        )

        probability = hist[
            hist > 0
        ].astype(np.float64)

        probability /= probability.sum()

        return float(
            entropy(
                probability,
                base=2
            )
        )

    def calculate_statistics(self, matrix):

        values = matrix.astype(
            np.float64
        ).flatten()

        if values.size == 0:

            return {

                "mean": 0,
                "median": 0,
                "variance": 0,
                "std": 0,
                "minimum": 0,
                "maximum": 0

            }

        return {

            "mean": float(np.mean(values)),

            "median": float(np.median(values)),

            "variance": float(np.var(values)),

            "std": float(np.std(values)),

            "minimum": float(np.min(values)),

            "maximum": float(np.max(values))

        }


    def build_basic_region_statistics(self):

        self.log("\nBuilding Region Statistics...\n")

        for region_id, region in self.region_database.items():

            matrix = self.extract_region_matrix(region)

            statistics = self.calculate_statistics(
                matrix
            )

            statistics["entropy"] = self.calculate_shannon_entropy(
                matrix
            )

            statistics["shape"] = list(
                matrix.shape
            )

            statistics["pixels"] = int(
                matrix.size
            )

            region["statistics"] = statistics

        self.log(

            f"Statistics Generated : {len(self.region_database)} Regions"

        )

    def calculate_extended_statistics(self, matrix):

        values = matrix.astype(
            np.float64
        ).flatten()

        if values.size == 0:

            return {

                "mad": 0.0,
                "dynamic_range": 0.0,
                "rms": 0.0,
                "energy": 0.0,
                "sparsity": 0.0,
                "density": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "entropy_utilization": 0.0,
                "local_randomness": 0.0

            }

        maximum = float(
            np.max(values)
        )

        minimum = float(
            np.min(values)
        )

        mean_value = float(
            np.mean(values)
        )

        mad = float(
            np.mean(
                np.abs(
                    values - mean_value
                )
            )
        )

        rms = float(
            np.sqrt(
                np.mean(
                    values ** 2
                )
            )
        )

        energy = float(
            np.sum(
                values ** 2
            )
        )

        zero_count = int(
            np.count_nonzero(
                values == 0
            )
        )

        non_zero_count = int(
            np.count_nonzero(
                values
            )
        )

        density = float(
            non_zero_count /
            values.size
        )

        sparsity = float(
            zero_count /
            values.size
        )

        entropy_value = self.calculate_shannon_entropy(
            matrix
        )

        entropy_utilization = float(
            np.clip(
                entropy_value /
                np.log2(
                    max(
                        len(
                            np.unique(values)
                        ),
                        2
                    )
                ),
                0.0,
                1.0
            )
        )

        entropy_component = entropy_utilization

        variation_component = float(
            np.clip(
                mad /
                (
                    np.std(values) +
                    1e-12
                ),
                0.0,
                1.0
            )
        )

        distribution_component = float(
            np.clip(
                1.0 -
                abs(
                    float(
                        skew(values)
                    )
                ) /
                5.0,
                0.0,
                1.0
            )
        )

        randomness = float(
            np.mean(
                [
                    entropy_component,
                    variation_component,
                    distribution_component
                ]
            ) * 100.0
        )

        try:

            skewness = float(
                skew(values)
            )

        except Exception:

            skewness = 0.0

        try:

            kurt = float(
                kurtosis(values)
            )

        except Exception:

            kurt = 0.0

        return {

            "mad":
                mad,

            "dynamic_range":
                maximum - minimum,

            "rms":
                rms,

            "energy":
                energy,

            "sparsity":
                sparsity,

            "density":
                density,

            "skewness":
                skewness,

            "kurtosis":
                kurt,

            "entropy_utilization":
                entropy_utilization,

            "local_randomness":
                randomness

        }

    def enrich_region_statistics(self):

        self.log("\nComputing Advanced Region Statistics...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(

                region

            )

            region["statistics"].update(

                self.calculate_extended_statistics(

                    matrix

                )

            )

        self.readiness["region_statistics"] = True

        self.save_json(

            self.output_files["readiness"],

            self.readiness

        )

        self.log(

            f"Advanced Statistics Generated : {len(self.region_database)} Regions"

        )

    def compute_global_statistics(self):

        self.log("\nComputing Global Image Statistics...\n")

        complete_image = np.concatenate(

            [

                self.ll.flatten(),

                self.lh.flatten(),

                self.hl.flatten(),

                self.hh.flatten()

            ]

        ).astype(np.float64)

        global_entropy = self.calculate_shannon_entropy(
            complete_image
        )

        self.image_statistics["global_statistics"] = {

            "mean": float(np.mean(complete_image)),

            "median": float(np.median(complete_image)),

            "variance": float(np.var(complete_image)),

            "standard_deviation": float(np.std(complete_image)),

            "minimum": float(np.min(complete_image)),

            "maximum": float(np.max(complete_image)),

            "dynamic_range": float(

                np.max(complete_image) -

                np.min(complete_image)

            ),

            "rms": float(

                np.sqrt(

                    np.mean(

                        complete_image ** 2

                    )

                )

            ),

            "energy": float(

                np.sum(

                    complete_image ** 2

                )

            ),

            "entropy": global_entropy,

            "entropy_utilization": float(
                np.clip(
                    global_entropy /
                    np.log2(
                        max(
                            self.ll.size +
                            self.lh.size +
                            self.hl.size +
                            self.hh.size,
                            2
                        )
                    ),
                    0.0,
                    1.0
                ) * 100.0
            ),

            "total_pixels": int(

                complete_image.size

            )

        }

        self.log("Global Statistics Generated")


    def compute_relative_region_statistics(self):

        self.log("\nComputing Relative Region Intelligence...\n")

        global_stats = self.image_statistics[

            "global_statistics"

        ]

        g_entropy = global_stats["entropy"]

        g_mean = global_stats["mean"]

        g_std = global_stats["standard_deviation"]

        g_energy = global_stats["energy"]

        for region in self.region_database.values():

            stats = region["statistics"]

            region["relative_statistics"] = {

                "entropy_ratio": float(

                    stats["entropy"] / g_entropy

                ) if g_entropy else 0.0,

                "mean_ratio": float(

                    stats["mean"] / g_mean

                ) if g_mean else 0.0,

                "std_ratio": float(

                    stats["std"] / g_std

                ) if g_std else 0.0,

                "energy_ratio": float(

                    stats["energy"] / g_energy

                ) if g_energy else 0.0,

                "entropy_difference": float(

                    stats["entropy"] - g_entropy

                ),

                "mean_difference": float(

                    stats["mean"] - g_mean

                ),

                "std_difference": float(

                    stats["std"] - g_std

                )

            }

        self.readiness["image_statistics"] = True

        self.save_json(

            self.output_files["readiness"],

            self.readiness

        )

        self.log(

            f"Relative Intelligence Generated : {len(self.region_database)} Regions"

        )

    def assign_entropy_intelligence_ids(self):

        self.log("\nAssigning Entropy Intelligence IDs...\n")

        counters = {

            "LL": 1,

            "LH": 1,

            "HL": 1,

            "HH": 1

        }

        for region in self.region_database.values():

            band = region["band"].upper()

            eid = f"EI-{band}-{counters[band]:05d}"

            region["entropy_intelligence_id"] = eid

            counters[band] += 1

        self.log("Entropy Intelligence IDs Assigned")


    def classify_entropy_regions(self):

        self.log("\nClassifying Regions...\n")

        global_entropy = self.image_statistics[

            "global_statistics"

        ]["entropy"]

        global_std = self.image_statistics[

            "global_statistics"

        ]["standard_deviation"]

        for region in self.region_database.values():

            stats = region["statistics"]

            entropy_score = stats["entropy"]

            randomness = stats["local_randomness"]

            std = stats["std"]

            if entropy_score >= global_entropy * 1.15:

                entropy_level = "VERY_HIGH"

            elif entropy_score >= global_entropy:

                entropy_level = "HIGH"

            elif entropy_score >= global_entropy * 0.80:

                entropy_level = "MEDIUM"

            else:

                entropy_level = "LOW"

            if randomness >= 85:

                randomness_level = "VERY_HIGH"

            elif randomness >= 70:

                randomness_level = "HIGH"

            elif randomness >= 50:

                randomness_level = "MEDIUM"

            else:

                randomness_level = "LOW"

            if std >= global_std * 1.20:

                texture_level = "HIGH"

            elif std >= global_std * 0.80:

                texture_level = "MEDIUM"

            else:

                texture_level = "LOW"

            region["classification"] = {

                "entropy_level": entropy_level,

                "randomness_level": randomness_level,

                "texture_level": texture_level

            }

        self.log("Region Classification Completed")


    def save_statistics_database(self):

        self.save_json(

            self.output_files["region_statistics"],

            self.region_database

        )

        self.save_json(

            self.output_files["image_statistics"],

            self.image_statistics

        )

        self.save_json(

            self.output_files["entropy_database"],

            self.region_database

        )

        self.readiness["entropy_database"] = True

        self.save_json(

            self.output_files["readiness"],

            self.readiness

        )

        self.log("\nRegion Statistics Saved")

        self.log("Image Statistics Saved")

        self.log("Entropy Database Saved")


    def execute_part2(self):

        self.build_basic_region_statistics()

        self.enrich_region_statistics()

        self.compute_global_statistics()

        self.compute_relative_region_statistics()

        self.assign_entropy_intelligence_ids()

        self.classify_entropy_regions()

        self.save_statistics_database()

        self.log("\n" + "=" * 100)

        self.log("PART 2 COMPLETED")

        self.log("=" * 100)

        self.log(f"Regions Processed : {len(self.region_database)}")

        self.log("Entropy Statistics : READY")

        self.log("Relative Metrics : READY")

        self.log("Classification : READY")

        self.log("Entropy Database : READY")

        self.save_report()

    def build_entropy_relationship_graph(self):

        self.log(
            "\nBuilding Entropy Relationship Graph...\n"
        )

        regions = list(
            self.region_database.values()
        )

        for region in regions:

            current_entropy = float(
                region["statistics"]["entropy"]
            )

            candidates = []

            x1 = float(
                region["coordinates"]["x"]
            )

            y1 = float(
                region["coordinates"]["y"]
            )

            w1 = float(
                region["coordinates"]["width"]
            )

            h1 = float(
                region["coordinates"]["height"]
            )

            cx1 = x1 + w1 / 2.0
            cy1 = y1 + h1 / 2.0

            for candidate in regions:

                if (
                    candidate["region_id"] ==
                    region["region_id"]
                ):
                    continue

                x2 = float(
                    candidate["coordinates"]["x"]
                )

                y2 = float(
                    candidate["coordinates"]["y"]
                )

                w2 = float(
                    candidate["coordinates"]["width"]
                )

                h2 = float(
                    candidate["coordinates"]["height"]
                )

                cx2 = x2 + w2 / 2.0
                cy2 = y2 + h2 / 2.0

                distance = math.sqrt(
                    (
                        cx1 - cx2
                    ) ** 2 +
                    (
                        cy1 - cy2
                    ) ** 2
                )

                entropy_difference = abs(
                    current_entropy -
                    float(
                        candidate["statistics"]["entropy"]
                    )
                )

                candidates.append({

                    "region":
                        candidate["region_id"],

                    "distance":
                        distance,

                    "entropy_difference":
                        entropy_difference,

                    "similarity_score":
                        max(
                            0.0,
                            100.0 -
                            (
                                entropy_difference *
                                12.5
                            )
                        )

                })

            candidates.sort(
                key=lambda item: (
                    item["distance"],
                    item["entropy_difference"]
                )
            )

            selected = candidates[:10]

            region["neighbors"] = [

                {
                    "region":
                        item["region"],

                    "entropy_difference":
                        round(
                            item["entropy_difference"],
                            6
                        ),

                    "similarity_score":
                        round(
                            item["similarity_score"],
                            4
                        ),

                    "spatial_distance":
                        round(
                            item["distance"],
                            6
                        )

                }

                for item in selected

            ]

            region["randomization"] = {

                "entropy_index":
                    self.get_random_integer(
                        "entropy",
                        100000,
                        999999
                    ),

                "region_index":
                    self.get_random_integer(
                        "region",
                        100000,
                        999999
                    )

            }

        self.log(
            f"Relationship Graph Generated : "
            f"{len(self.region_database)} Regions"
        )

    def compute_region_priority(self):

        self.log("\nComputing Region Priority Scores...\n")

        for region in self.region_database.values():

            stats = region["statistics"]

            relative = region["relative_statistics"]

            entropy_score = float(
                np.clip(
                    stats["entropy"] /
                    np.log2(
                        max(
                            stats["pixels"],
                            2
                        )
                    ),
                    0.0,
                    1.0
                ) * 100.0
            )

            randomness_score = stats["local_randomness"]

            relative_score = min(
                relative["entropy_ratio"] * 100,
                100
            )

            density_score = stats["density"] * 100

            safety_score = region["risk"]["detectability_index"] * 100.0

            score = (

                entropy_score * 0.30 +

                randomness_score * 0.20 +

                relative_score * 0.15 +

                density_score * 0.15 +

                safety_score * 0.20

            )

            score = max(0.0, min(score, 100.0))

            region["priority"] = {

                "score": round(

                    score,

                    4

                ),

                "rank": 0

            }

        ranking = sorted(

            self.region_database.items(),

            key=lambda item: item[1]["priority"]["score"],

            reverse=True

        )

        for rank, (_, region) in enumerate(

                ranking,

                start=1

        ):

            region["priority"]["rank"] = rank

        self.log("Priority Ranking Completed")

    def compute_detectability_index(self):

        self.log("\nComputing Detectability Intelligence...\n")

        global_entropy = self.image_statistics[
            "global_statistics"
        ]["entropy"]

        for region in self.region_database.values():

            stats = region["statistics"]

            entropy_score = stats["entropy"] / global_entropy if global_entropy else 0

            variance_score = min(
                stats["variance"] / 1000.0,
                1.0
            )

            randomness_score = (
                stats["local_randomness"] / 100.0
            )

            density_score = stats[
                "density"
            ]

            detectability = (

                entropy_score * 0.35 +

                randomness_score * 0.30 +

                variance_score * 0.20 +

                density_score * 0.15

            )

            detectability = max(
                0.0,
                min(
                    detectability,
                    1.0
                )
            )

            if detectability >= 0.85:

                level = "VERY_SAFE"

            elif detectability >= 0.70:

                level = "SAFE"

            elif detectability >= 0.55:

                level = "MODERATE"

            elif detectability >= 0.40:

                level = "RISKY"

            else:

                level = "HIGH_RISK"

            region["risk"] = {

                "detectability_index": round(
                    detectability,
                    6
                ),

                "risk_score": round(
                    (1.0 - detectability) * 100,
                    4
                ),

                "risk_level": level

            }

        self.log("Detectability Analysis Completed")

    def build_embedding_intelligence(self):

        self.log(
            "\nBuilding Embedding Intelligence...\n"
        )

        feature_definitions = [

            (
                "entropy",
                lambda r:
                    r["statistics"]["entropy"],
                True
            ),

            (
                "local_entropy",
                lambda r:
                    r["local_entropy"]["average"],
                True
            ),

            (
                "neighbor_entropy",
                lambda r:
                    r["neighbor_entropy"]["neighbor_average"],
                True
            ),

            (
                "five_neighbor_entropy",
                lambda r:
                    r["multi_scale_entropy"]["five_neighbor_average"],
                True
            ),

            (
                "ten_neighbor_entropy",
                lambda r:
                    r["multi_scale_entropy"]["ten_neighbor_average"],
                True
            ),

            (
                "band_entropy",
                lambda r:
                    r["multi_scale_entropy"]["band_average"],
                True
            ),

            (
                "entropy_stability",
                lambda r:
                    r["entropy_stability"]["stability_index"],
                True
            ),

            (
                "entropy_gradient",
                lambda r:
                    abs(
                        r["entropy_gradient"]["average_gradient"]
                    ),
                True
            ),

            (
                "energy",
                lambda r:
                    r["energy_intelligence"]["global_energy_ratio"],
                True
            ),

            (
                "variance",
                lambda r:
                    r["variance_intelligence"]["relative_variance"],
                True
            ),

            (
                "gradient",
                lambda r:
                    r["gradient_intelligence"]["average_gradient"],
                True
            ),

            (
                "noise",
                lambda r:
                    r["noise_intelligence"]["noise_variance"],
                True
            ),

            (
                "texture_complexity",
                lambda r:
                    r["texture_complexity"]["complexity_score"],
                True
            ),

            (
                "gradient_consistency",
                lambda r:
                    r["gradient_consistency"]["consistency_index"],
                True
            ),

            (
                "edge_consistency",
                lambda r:
                    r["edge_consistency"]["consistency_score"],
                True
            )

        ]

        regions = list(
            self.region_database.values()
        )

        feature_values = {}

        for name, extractor, higher_is_better in feature_definitions:

            values = []

            for region in regions:

                try:

                    value = float(
                        extractor(region)
                    )

                except (
                    KeyError,
                    TypeError,
                    ValueError
                ):

                    value = 0.0

                if not np.isfinite(value):

                    value = 0.0

                values.append(value)

            feature_values[name] = {

                "values": np.asarray(
                    values,
                    dtype=np.float64
                ),

                "higher_is_better":
                    higher_is_better

            }

        def normalize_feature(
            values,
            value,
            higher_is_better
        ):

            finite_values = values[
                np.isfinite(values)
            ]

            if finite_values.size == 0:

                return 0.5

            minimum = float(
                np.min(finite_values)
            )

            maximum = float(
                np.max(finite_values)
            )

            if maximum - minimum <= 1e-12:

                return 1.0

            normalized = (
                float(value) - minimum
            ) / (
                maximum - minimum
            )

            normalized = float(
                np.clip(
                    normalized,
                    0.0,
                    1.0
                )
            )

            if not higher_is_better:

                normalized = 1.0 - normalized

            return normalized

        for region in regions:

            evidence = {}

            evidence_values = []

            for name, extractor, higher_is_better in feature_definitions:

                try:

                    raw_value = float(
                        extractor(region)
                    )

                except (
                    KeyError,
                    TypeError,
                    ValueError
                ):

                    raw_value = 0.0

                if not np.isfinite(raw_value):

                    raw_value = 0.0

                normalized_value = normalize_feature(

                    feature_values[name]["values"],

                    raw_value,

                    higher_is_better

                )

                evidence[name] = {

                    "raw":
                        raw_value,

                    "normalized":
                        normalized_value

                }

                evidence_values.append(
                    normalized_value
                )

            evidence_values = np.asarray(
                evidence_values,
                dtype=np.float64
            )

            valid_values = evidence_values[
                np.isfinite(evidence_values)
            ]

            if valid_values.size == 0:

                capacity_fraction = 1.0

                evidence_spread = 0.0
                evidence_minimum = 0.0
                evidence_maximum = 0.0

            else:

                # Use ALL 15 measurements as intelligence evidence.
                # Do not aggressively compress the raw coefficient capacity.
                evidence_score = float(
                    np.mean(valid_values)
                )

                evidence_spread = float(
                    np.std(valid_values)
                )

                evidence_minimum = float(
                    np.min(valid_values)
                )

                evidence_maximum = float(
                    np.max(valid_values)
                )

                # Entropy stage gives an initial estimate.
                # Keep most of the 1-bit-per-coefficient capacity.
                capacity_fraction = float(
                    np.clip(
                        0.80 +
                        (evidence_score * 0.15),
                        0.80,
                        0.95
                    )
                )

            maximum_payload = max(
                int(
                    region["original_information"].get(
                        "estimated_capacity",
                        region["statistics"]["pixels"]
                    )
                ),
                1
            )

            recommended_payload = int(
                round(
                    maximum_payload *
                    capacity_fraction
                )
            )

            recommended_payload = max(
                1,
                min(
                    recommended_payload,
                    maximum_payload
                )
            )

            region["embedding"] = {

                "recommended_payload":
                    recommended_payload,

                "maximum_payload":
                    maximum_payload,

                "payload_percentage":
                    round(
                        (
                            recommended_payload /
                            maximum_payload
                        ) * 100.0,
                        4
                    ),

                "capacity_fraction":
                    round(
                        capacity_fraction,
                        6
                    ),
                "evidence_score":
                    round(
                        evidence_score,
                        6
                    ),

                "entropy_measure_count":
                    len(feature_definitions),

                "entropy_evidence":
                    evidence,

                "recommended_threshold":
                    self.get_random_integer(
                        "threshold",
                        2,
                        8
                    ),

                "recommended_strength":
                    round(
                        capacity_fraction * 100.0,
                        4
                    ),
                "evidence_spread":
                    round(
                        evidence_spread,
                        6
                    ),

                "evidence_minimum":
                    round(
                        evidence_minimum,
                        6
                    ),

                "evidence_maximum":
                    round(
                        evidence_maximum,
                        6
                    ),

                "payload_seed":
                    self.get_random_integer(
                        "payload",
                        100000,
                        999999
                    ),

                "coefficient_seed":
                    self.get_random_integer(
                        "coefficient",
                        100000,
                        999999
                    )

            }

        self.log(
            "Embedding Intelligence Generated"
        )

    def build_entropy_decision_engine(self):

        self.log("\nBuilding Entropy Decision Engine...\n")

        import numpy as np

        # Use the SAME score for threshold calculation
        # and final classification.
        decision_scores = []

        for region in self.region_database.values():

            score = (
                region["embedding"]["evidence_score"]
                * 100.0
            )

            score = max(
                0.0,
                min(
                    100.0,
                    score
                )
            )

            decision_scores.append(score)

        primary_threshold = np.percentile(
            decision_scores,
            85
        )

        secondary_threshold = np.percentile(
            decision_scores,
            55
        )

        tertiary_threshold = np.percentile(
            decision_scores,
            25
        )

        self.log(
            f"Primary Threshold   : {primary_threshold:.4f}"
        )

        self.log(
            f"Secondary Threshold : {secondary_threshold:.4f}"
        )

        self.log(
            f"Tertiary Threshold  : {tertiary_threshold:.4f}"
        )

        for region in self.region_database.values():

            embedding = region["embedding"]

            # SAME score used above
            score = (
                embedding["evidence_score"]
                * 100.0
            )

            score = max(
                0.0,
                min(
                    100.0,
                    score
                )
            )

            if score >= primary_threshold:

                decision = "PRIMARY"

            elif score >= secondary_threshold:

                decision = "SECONDARY"

            elif score >= tertiary_threshold:

                decision = "TERTIARY"

            else:

                decision = "REJECT"

            region["decision"] = {

                "decision":
                    decision,

                "confidence":
                    round(
                        score,
                        4
                    ),

                "decision_score":
                    round(
                        score,
                        4
                    ),

                "priority_score":
                    round(
                        region["priority"]["score"],
                        4
                    ),

                "recommended_payload":
                    embedding["recommended_payload"],

                "recommended_strength":
                    embedding["recommended_strength"]

            }

        self.log("Decision Engine Completed")

    def generate_entropy_summary(self):

        self.log("\nGenerating Intelligence Summary...\n")

        summary = {

            "PRIMARY": 0,

            "SECONDARY": 0,

            "TERTIARY": 0,

            "REJECT": 0

        }

        highest = None

        highest_score = -1

        for region_id, region in self.region_database.items():

            decision = region["decision"]["decision"]

            score = region["decision"]["decision_score"]

            summary[decision] += 1

            if score > highest_score:

                highest_score = score

                highest = region_id

        self.image_statistics["decision_summary"] = {

            "primary_regions":

                summary["PRIMARY"],

            "secondary_regions":

                summary["SECONDARY"],

            "tertiary_regions":

                summary["TERTIARY"],

            "rejected_regions":

                summary["REJECT"],

            "best_region":

                highest,

            "highest_score":

                round(

                    highest_score,

                    4

                )

        }

        self.log("Summary Generated")


    def save_decision_database(self):

        self.save_json(

            self.output_files["entropy_database"],

            self.region_database

        )

        self.save_json(

            self.output_files["image_statistics"],

            self.image_statistics

        )

        self.readiness["relationship_analysis"] = True

        self.readiness["detectability_analysis"] = True

        self.save_json(

            self.output_files["readiness"],

            self.readiness

        )

        self.log("Decision Database Saved")


    def execute_part3(self):

        self.build_entropy_relationship_graph()

        self.compute_detectability_index()

        self.compute_region_priority()

        self.log("\n" + "=" * 100)

        self.log("PART 3 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def build_entropy_heatmap(self):

        self.log("\nGenerating Entropy Heatmap...\n")

        rows = self.ll.shape[0]
        cols = self.ll.shape[1]

        entropy_map = np.zeros(
            (rows, cols),
            dtype=np.float64
        )

        count_map = np.zeros(
            (rows, cols),
            dtype=np.float64
        )

        for region in self.region_database.values():

            coordinate = region["coordinates"]

            x = int(coordinate["x"])
            y = int(coordinate["y"])
            w = int(coordinate["width"])
            h = int(coordinate["height"])

            entropy_value = region["statistics"]["entropy"]

            entropy_map[
                y:y+h,
                x:x+w
            ] += entropy_value

            count_map[
                y:y+h,
                x:x+w
            ] += 1

        count_map[
            count_map == 0
        ] = 1

        entropy_map /= count_map

        self.visualization_maps["entropy_heatmap"] = entropy_map

        self.image_statistics["entropy_heatmap"] = {

            "minimum_entropy":
                float(np.min(entropy_map)),

            "maximum_entropy":
                float(np.max(entropy_map)),

            "average_entropy":
                float(np.mean(entropy_map))

        }

        np.save(

            self.module_root /
            "entropy_heatmap.npy",

            entropy_map

        )

        self.log("Entropy Heatmap Generated")


    def build_priority_heatmap(self):

        self.log("\nGenerating Priority Heatmap...\n")

        rows = self.ll.shape[0]
        cols = self.ll.shape[1]

        priority_map = np.zeros(
            (rows, cols),
            dtype=np.float64
        )

        count_map = np.zeros(
            (rows, cols),
            dtype=np.float64
        )

        for region in self.region_database.values():

            coordinate = region["coordinates"]

            x = int(coordinate["x"])
            y = int(coordinate["y"])
            w = int(coordinate["width"])
            h = int(coordinate["height"])

            value = region["priority"]["score"]

            priority_map[
                y:y+h,
                x:x+w
            ] += value

            count_map[
                y:y+h,
                x:x+w
            ] += 1

        count_map[
            count_map == 0
        ] = 1

        priority_map /= count_map

        self.visualization_maps["priority_heatmap"] = priority_map

        np.save(

            self.module_root /
            "priority_heatmap.npy",

            priority_map

        )

        self.log("Priority Heatmap Generated")

    def generate_entropy_packages(self):

        self.log("\nGenerating Entropy Intelligence Packages...\n")

        ranked_regions = sorted(

            self.region_database.values(),

            key=lambda region: region["priority"]["score"],

            reverse=True

        )

        primary = []
        secondary = []
        tertiary = []
        rejected = []

        for region in ranked_regions:

            package = {

                "entropy_id":
                    region["entropy_intelligence_id"],

                "region_id":
                    region["region_id"],

                "band":
                    region["band"],

                "priority_score":
                    region["priority"]["score"],

                "priority_rank":
                    region["priority"]["rank"],

                "entropy":
                    region["statistics"]["entropy"],

                "randomness":
                    region["statistics"]["local_randomness"],

                "detectability":
                    region["risk"]["detectability_index"],

                "decision":
                    region["decision"]["decision"],

                "recommended_payload":
                    region["embedding"]["recommended_payload"],

                "recommended_strength":
                    region["embedding"]["recommended_strength"]

            }

            decision = region["decision"]["decision"]

            if decision == "PRIMARY":

                primary.append(package)

            elif decision == "SECONDARY":

                secondary.append(package)

            elif decision == "TERTIARY":

                tertiary.append(package)

            else:

                rejected.append(package)

        package_directory = self.module_root / "packages"

        package_directory.mkdir(

            exist_ok=True

        )

        self.save_json(

            package_directory /

            "primary_regions.json",

            primary

        )

        self.save_json(

            package_directory /

            "secondary_regions.json",

            secondary

        )

        self.save_json(

            package_directory /

            "tertiary_regions.json",

            tertiary

        )

        self.save_json(

            package_directory /

            "rejected_regions.json",

            rejected

        )

        self.image_statistics["package_summary"] = {

            "primary": len(primary),

            "secondary": len(secondary),

            "tertiary": len(tertiary),

            "rejected": len(rejected)

        }

        self.log("Entropy Packages Generated")


    def generate_engine_report(self):

        self.log("\nGenerating Engine Report...\n")

        report = {

            "module": self.module_name,

            "version": self.version,

            "timestamp": self.timestamp,

            "total_regions":

                len(self.region_database),

            "global_statistics":

                self.image_statistics["global_statistics"],

            "decision_summary":

                self.image_statistics["decision_summary"],

            "package_summary":

                self.image_statistics["package_summary"]

        }

        self.save_json(

            self.module_root /

            "engine_report.json",

            report

        )

        self.log("Engine Report Generated")

    def finalize_engine(self):

        self.save_pointer_state()

        self.save_json(

            self.output_files["image_statistics"],

            self.image_statistics

        )

        self.save_json(

            self.output_files["entropy_database"],

            self.region_database

        )

        self.readiness["graph_generation"] = True

        self.readiness["heatmap_generation"] = True

        self.readiness["final_report"] = True

        self.save_json(

            self.output_files["readiness"],

            self.readiness

        )

        self.log("\n" + "=" * 100)

        self.log("ENTROPY INTELLIGENCE ENGINE SUMMARY")

        self.log("=" * 100)

        summary = self.image_statistics["decision_summary"]

        packages = self.image_statistics["package_summary"]

        global_stats = self.image_statistics["global_statistics"]

        self.log(f"Total Regions             : {len(self.region_database)}")

        self.log(f"Global Entropy            : {global_stats['entropy']:.6f}")

        self.log(f"Global Mean               : {global_stats['mean']:.6f}")

        self.log(f"Global Variance           : {global_stats['variance']:.6f}")

        self.log(f"Primary Regions           : {summary['primary_regions']}")

        self.log(f"Secondary Regions         : {summary['secondary_regions']}")

        self.log(f"Tertiary Regions          : {summary['tertiary_regions']}")

        self.log(f"Rejected Regions          : {summary['rejected_regions']}")

        self.log(f"Best Region               : {summary['best_region']}")

        self.log(f"Highest Score             : {summary['highest_score']}")

        self.log(f"Primary Package Size      : {packages['primary']}")

        self.log(f"Secondary Package Size    : {packages['secondary']}")

        self.log(f"Tertiary Package Size     : {packages['tertiary']}")

        self.log(f"Rejected Package Size     : {packages['rejected']}")

        self.log("=" * 100)

        self.save_report()

    def compute_local_entropy_intelligence(self):

        self.log("\nComputing Local Entropy Intelligence...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(region)

            h, w = matrix.shape

            if h < 4 or w < 4:

                local_entropy = self.calculate_shannon_entropy(matrix)

                entropy_map = [local_entropy]

            else:

                entropy_map = []

                step_y = max(2, h // 4)

                step_x = max(2, w // 4)

                for y in range(0, h, step_y):

                    for x in range(0, w, step_x):

                        block = matrix[
                            y:min(y + step_y, h),
                            x:min(x + step_x, w)
                        ]

                        entropy_map.append(
                            self.calculate_shannon_entropy(block)
                        )

                local_entropy = float(np.mean(entropy_map))

            region["local_entropy"] = {

                "average": local_entropy,

                "minimum": float(np.min(entropy_map)),

                "maximum": float(np.max(entropy_map)),

                "standard_deviation": float(np.std(entropy_map)),

                "samples": len(entropy_map),

                "distribution": [

                    round(v, 6)

                    for v in entropy_map

                ]

            }

        self.log(

            f"Local Entropy Computed : {len(self.region_database)} Regions"

        )


    def compute_neighbor_entropy(self):

        self.log("\nComputing Neighbor Entropy...\n")

        ids = list(self.region_database.keys())

        for region_id, region in self.region_database.items():

            current = region["statistics"]["entropy"]

            neighbours = []

            for neighbour in region["neighbors"]:

                rid = neighbour["region"]

                if rid in self.region_database:

                    neighbours.append(

                        self.region_database[rid]["statistics"]["entropy"]

                    )

            if len(neighbours) == 0:

                average = current

                deviation = 0.0

            else:

                average = float(np.mean(neighbours))

                deviation = float(np.std(neighbours))

            region["neighbor_entropy"] = {

                "current_entropy": current,

                "neighbor_average": average,

                "neighbor_std": deviation,

                "neighbor_count": len(neighbours),

                "difference": current - average

            }

        self.log("Neighbor Entropy Completed")


    def compute_multi_neighborhood_entropy(self):

        self.log("\nComputing Multi-Scale Neighborhood Entropy...\n")

        global_entropy = self.image_statistics[

            "global_statistics"

        ]["entropy"]

        band_groups = {

            "LL": [],

            "LH": [],

            "HL": [],

            "HH": []

        }

        for region in self.region_database.values():

            band_groups[

                region["band"]

            ].append(

                region["statistics"]["entropy"]

            )

        band_average = {}

        for band, values in band_groups.items():

            if len(values):

                band_average[band] = float(np.mean(values))

            else:

                band_average[band] = 0.0

        for region in self.region_database.values():

            entropy_value = region["statistics"]["entropy"]

            neighbours = region["neighbors"]

            first5 = []

            first10 = []

            for n in neighbours[:5]:

                first5.append(

                    self.region_database[

                        n["region"]

                    ]["statistics"]["entropy"]

                )

            for n in neighbours[:10]:

                first10.append(

                    self.region_database[

                        n["region"]

                    ]["statistics"]["entropy"]

                )

            region["multi_scale_entropy"] = {

                "five_neighbor_average":

                    float(np.mean(first5))

                    if len(first5)

                    else entropy_value,

                "ten_neighbor_average":

                    float(np.mean(first10))

                    if len(first10)

                    else entropy_value,

                "band_average":

                    band_average[

                        region["band"]

                    ],

                "global_average":

                    global_entropy,

                "relative_entropy":

                    entropy_value /

                    global_entropy

                    if global_entropy

                    else 0.0,

                "entropy_deviation":

                    entropy_value -

                    global_entropy

            }

        self.log("Multi-Scale Entropy Completed")


    def execute_part5_1(self):

        self.compute_local_entropy_intelligence()

        self.compute_neighbor_entropy()

        self.compute_multi_neighborhood_entropy()

        self.log("\n" + "=" * 100)

        self.log("PART 5.1 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def compute_entropy_stability(self):

        self.log("\nComputing Entropy Stability...\n")

        for region in self.region_database.values():

            current_entropy = region["statistics"]["entropy"]

            neighbors = region["neighbors"]

            values = []

            for neighbor in neighbors:

                rid = neighbor["region"]

                if rid in self.region_database:

                    values.append(

                        self.region_database[rid]["statistics"]["entropy"]

                    )

            if len(values) == 0:

                stability = 100.0

                variance = 0.0

                deviation = 0.0

            else:

                variance = float(np.var(values))

                deviation = float(np.std(values))

                difference = abs(

                    current_entropy -

                    np.mean(values)

                )

                stability = max(

                    0.0,

                    100.0 -

                    (

                        difference * 10 +

                        deviation * 5

                    )

                )

            region["entropy_stability"] = {

                "stability_index": round(

                    stability,

                    4

                ),

                "variance": variance,

                "standard_deviation": deviation

            }

        self.log("Entropy Stability Completed")


    def compute_entropy_gradient(self):

        self.log("\nComputing Entropy Gradient...\n")

        for region in self.region_database.values():

            current_entropy = region["statistics"]["entropy"]

            neighbors = region["neighbors"]

            higher = 0

            lower = 0

            equal = 0

            gradient = []

            for neighbor in neighbors:

                rid = neighbor["region"]

                if rid not in self.region_database:

                    continue

                entropy = self.region_database[

                    rid

                ]["statistics"]["entropy"]

                diff = entropy - current_entropy

                gradient.append(diff)

                if diff > 0:

                    higher += 1

                elif diff < 0:

                    lower += 1

                else:

                    equal += 1

            if len(gradient):

                average_gradient = float(

                    np.mean(gradient)

                )

            else:

                average_gradient = 0.0

            if average_gradient > 0.20:

                direction = "INCREASING"

            elif average_gradient < -0.20:

                direction = "DECREASING"

            else:

                direction = "STABLE"

            region["entropy_gradient"] = {

                "direction": direction,

                "average_gradient": average_gradient,

                "higher_neighbors": higher,

                "lower_neighbors": lower,

                "equal_neighbors": equal

            }

        self.log("Entropy Gradient Completed")


    def build_entropy_hierarchy(self):

        self.log("\nBuilding Entropy Hierarchy...\n")

        ranking = sorted(

            self.region_database.values(),

            key=lambda region:

            region["statistics"]["entropy"],

            reverse=True

        )

        total = len(ranking)

        hierarchy = []

        for rank, region in enumerate(

            ranking,

            start=1

        ):

            percentage = rank / total

            if percentage <= 0.10:

                level = "ELITE"

            elif percentage <= 0.30:

                level = "HIGH"

            elif percentage <= 0.60:

                level = "MEDIUM"

            elif percentage <= 0.85:

                level = "LOW"

            else:

                level = "VERY_LOW"

            region["entropy_hierarchy"] = {

                "rank": rank,

                "level": level

            }

            hierarchy.append({

                "region_id":

                    region["region_id"],

                "entropy_id":

                    region["entropy_intelligence_id"],

                "entropy":

                    region["statistics"]["entropy"],

                "rank":

                    rank,

                "level":

                    level

            })

        self.image_statistics["entropy_hierarchy"] = hierarchy

        self.log("Entropy Hierarchy Completed")


    def execute_part5_2(self):

        self.compute_entropy_stability()

        self.compute_entropy_gradient()

        self.build_entropy_hierarchy()

        self.log("\n" + "=" * 100)

        self.log("PART 5.2 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def compute_local_energy_intelligence(self):

        self.log("\nComputing Local Energy Intelligence...\n")

        global_energy = self.image_statistics[
            "global_statistics"
        ]["energy"]

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(region)

            values = matrix.astype(
                np.float64
            )

            local_energy = float(
                np.sum(values ** 2)
            )

            normalized = float(
                np.mean(values ** 2)
            )

            rms = float(
                np.sqrt(
                    normalized
                )
            )

            region["energy_intelligence"] = {

                "local_energy":
                    local_energy,

                "normalized_energy":
                    normalized,

                "rms_energy":
                    rms,

                "global_energy_ratio":

                    local_energy /

                    global_energy

                    if global_energy

                    else 0.0

            }

        self.log("Local Energy Intelligence Completed")


    def compute_variance_intelligence(self):

        self.log("\nComputing Variance Intelligence...\n")

        global_variance = self.image_statistics[
            "global_statistics"
        ]["variance"]

        for region in self.region_database.values():

            values = self.extract_region_matrix(
                region
            ).astype(
                np.float64
            ).flatten()

            variance = float(
                np.var(values)
            )

            relative = (

                variance /

                global_variance

                if global_variance

                else 0.0

            )

            mad = float(

                np.median(

                    np.abs(

                        values -

                        np.median(values)

                    )

                )

            )

            region["variance_intelligence"] = {

                "variance":
                    variance,

                "relative_variance":
                    relative,

                "median_absolute_deviation":
                    mad

            }

        self.log("Variance Intelligence Completed")


    def compute_coefficient_density(self):

        self.log("\nComputing Coefficient Density...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(
                region
            )

            total = matrix.size

            non_zero = np.count_nonzero(
                matrix
            )

            positive = np.count_nonzero(
                matrix > 0
            )

            negative = np.count_nonzero(
                matrix < 0
            )

            density = float(
                non_zero / total
            )

            sparsity = float(
                1.0 - density
            )

            region["coefficient_density"] = {

                "total_coefficients":
                    int(total),

                "non_zero":
                    int(non_zero),

                "positive":
                    int(positive),

                "negative":
                    int(negative),

                "density":
                    density,

                "sparsity":
                    sparsity

            }

        self.log("Coefficient Density Completed")


    def compute_position_intelligence(self):

        self.log("\nComputing Position Intelligence...\n")

        rows = self.ll.shape[0]

        cols = self.ll.shape[1]

        for region in self.region_database.values():

            c = region["coordinates"]

            center_x = c["x"] + c["width"] / 2

            center_y = c["y"] + c["height"] / 2

            horizontal = center_x / cols

            vertical = center_y / rows

            priority = self.get_random_integer(
                "position",
                1,
                100
            )

            scan_order = self.get_random_integer(
                "position",
                1,
                4
            )

            embedding_direction = self.get_random_integer(
                "position",
                1,
                8
            )

            start_offset_x = self.get_random_integer(
                "position",
                0,
                c["width"] - 1
            )

            start_offset_y = self.get_random_integer(
                "position",
                0,
                c["height"] - 1
            )

            end_offset_x = self.get_random_integer(
                "position",
                start_offset_x,
                c["width"] - 1
            )

            end_offset_y = self.get_random_integer(
                "position",
                start_offset_y,
                c["height"] - 1
            )

            position_weight = self.random_float(
                "position"
            )

            position_confidence = self.random_float(
                "position"
            )

            region["position_intelligence"] = {

                "center_x": round(center_x,4),

                "center_y": round(center_y,4),

                "horizontal_ratio": round(horizontal,6),

                "vertical_ratio": round(vertical,6),

                "distance_from_center": round(

                    math.sqrt(

                        (center_x-cols/2)**2 +

                        (center_y-rows/2)**2

                    ),

                    6

                ),

                "position_priority": priority,

            "scan_order": scan_order,

            "embedding_direction": embedding_direction,

            "position_weight": position_weight,

            "position_confidence": position_confidence,

            "recommended_start": [

                c["x"] + start_offset_x,

                c["y"] + start_offset_y

            ],

            "recommended_end": [

                c["x"] + end_offset_x,

                c["y"] + end_offset_y

            ]

            }

        self.log("Position Intelligence Completed")

    def compute_gradient_intelligence(self):

        self.log("\nComputing Gradient Intelligence...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(region)

            gx = cv2.Sobel(

                matrix,

                cv2.CV_64F,

                1,

                0,

                ksize=3

            )

            gy = cv2.Sobel(

                matrix,

                cv2.CV_64F,

                0,

                1,

                ksize=3

            )

            magnitude = np.sqrt(

                gx**2 +

                gy**2

            )

            direction = np.arctan2(

                gy,

                gx

            )

            region["gradient_intelligence"] = {

                "average_gradient":

                    float(np.mean(magnitude)),

                "maximum_gradient":

                    float(np.max(magnitude)),

                "minimum_gradient":

                    float(np.min(magnitude)),

                "gradient_variance":

                    float(np.var(magnitude)),

                "gradient_strength":
                    self.get_random_integer("gradient", 1, 100),

                "gradient_priority":
                    self.get_random_integer("gradient", 1, 10),

                "gradient_rank":
                    self.get_random_integer("gradient", 1, 100),

                "edge_threshold":
                    self.get_random_integer("gradient", 5, 50),

                "direction_priority":
                    self.get_random_integer("gradient", 1, 8),

                "gradient_weight":
                    self.random_float("gradient"),

                "adaptive_threshold":
                    self.random_float("gradient"),

                "embedding_weight":
                    self.random_float("gradient"),

                "edge_density":

                    float(

                        np.count_nonzero(

                            magnitude>np.mean(magnitude)

                        )/

                        magnitude.size

                    ),

                "average_direction":

                    float(

                        np.mean(direction)

                    )

            }

        self.log("Gradient Intelligence Completed")

    def compute_noise_intelligence(self):

        self.log("\nComputing Noise Intelligence...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(region)

            blur = cv2.GaussianBlur(

                matrix,

                (3,3),

                0

            )

            noise = matrix.astype(

                np.float64

            )-blur.astype(

                np.float64

            )

            region["noise_intelligence"] = {

                "noise_mean":

                    float(np.mean(noise)),

                "noise_variance":

                    float(np.var(noise)),

                "noise_std":

                    float(np.std(noise)),

                "noise_energy":

                    float(

                        np.sum(

                            noise**2

                        )

                    ),

                "noise_density":

                    float(

                        np.count_nonzero(

                            np.abs(noise)>1

                        )/

                        noise.size

                    ),

                "estimated_noise_level":
                    self.get_random_integer("noise", 1, 100),

                "noise_budget":
                    self.get_random_integer("noise", 1, 255),

                "noise_priority":
                    self.get_random_integer("noise", 1, 10),

                "noise_rank":
                    self.get_random_integer("noise", 1, 100),

                "embedding_budget":
                    self.get_random_integer("noise", 5, 50),

                "filter_strength":
                    self.get_random_integer("noise", 1, 20),

                "adaptive_budget":
                    self.random_float("noise"),

                "noise_weight":
                    self.random_float("noise")

            }

        self.log("Noise Intelligence Completed")


    def execute_part5_3(self):

        self.compute_local_energy_intelligence()

        self.compute_variance_intelligence()

        self.compute_coefficient_density()

        self.compute_position_intelligence()

        self.compute_gradient_intelligence()

        self.compute_noise_intelligence()

        self.log("\n" + "=" * 100)

        self.log("PART 5.3 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def compute_texture_complexity(self):

        self.log("\nComputing Texture Complexity...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(region).astype(np.float64)

            if matrix.shape[1] > 1:
                gx_mean = float(np.mean(np.abs(np.diff(matrix, axis=1))))
            else:
                gx_mean = 0.0

            if matrix.shape[0] > 1:
                gy_mean = float(np.mean(np.abs(np.diff(matrix, axis=0))))
            else:
                gy_mean = 0.0

            gradient_strength = float(gx_mean + gy_mean)

            coefficient_variation = float(
                np.std(matrix) if matrix.size > 0 else 0.0
            )

            complexity = (
                gradient_strength * 0.60 +
                coefficient_variation * 0.40
            )

            region["texture_complexity"] = {
                "gradient_strength": gradient_strength,
                "coefficient_variation": coefficient_variation,
                "complexity_score": complexity
            }

        self.log("Texture Complexity Completed")

    def compute_gradient_consistency(self):

        self.log("\nComputing Gradient Consistency...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(region).astype(np.float64)

            if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                avg_grad = 0.0
                grad_std = 0.0
                consistency = 100.0
            else:
                gx = np.gradient(matrix, axis=1)
                gy = np.gradient(matrix, axis=0)
                magnitude = np.sqrt(gx ** 2 + gy ** 2)

                avg_grad = float(np.mean(magnitude))
                grad_std = float(np.std(magnitude))
                consistency = float(max(0.0, min(100.0 - grad_std, 100.0)))

            region["gradient_consistency"] = {
                "average_gradient": avg_grad,
                "gradient_std": grad_std,
                "consistency_index": consistency
            }

        self.log("Gradient Consistency Completed")

    def compute_edge_consistency(self):

        self.log("\nComputing Edge Consistency...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(region)

            matrix_float = matrix.astype(
                np.float32
            )

            global_min = float(
                np.min(
                    matrix_float
                )
            )

            global_max = float(
                np.max(
                    matrix_float
                )
            )

            if global_max - global_min <= 1e-12:

                normalized = np.zeros_like(
                    matrix_float,
                    dtype=np.uint8
                )

            else:

                normalized = (
                    (
                        matrix_float -
                        global_min
                    ) /
                    (
                        global_max -
                        global_min
                    ) *
                    255.0
                ).astype(
                    np.uint8
                )

            edges = cv2.Canny(

                normalized,

                50,

                150

            )

            total = edges.size

            edge_pixels = np.count_nonzero(edges)

            density = float(

                edge_pixels /

                total

            )

            region["edge_consistency"] = {

                "edge_pixels":

                    int(edge_pixels),

                "edge_density":

                    density,

                "consistency_score":

                    density * 100

            }

        self.log("Edge Consistency Completed")


    def execute_part5_4(self):

        self.compute_texture_complexity()

        self.compute_gradient_consistency()

        self.compute_edge_consistency()

        self.log("\n" + "=" * 100)

        self.log("PART 5.4 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def compute_histogram_similarity(self):

        self.log("\nComputing Histogram Similarity...\n")

        histograms = {}

        for region_id, region in self.region_database.items():

            matrix = self.extract_region_matrix(region)

            matrix = matrix.astype(np.float32)

            histogram, _ = np.histogram(

                matrix.flatten(),

                bins=64,

                density=True

            )

            histograms[region_id] = histogram

        for region_id, region in self.region_database.items():

            similarities = []

            for neighbor in region["neighbors"]:

                rid = neighbor["region"]

                if rid not in histograms:

                    continue

                similarity = np.corrcoef(

                    histograms[region_id],

                    histograms[rid]

                )[0, 1]

                if np.isnan(similarity):

                    similarity = 0.0

                similarities.append(float(similarity))

            if similarities:

                region["histogram_similarity"] = {

                    "average_similarity": float(np.mean(similarities)),

                    "maximum_similarity": float(np.max(similarities)),

                    "minimum_similarity": float(np.min(similarities)),

                    "neighbor_count": len(similarities)

                }

            else:

                region["histogram_similarity"] = {

                    "average_similarity": 0.0,

                    "maximum_similarity": 0.0,

                    "minimum_similarity": 0.0,

                    "neighbor_count": 0

                }

        self.log("Histogram Similarity Completed")


    def compute_coefficient_similarity(self):

        self.log("\nComputing Coefficient Similarity...\n")

        for region_id, region in self.region_database.items():

            matrix = self.extract_region_matrix(region)

            source = matrix.flatten().astype(np.float64)

            similarities = []

            for neighbor in region["neighbors"]:

                rid = neighbor["region"]

                if rid not in self.region_database:

                    continue

                target = self.extract_region_matrix(

                    self.region_database[rid]

                ).flatten().astype(np.float64)

                minimum = min(

                    len(source),

                    len(target)

                )

                if minimum > 1:
                    s_slice = source[:minimum]
                    t_slice = target[:minimum]
                    if np.std(s_slice) > 1e-12 and np.std(t_slice) > 1e-12:
                        corr_val = np.corrcoef(s_slice, t_slice)[0, 1]
                        correlation = 0.0 if np.isnan(corr_val) else float(corr_val)
                    else:
                        correlation = 0.0
                else:
                    correlation = 0.0

                similarities.append(float(correlation))

            if similarities:

                average = float(np.mean(similarities))

            else:

                average = 0.0

            region["coefficient_similarity"] = {

                "average_similarity": average,

                "maximum_similarity":

                    float(max(similarities))

                    if similarities

                    else 0.0,

                "minimum_similarity":

                    float(min(similarities))

                    if similarities

                    else 0.0

            }

        self.log("Coefficient Similarity Completed")


    def compute_frequency_overlap(self):

        self.log("\nComputing Frequency Overlap...\n")

        global_energy = self.image_statistics[

            "global_statistics"

        ]["energy"]

        for region in self.region_database.values():

            energy = region["energy_intelligence"][

                "local_energy"

            ]

            overlap = (

                energy /

                global_energy

                if global_energy

                else 0.0

            )

            overlap = min(

                overlap,

                1.0

            )

            region["frequency_overlap"] = {

                "overlap_index": overlap,

                "percentage": overlap * 100

            }

        self.log("Frequency Overlap Completed")


    def execute_part5_5(self):

        self.compute_histogram_similarity()

        self.compute_coefficient_similarity()

        self.compute_frequency_overlap()

        self.log("\n" + "=" * 100)

        self.log("PART 5.5 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def compute_spectral_similarity(self):

        self.log("\nComputing Spectral Similarity...\n")

        spectra = {}

        for region_id, region in self.region_database.items():

            matrix = self.extract_region_matrix(region).astype(np.float64)

            spectrum = np.abs(

                np.fft.fft2(matrix)

            )

            spectrum = spectrum.flatten()

            norm = np.linalg.norm(spectrum)

            if norm != 0:

                spectrum /= norm

            spectra[region_id] = spectrum

        for region_id, region in self.region_database.items():

            similarities = []

            source = spectra[region_id]

            for neighbor in region["neighbors"]:

                rid = neighbor["region"]

                if rid not in spectra:

                    continue

                target = spectra[rid]

                minimum = min(

                    len(source),

                    len(target)

                )

                similarity = np.dot(

                    source[:minimum],

                    target[:minimum]

                )

                similarities.append(float(similarity))

            if similarities:

                average = float(np.mean(similarities))

                maximum = float(np.max(similarities))

                minimum = float(np.min(similarities))

            else:

                average = 0.0

                maximum = 0.0

                minimum = 0.0

            region["spectral_similarity"] = {

                "average_similarity": average,

                "maximum_similarity": maximum,

                "minimum_similarity": minimum

            }

        self.log("Spectral Similarity Completed")


    def compute_structural_similarity_index(self):

        self.log("\nComputing Structural Similarity Intelligence...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(

                region

            ).astype(np.float64)

            source_mean = np.mean(matrix)

            source_std = np.std(matrix)

            similarities = []

            for neighbor in region["neighbors"]:

                rid = neighbor["region"]

                if rid not in self.region_database:

                    continue

                target = self.extract_region_matrix(

                    self.region_database[rid]

                ).astype(np.float64)

                target_mean = np.mean(target)

                target_std = np.std(target)

                c1 = 0.01

                c2 = 0.03

                similarity = (

                    (

                        2 * source_mean * target_mean + c1

                    )

                    *

                    (

                        2 * source_std * target_std + c2

                    )

                ) / (

                    (

                        source_mean ** 2 +

                        target_mean ** 2 +

                        c1

                    )

                    *

                    (

                        source_std ** 2 +

                        target_std ** 2 +

                        c2

                    )

                )

                similarities.append(float(similarity))

            if similarities:

                region["structural_similarity"] = {

                    "average": float(np.mean(similarities)),

                    "maximum": float(np.max(similarities)),

                    "minimum": float(np.min(similarities))

                }

            else:

                region["structural_similarity"] = {

                    "average": 0.0,

                    "maximum": 0.0,

                    "minimum": 0.0

                }

        self.log("Structural Similarity Completed")


    def compute_region_confidence_index(self):

        self.log("\nComputing Region Confidence Index...\n")

        for region in self.region_database.values():

            confidence = (

                region["priority"]["score"] * 0.25 +

                region["risk"]["detectability_index"] * 100 * 0.25 +

                region["statistics"]["local_randomness"] * 0.20 +

                region["entropy_stability"]["stability_index"] * 0.15 +

                region["gradient_consistency"]["consistency_index"] * 0.15

            )

            confidence = max(

                0.0,

                min(

                    confidence,

                    100.0

                )

            )

            region["confidence_index"] = {

                "confidence_score": round(

                    confidence,

                    4

                ),

                "confidence_level":

                    "VERY_HIGH"

                    if confidence >= 90

                    else

                    "HIGH"

                    if confidence >= 75

                    else

                    "MEDIUM"

                    if confidence >= 55

                    else

                    "LOW"

            }

        self.log("Region Confidence Index Completed")


    def execute_part5_6(self):

        self.compute_spectral_similarity()

        self.compute_structural_similarity_index()

        self.compute_region_confidence_index()

        self.log("\n" + "=" * 100)

        self.log("PART 5.6 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def build_region_correlation_matrix(self):

        self.log("\nBuilding Region Correlation Matrix...\n")

        self.region_correlation_matrix = {}

        cached_vectors = {}
        for r_id, r_data in self.region_database.items():
            vec = self.extract_region_matrix(r_data).astype(np.float64).flatten()
            cached_vectors[r_id] = vec

        for source_id, source_data in self.region_database.items():
            self.region_correlation_matrix[source_id] = {}
            source_vec = cached_vectors[source_id]
            s_len = len(source_vec)

            self.region_correlation_matrix[source_id][source_id] = 1.0

            neighbor_ids = [n["region"] for n in source_data.get("neighbors", [])]

            for target_id in neighbor_ids:
                if target_id not in cached_vectors:
                    continue

                target_vec = cached_vectors[target_id]
                min_len = min(s_len, len(target_vec))

                if min_len > 1:
                    s_sub = source_vec[:min_len]
                    t_sub = target_vec[:min_len]
                    s_std = np.std(s_sub)
                    t_std = np.std(t_sub)

                    if s_std > 1e-12 and t_std > 1e-12:
                        corr_val = np.corrcoef(s_sub, t_sub)[0, 1]
                        correlation = 0.0 if np.isnan(corr_val) else float(corr_val)
                    else:
                        correlation = 0.0
                else:
                    correlation = 0.0

                self.region_correlation_matrix[source_id][target_id] = float(correlation)

        self.image_statistics[
            "region_correlation_matrix"
        ] = self.region_correlation_matrix

        self.log("Region Correlation Matrix Completed")


    def compute_global_correlation_statistics(self):

        self.log("\nComputing Global Correlation Statistics...\n")

        values = []

        for source in self.region_correlation_matrix.values():

            values.extend(

                source.values()

            )

        if not values:
            values = [1.0]

        values = np.array(values, dtype=np.float64)

        self.image_statistics[

            "correlation_statistics"

        ] = {

            "mean":

                float(np.mean(values)),

            "std":

                float(np.std(values)),

            "minimum":

                float(np.min(values)),

            "maximum":

                float(np.max(values))

        }

        self.log("Global Correlation Statistics Completed")


    def compute_neighbor_correlation_statistics(self):

        self.log("\nComputing Neighbor Correlation Statistics...\n")

        for region_id, region in self.region_database.items():

            values = []

            for neighbor in region["neighbors"]:

                nid = neighbor["region"]

                values.append(

                    self.region_correlation_matrix[

                        region_id

                    ][

                        nid

                    ]

                )

            if values:

                region["neighbor_correlation"] = {

                    "average":

                        float(np.mean(values)),

                    "maximum":

                        float(np.max(values)),

                    "minimum":

                        float(np.min(values)),

                    "std":

                        float(np.std(values))

                }

            else:

                region["neighbor_correlation"] = {

                    "average": 0.0,

                    "maximum": 0.0,

                    "minimum": 0.0,

                    "std": 0.0

                }

        self.log("Neighbor Correlation Statistics Completed")



    def execute_part6_1(self):

        self.build_region_correlation_matrix()

        self.compute_neighbor_correlation_statistics()

        self.compute_global_correlation_statistics()

        self.log("\n" + "=" * 100)

        self.log("PART 6.1 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def build_weighted_region_graph(self):

        self.log("\nBuilding Weighted Region Graph...\n")

        self.weighted_region_graph = {}

        for region_id, region in self.region_database.items():

            self.weighted_region_graph[region_id] = []

            source_entropy = region["statistics"]["entropy"]

            source_priority = region["priority"]["score"]

            for neighbor in region["neighbors"]:

                nid = neighbor["region"]

                if nid not in self.region_database:

                    continue

                target = self.region_database[nid]

                target_entropy = target["statistics"]["entropy"]

                target_priority = target["priority"]["score"]

                correlation = self.region_correlation_matrix[
                    region_id
                ][
                    nid
                ]

                entropy_similarity = 1.0 - min(
                    abs(source_entropy - target_entropy) / 8.0,
                    1.0
                )

                priority_similarity = 1.0 - min(
                    abs(source_priority - target_priority) / 100.0,
                    1.0
                )

                edge_weight = (

                    correlation * 0.50 +

                    entropy_similarity * 0.30 +

                    priority_similarity * 0.20

                )

                self.weighted_region_graph[region_id].append({

                    "neighbor": nid,

                    "weight": float(edge_weight),

                    "correlation": float(correlation),

                    "entropy_similarity": float(entropy_similarity),

                    "priority_similarity": float(priority_similarity)

                })

        self.image_statistics[
            "weighted_region_graph"
        ] = self.weighted_region_graph

        self.log("Weighted Region Graph Completed")


    def compute_region_connectivity(self):

        self.log("\nComputing Region Connectivity...\n")

        for region_id, region in self.region_database.items():

            edges = self.weighted_region_graph[region_id]

            if edges:

                weights = [

                    edge["weight"]

                    for edge in edges

                ]

                region["graph_connectivity"] = {

                    "degree":

                        len(edges),

                    "average_weight":

                        float(np.mean(weights)),

                    "maximum_weight":

                        float(np.max(weights)),

                    "minimum_weight":

                        float(np.min(weights))

                }

            else:

                region["graph_connectivity"] = {

                    "degree": 0,

                    "average_weight": 0.0,

                    "maximum_weight": 0.0,

                    "minimum_weight": 0.0

                }

        self.log("Region Connectivity Completed")


    def compute_graph_importance(self):

        self.log("\nComputing Graph Importance...\n")

        for region in self.region_database.values():

            connectivity = region["graph_connectivity"]

            confidence = region["confidence_index"][
                "confidence_score"
            ]

            importance = (

                connectivity["degree"] * 0.30 +

                connectivity["average_weight"] * 35 +

                confidence * 0.35

            )

            region["graph_importance"] = {

                "importance_score":

                    round(importance, 4),

                "importance_level":

                    "VERY_HIGH"

                    if importance >= 85

                    else

                    "HIGH"

                    if importance >= 65

                    else

                    "MEDIUM"

                    if importance >= 45

                    else

                    "LOW"

            }

        self.log("Graph Importance Completed")


    def execute_part6_2(self):

        self.build_weighted_region_graph()

        self.compute_region_connectivity()

        self.compute_graph_importance()

        self.log("\n" + "=" * 100)

        self.log("PART 6.2 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def compute_multi_region_correlation(self):

        self.log("\nComputing Multi Region Correlation...\n")

        for region_id, region in self.region_database.items():

            correlations = []

            visited = set()

            queue = [region_id]

            depth = 2

            current_depth = 0

            while queue and current_depth <= depth:

                next_queue = []

                for node in queue:

                    if node in visited:
                        continue

                    visited.add(node)

                    for edge in self.weighted_region_graph.get(node, []):

                        neighbor = edge["neighbor"]

                        correlations.append(edge["correlation"])

                        if neighbor not in visited:

                            next_queue.append(neighbor)

                queue = next_queue

                current_depth += 1

            if correlations:

                region["multi_region_correlation"] = {

                    "average":

                        float(np.mean(correlations)),

                    "maximum":

                        float(np.max(correlations)),

                    "minimum":

                        float(np.min(correlations)),

                    "std":

                        float(np.std(correlations)),

                    "count":

                        len(correlations)

                }

            else:

                region["multi_region_correlation"] = {

                    "average":0.0,

                    "maximum":0.0,

                    "minimum":0.0,

                    "std":0.0,

                    "count":0

                }

        self.log("Multi Region Correlation Completed")


    def compute_graph_clustering(self):

        self.log("\nComputing Graph Clustering...\n")

        for region_id, region in self.region_database.items():

            neighbors = [

                edge["neighbor"]

                for edge in self.weighted_region_graph[region_id]

            ]

            links = 0

            possible = len(neighbors) * (len(neighbors)-1)

            if possible <= 0:

                coefficient = 0.0

            else:

                for node in neighbors:

                    connected = [

                        x["neighbor"]

                        for x in self.weighted_region_graph[node]

                    ]

                    for other in neighbors:

                        if other in connected:

                            links += 1

                coefficient = links / possible

            region["graph_clustering"] = {

                "neighbor_count":

                    len(neighbors),

                "clustering_coefficient":

                    float(coefficient)

            }

        self.log("Graph Clustering Completed")


    def compute_graph_centrality(self):

        self.log("\nComputing Graph Centrality...\n")

        total_regions = len(self.region_database)

        for region in self.region_database.values():

            degree = region["graph_connectivity"]["degree"]

            avg_weight = region["graph_connectivity"]["average_weight"]

            centrality = (

                degree /

                total_regions

            ) * avg_weight

            region["graph_centrality"] = {

                "centrality_score":

                    float(centrality),

                "normalized":

                    float(

                        centrality *

                        100

                    )

            }

        self.log("Graph Centrality Completed")


    def execute_part6_3(self):

        self.compute_multi_region_correlation()

        self.compute_graph_clustering()

        self.compute_graph_centrality()

        self.log("\n" + "=" * 100)

        self.log("PART 6.3 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def compute_glcm_relationship_intelligence(self):

        self.log("\nComputing GLCM Relationship Intelligence...\n")

        for region in self.region_database.values():

            matrix = self.extract_region_matrix(region)

            normalized = cv2.normalize(
                matrix,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            glcm = graycomatrix(
                normalized,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True
            )

            contrast = float(
                graycoprops(glcm, "contrast")[0, 0]
            )

            homogeneity = float(
                graycoprops(glcm, "homogeneity")[0, 0]
            )

            energy = float(
                graycoprops(glcm, "energy")[0, 0]
            )

            correlation = float(
                graycoprops(glcm, "correlation")[0, 0]
            )

            asm = float(
                graycoprops(glcm, "ASM")[0, 0]
            )

            region["glcm_relationship"] = {

                "contrast": contrast,

                "homogeneity": homogeneity,

                "energy": energy,

                "correlation": correlation,

                "asm": asm

            }

        self.log("GLCM Relationship Intelligence Completed")


    def compute_complexity_intelligence(self):

        self.log("\nComputing Complexity Intelligence...\n")

        for region in self.region_database.values():

            entropy = region["statistics"]["entropy"]

            gradient = region[
                "gradient_consistency"
            ]["average_gradient"]

            variance = region[
                "variance_intelligence"
            ]["relative_variance"]

            texture = region[
                "texture_complexity"
            ]["complexity_score"]

            complexity = (

                entropy * 0.30 +

                gradient * 0.20 +

                variance * 25 +

                texture * 0.25

            )

            region["complexity_intelligence"] = {

                "complexity_score":

                    round(complexity, 4),

                "complexity_level":

                    "VERY_HIGH"

                    if complexity >= 80

                    else

                    "HIGH"

                    if complexity >= 60

                    else

                    "MEDIUM"

                    if complexity >= 40

                    else

                    "LOW"

            }

        self.log("Complexity Intelligence Completed")


    def compute_neighbor_database(self):

        self.log("\nBuilding Neighbor Intelligence Database...\n")

        database = {}

        for region_id, region in self.region_database.items():

            database[region_id] = {

                "neighbor_count":

                    len(region["neighbors"]),

                "neighbors":[

                    {

                        "region":

                            edge["neighbor"],

                        "weight":

                            edge["weight"],

                        "correlation":

                            edge["correlation"]

                    }

                    for edge in self.weighted_region_graph[region_id]

                ]

            }

        self.image_statistics[
            "neighbor_database"
        ] = database

        self.log("Neighbor Database Completed")


    def execute_part6_4(self):

        self.compute_glcm_relationship_intelligence()

        self.compute_complexity_intelligence()

        self.compute_neighbor_database()

        self.log("\n" + "=" * 100)

        self.log("PART 6.4 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def compute_region_risk_index(self):

        self.log("\nComputing Region Risk Index...\n")

        for region in self.region_database.values():

            detectability = region["risk"]["detectability_index"]

            correlation = region[
                "neighbor_correlation"
            ]["average"]

            confidence = region[
                "confidence_index"
            ]["confidence_score"]

            complexity = region[
                "complexity_intelligence"
            ]["complexity_score"]

            graph = region[
                "graph_importance"
            ]["importance_score"]

            risk = (

                (1.0 - detectability) * 100.0 * 0.30 +

                (1.0 - correlation) * 100.0 * 0.20 +

                (100.0 - confidence) * 0.20 +

                complexity * 0.15 +

                graph * 0.15

            )

            risk = max(
                0.0,
                min(
                    risk,
                    100.0
                )
            )

            region["region_risk_index"] = {

                "risk_score":
                    round(risk, 4),

                "risk_level":

                    "VERY_HIGH"

                    if risk >= 85

                    else

                    "HIGH"

                    if risk >= 65

                    else

                    "MEDIUM"

                    if risk >= 45

                    else

                    "LOW"

            }

        self.log("Region Risk Index Completed")


    def compute_detectability_cost_map(self):

        self.log("\nComputing Detectability Cost Map...\n")

        cost_map = {}

        for region_id, region in self.region_database.items():

            entropy = region["statistics"]["entropy"]

            variance = region[
                "variance_intelligence"
            ]["relative_variance"]

            gradient = region[
                "gradient_consistency"
            ]["consistency_index"]

            detectability = region[
                "risk"
            ]["detectability_index"]

            cost = (

                entropy * 0.25 +

                variance * 25 * 0.20 +

                gradient * 0.20 +

                detectability * 100 * 0.35

            )

            cost_map[region_id] = {

                "cost":

                    round(cost, 4),

                "detectability":

                    detectability

            }

        self.image_statistics[
            "detectability_cost_map"
        ] = cost_map

        self.log("Detectability Cost Map Completed")


    def compute_statistical_summary(self):

        self.log("\nComputing Statistical Summary...\n")

        entropy = []

        confidence = []

        complexity = []

        risk = []

        for region in self.region_database.values():

            entropy.append(
                region["statistics"]["entropy"]
            )

            confidence.append(
                region["confidence_index"]["confidence_score"]
            )

            complexity.append(
                region["complexity_intelligence"]["complexity_score"]
            )

            risk.append(
                region["region_risk_index"]["risk_score"]
            )

        self.image_statistics[
            "advanced_statistics"
        ] = {

            "entropy":{

                "mean":float(np.mean(entropy)),
                "std":float(np.std(entropy)),
                "min":float(np.min(entropy)),
                "max":float(np.max(entropy))

            },

            "confidence":{

                "mean":float(np.mean(confidence)),
                "std":float(np.std(confidence))

            },

            "complexity":{

                "mean":float(np.mean(complexity)),
                "std":float(np.std(complexity))

            },

            "risk":{

                "mean":float(np.mean(risk)),
                "std":float(np.std(risk))

            }

        }

        self.log("Statistical Summary Completed")


    def execute_part6_5(self):

        self.compute_region_risk_index()

        self.compute_detectability_cost_map()

        self.compute_statistical_summary()

        self.log("\n" + "=" * 100)

        self.log("PART 6.5 COMPLETED")

        self.log("=" * 100)

        self.save_report()

    def build_feature_dataset(self):

        self.log("\nBuilding Feature Dataset...\n")

        feature_dataset = []

        export_bands = {

            "LH",

            "HL",

            "HH"

        }

        for region in self.region_database.values():

            if region["band"].upper() not in export_bands:

                continue

            coordinates = region["coordinates"]

            statistics = region["statistics"]

            relative = region["relative_statistics"]

            classification = region["classification"]

            local_entropy = region["local_entropy"]

            neighbor_entropy = region["neighbor_entropy"]

            multi_entropy = region["multi_scale_entropy"]

            stability = region["entropy_stability"]

            gradient = region["entropy_gradient"]

            hierarchy = region["entropy_hierarchy"]

            energy = region["energy_intelligence"]

            variance = region["variance_intelligence"]

            density = region["coefficient_density"]

            texture = region["texture_complexity"]

            gradient_consistency = region["gradient_consistency"]

            edge = region["edge_consistency"]

            histogram = region["histogram_similarity"]

            coefficient = region["coefficient_similarity"]

            frequency = region["frequency_overlap"]

            spectral = region["spectral_similarity"]

            structural = region["structural_similarity"]

            glcm = region["glcm_relationship"]

            confidence = region["confidence_index"]

            graph_connectivity = region["graph_connectivity"]

            graph_importance = region["graph_importance"]

            graph_centrality = region["graph_centrality"]

            graph_clustering = region["graph_clustering"]

            complexity = region["complexity_intelligence"]

            risk = region["region_risk_index"]

            record = {

                "entropy_intelligence_id":
                    region["entropy_intelligence_id"],

                "region_id":
                    region["region_id"],

                "band":
                    region["band"],

                "coordinates": {

                    "x":
                        coordinates["x"],

                    "y":
                        coordinates["y"],

                    "width":
                        coordinates["width"],

                    "height":
                        coordinates["height"]

                },

                "feature_vector": {

                    "mean":
                        statistics["mean"],

                    "median":
                        statistics["median"],

                    "variance":
                        statistics["variance"],

                    "standard_deviation":
                        statistics["std"],

                    "minimum":
                        statistics["minimum"],

                    "maximum":
                        statistics["maximum"],

                    "dynamic_range":
                        statistics["dynamic_range"],

                    "entropy":
                        statistics["entropy"],

                    "entropy_utilization":
                        statistics["entropy_utilization"],

                    "local_randomness":
                        statistics["local_randomness"],

                    "mad":
                        statistics["mad"],

                    "rms":
                        statistics["rms"],

                    "energy":
                        statistics["energy"],

                    "density":
                        statistics["density"],

                    "sparsity":
                        statistics["sparsity"],

                    "skewness":
                        statistics["skewness"],

                    "kurtosis":
                        statistics["kurtosis"],

                    "entropy_ratio":
                        relative["entropy_ratio"],

                    "mean_ratio":
                        relative["mean_ratio"],

                    "std_ratio":
                        relative["std_ratio"],

                    "energy_ratio":
                        relative["energy_ratio"],

                    "entropy_difference":
                        relative["entropy_difference"],

                    "mean_difference":
                        relative["mean_difference"],

                    "std_difference":
                        relative["std_difference"],

                    "local_entropy":
                        local_entropy["average"],

                    "neighbor_entropy":
                        neighbor_entropy["neighbor_average"],

                    "neighbor_entropy_std":
                        neighbor_entropy["neighbor_std"],

                    "five_neighbor_entropy":
                        multi_entropy["five_neighbor_average"],

                    "ten_neighbor_entropy":
                        multi_entropy["ten_neighbor_average"],

                    "band_entropy":
                        multi_entropy["band_average"],

                    "global_entropy":
                        multi_entropy["global_average"],

                    "relative_entropy":
                        multi_entropy["relative_entropy"],

                    "entropy_deviation":
                        multi_entropy["entropy_deviation"],

                    "entropy_stability":
                        stability["stability_index"],

                    "entropy_gradient":
                        gradient["average_gradient"],

                    "local_energy":
                        energy["local_energy"],

                    "normalized_energy":
                        energy["normalized_energy"],

                    "global_energy_ratio":
                        energy["global_energy_ratio"],

                    "relative_variance":
                        variance["relative_variance"],

                    "median_absolute_deviation":
                        variance["median_absolute_deviation"],

                    "coefficient_density":
                        density["density"],

                    "texture_complexity":
                        texture["complexity_score"],

                    "gradient_consistency":
                        gradient_consistency["consistency_index"],

                    "edge_density":
                        edge["edge_density"],

                    "edge_consistency":
                        edge["consistency_score"],

                    "histogram_similarity":
                        histogram["average_similarity"],

                    "coefficient_similarity":
                        coefficient["average_similarity"],

                    "frequency_overlap":
                        frequency["overlap_index"],

                    "spectral_similarity":
                        spectral["average_similarity"],

                    "structural_similarity":
                        structural["average"],

                    "glcm_contrast":
                        glcm["contrast"],

                    "glcm_homogeneity":
                        glcm["homogeneity"],

                    "glcm_energy":
                        glcm["energy"],

                    "glcm_correlation":
                        glcm["correlation"],

                    "graph_degree":
                        graph_connectivity["degree"],

                    "graph_average_weight":
                        graph_connectivity["average_weight"],

                    "graph_importance":
                        graph_importance["importance_score"],

                    "graph_centrality":
                        graph_centrality["normalized"],

                    "graph_clustering":
                        graph_clustering["clustering_coefficient"],

                    "complexity":
                        complexity["complexity_score"],

                    "confidence":
                        confidence["confidence_score"],

                    "risk":
                        risk["risk_score"]

                },

                "labels": {

                    "entropy_level":
                        classification["entropy_level"],

                    "randomness_level":
                        classification["randomness_level"],

                    "texture_level":
                        classification["texture_level"],

                    "hierarchy":
                        hierarchy["level"],

                    "confidence":
                        confidence["confidence_level"],

                    "complexity":
                        complexity["complexity_level"],

                    "risk":
                        risk["risk_level"]

                }

            }

            feature_dataset.append(record)

        self.feature_dataset = feature_dataset

        self.save_json(

            self.module_root /

            "feature_dataset.json",

            feature_dataset

        )

        prediction_features = {}

        for item in self.feature_dataset:

            prediction_features[

                item["region_id"]

            ] = item

        self.save_json(

            self.module_root /

            "prediction_features.json",

            prediction_features

        )

        self.log(

            "Prediction Features Generated Successfully"

        )

        self.log(

            f"Feature Dataset Generated : {len(feature_dataset)} Records"

        )

    def normalize_feature_dataset(self):

        self.log("\nNormalizing Feature Dataset...\n")

        if not hasattr(self, "feature_dataset"):

            raise RuntimeError(
                "Feature dataset not available."
            )

        normalized_dataset = []

        excluded = {

            "entropy_intelligence_id",

            "region_id",

            "band",

            "coordinates",

            "labels"

        }

        feature_names = []

        feature_values = {}

        sample = self.feature_dataset[0]

        for key in sample["feature_vector"]:

            feature_names.append(key)

            feature_values[key] = []

        for record in self.feature_dataset:

            for feature in feature_names:

                feature_values[feature].append(

                    float(

                        record["feature_vector"][feature]

                    )

                )

        feature_minimum = {}

        feature_maximum = {}

        for feature in feature_names:

            feature_minimum[feature] = min(

                feature_values[feature]

            )

            feature_maximum[feature] = max(

                feature_values[feature]

            )

        for record in self.feature_dataset:

            normalized_vector = {}

            for feature in feature_names:

                value = float(

                    record["feature_vector"][feature]

                )

                minimum = feature_minimum[feature]

                maximum = feature_maximum[feature]

                if maximum == minimum:

                    normalized = 0.0

                else:

                    normalized = (

                        value - minimum

                    ) / (

                        maximum - minimum

                    )

                normalized_vector[feature] = round(

                    normalized,

                    8

                )

            normalized_dataset.append({

                "entropy_intelligence_id":

                    record["entropy_intelligence_id"],

                "region_id":

                    record["region_id"],

                "band":

                    record["band"],

                "coordinates":

                    record["coordinates"],

                "normalized_feature_vector":

                    normalized_vector,

                "labels":

                    record["labels"]

            })

        self.normalized_dataset = normalized_dataset

        self.normalization_information = {

            feature: {

                "minimum":

                    feature_minimum[feature],

                "maximum":

                    feature_maximum[feature]

            }

            for feature in feature_names

        }

        self.save_json(

            self.module_root /

            "normalized_dataset.json",

            normalized_dataset

        )

        self.save_json(

            self.module_root /

            "normalization_information.json",

            self.normalization_information

        )

        self.log(

            f"Normalized Dataset Generated : {len(normalized_dataset)} Records"

        )

    def generate_feature_statistics(self):

        self.log("\nGenerating Feature Statistics...\n")

        if not hasattr(self, "feature_dataset"):

            raise RuntimeError(
                "Feature dataset not available."
            )

        import math

        feature_statistics = {}

        feature_names = list(

            self.feature_dataset[0][
                "feature_vector"
            ].keys()

        )

        for feature in feature_names:

            values = [

                float(

                    record["feature_vector"][feature]

                )

                for record in self.feature_dataset

            ]

            count = len(values)

            minimum = min(values)

            maximum = max(values)

            mean = sum(values) / count

            sorted_values = sorted(values)

            if count % 2 == 0:

                median = (

                    sorted_values[count // 2 - 1]

                    +

                    sorted_values[count // 2]

                ) / 2

            else:

                median = sorted_values[count // 2]

            variance = sum(

                (

                    x - mean

                ) ** 2

                for x in values

            ) / count

            std = math.sqrt(variance)

            value_range = maximum - minimum

            if std == 0:

                coefficient_variation = 0.0

            else:

                coefficient_variation = std / mean if mean != 0 else 0.0

            feature_statistics[feature] = {

                "count":
                    count,

                "minimum":
                    round(minimum, 8),

                "maximum":
                    round(maximum, 8),

                "range":
                    round(value_range, 8),

                "mean":
                    round(mean, 8),

                "median":
                    round(median, 8),

                "variance":
                    round(variance, 8),

                "standard_deviation":
                    round(std, 8),

                "coefficient_of_variation":
                    round(coefficient_variation, 8)

            }

        band_statistics = {}

        bands = [

            "LH",

            "HL",

            "HH"

        ]

        for band in bands:

            records = [

                record

                for record in self.feature_dataset

                if record["band"] == band

            ]

            band_statistics[band] = {

                "regions":
                    len(records)

            }

        dataset_statistics = {

            "total_regions":

                len(self.feature_dataset),

            "bands":

                band_statistics,

            "total_features":

                len(feature_names),

            "feature_statistics":

                feature_statistics

        }

        self.feature_statistics = dataset_statistics

        self.save_json(

            self.module_root /

            "feature_statistics.json",

            dataset_statistics

        )

        self.log(

            f"Feature Statistics Generated : {len(feature_names)} Features"

        )

    def validate_feature_dataset(self):

        self.log("\nValidating Feature Dataset...\n")

        if not hasattr(self, "feature_dataset"):

            raise RuntimeError(
                "Feature dataset not available."
            )

        import math

        validation = {

            "dataset_valid": True,

            "total_records": len(self.feature_dataset),

            "validated_records": 0,

            "invalid_records": 0,

            "duplicate_region_ids": [],

            "missing_values": [],

            "nan_values": [],

            "infinite_values": [],

            "invalid_coordinates": [],

            "invalid_bands": [],

            "invalid_feature_ranges": []

        }

        region_ids = set()

        valid_bands = {

            "LH",

            "HL",

            "HH"

        }

        for record in self.feature_dataset:

            region_id = record["region_id"]

            if region_id in region_ids:

                validation["duplicate_region_ids"].append(

                    region_id

                )

            else:

                region_ids.add(

                    region_id

                )

            if record["band"] not in valid_bands:

                validation["invalid_bands"].append(

                    region_id

                )

            coordinates = record["coordinates"]

            if (

                coordinates["x"] < 0 or

                coordinates["y"] < 0 or

                coordinates["width"] <= 0 or

                coordinates["height"] <= 0

            ):

                validation["invalid_coordinates"].append(

                    region_id

                )

            for feature, value in record["feature_vector"].items():

                if value is None:

                    validation["missing_values"].append({

                        "region_id": region_id,

                        "feature": feature

                    })

                    continue

                if isinstance(value, float):

                    if math.isnan(value):

                        validation["nan_values"].append({

                            "region_id": region_id,

                            "feature": feature

                        })

                        continue

                    if math.isinf(value):

                        validation["infinite_values"].append({

                            "region_id": region_id,

                            "feature": feature

                        })

                        continue

                if not isinstance(

                    value,

                    (int, float)

                ):

                    validation["invalid_feature_ranges"].append({

                        "region_id": region_id,

                        "feature": feature,

                        "value": value

                    })

                    continue

                if abs(float(value)) > 1e12:

                    validation["invalid_feature_ranges"].append({

                        "region_id": region_id,

                        "feature": feature,

                        "value": value

                    })

            validation["validated_records"] += 1

        validation["invalid_records"] = (

            len(validation["duplicate_region_ids"])

            +

            len(validation["missing_values"])

            +

            len(validation["nan_values"])

            +

            len(validation["infinite_values"])

            +

            len(validation["invalid_coordinates"])

            +

            len(validation["invalid_bands"])

            +

            len(validation["invalid_feature_ranges"])

        )

        if validation["invalid_records"] > 0:

            validation["dataset_valid"] = False

        validation["summary"] = {

            "duplicate_region_ids":

                len(validation["duplicate_region_ids"]),

            "missing_values":

                len(validation["missing_values"]),

            "nan_values":

                len(validation["nan_values"]),

            "infinite_values":

                len(validation["infinite_values"]),

            "invalid_coordinates":

                len(validation["invalid_coordinates"]),

            "invalid_bands":

                len(validation["invalid_bands"]),

            "invalid_feature_ranges":

                len(validation["invalid_feature_ranges"])

        }

        self.dataset_validation = validation

        self.save_json(

            self.module_root /

            "dataset_validation.json",

            validation

        )

        self.log(

            f"Dataset Validation Completed : {validation['dataset_valid']}"

        )

    def generate_dataset_manifest(self):

        self.log("\nGenerating Dataset Manifest...\n")

        if not hasattr(self, "feature_dataset"):

            raise RuntimeError(
                "Feature dataset not available."
            )

        if not hasattr(self, "normalized_dataset"):

            raise RuntimeError(
                "Normalized dataset not available."
            )

        if not hasattr(self, "feature_statistics"):

            raise RuntimeError(
                "Feature statistics not available."
            )

        if not hasattr(self, "dataset_validation"):

            raise RuntimeError(
                "Dataset validation not available."
            )

        feature_names = list(

            self.feature_dataset[0][
                "feature_vector"
            ].keys()

        )

        manifest = {

            "module":

                "Entropy Intelligence Engine",

            "module_version":

                "7.0",

            "dataset_name":

                "Feature Dataset",

            "dataset_type":

                "Machine Learning Feature Repository",

            "total_records":

                len(self.feature_dataset),

            "normalized_records":

                len(self.normalized_dataset),

            "bands": [

                "LH",

                "HL",

                "HH"

            ],

            "feature_count":

                len(feature_names),

            "feature_names":

                feature_names,

            "statistics_file":

                "feature_statistics.json",

            "dataset_file":

                "feature_dataset.json",

            "normalized_dataset_file":

                "normalized_dataset.json",

            "normalization_information_file":

                "normalization_information.json",

            "validation_file":

                "dataset_validation.json",

            "dataset_validation":

                self.dataset_validation["dataset_valid"],

            "validated_records":

                self.dataset_validation["validated_records"],

            "invalid_records":

                self.dataset_validation["invalid_records"],

            "machine_learning_ready":

                self.dataset_validation["dataset_valid"],

            "supported_models": [

                "Linear Regression",

                "Quantum GAN",

                "Genetic Algorithm",

                "Ant Colony Optimization",

                "Adaptive Embedding"

            ],

            "generated_outputs": [

                "feature_dataset.json",

                "prediction_features.json",

                "normalized_dataset.json",

                "normalization_information.json",

                "feature_statistics.json",

                "dataset_validation.json",

                "dataset_manifest.json"

            ]

        }

        self.dataset_manifest = manifest

        self.save_json(

            self.module_root /

            "dataset_manifest.json",

            manifest

        )

        self.log(

            "Dataset Manifest Generated Successfully"

        )

    def estimate_embedding_distortion(self):

        self.log("\nEstimating Embedding Distortion...\n")

        if not hasattr(self, "feature_dataset"):

            raise RuntimeError(
                "Feature dataset not available."
            )

        distortion_database = []

        global_statistics = self.image_statistics["global_statistics"]

        image_entropy = global_statistics["entropy"]

        image_variance = global_statistics["variance"]

        image_energy = global_statistics["energy"]

        for record in self.feature_dataset:

            features = record["feature_vector"]

            entropy = features["entropy"]

            variance = features["variance"]

            energy = features["energy"]

            texture = features["texture_complexity"]

            coefficient_density = features["coefficient_density"]

            edge_density = features["edge_density"]

            confidence = features["confidence"]

            risk = features["risk"]

            entropy_ratio = entropy / image_entropy if image_entropy > 0 else 0.0

            variance_ratio = variance / image_variance if image_variance > 0 else 0.0

            energy_ratio = energy / image_energy if image_energy > 0 else 0.0

            estimated_entropy_change = (

                abs(1.0 - entropy_ratio) * 0.30 +

                variance_ratio * 0.20 +

                coefficient_density * 0.15 +

                texture * 0.15 +

                edge_density * 0.10 +

                risk * 0.10

            )

            estimated_mse = (

                variance_ratio * 0.40 +

                coefficient_density * 0.30 +

                texture * 0.20 +

                edge_density * 0.10

            )

            estimated_psnr_loss = (

                estimated_mse * 8.0

            )

            estimated_ssim_loss = (

                estimated_mse * 0.08

            )

            distortion_score = (

                estimated_psnr_loss * 0.25 +

                estimated_ssim_loss * 25.0 +

                estimated_entropy_change * 40.0

            )

            embedding_cost = (

                distortion_score * (1.0 + risk)

            )

            distortion_confidence = (

                confidence *

                (1.0 - min(risk, 1.0))

            )

            distortion_database.append({

                "region_id":

                    record["region_id"],

                "band":

                    record["band"],

                "estimated_distortion": {

                    "estimated_psnr_loss":

                        round(

                            estimated_psnr_loss,

                            8

                        ),

                    "estimated_ssim_loss":

                        round(

                            estimated_ssim_loss,

                            8

                        ),

                    "estimated_entropy_change":

                        round(

                            estimated_entropy_change,

                            8

                        ),

                    "estimated_mse":

                        round(

                            estimated_mse,

                            8

                        ),

                    "distortion_score":

                        round(

                            distortion_score,

                            8

                        ),

                    "embedding_cost":

                        round(

                            embedding_cost,

                            8

                        ),

                    "distortion_confidence":

                        round(

                            distortion_confidence,

                            8

                        )

                }

            })

        self.distortion_database = distortion_database

        self.save_json(

            self.module_root /

            "distortion_database.json",

            distortion_database

        )

        self.log(

            f"Embedding Distortion Database Generated : {len(distortion_database)} Regions"

        )

    def generate_distortion_statistics(self):

        self.log("\nGenerating Distortion Statistics...\n")

        if not hasattr(self, "distortion_database"):

            raise RuntimeError(
                "Distortion database not available."
            )

        import math

        metrics = {

            "estimated_psnr_loss": [],

            "estimated_ssim_loss": [],

            "estimated_entropy_change": [],

            "estimated_mse": [],

            "distortion_score": [],

            "embedding_cost": [],

            "distortion_confidence": []

        }

        for record in self.distortion_database:

            distortion = record["estimated_distortion"]

            for metric in metrics:

                metrics[metric].append(

                    float(

                        distortion[metric]

                    )

                )

        statistics = {}

        for metric, values in metrics.items():

            count = len(values)

            minimum = min(values)

            maximum = max(values)

            mean = sum(values) / count

            variance = sum(

                (

                    value - mean

                ) ** 2

                for value in values

            ) / count

            standard_deviation = math.sqrt(

                variance

            )

            sorted_values = sorted(

                values

            )

            if count % 2 == 0:

                median = (

                    sorted_values[count // 2 - 1]

                    +

                    sorted_values[count // 2]

                ) / 2

            else:

                median = sorted_values[count // 2]

            statistics[metric] = {

                "minimum":

                    round(

                        minimum,

                        8

                    ),

                "maximum":

                    round(

                        maximum,

                        8

                    ),

                "mean":

                    round(

                        mean,

                        8

                    ),

                "median":

                    round(

                        median,

                        8

                    ),

                "variance":

                    round(

                        variance,

                        8

                    ),

                "standard_deviation":

                    round(

                        standard_deviation,

                        8

                    )

            }

        risk_distribution = {

            "very_low": 0,

            "low": 0,

            "medium": 0,

            "high": 0,

            "very_high": 0

        }

        for record in self.distortion_database:

            score = record["estimated_distortion"][

                "distortion_score"

            ]

            if score < 10:

                risk_distribution["very_low"] += 1

            elif score < 20:

                risk_distribution["low"] += 1

            elif score < 30:

                risk_distribution["medium"] += 1

            elif score < 40:

                risk_distribution["high"] += 1

            else:

                risk_distribution["very_high"] += 1

        distortion_statistics = {

            "total_regions":

                len(

                    self.distortion_database

                ),

            "metric_statistics":

                statistics,

            "distortion_distribution":

                risk_distribution

        }

        self.distortion_statistics = distortion_statistics

        self.save_json(

            self.module_root /

            "distortion_statistics.json",

            distortion_statistics

        )

        self.log(

            "Distortion Statistics Generated Successfully"

        )

    def classify_distortion_levels(self):

        self.log("\nClassifying Distortion Levels...\n")

        if not hasattr(self, "distortion_database"):

            raise RuntimeError(
                "Distortion database not available."
            )

        classified_database = []

        for record in self.distortion_database:

            distortion = record["estimated_distortion"]

            score = distortion["distortion_score"]

            confidence = distortion["distortion_confidence"]

            if score < 10:

                distortion_level = "Very Low"

            elif score < 20:

                distortion_level = "Low"

            elif score < 30:

                distortion_level = "Moderate"

            elif score < 40:

                distortion_level = "High"

            else:

                distortion_level = "Very High"

            if confidence >= 0.90:

                confidence_level = "Very High"

            elif confidence >= 0.75:

                confidence_level = "High"

            elif confidence >= 0.50:

                confidence_level = "Moderate"

            elif confidence >= 0.25:

                confidence_level = "Low"

            else:

                confidence_level = "Very Low"

            if score < 15 and confidence >= 0.75:

                embedding_recommendation = "Highly Suitable"

            elif score < 25 and confidence >= 0.50:

                embedding_recommendation = "Suitable"

            elif score < 35:

                embedding_recommendation = "Use With Caution"

            else:

                embedding_recommendation = "Avoid"

            classified_database.append({

                "region_id":

                    record["region_id"],

                "band":

                    record["band"],

                "estimated_distortion":

                    distortion,

                "classification": {

                    "distortion_level":

                        distortion_level,

                    "confidence_level":

                        confidence_level,

                    "embedding_recommendation":

                        embedding_recommendation

                }

            })

        self.distortion_database = classified_database

        self.save_json(

            self.module_root /

            "distortion_classification.json",

            classified_database

        )

        self.log(

            f"Distortion Classification Completed : {len(classified_database)} Regions"

        )

    def generate_distortion_manifest(self):

        self.log("\nGenerating Distortion Manifest...\n")

        if not hasattr(self, "distortion_database"):

            raise RuntimeError(
                "Distortion database not available."
            )

        if not hasattr(self, "distortion_statistics"):

            raise RuntimeError(
                "Distortion statistics not available."
            )

        total_regions = len(

            self.distortion_database

        )

        recommendation_summary = {

            "Highly Suitable": 0,

            "Suitable": 0,

            "Use With Caution": 0,

            "Avoid": 0

        }

        distortion_summary = {

            "Very Low": 0,

            "Low": 0,

            "Moderate": 0,

            "High": 0,

            "Very High": 0

        }

        confidence_summary = {

            "Very Low": 0,

            "Low": 0,

            "Moderate": 0,

            "High": 0,

            "Very High": 0

        }

        for record in self.distortion_database:

            classification = record["classification"]

            recommendation_summary[

                classification["embedding_recommendation"]

            ] += 1

            distortion_summary[

                classification["distortion_level"]

            ] += 1

            confidence_summary[

                classification["confidence_level"]

            ] += 1

        manifest = {

            "module":

                "Embedding Distortion Intelligence",

            "module_version":

                "8.0",

            "total_regions":

                total_regions,

            "database_file":

                "distortion_database.json",

            "classification_file":

                "distortion_classification.json",

            "statistics_file":

                "distortion_statistics.json",

            "supported_metrics": [

                "estimated_psnr_loss",

                "estimated_ssim_loss",

                "estimated_entropy_change",

                "estimated_mse",

                "distortion_score",

                "embedding_cost",

                "distortion_confidence"

            ],

            "distortion_distribution":

                distortion_summary,

            "confidence_distribution":

                confidence_summary,

            "embedding_recommendation_distribution":

                recommendation_summary,

            "generated_outputs": [

                "distortion_database.json",

                "distortion_statistics.json",

                "distortion_classification.json",

                "distortion_manifest.json"

            ]

        }

        self.distortion_manifest = manifest

        self.save_json(

            self.module_root /

            "distortion_manifest.json",

            manifest

        )

        self.log(

            "Distortion Manifest Generated Successfully"

        )

    def generate_region_combination_candidates(self):

        self.log("\nGenerating Region Combination Candidates...\n")

        if not hasattr(self, "feature_dataset"):

            raise RuntimeError(
                "Feature dataset not available."
            )

        if not hasattr(self, "distortion_database"):

            raise RuntimeError(
                "Distortion database not available."
            )

        distortion_lookup = {}

        for item in self.distortion_database:

            distortion_lookup[

                item["region_id"]

            ] = item

        candidate_database = []

        region_lookup = {}

        for record in self.feature_dataset:

            region_lookup[

                record["region_id"]

            ] = record

        region_ids = list(

            region_lookup.keys()

        )

        for record in self.feature_dataset:

            source_id = record["region_id"]

            source = record["feature_vector"]

            source_coordinates = record["coordinates"]

            source_band = record["band"]

            candidates = []

            for candidate_id in region_ids:

                if candidate_id == source_id:

                    continue

                target = region_lookup[

                    candidate_id

                ]

                target_features = target["feature_vector"]

                target_coordinates = target["coordinates"]

                target_band = target["band"]

                if target_band == source_band:

                    continue

                dx = abs(

                    source_coordinates["x"]

                    -

                    target_coordinates["x"]

                )

                dy = abs(

                    source_coordinates["y"]

                    -

                    target_coordinates["y"]

                )

                if dx < source_coordinates["width"] and dy < source_coordinates["height"]:

                    continue

                entropy_gap = abs(

                    source["entropy"]

                    -

                    target_features["entropy"]

                )

                texture_gap = abs(

                    source["texture_complexity"]

                    -

                    target_features["texture_complexity"]

                )

                correlation_gap = abs(

                    source["coefficient_similarity"]

                    -

                    target_features["coefficient_similarity"]

                )

                confidence = (

                    source["confidence"]

                    +

                    target_features["confidence"]

                ) / 2.0

                distortion = (

                    distortion_lookup[source_id]["estimated_distortion"]["distortion_score"]

                    +

                    distortion_lookup[candidate_id]["estimated_distortion"]["distortion_score"]

                ) / 2.0

                compatibility = (

                    confidence * 35.0 +

                    entropy_gap * 20.0 +

                    texture_gap * 15.0 +

                    (1.0 - correlation_gap) * 15.0 +

                    (100.0 - distortion) * 15.0

                )

                candidates.append({

                    "region_id":

                        candidate_id,

                    "band":

                        target_band,

                    "compatibility_score":

                        round(

                            compatibility,

                            8

                        ),

                    "entropy_difference":

                        round(

                            entropy_gap,

                            8

                        ),

                    "texture_difference":

                        round(

                            texture_gap,

                            8

                        ),

                    "correlation_difference":

                        round(

                            correlation_gap,

                            8

                        ),

                    "average_distortion":

                        round(

                            distortion,

                            8

                        )

                })

            candidates.sort(

                key=lambda x:

                x["compatibility_score"],

                reverse=True

            )

            candidate_database.append({

                "source_region":

                    source_id,

                "source_band":

                    source_band,

                "candidate_count":

                    len(candidates),

                "candidate_regions":

                    candidates[:15]

            })

        self.region_combination_database = candidate_database

        self.save_json(

            self.module_root /

            "region_combination_database.json",

            candidate_database

        )

        self.log(

            f"Region Combination Database Generated : {len(candidate_database)} Source Regions"

        )

    def validate_region_combinations(self):

        self.log("\nValidating Region Combination Candidates...\n")

        if not hasattr(self, "region_combination_database"):

            raise RuntimeError(
                "Region combination database not available."
            )

        validation = {

            "total_source_regions": 0,

            "total_candidate_groups": 0,

            "valid_candidates": 0,

            "invalid_candidates": 0,

            "duplicate_candidates": [],

            "same_band_candidates": [],

            "invalid_scores": [],

            "empty_candidate_lists": [],

            "validation_passed": True

        }

        for entry in self.region_combination_database:

            validation["total_source_regions"] += 1

            source_region = entry["source_region"]

            source_band = entry["source_band"]

            candidates = entry["candidate_regions"]

            if len(candidates) == 0:

                validation["empty_candidate_lists"].append(

                    source_region

                )

            seen = set()

            for candidate in candidates:

                validation["total_candidate_groups"] += 1

                region_id = candidate["region_id"]

                if region_id in seen:

                    validation["duplicate_candidates"].append({

                        "source_region":

                            source_region,

                        "candidate_region":

                            region_id

                    })

                else:

                    seen.add(

                        region_id

                    )

                if candidate["band"] == source_band:

                    validation["same_band_candidates"].append({

                        "source_region":

                            source_region,

                        "candidate_region":

                            region_id

                    })

                score = candidate["compatibility_score"]

                if not isinstance(

                    score,

                    (int, float)

                ):

                    validation["invalid_scores"].append({

                        "source_region":

                            source_region,

                        "candidate_region":

                            region_id

                    })

                    continue

                validation["valid_candidates"] += 1

        validation["invalid_candidates"] = (

            len(validation["duplicate_candidates"])

            +

            len(validation["same_band_candidates"])

            +

            len(validation["invalid_scores"])

            +

            len(validation["empty_candidate_lists"])

        )

        if validation["invalid_candidates"] > 0:

            validation["validation_passed"] = False

        self.region_combination_validation = validation

        self.save_json(

            self.module_root /

            "region_combination_validation.json",

            validation

        )

        self.log(

            "Region Combination Validation Completed"

        )

    def generate_region_combination_statistics(self):

        self.log("\nGenerating Region Combination Statistics...\n")

        if not hasattr(self, "region_combination_database"):

            raise RuntimeError(
                "Region combination database not available."
            )

        import math

        candidate_counts = []

        compatibility_scores = []

        entropy_gaps = []

        texture_gaps = []

        correlation_gaps = []

        distortion_scores = []

        band_pairs = {}

        for entry in self.region_combination_database:

            source_band = entry["source_band"]

            candidate_counts.append(

                entry["candidate_count"]

            )

            for candidate in entry["candidate_regions"]:

                compatibility_scores.append(

                    candidate["compatibility_score"]

                )

                entropy_gaps.append(

                    candidate["entropy_difference"]

                )

                texture_gaps.append(

                    candidate["texture_difference"]

                )

                correlation_gaps.append(

                    candidate["correlation_difference"]

                )

                distortion_scores.append(

                    candidate["average_distortion"]

                )

                pair = f"{source_band}->{candidate['band']}"

                band_pairs[pair] = band_pairs.get(

                    pair,

                    0

                ) + 1

        def calculate_statistics(values):

            if len(values) == 0:

                return {

                    "minimum": 0,

                    "maximum": 0,

                    "mean": 0,

                    "median": 0,

                    "standard_deviation": 0

                }

            values = sorted(values)

            count = len(values)

            mean = sum(values) / count

            variance = sum(

                (x - mean) ** 2

                for x in values

            ) / count

            if count % 2 == 0:

                median = (

                    values[count // 2 - 1]

                    +

                    values[count // 2]

                ) / 2

            else:

                median = values[count // 2]

            return {

                "minimum":

                    round(values[0], 8),

                "maximum":

                    round(values[-1], 8),

                "mean":

                    round(mean, 8),

                "median":

                    round(median, 8),

                "standard_deviation":

                    round(

                        math.sqrt(variance),

                        8

                    )

            }

        statistics = {

            "total_source_regions":

                len(

                    self.region_combination_database

                ),

            "total_candidate_relationships":

                len(

                    compatibility_scores

                ),

            "candidate_count_statistics":

                calculate_statistics(

                    candidate_counts

                ),

            "compatibility_statistics":

                calculate_statistics(

                    compatibility_scores

                ),

            "entropy_gap_statistics":

                calculate_statistics(

                    entropy_gaps

                ),

            "texture_gap_statistics":

                calculate_statistics(

                    texture_gaps

                ),

            "correlation_gap_statistics":

                calculate_statistics(

                    correlation_gaps

                ),

            "distortion_statistics":

                calculate_statistics(

                    distortion_scores

                ),

            "band_pair_distribution":

                band_pairs

        }

        self.region_combination_statistics = statistics

        self.save_json(

            self.module_root /

            "region_combination_statistics.json",

            statistics

        )

        self.log(

            "Region Combination Statistics Generated Successfully"

        )

    def generate_region_combination_manifest(self):

        self.log("\nGenerating Region Combination Manifest...\n")

        if not hasattr(self, "region_combination_database"):

            raise RuntimeError(
                "Region combination database not available."
            )

        if not hasattr(self, "region_combination_statistics"):

            raise RuntimeError(
                "Region combination statistics not available."
            )

        if not hasattr(self, "region_combination_validation"):

            raise RuntimeError(
                "Region combination validation not available."
            )

        band_distribution = {}

        recommendation_distribution = {

            "Excellent": 0,

            "Good": 0,

            "Moderate": 0,

            "Weak": 0

        }

        total_candidates = 0

        for entry in self.region_combination_database:

            band = entry["source_band"]

            band_distribution[band] = (

                band_distribution.get(

                    band,

                    0

                ) + 1

            )

            for candidate in entry["candidate_regions"]:

                total_candidates += 1

                score = candidate["compatibility_score"]

                if score >= 75:

                    recommendation_distribution["Excellent"] += 1

                elif score >= 60:

                    recommendation_distribution["Good"] += 1

                elif score >= 40:

                    recommendation_distribution["Moderate"] += 1

                else:

                    recommendation_distribution["Weak"] += 1

        manifest = {

            "module":

                "Region Combination Intelligence",

            "module_version":

                "9.0",

            "total_source_regions":

                len(

                    self.region_combination_database

                ),

            "total_candidate_relationships":

                total_candidates,

            "band_distribution":

                band_distribution,

            "candidate_recommendation_distribution":

                recommendation_distribution,

            "validation_status":

                self.region_combination_validation["validation_passed"],

            "validated_candidates":

                self.region_combination_validation["valid_candidates"],

            "invalid_candidates":

                self.region_combination_validation["invalid_candidates"],

            "database_file":

                "region_combination_database.json",

            "statistics_file":

                "region_combination_statistics.json",

            "validation_file":

                "region_combination_validation.json",

            "generated_outputs": [

                "region_combination_database.json",

                "region_combination_statistics.json",

                "region_combination_validation.json",

                "region_combination_manifest.json"

            ]

        }

        self.region_combination_manifest = manifest

        self.save_json(

            self.module_root /

            "region_combination_manifest.json",

            manifest

        )

        self.log(

            "Region Combination Manifest Generated Successfully"

        )

    def generate_qrng_randomization_plan(self):

        self.log("\nGenerating QRNG Randomization Intelligence...\n")

        if not hasattr(self, "feature_dataset"):

            raise RuntimeError(
                "Feature dataset not available."
            )

        if not hasattr(self, "region_combination_database"):

            raise RuntimeError(
                "Region combination database not available."
            )

        region_ids = [

            record["region_id"]

            for record in self.feature_dataset

        ]

        chunk_ids = [

            f"CHUNK_{i + 1}"

            for i in range(

                len(self.feature_dataset)

            )

        ]

        payload_indices = list(

            range(

                len(self.feature_dataset)

            )

        )

        coefficient_indices = list(

            range(64)

        )

        position_indices = list(

            range(64)

        )

        region_permutation = self.random_permutation(

            
            "region",

            region_ids

        )

        chunk_permutation = self.random_permutation(

            
            "permutation",

            chunk_ids

        )

        payload_permutation = self.random_permutation(

             "payload",

            payload_indices

        )

        coefficient_permutation = self.random_permutation(

            "coefficient",

            coefficient_indices

        )

        position_permutation = self.random_permutation(

            "position",

            position_indices

        )

        synchronization_token = self.get_random_integer(

             "entropy",

            100000000,

            999999999

        )

        randomization_plan = {

            "seed_responsibilities": {

                "master_seed":

                    "Global synchronization",

                "region_seed":

                    "Region permutation",

                "chunk_seed":

                    "Chunk permutation",

                "payload_seed":

                    "Payload permutation",

                "coefficient_seed":

                    "Coefficient permutation",

                "position_seed":

                    "Embedding position permutation"

            },

            "master_synchronization": {

                "synchronization_token":

                    synchronization_token,

                "total_regions":

                    len(region_ids),

                "total_chunks":

                    len(chunk_ids)

            },

            "region_permutation": {

                "total_regions":

                    len(region_permutation),

                "permutation":

                    region_permutation

            },

            "chunk_permutation": {

                "total_chunks":

                    len(chunk_permutation),

                "permutation":

                    chunk_permutation

            },

            "payload_permutation": {

                "payload_elements":

                    len(payload_permutation),

                "permutation":

                    payload_permutation

            },

            "coefficient_permutation": {

                "coefficients":

                    len(coefficient_permutation),

                "permutation":

                    coefficient_permutation

            },

            "position_permutation": {

                "positions":

                    len(position_permutation),

                "permutation":

                    position_permutation

            }

        }

        self.randomization_plan = randomization_plan

        self.save_json(

            self.module_root /

            "randomization_plan.json",

            randomization_plan

        )

        self.log(

            "QRNG Randomization Plan Generated Successfully"

        )

    def validate_randomization_plan(self):

        self.log("\nValidating QRNG Randomization Plan...\n")

        if not hasattr(self, "randomization_plan"):

            raise RuntimeError(
                "Randomization plan not available."
            )

        validation = {

            "validation_passed": True,

            "seed_validation": {},

            "permutation_validation": {},

            "master_validation": {}

        }

        responsibilities = self.randomization_plan[

            "seed_responsibilities"

        ]

        required_seeds = [

            "master_seed",

            "region_seed",

            "chunk_seed",

            "payload_seed",

            "coefficient_seed",

            "position_seed"

        ]

        for seed in required_seeds:

            validation["seed_validation"][seed] = {

                "available":

                    seed in responsibilities,

                "responsibility":

                    responsibilities.get(

                        seed,

                        None

                    )

            }

            if seed not in responsibilities:

                validation["validation_passed"] = False

        permutation_sections = [

            "region_permutation",

            "chunk_permutation",

            "payload_permutation",

            "coefficient_permutation",

            "position_permutation"

        ]

        for section in permutation_sections:

            permutation = self.randomization_plan[

                section

            ]["permutation"]

            unique = len(

                permutation

            ) == len(

                set(permutation)

            )

            validation["permutation_validation"][section] = {

                "total_elements":

                    len(permutation),

                "unique":

                    unique

            }

            if not unique:

                validation["validation_passed"] = False

        synchronization = self.randomization_plan[

            "master_synchronization"

        ]

        token = synchronization[

            "synchronization_token"

        ]

        validation["master_validation"] = {

            "token_available":

                token is not None,

            "region_count":

                synchronization["total_regions"],

            "chunk_count":

                synchronization["total_chunks"]

        }

        if token is None:

            validation["validation_passed"] = False

        self.randomization_validation = validation

        self.save_json(

            self.module_root /

            "randomization_validation.json",

            validation

        )

        self.log(

            "QRNG Randomization Validation Completed"

        )

    def generate_randomization_statistics(self):

        self.log("\nGenerating QRNG Randomization Statistics...\n")

        if not hasattr(self, "randomization_plan"):

            raise RuntimeError(
                "Randomization plan not available."
            )

        statistics = {}

        permutation_sections = {

            "region_permutation":

                "total_regions",

            "chunk_permutation":

                "total_chunks",

            "payload_permutation":

                "payload_elements",

            "coefficient_permutation":

                "coefficients",

            "position_permutation":

                "positions"

        }

        for section, count_key in permutation_sections.items():

            permutation = self.randomization_plan[

                section

            ]["permutation"]

            statistics[section] = {

                "total_elements":

                    self.randomization_plan[

                        section

                    ][count_key],

                "minimum":

                    min(permutation),

                "maximum":

                    max(permutation),

                "unique_elements":

                    len(

                        set(permutation)

                    ),

                "duplicates":

                    len(permutation)

                    -

                    len(set(permutation))

            }

        synchronization = self.randomization_plan[

            "master_synchronization"

        ]

        statistics["master_synchronization"] = {

            "synchronization_token":

                synchronization["synchronization_token"],

            "total_regions":

                synchronization["total_regions"],

            "total_chunks":

                synchronization["total_chunks"]

        }

        statistics["seed_summary"] = {

            "total_seeds":

                len(

                    self.randomization_plan[

                        "seed_responsibilities"

                    ]

                ),

            "seed_names":

                list(

                    self.randomization_plan[

                        "seed_responsibilities"

                    ].keys()

                )

        }

        self.randomization_statistics = statistics

        self.save_json(

            self.module_root /

            "randomization_statistics.json",

            statistics

        )

        self.log(

            "QRNG Randomization Statistics Generated Successfully"

        )

    def generate_randomization_manifest(self):

        self.log("\nGenerating QRNG Randomization Manifest...\n")

        if not hasattr(self, "randomization_plan"):

            raise RuntimeError(
                "Randomization plan not available."
            )

        if not hasattr(self, "randomization_statistics"):

            raise RuntimeError(
                "Randomization statistics not available."
            )

        if not hasattr(self, "randomization_validation"):

            raise RuntimeError(
                "Randomization validation not available."
            )

        manifest = {

            "module":

                "QRNG Randomization Intelligence",

            "module_version":

                "10.0",

            "validation_passed":

                self.randomization_validation[

                    "validation_passed"

                ],

            "seed_count":

                len(

                    self.randomization_plan[

                        "seed_responsibilities"

                    ]

                ),

            "seed_responsibilities":

                self.randomization_plan[

                    "seed_responsibilities"

                ],

            "master_synchronization":

                self.randomization_plan[

                    "master_synchronization"

                ],

            "permutation_summary": {

                "region_permutation":

                    self.randomization_statistics[

                        "region_permutation"

                    ]["total_elements"],

                "chunk_permutation":

                    self.randomization_statistics[

                        "chunk_permutation"

                    ]["total_elements"],

                "payload_permutation":

                    self.randomization_statistics[

                        "payload_permutation"

                    ]["total_elements"],

                "coefficient_permutation":

                    self.randomization_statistics[

                        "coefficient_permutation"

                    ]["total_elements"],

                "position_permutation":

                    self.randomization_statistics[

                        "position_permutation"

                    ]["total_elements"]

            },

            "database_file":

                "randomization_plan.json",

            "statistics_file":

                "randomization_statistics.json",

            "validation_file":

                "randomization_validation.json",

            "generated_outputs": [

                "randomization_plan.json",

                "randomization_statistics.json",

                "randomization_validation.json",

                "randomization_manifest.json"

            ]

        }

        self.randomization_manifest = manifest

        self.save_json(

            self.module_root /

            "randomization_manifest.json",

            manifest

        )

        self.log(

            "QRNG Randomization Manifest Generated Successfully"

        )

    def generate_chunk_region_candidates(self):

        self.log("\nGenerating Chunk-to-Region Candidate Intelligence...\n")

        if not hasattr(self, "feature_dataset"):

            raise RuntimeError(
                "Feature dataset not available."
            )

        if not hasattr(self, "region_combination_database"):

            raise RuntimeError(
                "Region combination database not available."
            )

        if not hasattr(self, "randomization_plan"):

            raise RuntimeError(
                "Randomization plan not available."
            )

        feature_lookup = {}

        for record in self.feature_dataset:

            feature_lookup[

                record["region_id"]

            ] = record

        randomized_regions = self.randomization_plan[

            "region_permutation"

        ][

            "permutation"

        ]

        candidate_database = []

        for chunk_index, region_id in enumerate(randomized_regions):

            if region_id not in feature_lookup:

                continue

            source = feature_lookup[

                region_id

            ]

            source_features = source[

                "feature_vector"

            ]

            combinations = next(

                (

                    item

                    for item in self.region_combination_database

                    if item["source_region"] == region_id

                ),

                None

            )

            if combinations is None:

                continue

            candidate_groups = []

            priority = 1

            for candidate in combinations["candidate_regions"]:

                target = feature_lookup[

                    candidate["region_id"]

                ]

                target_features = target[

                    "feature_vector"

                ]

                confidence = (

                    source_features["confidence"]

                    +

                    target_features["confidence"]

                ) / 2.0

                compatibility = candidate[

                    "compatibility_score"

                ]

                diversity = (

                    abs(

                        source_features["entropy"]

                        -

                        target_features["entropy"]

                    )

                    +

                    abs(

                        source_features["texture_complexity"]

                        -

                        target_features["texture_complexity"]

                    )

                    +

                    abs(

                        source_features["risk"]

                        -

                        target_features["risk"]

                    )

                ) / 3.0

                weights = self.priority_weights

                priority_score = (

                    compatibility * weights["compatibility"] +

                    confidence * weights["confidence"] +

                    diversity * weights["diversity"]

                )

                candidate_groups.append({

                    "priority":

                        priority,

                    "candidate_region":

                        candidate["region_id"],

                    "candidate_band":

                        candidate["band"],

                    "priority_score":

                        round(

                            priority_score,

                            8

                        ),

                    "confidence_score":

                        round(

                            confidence,

                            8

                        ),

                    "compatibility_score":

                        round(

                            compatibility,

                            8

                        ),

                    "diversity_score":

                        round(

                            diversity,

                            8

                        )

                })

                priority += 1

            candidate_groups.sort(

                key=lambda x:

                x["priority_score"],

                reverse=True

            )

            candidate_database.append({

                "chunk_id":

                    f"CHUNK_{chunk_index + 1}",

                "source_region":

                    region_id,

                "candidate_count":

                    len(candidate_groups),

                "candidate_groups":

                    candidate_groups

            })

        self.chunk_region_candidates = candidate_database

        self.save_json(

            self.module_root /

            "chunk_region_candidates.json",

            candidate_database

        )

        self.log(

            f"Chunk-to-Region Candidate Database Generated : {len(candidate_database)} Chunks"

        )

    def validate_chunk_region_candidates(self):

        self.log("\nValidating Chunk-to-Region Candidate Intelligence...\n")

        if not hasattr(self, "chunk_region_candidates"):

            raise RuntimeError(
                "Chunk-region candidate database not available."
            )

        validation = {

            "validation_passed": True,

            "total_chunks": 0,

            "total_candidates": 0,

            "valid_candidates": 0,

            "invalid_candidates": 0,

            "empty_candidate_groups": [],

            "duplicate_regions": [],

            "invalid_scores": [],

            "missing_regions": []

        }

        for chunk in self.chunk_region_candidates:

            validation["total_chunks"] += 1

            seen = set()

            if len(chunk["candidate_groups"]) == 0:

                validation["empty_candidate_groups"].append(

                    chunk["chunk_id"]

                )

                validation["validation_passed"] = False

            for candidate in chunk["candidate_groups"]:

                validation["total_candidates"] += 1

                region = candidate["candidate_region"]

                if region in seen:

                    validation["duplicate_regions"].append({

                        "chunk_id":

                            chunk["chunk_id"],

                        "region_id":

                            region

                    })

                    validation["validation_passed"] = False

                else:

                    seen.add(

                        region

                    )

                scores = [

                    candidate["priority_score"],

                    candidate["confidence_score"],

                    candidate["compatibility_score"],

                    candidate["diversity_score"]

                ]

                valid = True

                for score in scores:

                    if not isinstance(score, (int, float)):
                        valid = False
                        break

                    if math.isnan(score):
                        valid = False
                        break

                    if math.isinf(score):
                        valid = False
                        break

                if not (0 <= candidate["confidence_score"] <= 1):
                    valid = False

                if candidate["priority_score"] < 0:
                    valid = False

                if candidate["compatibility_score"] < 0:
                    valid = False

                if candidate["diversity_score"] < 0:
                    valid = False

                if valid:

                    validation["valid_candidates"] += 1

                else:

                    validation["invalid_scores"].append({

                        "chunk_id":

                            chunk["chunk_id"],

                        "region_id":

                            region

                    })

                    validation["validation_passed"] = False

        validation["invalid_candidates"] = (

            len(validation["empty_candidate_groups"])

            +

            len(validation["duplicate_regions"])

            +

            len(validation["invalid_scores"])

            +

            len(validation["missing_regions"])

        )

        self.chunk_region_candidate_validation = validation

        self.save_json(

            self.module_root /

            "chunk_region_candidate_validation.json",

            validation

        )

        self.log(

            "Chunk-to-Region Candidate Validation Completed"

        )

    def generate_chunk_region_candidate_statistics(self):

        self.log("\nGenerating Chunk-to-Region Candidate Statistics...\n")

        if not hasattr(self, "chunk_region_candidates"):

            raise RuntimeError(
                "Chunk-region candidate database not available."
            )

        import math

        candidate_counts = []

        priority_scores = []

        confidence_scores = []

        compatibility_scores = []

        diversity_scores = []

        band_distribution = {}

        for chunk in self.chunk_region_candidates:

            candidate_counts.append(

                chunk["candidate_count"]

            )

            for candidate in chunk["candidate_groups"]:

                priority_scores.append(

                    candidate["priority_score"]

                )

                confidence_scores.append(

                    candidate["confidence_score"]

                )

                compatibility_scores.append(

                    candidate["compatibility_score"]

                )

                diversity_scores.append(

                    candidate["diversity_score"]

                )

                band = candidate["candidate_band"]

                band_distribution[band] = (

                    band_distribution.get(

                        band,

                        0

                    ) + 1

                )

        def calculate_statistics(values):

            if len(values) == 0:

                return {

                    "minimum": 0,

                    "maximum": 0,

                    "mean": 0,

                    "median": 0,

                    "standard_deviation": 0

                }

            values = sorted(values)

            count = len(values)

            mean = sum(values) / count

            variance = sum(

                (

                    value - mean

                ) ** 2

                for value in values

            ) / count

            if count % 2 == 0:

                median = (

                    values[count // 2 - 1]

                    +

                    values[count // 2]

                ) / 2

            else:

                median = values[count // 2]

            return {

                "minimum":

                    round(

                        values[0],

                        8

                    ),

                "maximum":

                    round(

                        values[-1],

                        8

                    ),

                "mean":

                    round(

                        mean,

                        8

                    ),

                "median":

                    round(

                        median,

                        8

                    ),

                "standard_deviation":

                    round(

                        math.sqrt(

                            variance

                        ),

                        8

                    )

            }

        statistics = {

            "total_chunks":

                len(

                    self.chunk_region_candidates

                ),

            "total_candidate_groups":

                len(

                    priority_scores

                ),

            "candidate_count_statistics":

                calculate_statistics(

                    candidate_counts

                ),

            "priority_score_statistics":

                calculate_statistics(

                    priority_scores

                ),

            "confidence_score_statistics":

                calculate_statistics(

                    confidence_scores

                ),

            "compatibility_score_statistics":

                calculate_statistics(

                    compatibility_scores

                ),

            "diversity_score_statistics":

                calculate_statistics(

                    diversity_scores

                ),

            "candidate_band_distribution":

                band_distribution

        }

        self.chunk_region_candidate_statistics = statistics

        self.save_json(

            self.module_root /

            "chunk_region_candidate_statistics.json",

            statistics

        )

        self.log(

            "Chunk-to-Region Candidate Statistics Generated Successfully"

        )

    def generate_chunk_region_candidate_manifest(self):

        self.log("\nGenerating Chunk-to-Region Candidate Manifest...\n")

        if not hasattr(self, "chunk_region_candidates"):

            raise RuntimeError(
                "Chunk-region candidate database not available."
            )

        if not hasattr(self, "chunk_region_candidate_statistics"):

            raise RuntimeError(
                "Chunk-region candidate statistics not available."
            )

        if not hasattr(self, "chunk_region_candidate_validation"):

            raise RuntimeError(
                "Chunk-region candidate validation not available."
            )

        recommendation_distribution = {

            "Excellent": 0,

            "Good": 0,

            "Moderate": 0,

            "Weak": 0

        }

        total_candidate_groups = 0

        for chunk in self.chunk_region_candidates:

            total_candidate_groups += len(

                chunk["candidate_groups"]

            )

            for candidate in chunk["candidate_groups"]:

                score = candidate[

                    "priority_score"

                ]

                if score >= 75:

                    recommendation_distribution[

                        "Excellent"

                    ] += 1

                elif score >= 60:

                    recommendation_distribution[

                        "Good"

                    ] += 1

                elif score >= 40:

                    recommendation_distribution[

                        "Moderate"

                    ] += 1

                else:

                    recommendation_distribution[

                        "Weak"

                    ] += 1

        manifest = {

            "module":

                "Chunk-to-Region Candidate Intelligence",

            "module_version":

                "11.0",

            "total_chunks":

                len(

                    self.chunk_region_candidates

                ),

            "total_candidate_groups":

                total_candidate_groups,

            "validation_passed":

                self.chunk_region_candidate_validation[

                    "validation_passed"

                ],

            "validated_candidates":

                self.chunk_region_candidate_validation[

                    "valid_candidates"

                ],

            "invalid_candidates":

                self.chunk_region_candidate_validation[

                    "invalid_candidates"

                ],

            "candidate_recommendation_distribution":

                recommendation_distribution,

            "database_file":

                "chunk_region_candidates.json",

            "statistics_file":

                "chunk_region_candidate_statistics.json",

            "validation_file":

                "chunk_region_candidate_validation.json",

            "generated_outputs": [

                "chunk_region_candidates.json",

                "chunk_region_candidate_statistics.json",

                "chunk_region_candidate_validation.json",

                "chunk_region_candidate_manifest.json"

            ]

        }

        self.chunk_region_candidate_manifest = manifest

        self.save_json(

            self.module_root /

            "chunk_region_candidate_manifest.json",

            manifest

        )

        self.log(

            "Chunk-to-Region Candidate Manifest Generated Successfully"

        )

    def generate_entropy_intelligence_manifest(self):

        self.log("\nGenerating Final Entropy Intelligence Manifest...\n")

        required_components = {

            "feature_dataset":
                hasattr(self, "feature_dataset"),

            "normalized_dataset":
                hasattr(self, "normalized_dataset"),

            "feature_statistics":
                hasattr(self, "feature_statistics"),

            "dataset_validation":
                hasattr(self, "dataset_validation"),

            "dataset_manifest":
                hasattr(self, "dataset_manifest"),

            "distortion_database":
                hasattr(self, "distortion_database"),

            "distortion_statistics":
                hasattr(self, "distortion_statistics"),

            "distortion_manifest":
                hasattr(self, "distortion_manifest"),

            "region_combination_database":
                hasattr(self, "region_combination_database"),

            "region_combination_statistics":
                hasattr(self, "region_combination_statistics"),

            "region_combination_manifest":
                hasattr(self, "region_combination_manifest"),

            "randomization_plan":
                hasattr(self, "randomization_plan"),

            "randomization_statistics":
                hasattr(self, "randomization_statistics"),

            "randomization_manifest":
                hasattr(self, "randomization_manifest"),

            "chunk_region_candidates":
                hasattr(self, "chunk_region_candidates"),

            "chunk_region_candidate_statistics":
                hasattr(self, "chunk_region_candidate_statistics"),

            "chunk_region_candidate_manifest":
                hasattr(self, "chunk_region_candidate_manifest")

        }

        completed = sum(

            required_components.values()

        )

        total = len(

            required_components

        )

        ready = (

            completed == total

            and self.dataset_validation["dataset_valid"]

            and self.region_combination_validation["validation_passed"]

            and self.randomization_validation["validation_passed"]

            and self.chunk_region_candidate_validation["validation_passed"]

        )

        readiness = {

            "linear_regression": ready,

            "quantum_gan": ready,

            "genetic_algorithm": ready,

            "ant_colony_optimization": ready,

            "adaptive_embedding": ready

        }

        manifest = {

            "module":

                "Entropy Intelligence Engine",

            "module_version":

                "12.0",

            "module_status":

                "Completed" if completed == total else "Incomplete",

            "completion_percentage":

                round(

                    (completed / total) * 100,

                    2

                ),

            "completed_components":

                completed,

            "total_components":

                total,

            "component_status":

                required_components,

            "generated_datasets": [

                "feature_dataset.json",

                "prediction_features.json",

                "normalized_dataset.json",

                "feature_statistics.json",

                "dataset_validation.json",

                "dataset_manifest.json",

                "distortion_database.json",

                "distortion_statistics.json",

                "distortion_classification.json",

                "distortion_manifest.json",

                "region_combination_database.json",

                "region_combination_statistics.json",

                "region_combination_validation.json",

                "region_combination_manifest.json",

                "randomization_plan.json",

                "randomization_statistics.json",

                "randomization_validation.json",

                "randomization_manifest.json",

                "chunk_region_candidates.json",

                "chunk_region_candidate_statistics.json",

                "chunk_region_candidate_validation.json",

                "chunk_region_candidate_manifest.json"

            ],

            "downstream_module_readiness":

                readiness,

            "next_pipeline_modules": [

                "Linear Regression",

                "Quantum GAN",

                "Genetic Algorithm",

                "Ant Colony Optimization",

                "Adaptive Embedding"

            ]

        }

        self.entropy_intelligence_manifest = manifest

        self.save_json(

            self.module_root /

            "entropy_intelligence_manifest.json",

            manifest

        )

        self.log(

            "Final Entropy Intelligence Manifest Generated Successfully"

        )

    def generate_entropy_intelligence_report(self):

        self.log("\nGenerating Final Entropy Intelligence Report...\n")

        if not hasattr(self, "entropy_intelligence_manifest"):

            raise RuntimeError(
                "Entropy intelligence manifest not available."
            )

        report = {

            "entropy_intelligence_engine": {

                "status":

                    self.entropy_intelligence_manifest[

                        "module_status"

                    ],

                "completion_percentage":

                    self.entropy_intelligence_manifest[

                        "completion_percentage"

                    ],

                "completed_components":

                    self.entropy_intelligence_manifest[

                        "completed_components"

                    ],

                "total_components":

                    self.entropy_intelligence_manifest[

                        "total_components"

                    ]

            },

            "datasets": {

                "feature_dataset":

                    len(self.feature_dataset),

                "normalized_dataset":

                    len(self.normalized_dataset),

                "distortion_database":

                    len(self.distortion_database),

                "region_combination_database":

                    len(self.region_combination_database),

                "chunk_region_candidates":

                    len(self.chunk_region_candidates)

            },

            "validation": {

                "feature_dataset":

                    self.dataset_validation[

                        "dataset_valid"

                    ],

                "region_combinations":

                    self.region_combination_validation[

                        "validation_passed"

                    ],

                "randomization":

                    self.randomization_validation[

                        "validation_passed"

                    ],

                "chunk_region_candidates":

                    self.chunk_region_candidate_validation[

                        "validation_passed"

                    ]

            },

            "downstream_readiness":

                self.entropy_intelligence_manifest[

                    "downstream_module_readiness"

                ],

            "generated_outputs":

                self.entropy_intelligence_manifest[

                    "generated_datasets"

                ],

            "pipeline": [

                "Entropy Intelligence Engine",

                "Linear Regression",

                "Quantum GAN",

                "Genetic Algorithm",

                "Ant Colony Optimization",

                "Adaptive Embedding"

            ],

            "overall_status": {

                "engine_ready":

                    all(

                        self.entropy_intelligence_manifest[

                            "downstream_module_readiness"

                        ].values()

                    ),

                "next_module":

                    "Linear Regression",

                "final_status":

                    "READY"

                    if all(

                        self.entropy_intelligence_manifest[

                            "downstream_module_readiness"

                        ].values()

                    )

                    else

                    "NOT_READY"

            }

        }

        self.entropy_intelligence_report = report

        self.save_json(

            self.module_root /

            "entropy_intelligence_report.json",

            report

        )

        self.log(

            "Entropy Intelligence Report Generated Successfully"

        )

    def execute_part7_1(self):

        self.log("\n" + "=" * 80)
        self.log("PART 7.1 : FEATURE DATASET BUILDER")
        self.log("=" * 80)

        self.build_feature_dataset()


    def execute_part7_2(self):

        self.log("\n" + "=" * 80)
        self.log("PART 7.2 : DATASET NORMALIZATION")
        self.log("=" * 80)

        self.normalize_feature_dataset()


    def execute_part7_3(self):

        self.log("\n" + "=" * 80)
        self.log("PART 7.3 : FEATURE STATISTICS")
        self.log("=" * 80)

        self.generate_feature_statistics()


    def execute_part7_4(self):

        self.log("\n" + "=" * 80)
        self.log("PART 7.4 : DATASET VALIDATION")
        self.log("=" * 80)

        self.validate_feature_dataset()


    def execute_part7_5(self):

        self.log("\n" + "=" * 80)
        self.log("PART 7.5 : DATASET MANIFEST")
        self.log("=" * 80)

        self.generate_dataset_manifest()


    def execute_part8_1(self):

        self.log("\n" + "=" * 80)
        self.log("PART 8.1 : EMBEDDING DISTORTION ESTIMATION")
        self.log("=" * 80)

        self.estimate_embedding_distortion()


    def execute_part8_2(self):

        self.log("\n" + "=" * 80)
        self.log("PART 8.2 : DISTORTION STATISTICS")
        self.log("=" * 80)

        self.generate_distortion_statistics()


    def execute_part8_3(self):

        self.log("\n" + "=" * 80)
        self.log("PART 8.3 : DISTORTION CLASSIFICATION")
        self.log("=" * 80)

        self.classify_distortion_levels()


    def execute_part8_4(self):

        self.log("\n" + "=" * 80)
        self.log("PART 8.4 : DISTORTION MANIFEST")
        self.log("=" * 80)

        self.generate_distortion_manifest()


    def execute_part9_1(self):

        self.log("\n" + "=" * 80)
        self.log("PART 9.1 : REGION COMBINATION CANDIDATES")
        self.log("=" * 80)

        self.generate_region_combination_candidates()


    def execute_part9_2(self):

        self.log("\n" + "=" * 80)
        self.log("PART 9.2 : REGION COMBINATION VALIDATION")
        self.log("=" * 80)

        self.validate_region_combinations()


    def execute_part9_3(self):

        self.log("\n" + "=" * 80)
        self.log("PART 9.3 : REGION COMBINATION STATISTICS")
        self.log("=" * 80)

        self.generate_region_combination_statistics()


    def execute_part9_4(self):

        self.log("\n" + "=" * 80)
        self.log("PART 9.4 : REGION COMBINATION MANIFEST")
        self.log("=" * 80)

        self.generate_region_combination_manifest()


    def execute_part10_1(self):

        self.log("\n" + "=" * 80)
        self.log("PART 10.1 : QRNG RANDOMIZATION PLAN")
        self.log("=" * 80)

        self.generate_qrng_randomization_plan()


    def execute_part10_2(self):

        self.log("\n" + "=" * 80)
        self.log("PART 10.2 : RANDOMIZATION VALIDATION")
        self.log("=" * 80)

        self.validate_randomization_plan()


    def execute_part10_3(self):

        self.log("\n" + "=" * 80)
        self.log("PART 10.3 : RANDOMIZATION STATISTICS")
        self.log("=" * 80)

        self.generate_randomization_statistics()


    def execute_part10_4(self):

        self.log("\n" + "=" * 80)
        self.log("PART 10.4 : RANDOMIZATION MANIFEST")
        self.log("=" * 80)

        self.generate_randomization_manifest()


    def execute_part11_1(self):

        self.log("\n" + "=" * 80)
        self.log("PART 11.1 : CHUNK-REGION CANDIDATES")
        self.log("=" * 80)

        self.generate_chunk_region_candidates()


    def execute_part11_2(self):

        self.log("\n" + "=" * 80)
        self.log("PART 11.2 : CHUNK-REGION VALIDATION")
        self.log("=" * 80)

        self.validate_chunk_region_candidates()


    def execute_part11_3(self):

        self.log("\n" + "=" * 80)
        self.log("PART 11.3 : CHUNK-REGION STATISTICS")
        self.log("=" * 80)

        self.generate_chunk_region_candidate_statistics()


    def execute_part11_4(self):

        self.log("\n" + "=" * 80)
        self.log("PART 11.4 : CHUNK-REGION MANIFEST")
        self.log("=" * 80)

        self.generate_chunk_region_candidate_manifest()


    def execute_part12_1(self):

        self.log("\n" + "=" * 80)
        self.log("PART 12.1 : ENTROPY INTELLIGENCE MANIFEST")
        self.log("=" * 80)

        self.generate_entropy_intelligence_manifest()


    def execute_part12_2(self):

        self.log("\n" + "=" * 80)
        self.log("PART 12.2 : ENTROPY INTELLIGENCE REPORT")
        self.log("=" * 80)

        self.generate_entropy_intelligence_report()

    def load_pointer_state(self):

        if os.path.exists(self.pointer_file):

            with open(
                self.pointer_file,
                "r",
                encoding="utf-8"
            ) as file:

                self.pointer_state = json.load(file)

        else:

            self.pointer_state = {

                "entropy": 0,

                "region": 0,

                "position": 0,

                "coefficient": 0,

                "permutation": 0,

                "noise": 0,

                "gradient": 0,

                "threshold": 0,

                "payload": 0

            }

            self.save_pointer_state()

        self.log(
            "QRNG Pointer State Loaded"
        )

    def save_pointer_state(self):

        with open(
            self.pointer_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(

                self.pointer_state,

                file,

                indent=4

            )

        self.log(
            "QRNG Pointer State Saved"
        )

    def run(self):

        self.log("\n" + "=" * 100)
        self.log("ENTROPY INTELLIGENCE ENGINE")
        self.log("=" * 100)

        # Part 1: Engine Initialization & Initial DB Setup
        self.initialize_part1()

        # Part 2: Core Region & Global Statistics
        self.execute_part2()

        # Part 3: Relationship Graph, Priority & Decision Engine
        # Part 3: Relationship Graph & Priority
        self.execute_part3()

        # Part 5: Advanced Entropy & Feature Metrics
        self.execute_part5_1()
        self.execute_part5_2()
        self.execute_part5_3()
        self.execute_part5_4()
        self.execute_part5_5()
        self.execute_part5_6()

        self.build_embedding_intelligence()

        # Decision Engine
        self.build_entropy_decision_engine()
        self.generate_entropy_summary()
        self.save_decision_database()

        # Part 4: Heatmaps & Package Generation
        self.build_entropy_heatmap()
        self.build_priority_heatmap()
        self.generate_entropy_packages()
        self.generate_engine_report()

        # Part 6: Correlation, Graphs & GLCM
        self.execute_part6_1()
        self.execute_part6_2()
        self.execute_part6_3()
        self.execute_part6_4()
        self.execute_part6_5()

        # Part 7: Feature Dataset Pipeline
        self.execute_part7_1()
        self.execute_part7_2()
        self.execute_part7_3()
        self.execute_part7_4()
        self.execute_part7_5()

        # Part 8: Embedding Distortion Analysis
        self.execute_part8_1()
        self.execute_part8_2()
        self.execute_part8_3()
        self.execute_part8_4()

        # Part 9: Region Combination Candidates
        self.execute_part9_1()
        self.execute_part9_2()
        self.execute_part9_3()
        self.execute_part9_4()

        # Part 10: QRNG Randomization Plan
        self.execute_part10_1()
        self.execute_part10_2()
        self.execute_part10_3()
        self.execute_part10_4()

        # Part 11: Chunk-to-Region Pipeline
        self.execute_part11_1()
        self.execute_part11_2()
        self.execute_part11_3()
        self.execute_part11_4()

        # Part 12: Final Manifest and Report
        self.execute_part12_1()
        self.execute_part12_2()

        # Final Summary
        self.finalize_engine()

        self.save_visualization_maps()

        self.log("\n" + "=" * 100)
        self.log("ENTROPY INTELLIGENCE ENGINE COMPLETED")
        self.log("=" * 100)
        

if __name__ == "__main__":

    engine = EntropyIntelligenceEngine()

    try:

        engine.run()

        print("\n" + "=" * 100)
        print("ENTROPY INTELLIGENCE ENGINE EXECUTED SUCCESSFULLY")
        print("=" * 100)

    except Exception as error:

        print("\n" + "=" * 100)
        print("ENTROPY INTELLIGENCE ENGINE FAILED")
        print("=" * 100)
        print(error)

        raise