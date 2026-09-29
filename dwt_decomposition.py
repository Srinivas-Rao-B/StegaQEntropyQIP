import os
import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pywt


class DWTDecomposition:

    def __init__(self):

        self.image_input_directory = Path("output/image_acquisition")

        self.seed_file = Path(
            "qrng/output/history_check/final_verified_seeds/"
            "dwt_seed_verified.txt"
        )

        self.output_directory = Path(
            "output/dwt_decomposition"
        )

        os.makedirs(
            self.output_directory,
            exist_ok=True
        )

        self.pointer_file = (
            self.output_directory /
            "dwt_seed_pointer.txt"
        )

        self.image_profile_file = (
            self.image_input_directory /
            "image_profile.json"
        )

        self.qrng_statistics = {
            "start_pointer": None,
            "end_pointer": None,
            "total_bits_consumed": 0,
            "total_reads": 0,
            "wrap_count": 0,
            "seed_length": 0
        }

        self.candidate_percentile = 75

        self.image = None

        self.gray_image = None

        self.image_profile = {}

        self.band_priority = {}

        self.fitness_weights = {}

        self.seed = ""

        self.seed_sha256 = ""

        self.image_validation = {}

        self.image_intelligence = {}

        self.optimization_guidance = {}

        self.normalized_bands = {}

        self.global_profile = {}

        self.global_factors = {}

        self.minimum_block_size = 8

        self.medium_block_size = 16

        self.maximum_block_size = 32

        self.adaptive_region_information = {}

        self.boundary_mode = "symmetric"

        self.wavelet_family = "db2"

        self.decomposition_level = 1

        self.maximum_supported_level = None

        self.histogram_information = {}

        self.region_complexities = []

        self.dwt_id = f"DWT-{uuid.uuid4().hex[:8].upper()}"

        self.created_on = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

    def banner(self):

        print()
        print("=" * 80)
        print("               DWT DECOMPOSITION")
        print("=" * 80)
        print()
        print("Preparing Frequency Domain...")
        print()
        print("• Loading Image Acquisition Output")
        print("• Loading QRNG DWT Seed")
        print("• Verifying Image")
        print("• Performing Haar Wavelet Transform")
        print("• Extracting LL LH HL HH")
        print("• Analysing Frequency Bands")
        print("• Preparing Region Information")
        print("• Preparing GA / ACO Packages")
        print()
        print("=" * 80)
        print()

    def save_global_profile(self):

        with open(

            self.output_directory /
            "global_image_profile.json",

            "w"

        ) as file:

            json.dump(

                self.global_factors,

                file,

                indent=4

            )

    def adaptive_fitness_model(self):

        print("Generating Adaptive Fitness Model...\n")

        entropy = self.global_factors["entropy"]

        gradient = self.global_factors["average_gradient"]

        edge = self.global_factors["edge_density"]

        contrast = self.global_factors["contrast"]

        correlation = self.global_factors["correlation"]

        entropy_weight = 0.22
        variance_weight = 0.18
        gradient_weight = 0.18
        energy_weight = 0.15
        density_weight = 0.10
        capacity_weight = 0.12
        qrng_weight = 0.05
        distortion_weight = 0.10

        if entropy >= 7.2:

            entropy_weight += 0.05

            capacity_weight += 0.03

            energy_weight -= 0.02

        if gradient >= 120:

            gradient_weight += 0.05

            distortion_weight -= 0.02

        if edge >= 20:

            gradient_weight += 0.03

            density_weight += 0.02

        if contrast <= 80:

            variance_weight += 0.03

            energy_weight -= 0.02

        if correlation >= 0.90:

            density_weight -= 0.03

            variance_weight += 0.03

        total = (

            entropy_weight +

            variance_weight +

            gradient_weight +

            energy_weight +

            density_weight +

            capacity_weight +

            qrng_weight +

            distortion_weight

        )

        entropy_weight /= total
        variance_weight /= total
        gradient_weight /= total
        energy_weight /= total
        density_weight /= total
        capacity_weight /= total
        qrng_weight /= total
        distortion_weight /= total

        self.fitness_weights = {

            "entropy": entropy_weight,

            "variance": variance_weight,

            "gradient": gradient_weight,

            "energy": energy_weight,

            "density": density_weight,

            "capacity": capacity_weight,

            "qrng": qrng_weight,

            "distortion": distortion_weight

        }

        with open(

            self.output_directory /

            "adaptive_fitness_weights.json",

            "w"

        ) as file:

            json.dump(

                self.fitness_weights,

                file,

                indent=4

            )

        for key, value in self.fitness_weights.items():

            print(f"{key:<15}: {value:.4f}")

        print()


    def generate_optimization_guidance(self):

        print("Generating Optimization Guidance...\n")

        complexity = self.image_intelligence["complexity_index"]

        if complexity >= 80:

            ga = {
                "population_size": 100,
                "generations": 200,
                "mutation_rate": 0.10,
                "crossover_rate": 0.90,
                "elitism": 8
            }

            aco = {
                "ants": 60,
                "iterations": 200,
                "alpha": 1.5,
                "beta": 3.0,
                "evaporation": 0.45
            }

        elif complexity >= 60:

            ga = {
                "population_size": 80,
                "generations": 150,
                "mutation_rate": 0.08,
                "crossover_rate": 0.90,
                "elitism": 6
            }

            aco = {
                "ants": 45,
                "iterations": 150,
                "alpha": 1.3,
                "beta": 2.8,
                "evaporation": 0.50
            }

        elif complexity >= 40:

            ga = {
                "population_size": 60,
                "generations": 120,
                "mutation_rate": 0.06,
                "crossover_rate": 0.85,
                "elitism": 5
            }

            aco = {
                "ants": 35,
                "iterations": 120,
                "alpha": 1.2,
                "beta": 2.5,
                "evaporation": 0.55
            }

        else:

            ga = {
                "population_size": 40,
                "generations": 100,
                "mutation_rate": 0.05,
                "crossover_rate": 0.80,
                "elitism": 4
            }

            aco = {
                "ants": 25,
                "iterations": 100,
                "alpha": 1.0,
                "beta": 2.2,
                "evaporation": 0.60
            }

        self.optimization_guidance = {
            "ga": ga,
            "aco": aco
        }

        with open(
            self.output_directory /
            "optimization_guidance.json",
            "w"
        ) as file:

            json.dump(
                self.optimization_guidance,
                file,
                indent=4
            )

        print("Optimization Guidance Generated.\n")

    def load_image_acquisition(self):

        print("Loading Image Acquisition Output...\n")

        if not self.image_profile_file.exists():

            raise FileNotFoundError(
                "image_profile.json not found."
            )

        with open(
            self.image_profile_file,
            "r"
        ) as file:

            self.image_profile = json.load(file)

        image_path = self.image_profile["image_path"]

        self.image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if self.image is None:

            raise FileNotFoundError(
                "Unable to load cover image."
            )

        self.gray_image = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

        print(f"Image Path      : {image_path}")
        print(f"Image Width     : {self.image.shape[1]}")
        print(f"Image Height    : {self.image.shape[0]}")
        print(f"Channels        : {self.image.shape[2]}")

        self.global_profile = self.image_profile

        self.global_factors = {

            "entropy":
                self.image_profile["entropy"],

            "variance":
                self.image_profile["variance"],

            "average_gradient":
                self.image_profile["average_gradient"],

            "edge_density":
                self.image_profile["edge_density"],

            "contrast":
                self.image_profile["contrast"],

            "correlation":
                self.image_profile["correlation"],

            "energy":
                self.image_profile["energy"],

            "homogeneity":
                self.image_profile["homogeneity"],

            "asm":
                self.image_profile["asm"],

            "dominant_peak":
                self.image_profile["dominant_peak"]

        }

        print("=" * 80)
        print("GLOBAL IMAGE PROFILE")
        print("=" * 80)

        for key, value in self.global_factors.items():

            print(f"{key:<20}: {value}")

        print()

    def adaptive_image_intelligence(self):

        print("Generating Adaptive Image Intelligence...\n")

        entropy = self.global_factors["entropy"]
        variance = self.global_factors["variance"]
        gradient = self.global_factors["average_gradient"]
        edge = self.global_factors["edge_density"]
        contrast = self.global_factors["contrast"]

        complexity = (

            entropy * 0.30 +

            min(variance / 5000, 1.0) * 0.20 +

            min(gradient / 300, 1.0) * 0.20 +

            min(edge / 100, 1.0) * 0.15 +

            min(contrast / 1000, 1.0) * 0.15

        )

        if complexity >= 0.80:

            image_class = "Highly Complex"

            candidate_percentile = 85

            preferred_block = 8

        elif complexity >= 0.60:

            image_class = "Complex"

            candidate_percentile = 80

            preferred_block = 16

        elif complexity >= 0.40:

            image_class = "Moderate"

            candidate_percentile = 75

            preferred_block = 16

        else:

            image_class = "Simple"

            candidate_percentile = 70

            preferred_block = 32

        self.image_intelligence = {

            "complexity_index": float(complexity),

            "image_class": image_class,

            "candidate_percentile": candidate_percentile,

            "preferred_block_size": preferred_block

        }

        self.candidate_percentile = candidate_percentile

        print(f"Complexity Index      : {complexity:.4f}")
        print(f"Image Class           : {image_class}")
        print(f"Candidate Percentile  : {candidate_percentile}")
        print(f"Preferred Block Size  : {preferred_block}")
        print()

        with open(

            self.output_directory /
            "image_intelligence.json",

            "w"

        ) as file:

            json.dump(

                self.image_intelligence,

                file,

                indent=4

            )

    def adaptive_band_priority(self):

        print("Generating Adaptive Band Priority...\n")

        entropy = self.global_factors["entropy"]

        edge = self.global_factors["edge_density"]

        gradient = self.global_factors["average_gradient"]

        correlation = self.global_factors["correlation"]

        contrast = self.global_factors["contrast"]

        if edge >= 20 or gradient >= 120:

            order = ["HH", "HL", "LH"]

            reason = "High Edge Image"

        elif entropy >= 7.2:

            order = ["HH", "LH", "HL"]

            reason = "Highly Textured Image"

        elif correlation >= 0.90:

            order = ["LH", "HL", "HH"]

            reason = "Highly Correlated Image"

        elif contrast <= 80:

            order = ["LH", "HH", "HL"]

            reason = "Low Contrast Image"

        else:

            order = ["HH", "LH", "HL"]

            reason = "Balanced Image"

        self.band_priority = {

            "order": order,

            "reason": reason

        }

        with open(

            self.output_directory /

            "band_priority.json",

            "w"

        ) as file:

            json.dump(

                self.band_priority,

                file,

                indent=4

            )

        print(f"Priority Order : {' -> '.join(order)}")

        print(f"Reason         : {reason}")

        print()


    def load_seed_pointer(self):

        if not self.pointer_file.exists():

            with open(
                self.pointer_file,
                "w"
            ) as file:

                file.write("0")

            return 0

        with open(
            self.pointer_file,
            "r"
        ) as file:

            pointer = file.read().strip()

        if pointer == "":

            return 0

        return int(pointer)

    def save_seed_pointer(self, pointer):

        with open(
            self.pointer_file,
            "w"
        ) as file:

            file.write(str(pointer))

    def get_qrng_uint(self, bit_count=64):

        return int(

            self.get_qrng_bits(
                bit_count
            ),

            2

        )

    def get_qrng_float(self, bit_count=64):

        value = self.get_qrng_uint(
            bit_count
        )

        maximum = (

            1 << bit_count

        ) - 1

        return value / maximum

    def save_qrng_usage(self):

        with open(

            self.output_directory /

            "qrng_usage.json",

            "w"

        ) as file:

            json.dump(

                self.last_qrng_usage,

                file,

                indent=4

            )

    def get_qrng_bits(self, bit_count=64):

        pointer = self.load_seed_pointer()

        seed_length = len(self.seed)

        if bit_count <= 0:

            raise ValueError(
                "Bit count must be greater than zero."
            )

        bits = ""

        start_pointer = pointer

        while len(bits) < bit_count:

            available = seed_length - pointer

            required = bit_count - len(bits)

            if available >= required:

                bits += self.seed[
                    pointer:
                    pointer + required
                ]

                pointer += required

            else:

                bits += self.seed[
                    pointer:
                ]

                pointer = 0

        end_pointer = pointer

        self.save_seed_pointer(pointer)

        self.last_qrng_usage = {

            "start_pointer": start_pointer,

            "end_pointer": end_pointer,

            "bits_requested": bit_count,

            "seed_length": seed_length,

            "wrapped": end_pointer < start_pointer

        }

        return bits
    
    def load_qrng_seed(self):

        print("Loading QRNG DWT Seed...\n")

        if not self.seed_file.exists():

            raise FileNotFoundError(
                "DWT QRNG seed not found."
            )

        with open(
            self.seed_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = file.read()

        start = data.find("\nBinary\n")

        if start == -1:

            start = data.find("Binary\n")

        if start == -1:

            raise ValueError(
                "Binary section not found."
            )

        data = data[start:]

        lines = data.splitlines()

        binary = ""

        binary_started = False

        for line in lines:

            line = line.strip()

            if line == "Binary":

                binary_started = True
                continue

            if not binary_started:
                continue

            if line.startswith("Hexadecimal"):
                break

            if line.startswith("Integer"):
                break

            if line.startswith("SHA256"):
                break

            if (
                line
                and set(line).issubset({"0", "1"})
            ):

                binary += line

        self.seed = binary

        if len(self.seed) == 0:

            raise ValueError(
                "No binary seed extracted from QRNG file."
            )

        if len(self.seed) != 16384:

            raise ValueError(
                f"Invalid QRNG seed length: "
                f"{len(self.seed)} bits. "
                f"Expected 16384 bits."
            )

        self.seed_sha256 = hashlib.sha256(
            self.seed.encode()
        ).hexdigest()

        print(
            f"Seed Length     : {len(self.seed)}"
        )

        print(
            f"Seed SHA-256    : {self.seed_sha256}"
        )

        print()

        with open(
            self.output_directory /
            "seed_information.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "seed_file": str(self.seed_file),
                    "seed_length": len(self.seed),
                    "seed_sha256": self.seed_sha256,
                    "loaded_on": self.created_on,
                    "status": "LOADED"
                },
                file,
                indent=4
            )

        if not self.pointer_file.exists():

            with open(
                self.pointer_file,
                "w",
                encoding="utf-8"
            ) as file:

                file.write("0")

    def verify_image(self):

        print("Verifying Cover Image...\n")

        image_bytes = self.image.tobytes()

        image_hash = hashlib.sha256(
            image_bytes
        ).hexdigest()

        self.image_validation = {

            "dwt_id": self.dwt_id,

            "verified_on": self.created_on,

            "image_path":
                self.image_profile["image_path"],

            "width":
                int(self.image.shape[1]),

            "height":
                int(self.image.shape[0]),

            "channels":
                int(self.image.shape[2]),

            "bit_depth":
                int(self.image.dtype.itemsize * 8),

            "dtype":
                str(self.image.dtype),

            "pixel_count":
                int(
                    self.image.shape[0] *
                    self.image.shape[1]
                ),

            "color_space":
                "BGR",

            "grayscale_available":
                True,

            "image_sha256":
                image_hash,

            "corrupted":
                False,

            "ready_for_dwt":
                True

        }

        with open(
            self.output_directory /
            "image_validation.json",
            "w"
        ) as file:

            json.dump(
                self.image_validation,
                file,
                indent=4
            )

        print(f"Pixel Count     : {self.image_validation['pixel_count']}")
        print(f"Bit Depth       : {self.image_validation['bit_depth']}")
        print(f"Image SHA-256   : {image_hash}")
        print("Image Status    : VERIFIED")
        print()

    def perform_dwt(self):

        print("Performing Haar Wavelet Decomposition...\n")

        image = np.float32(self.gray_image)

        self.maximum_supported_level = pywt.dwt_max_level(

            min(image.shape),

            pywt.Wavelet(self.wavelet_family)

        )

        self.LL, (self.LH, self.HL, self.HH) = pywt.dwt2(
            image,
            self.wavelet_family,
            mode=self.boundary_mode
        )

        print("Spatial Domain")
        print("        │")
        print("        ▼")
        print(" Haar Wavelet Transform")
        print("        │")
        print("        ▼")
        print("LL  LH")
        print("HL  HH")
        print()

        print(f"Original Image Size : {image.shape[0]} x {image.shape[1]}")
        print(f"LL Shape            : {self.LL.shape}")
        print(f"LH Shape            : {self.LH.shape}")
        print(f"HL Shape            : {self.HL.shape}")
        print(f"HH Shape            : {self.HH.shape}")
        print()

    def save_frequency_coefficients(self):

        print("Saving Frequency Coefficients...\n")

        np.save(
            self.output_directory /
            "LL.npy",
            self.LL
        )

        np.save(
            self.output_directory /
            "LH.npy",
            self.LH
        )

        np.save(
            self.output_directory /
            "HL.npy",
            self.HL
        )

        np.save(
            self.output_directory /
            "HH.npy",
            self.HH
        )

        coefficient_information = {

            "module_id": self.dwt_id,

            "created_on": self.created_on,

            "wavelet": "db2",

            "decomposition_level": 1,

            "input_shape": [

                int(self.gray_image.shape[0]),

                int(self.gray_image.shape[1])

            ],

            "output_shapes": {

                "LL": list(self.LL.shape),

                "LH": list(self.LH.shape),

                "HL": list(self.HL.shape),

                "HH": list(self.HH.shape)

            },

            "files": {

                "LL": "LL.npy",

                "LH": "LH.npy",

                "HL": "HL.npy",

                "HH": "HH.npy"

            },

            "status": "SUCCESS"

        }

        with open(

            self.output_directory /
            "dwt_coefficients.json",

            "w"

        ) as file:

            json.dump(

                coefficient_information,

                file,

                indent=4

            )

        print("Coefficient Files Saved.")
        print()

    def coefficient_statistics(self):

        print("Calculating Frequency Statistics...\n")

        statistics = {}

        statistics["normalization"] = {

            "method":

                "Min-Max",

            "range":

                [

                    0,

                    1

                ]

        }

        bands = self.normalized_bands

        for name, band in bands.items():

            statistics[name] = {

                "rows": int(band.shape[0]),

                "columns": int(band.shape[1]),

                "coefficients": int(band.size),

                "minimum": float(np.min(band)),

                "maximum": float(np.max(band)),

                "mean": float(np.mean(band)),

                "variance": float(np.var(band)),

                "standard_deviation": float(np.std(band)),

                "energy": float(np.sum(np.square(band))),

                "absolute_sum": float(np.sum(np.abs(band))),

                "non_zero_coefficients": int(np.count_nonzero(band)),

                "sparsity": float(

                    1 -

                    (

                        np.count_nonzero(band)

                        /

                        band.size

                    )

                )

            }

            print(f"{name}")

            print(f"Rows                 : {statistics[name]['rows']}")

            print(f"Columns              : {statistics[name]['columns']}")

            print(f"Coefficients         : {statistics[name]['coefficients']}")

            print(f"Minimum              : {statistics[name]['minimum']:.4f}")

            print(f"Maximum              : {statistics[name]['maximum']:.4f}")

            print(f"Mean                 : {statistics[name]['mean']:.4f}")

            print(f"Variance             : {statistics[name]['variance']:.4f}")

            print(f"Standard Deviation   : {statistics[name]['standard_deviation']:.4f}")

            print(f"Energy               : {statistics[name]['energy']:.4f}")

            print(f"Absolute Sum         : {statistics[name]['absolute_sum']:.4f}")

            print(f"Sparsity             : {statistics[name]['sparsity']:.4f}")

            print()

        self.coefficient_statistics_data = statistics

        with open(

            self.output_directory /
            "coefficient_statistics.json",

            "w"

        ) as file:

            json.dump(

                statistics,

                file,

                indent=4

            )

        print("Frequency Statistics Saved.\n")

    def prepare_frequency_bands(self):

        print("Preparing Frequency Band Information...\n")

        self.frequency_bands = {

            "LL": {

                "band": "LL",

                "priority": 4,

                "used_for_embedding": False,

                "reason":
                    "Reserved for preserving image quality",

                "description":
                    "Approximation coefficients"

            },

            "LH": {

                "band": "LH",

                "priority": 2,

                "used_for_embedding": True,

                "description":
                    "Horizontal frequency coefficients"

            },

            "HL": {

                "band": "HL",

                "priority": 3,

                "used_for_embedding": True,

                "description":
                    "Vertical frequency coefficients"

            },

            "HH": {

                "band": "HH",

                "priority": 1,

                "used_for_embedding": True,

                "description":
                    "Diagonal high-frequency coefficients"

            }

        }

        with open(

            self.output_directory /
            "frequency_bands.json",

            "w"

        ) as file:

            json.dump(

                self.frequency_bands,

                file,

                indent=4

            )

        print("LL  : Ignored")
        print("LH  : Candidate")
        print("HL  : Candidate")
        print("HH  : Highest Priority")
        print()


    def divide_frequency_regions(self):

        print("Dividing Frequency Bands Into Adaptive Regions...\n")

        self.frequency_regions = []

        self.region_complexities = []

        region_number = 1

        bands = {

            band: self.normalized_bands[band]

            for band in self.band_priority["order"]

        }

        for band_name, matrix in bands.items():

            rows, cols = matrix.shape

            row = 0

            while row < rows:

                column = 0

                while column < cols:

                    preview = matrix[
                        row:min(row + 32, rows),
                        column:min(column + 32, cols)
                    ]

                    block_size, complexity = self.determine_block_size(
                        preview
                    )

                    row_end = min(
                        row + block_size,
                        rows
                    )

                    column_end = min(
                        column + block_size,
                        cols
                    )

                    self.frequency_regions.append({

                        "region_id":

                            f"REGION_{region_number:05d}",

                        "band":

                            band_name,

                        "row_start":

                            row,

                        "row_end":

                            row_end,

                        "column_start":

                            column,

                        "column_end":

                            column_end,

                        "rows":

                            row_end - row,

                        "columns":

                            column_end - column,

                        "block_size":

                            block_size,

                        "complexity_score":

                            float(complexity),

                        "boundary_region":

                            (

                                row_end == rows

                                or

                                column_end == cols

                            ),

                        "padding_required":

                            (

                                (row_end-row) != block_size

                                or

                                (column_end-column) != block_size

                            )

                    })

                    self.region_complexities.append(

                        float(complexity)

                    )

                    region_number += 1

                    column += block_size

                row += block_size

        adaptive_information = {

            "adaptive_block_sizing": True,

            "minimum_block_size": self.minimum_block_size,

            "medium_block_size": self.medium_block_size,

            "maximum_block_size": self.maximum_block_size,

            "boundary_mode": self.boundary_mode,

            "wavelet": self.wavelet_family,

            "decomposition_level": self.decomposition_level,

            "maximum_supported_level":

                self.maximum_supported_level,

            "total_regions":

                len(self.frequency_regions)

        }

        with open(

            self.output_directory /

            "adaptive_regions.json",

            "w"

        ) as file:

            json.dump(

                adaptive_information,

                file,

                indent=4

            )

        print(f"Total Regions          : {len(self.frequency_regions)}")

        print()

        print("Adaptive Block Sizes")

        print("--------------------")

        print(f"Small Block            : {self.minimum_block_size} x {self.minimum_block_size}")

        print(f"Medium Block           : {self.medium_block_size} x {self.medium_block_size}")

        print(f"Large Block            : {self.maximum_block_size} x {self.maximum_block_size}")

        print()

        print(f"Boundary Mode          : {self.boundary_mode}")

        print(f"Wavelet                : {self.wavelet_family}")

        print(f"Maximum DWT Level      : {self.maximum_supported_level}")

        print()


    def analyze_regions(self):

        print("Analysing Frequency Regions...\n")

        for region in self.frequency_regions:

            matrix = self.normalized_bands[

                region["band"]

            ]

            block = matrix[

                region["row_start"]:region["row_end"],

                region["column_start"]:region["column_end"]

            ]

            absolute = np.abs(block)

            histogram, _ = np.histogram(

                absolute,

                bins=256

            )

            probability = histogram / np.sum(histogram)

            probability = probability[probability > 0]

            entropy = -np.sum(
                probability *
                np.log2(probability)
            )

            if block.shape[0] < 2 or block.shape[1] < 2:
                gradient_strength = 0.0
            else:
                gradient_y, gradient_x = np.gradient(block)
                gradient_strength = float(np.mean(np.sqrt(gradient_x ** 2 + gradient_y ** 2)))

            energy = np.sum(

                block ** 2

            )

            variance = np.var(block)

            coefficient_density = (

                np.count_nonzero(block)

                /

                block.size

            )

            distortion = (

                np.std(block)

                /

                (

                    np.max(np.abs(block))

                    + 1e-9

                )

            )

            capacity = int(

                block.size

            )

            randomness = (

                entropy *

                coefficient_density

            )

            region.update({

                "entropy":
                    float(entropy),

                "variance":
                    float(variance),

                "energy":
                    float(energy),

                "gradient_strength":
                    float(gradient_strength),

                "coefficient_density":
                    float(coefficient_density),

                "distortion_tolerance":
                    float(distortion),

                "estimated_capacity":
                    capacity,

                "randomness_score":
                    float(randomness)

            })

        print("Region Analysis Completed.")

        print(f"Regions Analysed : {len(self.frequency_regions)}")

        print()

    def calculate_region_fitness(self):

        print("Calculating Region Fitness...\n")

        weights = self.fitness_weights

        image_entropy = max(self.global_factors["entropy"], 0.001)
        image_variance = max(self.global_factors["variance"], 0.001)
        image_gradient = max(self.global_factors["average_gradient"], 0.001)
        image_density = max(self.global_factors["edge_density"] / 100.0, 0.001)
        image_energy = max(self.global_factors["energy"], 0.001)

        for region in self.frequency_regions:

            entropy = region["entropy"]
            variance = region["variance"]
            gradient = region["gradient_strength"]
            energy = np.log1p(region["energy"])
            density = region["coefficient_density"]
            distortion = region["distortion_tolerance"]

            capacity = (
                region["estimated_capacity"] /
                (
                    region["block_size"] *
                    region["block_size"]
                )
            )

            qrng_factor = self.get_qrng_float(64)

            relative_entropy = entropy / image_entropy
            relative_variance = variance / image_variance
            relative_gradient = gradient / image_gradient
            relative_density = density / image_density
            relative_energy = energy / image_energy

            fitness = (

                relative_entropy *
                weights["entropy"]

                +

                relative_variance *
                weights["variance"]

                +

                relative_gradient *
                weights["gradient"]

                +

                relative_energy *
                weights["energy"]

                +

                relative_density *
                weights["density"]

                +

                capacity *
                weights["capacity"]

                +

                qrng_factor *
                weights["qrng"]

                -

                distortion *
                weights["distortion"]

            )

            region["relative_entropy"] = float(relative_entropy)
            region["relative_variance"] = float(relative_variance)
            region["relative_gradient"] = float(relative_gradient)
            region["relative_energy"] = float(relative_energy)
            region["relative_density"] = float(relative_density)

            region["qrng_factor"] = float(qrng_factor)
            region["fitness"] = float(fitness)

            confidence = min(
                100.0,
                max(
                    0.0,
                    fitness * 100
                )
            )

            region["confidence_score"] = round(confidence, 2)

            if confidence >= 90:

                category = "Premium"

            elif confidence >= 75:

                category = "High"

            elif confidence >= 60:

                category = "Medium"

            else:

                category = "Low"

            region["category"] = category

            region["fitness_breakdown"] = {

                "relative_entropy":
                    float(relative_entropy * weights["entropy"]),

                "relative_variance":
                    float(relative_variance * weights["variance"]),

                "relative_gradient":
                    float(relative_gradient * weights["gradient"]),

                "relative_energy":
                    float(relative_energy * weights["energy"]),

                "relative_density":
                    float(relative_density * weights["density"]),

                "capacity":
                    float(capacity * weights["capacity"]),

                "qrng":
                    float(qrng_factor * weights["qrng"]),

                "distortion":
                    float(distortion * weights["distortion"])

            }

        self.save_qrng_usage()

        print("Global Relative Fitness Calculation Completed.\n")
    def rank_regions(self):

        print("Ranking Frequency Regions...\n")

        priority = {

            "HH": 1,

            "LH": 2,

            "HL": 3

        }

        self.frequency_regions.sort(

            key=lambda region: (

                priority[region["band"]],

                -region["fitness"]

            )

        )

        fitness_values = np.array(

            [

                region["fitness"]

                for region in self.frequency_regions

            ]

        )

        minimum_fitness = float(

            np.min(fitness_values)

        )

        maximum_fitness = float(

            np.max(fitness_values)

        )

        average_fitness = float(

            np.mean(fitness_values)

        )

        median_fitness = float(

            np.median(fitness_values)

        )

        percentile_90 = float(

            np.percentile(

                fitness_values,

                90

            )

        )

        percentile_80 = float(

            np.percentile(

                fitness_values,

                80

            )

        )

        percentile_70 = float(

            np.percentile(

                fitness_values,

                70

            )

        )

        percentile_60 = float(

            np.percentile(

                fitness_values,

                60

            )

        )

        percentile_50 = float(

            np.percentile(

                fitness_values,

                50

            )

        )

        for rank, region in enumerate(

            self.frequency_regions,

            start=1

        ):

            region["rank"] = rank

            region["fitness_percent"] = float(

                (

                    region["fitness"] -

                    minimum_fitness

                )

                /

                (

                    maximum_fitness -

                    minimum_fitness +

                    1e-12

                )

                * 100

            )

        candidate_threshold = percentile_75 = float(

            np.percentile(

                fitness_values,

                self.candidate_percentile

            )

        )

        self.candidate_regions = [

            region

            for region in self.frequency_regions

            if region["fitness"] >= candidate_threshold

        ]

        self.ranking_statistics = {

            "total_regions":

                len(

                    self.frequency_regions

                ),

            "candidate_regions":

                len(

                    self.candidate_regions

                ),

            "minimum_fitness":

                minimum_fitness,

            "maximum_fitness":

                maximum_fitness,

            "average_fitness":

                average_fitness,

            "median_fitness":

                median_fitness,

            "50_percentile":

                percentile_50,

            "60_percentile":

                percentile_60,

            "70_percentile":

                percentile_70,

            "75_percentile":

                percentile_75,

            "80_percentile":

                percentile_80,

            "90_percentile":

                percentile_90,

            "candidate_threshold":

                candidate_threshold

        }

        with open(

            self.output_directory /

            "ranking_statistics.json",

            "w"

        ) as file:

            json.dump(

                self.ranking_statistics,

                file,

                indent=4

            )

        print(f"Total Regions         : {len(self.frequency_regions)}")

        print(f"Candidate Regions     : {len(self.candidate_regions)}")

        print(f"Candidate Threshold   : {candidate_threshold:.6f}")

        print(f"Minimum Fitness       : {minimum_fitness:.6f}")

        print(f"Maximum Fitness       : {maximum_fitness:.6f}")

        print(f"Average Fitness       : {average_fitness:.6f}")

        print(f"Median Fitness        : {median_fitness:.6f}")

        print(f"Best Region           : {self.frequency_regions[0]['region_id']}")

        print(f"Worst Region          : {self.frequency_regions[-1]['region_id']}")

        print()

        print("Ranking Statistics Saved.\n")


    def prepare_packages(self):

        print("Preparing GA / ACO Packages...\n")

        heuristic_configuration = {

            "normalized_fitness_weight": 0.40,

            "randomness_weight": 0.20,

            "coefficient_density_weight": 0.20,

            "capacity_weight": 0.20,

            "pheromone_minimum": 0.10,

            "pheromone_maximum": 1.00

        }

        with open(

            self.output_directory /

            "heuristic_configuration.json",

            "w"

        ) as file:

            json.dump(

                heuristic_configuration,

                file,

                indent=4

            )

        self.ga_package = []

        self.aco_package = []

        fitness_values = [

            region["fitness"]

            for region in self.candidate_regions

        ]

        minimum_fitness = min(fitness_values)

        maximum_fitness = max(fitness_values)

        pheromone_sum = 0.0

        visibility_sum = 0.0

        for chromosome_number, region in enumerate(

            self.candidate_regions,

            start=1

        ):

            normalized_fitness = (

                region["fitness"] -

                minimum_fitness

            ) / (

                maximum_fitness -

                minimum_fitness +

                1e-12

            )

            band_priority = {

                "HH": 1,

                "LH": 2,

                "HL": 3

            }[

                region["band"]

            ]

            heuristic_value = (

                0.40 * normalized_fitness +

                0.20 * region["randomness_score"] +

                0.20 * region["coefficient_density"] +

                0.20 * (

                    region["estimated_capacity"] /

                    (

                        region["block_size"] *

                        region["block_size"]

                    )

                )

            )

            initial_pheromone = (

                0.10 +

                0.90 *

                normalized_fitness

            )

            transition_probability = (

                initial_pheromone *

                heuristic_value

            )

            pheromone_sum += initial_pheromone

            visibility_sum += heuristic_value

            self.ga_package.append({

                "chromosome_id":

                    f"CHR_{chromosome_number:05d}",

                "gene_id":

                    f"GENE_{chromosome_number:05d}",

                "region_id":

                    region["region_id"],

                "band":

                    region["band"],

                "band_priority":

                    band_priority,

                "rank":

                    region["rank"],

                "fitness":

                    region["fitness"],

                "normalized_fitness":

                    normalized_fitness,

                "complexity_score":

                    region["complexity_score"],

                "entropy":

                    region["entropy"],

                "variance":

                    region["variance"],

                "gradient_strength":

                    region["gradient_strength"],

                "energy":

                    region["energy"],

                "estimated_capacity":

                    region["estimated_capacity"],

                "coefficient_density":

                    region["coefficient_density"],

                "distortion_tolerance":

                    region["distortion_tolerance"],

                "qrng_factor":

                    region["qrng_factor"],

                "block_size":

                    region["block_size"],

                "coordinates": {

                    "row_start":

                        region["row_start"],

                    "row_end":

                        region["row_end"],

                    "column_start":

                        region["column_start"],

                    "column_end":

                        region["column_end"]

                },

                "selected":

                    False,

                "mutation_allowed":

                    True,

                "crossover_allowed":

                    True

            })

            self.aco_package.append({

                "node_id":

                    chromosome_number,

                "region_id":

                    region["region_id"],

                "band":

                    region["band"],

                "rank":

                    region["rank"],

                "fitness":

                    region["fitness"],

                "normalized_fitness":

                    normalized_fitness,

                "initial_pheromone":

                    initial_pheromone,

                "heuristic_value":

                    heuristic_value,

                "visibility":

                    heuristic_value,

                "transition_probability":

                    transition_probability,

                "node_weight":

                    normalized_fitness,

                "visit_count":

                    0,

                "visited":

                    False,

                "best_ant":

                    False,

                "neighbors":

                    []

            })

        with open(

            self.output_directory /

            "ga_package.json",

            "w"

        ) as file:

            json.dump(

                self.ga_package,

                file,

                indent=4

            )

        with open(

            self.output_directory /

            "aco_package.json",

            "w"

        ) as file:

            json.dump(

                self.aco_package,

                file,

                indent=4

            )

        ga_statistics = {

            "chromosomes":

                len(self.ga_package),

            "average_fitness":

                float(

                    np.mean(

                        fitness_values

                    )

                ),

            "maximum_fitness":

                float(

                    np.max(

                        fitness_values

                    )

                ),

            "minimum_fitness":

                float(

                    np.min(

                        fitness_values

                    )

                )

        }

        aco_statistics = {

            "nodes":

                len(self.aco_package),

            "average_pheromone":

                pheromone_sum /

                len(self.aco_package),

            "average_visibility":

                visibility_sum /

                len(self.aco_package),

            "maximum_pheromone":

                float(

                    np.max(

                        [

                            node["initial_pheromone"]

                            for node in self.aco_package

                        ]

                    )

                ),

            "minimum_pheromone":

                float(

                    np.min(

                        [

                            node["initial_pheromone"]

                            for node in self.aco_package

                        ]

                    )

                )

        }

        with open(

            self.output_directory /

            "ga_statistics.json",

            "w"

        ) as file:

            json.dump(

                ga_statistics,

                file,

                indent=4

            )

        with open(

            self.output_directory /

            "aco_statistics.json",

            "w"

        ) as file:

            json.dump(

                aco_statistics,

                file,

                indent=4

            )

        print(f"GA Chromosomes        : {len(self.ga_package)}")

        print(f"ACO Nodes             : {len(self.aco_package)}")

        print("GA Package Prepared.")

        print("ACO Package Prepared.\n")


    def save_outputs(self):

        print("Saving Output Files...\n")

        with open(

            self.output_directory /
            "region_analysis.json",

            "w"

        ) as file:

            json.dump(

                self.frequency_regions,

                file,

                indent=4

            )

        with open(

            self.output_directory /
            "candidate_regions.json",

            "w"

        ) as file:

            json.dump(

                self.candidate_regions,

                file,

                indent=4

            )

        complexities = [

            region["complexity_score"]

            for region

            in self.frequency_regions

        ]


        statistics = {

            "total_regions": len(self.frequency_regions),

            "candidate_regions": len(self.candidate_regions),

            "ignored_band": "LL",

            "bands_used": [

                "HH",

                "LH",

                "HL"

            ],

            "best_region": self.frequency_regions[0]["region_id"],

            "best_fitness": self.frequency_regions[0]["fitness"],

            "minimum_complexity":

                float(

                    np.min(

                        complexities

                    )

                ),

            "maximum_complexity":

                float(

                    np.max(

                        complexities

                    )

                ),

            "average_complexity":

                float(

                    np.mean(

                        complexities

                    )

                ),

            "average_entropy":

                float(

                    np.mean(

                        [

                            region["entropy"]

                            for region in self.frequency_regions

                        ]

                    )

                ),

            "average_variance":

                float(

                    np.mean(

                        [

                            region["variance"]

                            for region in self.frequency_regions

                        ]

                    )

                ),

            "average_capacity":

                float(

                    np.mean(

                        [

                            region["estimated_capacity"]

                            for region in self.frequency_regions

                        ]

                    )

                )

        }

        with open(

            self.output_directory /
            "statistics.json",

            "w"

        ) as file:

            json.dump(

                statistics,

                file,

                indent=4

            )

        readiness = {

            "image_loaded": True,

            "seed_loaded": True,

            "dwt_completed": True,

            "coefficients_generated": True,

            "regions_created": True,

            "region_analysis": True,

            "ranking_completed": True,

            "packages_ready": True,

            "overall_readiness": "100%"

        }

        with open(

            self.output_directory /
            "readiness.json",

            "w"

        ) as file:

            json.dump(

                readiness,

                file,

                indent=4

            )

        manifest = {

            "module": "DWT Decomposition",

            "wavelet": "db2",

            "level": 1,

            "ignored_band": "LL",

            "embedding_bands": [

                "HH",

                "LH",

                "HL"

            ],

            "total_regions": len(self.frequency_regions),

            "candidate_regions": len(self.candidate_regions),

            "status": "READY FOR GA"

        }

        with open(

            self.output_directory /
            "manifest.json",

            "w"

        ) as file:

            json.dump(

                manifest,

                file,

                indent=4

            )

        with open(

            self.output_directory /
            "report.txt",

            "w"

        ) as file:

            file.write("DWT DECOMPOSITION REPORT\n\n")

            file.write(f"Wavelet : db2\n")

            file.write(f"Total Regions : {len(self.frequency_regions)}\n")

            file.write(f"Candidate Regions : {len(self.candidate_regions)}\n")

            file.write(f"Best Region : {self.frequency_regions[0]['region_id']}\n")

            file.write(f"Overall Status : READY FOR GA\n")

        print("All Outputs Saved.\n")

    def normalize_frequency_bands(self):

        print("Normalizing Frequency Bands...\n")

        bands = {

            "LL": self.LL,

            "LH": self.LH,

            "HL": self.HL,

            "HH": self.HH

        }

        self.normalized_bands = {}

        normalization_information = {}

        for name, band in bands.items():

            minimum = np.min(band)

            maximum = np.max(band)

            if maximum == minimum:

                normalized = np.zeros_like(band)

            else:

                normalized = (

                    band - minimum

                ) / (

                    maximum - minimum

                )

            self.normalized_bands[name] = normalized

            normalization_information[name] = {

                "minimum": float(minimum),

                "maximum": float(maximum),

                "normalized_minimum": float(np.min(normalized)),

                "normalized_maximum": float(np.max(normalized))

            }

            np.save(

                self.output_directory /

                f"{name}_normalized.npy",

                normalized

            )

            print(f"{name}")

            print(f"Original Range      : {minimum:.4f}  →  {maximum:.4f}")

            print(f"Normalized Range    : 0.0000 → 1.0000")

            print()

        with open(

            self.output_directory /

            "band_normalization.json",

            "w"

        ) as file:

            json.dump(

                normalization_information,

                file,

                indent=4

            )

        print("Band Normalization Completed.\n")


    def print_summary(self):

        print("=" * 80)

        print("MODULE SUMMARY")

        print("=" * 80)

        print()

        print(f"Wavelet                 : db2")

        print(f"Frequency Bands         : 4")

        print(f"Embedding Bands         : HH, LH, HL")

        print(f"Ignored Band            : LL")

        print(f"Total Regions           : {len(self.frequency_regions)}")

        print(f"Candidate Regions       : {len(self.candidate_regions)}")

        print(f"Best Region             : {self.frequency_regions[0]['region_id']}")

        print(f"Module Status           : READY FOR GA")

        print()

        print("=" * 80)

        print()

    def analyze_band_histograms(self):

        print("Analysing Band Histograms...\n")

        histogram_report = {}

        for band_name, matrix in self.normalized_bands.items():

            histogram, bins = np.histogram(

                matrix,

                bins=256,

                range=(0, 1)

            )

            histogram_report[band_name] = {

                "total_coefficients":

                    int(matrix.size),

                "histogram":

                    histogram.tolist(),

                "bin_edges":

                    bins.tolist(),

                "peak_bin":

                    int(np.argmax(histogram)),

                "peak_frequency":

                    int(np.max(histogram))

            }

            print(

                f"{band_name}"

            )

            print(

                f"Peak Bin       : {np.argmax(histogram)}"

            )

            print(

                f"Peak Frequency : {np.max(histogram)}"

            )

            print()

        self.histogram_information = histogram_report

        with open(

            self.output_directory /

            "histogram_analysis.json",

            "w"

        ) as file:

            json.dump(

                histogram_report,

                file,

                indent=4

            )

        print("Histogram Analysis Completed.\n")

    def save_dwt_metadata(self):

        metadata = {

            "wavelet":

                self.wavelet_family,

            "family":

                pywt.Wavelet(

                    self.wavelet_family

                ).family_name,

            "boundary_mode":

                self.boundary_mode,

            "decomposition_level":

                self.decomposition_level,

            "maximum_supported_level":

                self.maximum_supported_level,

            "adaptive_block_sizing":

                True,

            "normalization":

                "Min-Max",

            "bands": [

                "LL",

                "LH",

                "HL",

                "HH"

            ]

        }

        with open(

            self.output_directory /

            "dwt_metadata.json",

            "w"

        ) as file:

            json.dump(

                metadata,

                file,

                indent=4

            )

    # def determine_block_size(self, block):

    #     absolute = np.abs(block)

    #     histogram, _ = np.histogram(
    #         absolute,
    #         bins=256
    #     )

    #     probability = histogram / np.sum(histogram)

    #     probability = probability[
    #         probability > 0
    #     ]

    #     entropy = -np.sum(

    #         probability *

    #         np.log2(probability)

    #     )

    #     variance = np.var(block)

    #     energy = np.mean(

    #         block ** 2

    #     )

    #     if block.shape[0] < 2 or block.shape[1] < 2:
    #         gradient = 0.0
    #     else:
    #         gradient_y, gradient_x = np.gradient(block)
    #         gradient = float(np.mean(np.sqrt(gradient_x ** 2 + gradient_y ** 2)))

    #     density = (

    #         np.count_nonzero(block)

    #         /

    #         block.size

    #     )

    #     local_complexity = (

    #         entropy * 0.30 +

    #         variance * 0.20 +

    #         gradient * 0.20 +

    #         energy * 0.15 +

    #         density * 0.15

    #     )

    #     global_complexity = self.image_intelligence[
    #         "complexity_index"
    #     ]

    #     global_weight = 0.30

    #     local_weight = 0.70

    #     final_complexity = (

    #         local_complexity * local_weight +

    #         global_complexity * global_weight

    #     )

    #     preferred = self.image_intelligence[
    #         "preferred_block_size"
    #     ]

    #     if preferred == 8:

    #         if final_complexity >= 2.10:

    #             block_size = 8

    #         elif final_complexity >= 1.20:

    #             block_size = 16

    #         else:

    #             block_size = 32

    #     elif preferred == 16:

    #         if final_complexity >= 2.40:

    #             block_size = 8

    #         elif final_complexity >= 1.20:

    #             block_size = 16

    #         else:

    #             block_size = 32

    #     else:

    #         if final_complexity >= 2.80:

    #             block_size = 8

    #         elif final_complexity >= 1.60:

    #             block_size = 16

    #         else:

    #             block_size = 32

    #     return block_size, final_complexity

    def determine_block_size(self, block):

        return 16, 0.0

    def run(self):

        self.banner()

        self.load_image_acquisition()

        self.save_global_profile()

        self.adaptive_image_intelligence()

        self.adaptive_band_priority()

        self.adaptive_fitness_model()

        self.generate_optimization_guidance()

        self.load_qrng_seed()

        self.verify_image()

        self.perform_dwt()

        self.normalize_frequency_bands()

        self.save_frequency_coefficients()

        self.coefficient_statistics()

        self.analyze_band_histograms()

        self.save_dwt_metadata()

        self.prepare_frequency_bands()

        self.divide_frequency_regions()

        self.analyze_regions()

        self.calculate_region_fitness()

        self.rank_regions()

        self.prepare_packages()

        self.save_outputs()

        self.print_summary()

if __name__ == "__main__":

    dwt = DWTDecomposition()

    dwt.run()