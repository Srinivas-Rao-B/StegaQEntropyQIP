import os
import hashlib
import time


class SeedManager:

    def __init__(self, qrng_pool_file):

        self.qrng_pool_file = qrng_pool_file

        self.pointer_file = "qrng/output/qrng_pointer.txt"

        self.output_folder = "qrng/output/seeds"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

        self.pool = ""

        self.pool_pointer = 0

        self.total_pool_bits = 0

        self.total_consumed_bits = 0

        self.generated_seeds = {}

        self.seed_sizes = {

            "master_seed": 16384,

            "session_seed": 16384,

            "region_seed": 16384,

            "payload_seed": 16384,

            "block_seed": 16384,

            "chunk_seed": 16384,

            "dwt_seed": 16384,

            "entropy_seed": 16384,

            "gradient_seed": 16384,

            "threshold_seed": 16384,

            "adaptive_embedding_seed": 16384,

            "linear_regression_seed": 16384,

            "ga_seed": 16384,

            "aco_seed": 16384,

            "pso_seed": 16384,

            "qkd_seed": 16384,

            "permutation_seed": 16384,

            "position_seed": 16384,

            "coefficient_seed": 16384,

            "noise_seed": 16384,

            "authentication_seed": 16384,

            "synchronization_seed": 16384,

            "receiver_seed": 16384,

            "extraction_seed": 16384,

            "integrity_seed": 16384,

            "verification_seed": 16384

        }

    def banner(self):

        print("\n" + "=" * 70)

        print("QUANTUM SEED MANAGER")

        print("=" * 70)


    def load_qrng_pool(self):

        print("\nLoading QRNG Pool...")

        with open(
            self.qrng_pool_file,
            "r"
        ) as file:

            self.pool = "".join(
                line.strip()
                for line in file
            )

        self.total_pool_bits = len(self.pool)

        if os.path.exists(self.pointer_file):

            with open(self.pointer_file, "r") as file:

                value = file.read().strip()

                if value:

                    self.pool_pointer = int(value)

                else:

                    self.pool_pointer = 0

        else:

            self.pool_pointer = 0

            with open(self.pointer_file, "w") as file:

                file.write("0")

        self.total_consumed_bits = self.pool_pointer

        print("QRNG Pool Loaded Successfully.")
        print(f"Available Quantum Bits : {self.total_pool_bits:,}")
        print(f"Current Pointer        : {self.pool_pointer:,}")


    def pool_information(self):

        print("\n" + "-" * 70)

        print("POOL INFORMATION")

        print("-" * 70)

        print(f"Total Pool Bits        : {self.total_pool_bits:,}")

        print(f"Consumed Bits          : {self.total_consumed_bits:,}")

        print(f"Remaining Bits         : {self.total_pool_bits-self.total_consumed_bits:,}")

        print(f"Current Pointer        : {self.pool_pointer:,}")

        print(f"Configured Seeds       : {len(self.seed_sizes)}")


    def list_seed_configuration(self):

        print("\n" + "-" * 70)

        print("PROJECT SEED CONFIGURATION")

        print("-" * 70)

        for name, size in self.seed_sizes.items():

            print(f"{name:<35}{size:>6} bits")

    def consume_bits(self, number_of_bits):

        remaining = self.total_pool_bits - self.pool_pointer

        if remaining < number_of_bits:

            raise RuntimeError(
                "QRNG Pool Exhausted. Generate More Quantum Bits."
            )

        start = self.pool_pointer

        end = start + number_of_bits

        seed = self.pool[start:end]

        self.pool_pointer = end

        self.total_consumed_bits = self.pool_pointer

        with open(self.pointer_file, "w") as file:

            file.write(str(self.pool_pointer))

        return seed


    def binary_to_hex(self, binary_seed):

        integer = int(binary_seed, 2)

        digits = (len(binary_seed) + 3) // 4

        return format(integer, f"0{digits}X")


    def sha256_hash(self, binary_seed):

        return hashlib.sha256(

            binary_seed.encode()

        ).hexdigest()


    def generate_seed(self, seed_name):

        if seed_name not in self.seed_sizes:

            raise ValueError(
                f"{seed_name} is not defined."
            )

        print("\n" + "=" * 70)

        print(seed_name.upper())

        print("=" * 70)

        bits_required = self.seed_sizes[seed_name]

        start_time = time.time()

    
        start_bit = self.pool_pointer

        binary_seed = self.consume_bits(bits_required)

        end_bit = self.pool_pointer - 1

        hexadecimal_seed = self.binary_to_hex(

            binary_seed

        )


        fingerprint = self.sha256_hash(

            binary_seed

        )

        elapsed = time.time() - start_time

        self.generated_seeds[seed_name] = {

            "binary": binary_seed,

            "hex": hexadecimal_seed,

            "hash": fingerprint,

            "bits": bits_required,

            "generation_time": elapsed,
            "pool_start": start_bit,
            "pool_end": end_bit

        }

        print(f"Seed Length          : {bits_required} bits")

        print(f"Generation Time      : {elapsed:.6f} sec")

        print(f"Remaining Pool       : {self.total_pool_bits-self.total_consumed_bits:,}")

        print()

        print("Binary Preview")

        print(binary_seed[:128] + "...")

        print()

        print("Hex Preview")

        print(hexadecimal_seed[:64] + "...")

        print()

        print("Integer Preview")

        print("Skipped (16384-bit integer)")

        print()

        print("SHA256")

        print(fingerprint)

        self.save_seed(seed_name)

        return os.path.join(
            self.output_folder,
            f"{seed_name}.txt"
)


    def save_seed(self, seed_name):

        seed = self.generated_seeds[seed_name]

        filename = os.path.join(

            self.output_folder,

            f"{seed_name}.txt"

        )

        with open(filename, "w") as file:

            file.write("=" * 70 + "\n")

            file.write(f"{seed_name.upper()}\n")

            file.write("=" * 70 + "\n\n")

            file.write(f"Length : {seed['bits']} bits\n\n")

            file.write(f"Pool Start Bit : {seed['pool_start']}\n")

            file.write(f"Pool End Bit   : {seed['pool_end']}\n\n")

            file.write("Binary\n")

            file.write("-" * 40 + "\n")

            file.write(seed["binary"] + "\n\n")

            file.write("Hexadecimal\n")

            file.write("-" * 40 + "\n")

            file.write(seed["hex"] + "\n\n")

            file.write("Integer\n")

            file.write("-" * 40 + "\n")

            file.write("Skipped (16384-bit integer)\n\n")

            file.write("SHA256\n")

            file.write("-" * 40 + "\n")

            file.write(seed["hash"] + "\n")

        print(f"Saved : {seed_name}.txt")

    def generate_all_seeds(self):

        print("\n" + "=" * 70)

        print("GENERATING ALL QUANTUM PROJECT SEEDS")

        print("=" * 70)

        total_required = sum(self.seed_sizes.values())

        print(f"\nTotal Seeds           : {len(self.seed_sizes)}")

        print(f"Required QRNG Bits    : {total_required:,}")

        print(f"Available QRNG Bits   : {self.total_pool_bits-self.total_consumed_bits:,}")

        if total_required > (self.total_pool_bits-self.total_consumed_bits):

            raise RuntimeError(
                "Insufficient QRNG Pool."
            )

        print("\nStarting Automatic Seed Generation...\n")

        for seed_name in self.seed_sizes:

            self.generate_seed(seed_name)

            

        print("\nAll Quantum Seeds Generated Successfully.")


    def allocation_statistics(self):

        print("\n" + "=" * 70)

        print("QRNG ALLOCATION STATISTICS")

        print("=" * 70)

        total_required = sum(self.seed_sizes.values())

        remaining = self.total_pool_bits - self.total_consumed_bits

        utilization = (self.total_consumed_bits / self.total_pool_bits) * 100

        print(f"\nTotal Pool Bits          : {self.total_pool_bits:,}")

        print(f"Consumed Bits            : {self.total_consumed_bits:,}")

        print(f"Remaining Bits           : {remaining:,}")

        print(f"Pool Utilization         : {utilization:.4f}%")

        print(f"Generated Seeds          : {len(self.generated_seeds)}")


    def dependency_map(self):

        print("\n" + "=" * 70)

        print("SEED DEPENDENCY MAP")

        print("=" * 70)

        dependencies = {

            "master_seed"               : "Entire Project",

            "session_seed"              : "Runtime Session",

            "region_seed"               : "Region Selection",

            "payload_seed"              : "Payload Allocation",

            "block_seed"                : "Block Partitioning",

            "chunk_seed"                : "Chunk Division",

            "dwt_seed"                  : "Wavelet Transform",

            "entropy_seed"              : "Entropy Analysis",

            "gradient_seed"             : "Gradient Analysis",

            "threshold_seed"            : "Adaptive Threshold",

            "adaptive_embedding_seed"   : "Embedding Controller",

            "linear_regression_seed"    : "Prediction Module",

            "ga_seed"                   : "Genetic Algorithm",

            "aco_seed"                  : "Ant Colony Optimization",

            "pso_seed"                  : "Particle Swarm Optimization",

            "qkd_seed"                  : "Quantum Key Distribution",

            "permutation_seed"          : "Permutation Generator",

            "position_seed"             : "Pixel Selection",

            "coefficient_seed"          : "Coefficient Selection",

            "noise_seed"                : "Noise Estimation",

            "authentication_seed"       : "Authentication",

            "synchronization_seed"      : "Sender-Receiver Sync",

            "receiver_seed"             : "Receiver Module",

            "extraction_seed"           : "Extraction Module",

            "integrity_seed"            : "Integrity Verification",

            "verification_seed"         : "Final Verification"

        }

        for seed, purpose in dependencies.items():

            print(f"{seed:<35} {purpose}")


    def save_summary(self):

        filename = os.path.join(

            self.output_folder,

            "seed_summary.txt"

        )

        with open(filename, "w") as file:

            file.write("="*70+"\n")

            file.write("STEGAQENTROPY QUANTUM SEED SUMMARY\n")

            file.write("="*70+"\n\n")

            file.write(f"Generated Seeds : {len(self.generated_seeds)}\n")
            

            file.write(f"Total Pool Bits : {self.total_pool_bits}\n")

            file.write(f"Consumed Bits   : {self.total_consumed_bits}\n")

            file.write(f"Remaining Bits  : {self.total_pool_bits-self.total_consumed_bits}\n\n")

            file.write("Seed Allocation\n")

            file.write("-"*60+"\n")

            for seed in self.generated_seeds:

                file.write(f"{seed:<35}")

                file.write(f"{self.generated_seeds[seed]['bits']} bits\n")

        print("\nSaved : seed_summary.txt")

    def get_seed(self, seed_name):

        if seed_name not in self.generated_seeds:

            raise KeyError(
                f"{seed_name} has not been generated."
            )

        return self.generated_seeds[seed_name]


    def get_binary_seed(self, seed_name):

        return self.get_seed(seed_name)["binary"]


    def get_hex_seed(self, seed_name):

        return self.get_seed(seed_name)["hex"]


    def print_seed_summary(self):

        print("\n" + "=" * 70)
        print("PROJECT SEED SUMMARY")
        print("=" * 70)

        print()

        print(f"Generated Seeds          : {len(self.generated_seeds)}")

        print(f"Pool Size                : {self.total_pool_bits:,}")

        print(f"Consumed Bits            : {self.total_consumed_bits:,}")

        print(f"Remaining Bits           : {self.total_pool_bits-self.total_consumed_bits:,}")

        print()

        print("Generated Seed List")

        print("-"*70)

        for seed_name in self.generated_seeds:

            info = self.generated_seeds[seed_name]

            print(

                f"{seed_name:<35}"

                f"{info['bits']:>5} bits"

            )

        print()


    def export_seed_database(self):

        filename = os.path.join(

            self.output_folder,

            "seed_database.txt"

        )

        with open(filename, "w") as file:

            file.write("="*70+"\n")

            file.write("STEGAQENTROPY QUANTUM SEED DATABASE\n")

            file.write("="*70+"\n\n")

            for seed_name in self.generated_seeds:

                seed = self.generated_seeds[seed_name]

                file.write(f"{seed_name.upper()}\n")

                file.write("-"*70+"\n")

                file.write(f"Length : {seed['bits']} bits\n")

                file.write(f"Pool Start : {seed['pool_start']}\n")

                file.write(f"Pool End   : {seed['pool_end']}\n")

                file.write(f"SHA256 : {seed['hash']}\n")

                file.write(f"Binary : {seed['binary']}\n")

                file.write(f"Hex    : {seed['hex']}\n")

                file.write("Integer: Skipped (16384-bit integer)\n")

                file.write("\n")

        print("Seed Database Saved Successfully.")


    def final_summary(self):

        print("\n" + "=" * 70)
        print("QUANTUM SEED MANAGER COMPLETED")
        print("=" * 70)

        print()

        print("Generated Files")

        print("-"*70)

        print("master_seed.txt")

        print("session_seed.txt")

        print("payload_seed.txt")

        print("ga_seed.txt")

        print("aco_seed.txt")

        print("...")

        print("seed_summary.txt")

        print("seed_database.txt")

        print()

        print("Quantum Seed Manager Ready For")

        print("-"*70)

        print("Region Analysis")

        print("DWT")

        print("Linear Regression")

        print("GA")

        print("ACO")

        print("QKD")

        print("Adaptive Embedding")

        print("Extraction")

        print("Receiver")

        print("Authentication")

        print("Integrity Verification")

        print("Synchronization")

        print()


    def run(self):

        self.banner()

        self.load_qrng_pool()

        self.pool_information()

        self.list_seed_configuration()

        self.generate_all_seeds()

        self.allocation_statistics()

        self.dependency_map()

        self.save_summary()

        self.export_seed_database()

        self.print_seed_summary()

        self.final_summary()

        return self