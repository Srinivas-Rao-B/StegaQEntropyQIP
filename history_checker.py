import os
import time
import math
import json
from qrng.seed_manager import SeedManager


class HistoryChecker:

    def __init__(self):


        self.output_folder = "qrng/output/history_check"
        
        self.seed_manager = SeedManager(
            "qrng/output/qrng_pool.txt"
        )
        self.seed_manager.load_qrng_pool()

        print("QRNG Pool Loaded Successfully.")

        self.verified_folder = os.path.join(
            self.output_folder,
            "verified_seeds"
        )

        self.regenerated_folder = os.path.join(
            self.output_folder,
            "regenerated_seeds"
        )

        self.rejected_folder = os.path.join(
            self.output_folder,
            "rejected_seeds"
        )

        self.history_database = os.path.join(
            self.output_folder,
            "history_database.txt"
        )

        self.history_summary = os.path.join(
            self.output_folder,
            "history_summary.txt"
        )
        self.database_file = os.path.join(
            self.output_folder,
            "history_database.txt"
        )

        self.final_verified_folder = os.path.join(
            self.output_folder,
            "final_verified_seeds"
        )

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.verified_folder, exist_ok=True)
        os.makedirs(self.regenerated_folder, exist_ok=True)
        os.makedirs(self.rejected_folder, exist_ok=True)
        os.makedirs(self.final_verified_folder, exist_ok=True)

        self.database_records = []

        self.qrng_seed_folder = "qrng/output/seeds"

        self.qrng_seed_files = []

        self.retry_statistics = []

        self.maximum_retries = 6

        self.verified_count = 0

        self.rejected_count = 0

        self.regenerated_count = 0

        self.retry_count = 0

        self.start_time = time.time()

    def banner(self):

        print("\n" + "=" * 70)
        print("STEGAQENTROPY HISTORY CHECKER")
        print("=" * 70)

    def load_history_database(self):

        print("\nLoading History Database...")

        if not os.path.exists(self.history_database):

            open(
                self.history_database,
                "w"
            ).close()

        with open(
            self.history_database,
            "r"
        ) as file:

            self.database_records = []

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    self.database_records.append(
                        json.loads(line)
                    )

                except:

                    continue

        print("History Database Loaded.")

    def load_qrng_seeds(self):

        print("\nLoading Original QRNG Seeds...")

        excluded_files = {
            "seed_database.txt",
            "seed_summary.txt"
        }

        self.qrng_seed_files = sorted(
            [
                file
                for file in os.listdir(
                    self.qrng_seed_folder
                )
                if file.endswith(".txt")
                and file not in excluded_files
            ]
        )

        print(
            f"Original QRNG Seeds Loaded : "
            f"{len(self.qrng_seed_files)}"
        )

        for file in self.qrng_seed_files:
            print(
                f"  - {file}"
            )

    def initialization_summary(self):

        print("\n" + "-" * 70)
        print("INITIALIZATION SUMMARY")
        print("-" * 70)

        print(f"History Records          : {len(self.database_records)}")
        print(f"Original QRNG Seeds      : {len(self.qrng_seed_files)}")
        print()
        print(f"Maximum Retries          : {self.maximum_retries}")
        print()
        print(f"Verified Folder          : {self.verified_folder}")
        print(f"Rejected Folder          : {self.rejected_folder}")
        print(f"Regenerated Folder       : {self.regenerated_folder}")
        print()
        print("Initialization Complete.")

    def read_seed_file(self, filename):
        
        if os.path.exists(filename):

            path = filename

        else:

            path = os.path.join(
                self.qrng_seed_folder,
                os.path.basename(filename)
            )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read()

        data = {
            "seed_name": os.path.splitext(
                os.path.basename(filename)
            )[0],
            "binary": "",
            "sha256": "",
            "pool_start": 0,
            "pool_end": 0
        }

        lines = content.splitlines()

        for line in lines:

            line = line.strip()

            if line.startswith("Pool Start Bit"):

                data["pool_start"] = int(
                    line.split(":", 1)[1].strip()
                )

            elif line.startswith("Pool End Bit"):

                data["pool_end"] = int(
                    line.split(":", 1)[1].strip()
                )

            elif (
                len(line) > 1000
                and all(
                    char in "01"
                    for char in line
                )
            ):

                data["binary"] = line

            elif (
                len(line) == 64
                and all(
                    char in "0123456789abcdefABCDEF"
                    for char in line
                )
            ):

                data["sha256"] = line.lower()

        return data


    def extract_features(self, seed):

        binary = seed["binary"]

        bits = [int(bit) for bit in binary]

        length = len(bits)

        zeros = bits.count(0)

        ones = bits.count(1)

        mean = sum(bits) / length

        variance = sum(
            (bit - mean) ** 2
            for bit in bits
        ) / length

        std = math.sqrt(variance)

        transitions = sum(
            1
            for i in range(1, length)
            if bits[i] != bits[i - 1]
        )

        transition_rate = transitions / (length - 1)

        entropy = 0

        if zeros:
            p0 = zeros / length
            entropy -= p0 * math.log2(p0)

        if ones:
            p1 = ones / length
            entropy -= p1 * math.log2(p1)

        balance_ratio = min(zeros, ones) / max(zeros, ones)

        zero_runs = []
        one_runs = []

        current = bits[0]
        count = 1

        for bit in bits[1:]:

            if bit == current:
                count += 1
            else:
                if current == 0:
                    zero_runs.append(count)
                else:
                    one_runs.append(count)

                current = bit
                count = 1

        if current == 0:
            zero_runs.append(count)
        else:
            one_runs.append(count)

        longest_zero_run = max(zero_runs) if zero_runs else 0
        longest_one_run = max(one_runs) if one_runs else 0

        average_zero_run = sum(zero_runs) / len(zero_runs) if zero_runs else 0
        average_one_run = sum(one_runs) / len(one_runs) if one_runs else 0

        run_count = len(zero_runs) + len(one_runs)

        autocorrelation = 0

        for i in range(length - 1):

            if bits[i] == bits[i + 1]:
                autocorrelation += 1
            else:
                autocorrelation -= 1

        autocorrelation /= (length - 1)

        ideal_mean = 0.5

        if std == 0:
            gaussian_distance = 0.0
        else:
            gaussian_distance = abs(
                mean - ideal_mean
            ) / std

        gaussian_probability = math.exp(
            -(gaussian_distance ** 2) / 2
        )

        randomness_score = (

            entropy * 0.40 +

            transition_rate * 0.25 +

            balance_ratio * 0.20 +

            gaussian_probability * 0.15

        )

        if gaussian_distance <= 0.50:
            gaussian_strength = "VERY STRONG"
        elif gaussian_distance <= 1.00:
            gaussian_strength = "STRONG"
        elif gaussian_distance <= 2.00:
            gaussian_strength = "MODERATE"
        else:
            gaussian_strength = "WEAK"

        if entropy >= 0.995:
            nearest_region = "Excellent"
        elif entropy >= 0.985:
            nearest_region = "Good"
        elif entropy >= 0.970:
            nearest_region = "Moderate"
        else:
            nearest_region = "Poor"

        if gaussian_distance <= 0.50:
            possible_regions = ["Excellent"]
        elif gaussian_distance <= 1.00:
            possible_regions = ["Excellent", "Good"]
        elif gaussian_distance <= 2.00:
            possible_regions = ["Good", "Moderate"]
        else:
            possible_regions = ["Moderate", "Poor"]

        return {

            "seed_name": seed["seed_name"],

            "pool_start": seed["pool_start"],

            "pool_end": seed["pool_end"],

            "sha256": seed["sha256"],

            "binary": binary,

            "length": length,

            "zeros": zeros,

            "ones": ones,

            "mean": mean,

            "variance": variance,

            "std": std,

            "transition_rate": transition_rate,

            "entropy": entropy,
            "balance_ratio": balance_ratio,

            "longest_zero_run": longest_zero_run,

            "longest_one_run": longest_one_run,

            "average_zero_run": average_zero_run,

            "average_one_run": average_one_run,

            "run_count": run_count,

            "autocorrelation": autocorrelation,

            "gaussian_distance": gaussian_distance,

            "gaussian_probability": gaussian_probability,

            "gaussian_strength": gaussian_strength,

            "nearest_region": nearest_region,

            "possible_regions": possible_regions,

            "randomness_score": randomness_score

        }

    def feature_extraction(self):

        print("\n" + "=" * 70)

        print("FEATURE EXTRACTION")

        print("=" * 70)

        self.seed_features = []

        for filename in self.qrng_seed_files:

            seed = self.read_seed_file(

                filename

            )

            features = self.extract_features(

                seed

            )

            self.seed_features.append(

                features

            )

            print(f"\nSeed               : {features['seed_name']}")

            print(f"Pool Start         : {features['pool_start']}")

            print(f"Pool End           : {features['pool_end']}")

            print(f"Length             : {features['length']}")

            print(f"Zeros              : {features['zeros']}")

            print(f"Ones               : {features['ones']}")

            print(f"Mean               : {features['mean']:.6f}")

            print(f"Variance           : {features['variance']:.6f}")

            print(f"Std Dev            : {features['std']:.6f}")

            print(f"Transition Rate    : {features['transition_rate']:.6f}")

            print(f"Entropy            : {features['entropy']:.6f}")

            print(f"SHA256             : {features['sha256']}")


    def binary_similarity(self, binary1, binary2):

        matches = sum(

            1

            for a, b in zip(binary1, binary2)

            if a == b

        )

        return matches / len(binary1)


    def region_similarity(self, region1, region2):

        if region1 == region2:

            return 1.0

        return 0.0
    
    def quick_filter(self, seed1, seed2):

        if seed1["sha256"] == seed2["sha256"]:
            return True

        if seed1["binary"] == seed2["binary"]:
            return True

        if seed1["length"] != seed2["length"]:
            return False

        if abs(
            seed1["entropy"] -
            seed2["entropy"]
        ) > 0.05:
            return False

        if abs(
            seed1["mean"] -
            seed2["mean"]
        ) > 0.05:
            return False

        if abs(
            seed1["transition_rate"] -
            seed2["transition_rate"]
        ) > 0.10:
            return False

        return True

    
    def similarity_score(self, seed1, seed2):

        if seed1["sha256"] == seed2["sha256"]:

            return {
                "overall": 1.0,
                "duplicate": True
            }

        if seed1["binary"] == seed2["binary"]:

            return {
                "overall": 1.0,
                "duplicate": True
            }

        return {
            "overall": 0.0,
            "duplicate": False
        }

    def similarity_engine(self):

        print("\n" + "=" * 70)
        print("SIMILARITY ENGINE")
        print("=" * 70)

        self.similarity_results = []

        for seed in self.seed_features:

            best_score = 0.0
            best_match = None

            for record in self.database_records:

                if not self.quick_filter(
                    seed,
                    record
                ):
                    continue

                result = self.similarity_score(
                    seed,
                    record
                )

                if result["overall"] > best_score:

                    best_score = result["overall"]
                    best_match = result

            for other_seed in self.seed_features:

                if (
                    other_seed["seed_name"]
                    ==
                    seed["seed_name"]
                ):
                    continue

                result = self.similarity_score(
                    seed,
                    other_seed
                )

                if result["overall"] > best_score:

                    best_score = result["overall"]
                    best_match = result

            if best_match is None:

                best_match = {
                    "overall": 0.0,
                    "duplicate": False
                }

                best_score = 0.0

            self.similarity_results.append({

                "seed_name":
                    seed["seed_name"],

                "similarity":
                    best_score,

                "details":
                    best_match

            })

            print(
                f"\n{seed['seed_name']}"
            )

            print(
                f"Duplicate          : "
                f"{best_match['duplicate']}"
            )

            print(
                f"Similarity         : "
                f"{best_score:.4f}"
            )
    def decision_engine(self):

        print("\n" + "=" * 70)
        print("DECISION ENGINE")
        print("=" * 70)

        self.accepted_seeds = []
        self.duplicate_seeds = []

        for result in self.similarity_results:

            seed_name = result["seed_name"]

            feature = next(

                seed
                for seed in self.seed_features

                if seed["seed_name"]
                ==
                seed_name

            )

            duplicate_found = (
                result["details"]["duplicate"]
            )

            if not duplicate_found:

                for accepted_seed in self.accepted_seeds:

                    check = self.similarity_score(
                        feature,
                        accepted_seed
                    )

                    if check["duplicate"]:

                        duplicate_found = True
                        break

            if duplicate_found:

                print(
                    f"{seed_name} : DUPLICATE"
                )

                self.duplicate_seeds.append(
                    feature
                )

            else:

                print(
                    f"{seed_name} : ACCEPTED"
                )

                self.accepted_seeds.append(
                    feature
                )

        print(
            f"\nAccepted : "
            f"{len(self.accepted_seeds)}"
        )

        print(
            f"Duplicates : "
            f"{len(self.duplicate_seeds)}"
        )

    def regenerate_duplicates(self):

        print("\n" + "=" * 70)
        print("QUANTUM REGENERATION")
        print("=" * 70)

        self.regenerated_seeds = []

        if len(self.duplicate_seeds) == 0:

            print("\nNo duplicate seeds detected.")

            return

        for seed in self.duplicate_seeds:

            print("\n" + "-" * 70)
            print(seed["seed_name"])
            print("-" * 70)

            try:

                print("Requesting fresh QRNG pool bits...")
                print(
                    f"Generating new seed for : "
                    f"{seed['seed_name']}"
                )

                regenerated_seed = (
                    self.seed_manager.generate_seed(
                        seed["seed_name"]
                    )
                )

                destination = os.path.join(
                    self.regenerated_folder,
                    os.path.basename(regenerated_seed)
                )

                import shutil

                shutil.copy2(
                    regenerated_seed,
                    destination
                )

                print(
                    f"Generated Seed File : "
                    f"{destination}"
                )

                print("Extracting features...")

                seed_data = self.read_seed_file(
                    destination
                )

                if not seed_data["binary"]:

                    print(
                        "Status                : FAILED"
                    )

                    print(
                        "Generated seed contains "
                        "no readable binary data."
                    )

                    continue

                features = self.extract_features(
                    seed_data
                )

                self.regenerated_seeds.append(
                    features
                )

                print(
                    "Status                : SUCCESS"
                )

                print(
                    f"New Seed              : "
                    f"{features['seed_name']}"
                )

                print(
                    f"Pool Start            : "
                    f"{features['pool_start']}"
                )

                print(
                    f"Pool End              : "
                    f"{features['pool_end']}"
                )

            except Exception as error:

                print(
                    "Status                : FAILED"
                )

                print(error)

    def retry_manager(self):

        print("\n" + "=" * 70)
        print("RETRY MANAGER")
        print("=" * 70)

        self.final_verified_seeds = []
        self.final_rejected_seeds = []

        max_retry = self.maximum_retries

        for seed in self.regenerated_seeds:

            print("\n" + "-" * 70)
            print(seed["seed_name"])
            print("-" * 70)

            verified = False
            current_seed = seed
            retry = 0

            while retry < max_retry:

                best_similarity = 0

                for record in self.database_records:

                    if not self.quick_filter(
                        current_seed,
                        record
                    ):
                        continue

                    result = self.similarity_score(
                        current_seed,
                        record
                    )

                    if result["overall"] > best_similarity:

                        best_similarity = result["overall"]

                for accepted_seed in self.accepted_seeds:

                    result = self.similarity_score(
                        current_seed,
                        accepted_seed
                    )

                    if result["overall"] > best_similarity:

                        best_similarity = result["overall"]

                for verified_seed in self.final_verified_seeds:

                    result = self.similarity_score(
                        current_seed,
                        verified_seed
                    )

                    if result["overall"] > best_similarity:

                        best_similarity = result["overall"]

                print(
                    f"Attempt              : "
                    f"{retry + 1}"
                )

                print(
                    f"Similarity           : "
                    f"{best_similarity:.4f}"
                )

                if best_similarity == 0:

                    print(
                        "Status               : VERIFIED"
                    )

                    verified = True

                    self.retry_statistics.append(
                        retry + 1
                    )

                    self.final_verified_seeds.append(
                        current_seed
                    )

                    break

                print("Duplicate Found")
                print("Generating Fresh Seed...")

                regenerated = (
                    self.seed_manager.generate_seed(
                        current_seed["seed_name"]
                    )
                )

                destination = os.path.join(
                    self.regenerated_folder,
                    os.path.basename(regenerated)
                )

                import shutil

                shutil.copy2(
                    regenerated,
                    destination
                )

                seed_data = self.read_seed_file(
                    destination
                )

                if not seed_data["binary"]:

                    print(
                        "Generated seed could not "
                        "be read."
                    )

                    retry += 1

                    continue

                current_seed = self.extract_features(
                    seed_data
                )

                retry += 1

            if not verified:

                print("Retry Limit Reached")

                self.retry_statistics.append(
                    max_retry
                )

                self.final_rejected_seeds.append(
                    current_seed
                )

        print("\n" + "=" * 70)
        print("RETRY SUMMARY")
        print("=" * 70)

        print(
            f"Verified Seeds       : "
            f"{len(self.final_verified_seeds)}"
        )

        print(
            f"Rejected Seeds       : "
            f"{len(self.final_rejected_seeds)}"
        )

    def update_history_database(self):

        print("\n" + "=" * 70)

        print("DATABASE UPDATE")

        print("=" * 70)

        total_added = 0

        verified = []

        verified.extend(self.accepted_seeds)

        verified.extend(self.final_verified_seeds)

        with open(

            self.database_file,

            "a"

        ) as database:

            for seed in verified:

                database.write(

                    json.dumps(seed)

                )

                database.write("\n")

                total_added += 1

        print(f"\nNew Records Added      : {total_added}")

        print(f"Database File          : {self.database_file}")


    def organize_seed_files(self):

        print("\n" + "=" * 70)

        print("ORGANIZING OUTPUT FILES")

        print("=" * 70)

        import shutil

        for seed in self.accepted_seeds:

            source = os.path.join(

                self.qrng_seed_folder,

                seed["seed_name"] + ".txt"

            )

            destination = os.path.join(

                self.verified_folder,

                seed["seed_name"] + ".txt"

            )

            if os.path.exists(source):

                shutil.copy2(

                    source,

                    destination

                )

        for seed in self.final_verified_seeds:

            source = os.path.join(

                self.regenerated_folder,

                seed["seed_name"] + ".txt"

            )

            destination = os.path.join(

                self.verified_folder,

                seed["seed_name"] + ".txt"

            )

            if os.path.exists(source):

                shutil.copy2(

                    source,

                    destination

                )

        for seed in self.final_rejected_seeds:

            source = os.path.join(

                self.regenerated_folder,

                seed["seed_name"] + ".txt"

            )

            destination = os.path.join(

                self.rejected_folder,

                seed["seed_name"] + ".txt"

            )

            if os.path.exists(source):

                shutil.copy2(

                    source,

                    destination

                )

        for filename in os.listdir(self.verified_folder):

            source = os.path.join(

                self.verified_folder,

                filename

            )

            destination = os.path.join(

                self.final_verified_folder,

                filename.replace(

                    ".txt",

                    "_verified.txt"

                )

            )

            shutil.copy2(

                source,

                destination

            )

        print("Output folders updated successfully.")


    def save_summary(self):

        total_verified = len(self.accepted_seeds)

        total_verified += len(self.final_verified_seeds)

        total_duplicates = len(self.duplicate_seeds)

        total_rejected = len(self.final_rejected_seeds)

        with open(

            self.history_summary,

            "w"

        ) as summary:

            summary.write("=" * 70 + "\n")

            summary.write("HISTORY CHECKER SUMMARY\n")

            summary.write("=" * 70 + "\n\n")

            summary.write(f"History Records        : {len(self.database_records)}\n")

            summary.write(f"Verified Seeds         : {total_verified}\n")

            summary.write(f"Duplicate Seeds        : {total_duplicates}\n")

            summary.write(f"Rejected Seeds         : {total_rejected}\n")

        print("\nSummary saved successfully.")


    def final_summary(self):

        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        final_accepted = (
            len(self.accepted_seeds) +
            len(self.final_verified_seeds)
        )

        initial_seeds = len(self.qrng_seed_files)

        initial_verifications = initial_seeds

        accepted_initially = len(self.accepted_seeds)

        duplicates_detected = len(self.duplicate_seeds)

        regenerated_seeds = len(self.regenerated_seeds)

        retry_verifications = len(self.final_verified_seeds)

        accepted_after_retry = len(self.final_verified_seeds)

        rejected_seeds = len(self.final_rejected_seeds)

        total_retries = sum(self.retry_statistics)

        max_retry = (
            max(self.retry_statistics)
            if self.retry_statistics else 0
        )

        avg_retry = (
            total_retries /
            len(self.retry_statistics)
            if self.retry_statistics else 0
        )

        duplicate_sha = len(self.duplicate_seeds)

        verification_steps = (
            initial_verifications +
            retry_verifications
        )

        total_seed_operations = (
            initial_verifications +
            len(self.regenerated_seeds) +
            retry_verifications
        )

        overall_status = (
            "SUCCESS"
            if rejected_seeds == 0
            else "PARTIAL SUCCESS"
        )

        print(f"History Records          : {len(self.database_records)}")
        print()

        print(f"Initial Seeds            : {initial_seeds}")
        print(f"Initial Verifications    : {initial_verifications}")
        print()

        print(f"Accepted Initially       : {accepted_initially}")
        print(f"Duplicates Detected      : {duplicates_detected}")
        print()

        print(f"Regenerated Seeds        : {regenerated_seeds}")
        print(f"Retry Verifications      : {retry_verifications}")
        print()

        print(f"Accepted After Retry     : {accepted_after_retry}")
        print(f"Rejected Seeds           : {rejected_seeds}")
        print()

        print(f"Final Accepted Seeds     : {final_accepted}")
        print()

        print(f"Duplicate SHA Count      : {duplicate_sha}")
        print()

        print(f"Total Retry Attempts     : {total_retries}")
        print(f"Maximum Retry            : {max_retry}")
        print(f"Average Retry            : {avg_retry:.2f}")
        print()

        print(f"Total Verification Steps : {verification_steps}")
        print(f"Total Seed Operations    : {total_seed_operations}")
        print()

        print(f"Overall Status           : {overall_status}")

        print("\nHistory Checker Completed Successfully.")

    def run(self):

        self.banner()

        self.clear_output_folders()

        self.load_history_database()

        self.load_qrng_seeds()

        self.initialization_summary()

        self.feature_extraction()

        self.similarity_engine()

        self.decision_engine()

        self.regenerate_duplicates()

        self.retry_manager()

        self.update_history_database()

        self.organize_seed_files()

        self.save_summary()

        self.final_summary()

    def clear_output_folders(self):

        import shutil

        folders = [
            self.regenerated_folder,
            self.verified_folder,
            self.rejected_folder,
            self.final_verified_folder
        ]

        for folder in folders:

            os.makedirs(folder, exist_ok=True)

            for file in os.listdir(folder):

                path = os.path.join(folder, file)

                if os.path.isfile(path):

                    os.remove(path)

                elif os.path.isdir(path):

                    shutil.rmtree(path)

