import os
import json
import uuid
from pathlib import Path
from datetime import datetime

class QuantumKeyHierarchy:

    def __init__(self):

        self.input_directory = Path("output/message_preparation")

        self.output_directory = Path("output/quantum_key_hierarchy")

        self.configuration_file = (
            self.output_directory /
            "key_configuration.json"
        )

        self.mapping_file = (
            self.output_directory /
            "chunk_key_mapping.json"
        )

        self.hierarchy_file = (
            self.output_directory /
            "hierarchy.json"
        )

        self.manifest_file = (
            self.output_directory /
            "manifest.json"
        )

        self.qkd_package_file = (
            self.output_directory /
            "qkd_package.json"
        )

        self.report_file = (
            self.output_directory /
            "hierarchy_report.txt"
        )

        os.makedirs(self.output_directory, exist_ok=True)

        self.message_preparation_file = (
            self.input_directory /
            "message_preparation.json"
        )

        self.chunk_information = []

        self.total_chunks = 0

        self.master_key_configuration = {}

        self.subkey_configurations = []

        self.chunk_key_mapping = []

        self.hierarchy = {}

        self.manifest = {}

        self.qkd_package = {}

        self.statistics = {}

        self.embedding_readiness = {}

        self.security = {}

        self.dependencies = {}

        self.qkd_plan = []

        self.total_reserved_key_bits = 0

        self.minimum_key_size = 0

        self.maximum_key_size = 0

        self.average_key_size = 0

        self.master_key_size = 0

    def banner(self):

        print()
        print("=" * 70)
        print("                 MULTI-KEY QUANTUM HIERARCHY")
        print("=" * 70)
        print()
        print("Preparing Quantum Key Hierarchy...")
        print()
        print("• Loading Message Preparation Output")
        print("• Analysing Message Chunks")
        print("• Configuring Master Key")
        print("• Configuring Quantum Subkeys")
        print("• Building Key Hierarchy")
        print("• Preparing QKD Synchronization Package")
        print("• Reserving Embedding Placeholders")
        print()
        print("=" * 80)
        print()


    def load_message_preparation(self):

        print("Loading Message Preparation Output...\n")

        if not self.message_preparation_file.exists():

            raise FileNotFoundError(
                "message_preparation.json not found."
            )

        with open(self.message_preparation_file, "r") as file:

            data = json.load(file)

        self.chunk_information = data["chunks"]

        self.total_chunks = data["chunk_count"]

        self.binary_bits = data["binary_bits"]

        self.binary_sha256 = data["binary_sha256"]

        self.message_characters = data["message_characters"]

        self.ascii_bytes = data["ascii_bytes"]

        self.binary_file = data["binary_file"]

        self.qrng_information = data["qrng"]

        self.total_chunks = len(self.chunk_information)

        print(f"Chunks Loaded        : {self.total_chunks}")
        print(f"Binary Bits          : {self.binary_bits}")
        print(f"Binary SHA-256       : {self.binary_sha256}")
        print()


    def analyze_chunk_requirements(self):

        print("Analysing Chunk Requirements...\n")

        self.subkey_configurations = []

        for index in range(self.total_chunks):

            chunk = self.chunk_information[index]

            chunk_bits = chunk["size"]

            reserved_key_bits = ((chunk_bits + 127) // 128) * 128

            self.subkey_configurations.append({

                "subkey_id": f"SK-{uuid.uuid4().hex[:8].upper()}",

                "chunk_id": chunk["chunk_id"],

                "chunk_file": chunk["chunk_file"],

                "chunk_start_bit": chunk["start_bit"],

                "chunk_end_bit": chunk["end_bit"],

                "chunk_bits": chunk_bits,

                "required_key_bits": chunk["size"],

                "reserved_key_bits": reserved_key_bits,

                "priority": chunk["priority"],

                "priority_score": chunk["priority_score"],

                "entropy": chunk["entropy"],

                "density": chunk["density"],

                "transition_rate": chunk["transition_rate"],

                "chunk_sha256": chunk["sha256"],

                "required": True,

                "status": "NOT GENERATED",

                "generation_module": "QKD Synchronization",

                "distribution": "QKD",

                "key_level":"LEVEL-2",

                "created_on":
                datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

                "expected_qkd_protocol":"BB84",

                "fingerprint_algorithm": "SHA-256",

                "fingerprint": "PENDING",

                "embedding_status": "PENDING"

            })

        largest_subkey = max(
            item["reserved_key_bits"]
            for item in self.subkey_configurations
        )

        self.master_key_size = (
            ((int(largest_subkey * 1.25) + 127) // 128) * 128
        )

        self.master_key_configuration = {

            "master_key_id": f"MK-{uuid.uuid4().hex[:8].upper()}",

            "key_level": "LEVEL-1",

            "reserved_key_bits": self.master_key_size,

            "required": True,

            "status": "NOT GENERATED",

            "generation_module": "QKD Synchronization",

            "distribution": "QKD",

            "fingerprint_algorithm": "SHA-256",

            "fingerprint": "PENDING",

            "embedding_status": "PENDING",

            "created_on": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

            "expected_qkd_protocol": "BB84"

        }

        for subkey in self.subkey_configurations:

            subkey["parent_master_key"] = \
                self.master_key_configuration["master_key_id"]

        print(f"Master Keys Required : 1")
        print(f"Subkeys Required     : {len(self.subkey_configurations)}")
        print()


    def assign_subkeys_to_chunks(self):

        print("Assigning Subkeys To Chunks...\n")

        self.chunk_key_mapping = []

        for index in range(self.total_chunks):

            chunk = self.chunk_information[index]

            self.chunk_key_mapping.append({

                "chunk_id": chunk["chunk_id"],

                "chunk_file": chunk["chunk_file"],

                "chunk_bits": chunk["size"],

                "chunk_start_bit": chunk["start_bit"],

                "chunk_end_bit": chunk["end_bit"],

                "reserved_key_bits":self.subkey_configurations[index]["reserved_key_bits"],

                "priority": chunk["priority"],

                "priority_score": chunk["priority_score"],

                "entropy": chunk["entropy"],

                "density": chunk["density"],

                "transition_rate": chunk["transition_rate"],

                "chunk_sha256": chunk["sha256"],

                "subkey_id":self.subkey_configurations[index]["subkey_id"],

                "master_key_id":self.master_key_configuration["master_key_id"],

                "key_level": "LEVEL-2",

                "parent_master_key":self.master_key_configuration["master_key_id"],

                "qkd_status":"PENDING",

                "fingerprint_status":"PENDING",

                "embedding_status":"PENDING"

                })

        print(f"Mappings Created     : {len(self.chunk_key_mapping)}")
        print()


    def build_hierarchy(self):

        print("Building Quantum Key Hierarchy...\n")

        self.hierarchy = {

            "hierarchy_id": f"QKH-{uuid.uuid4().hex[:8].upper()}",

            "hierarchy_version": "1.0",

            "architecture":"Two-Level Adaptive Quantum Key Hierarchy",

            "created_on": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

            "master_key": self.master_key_configuration,

            "master_key_id":self.master_key_configuration["master_key_id"],

            "subkeys": self.subkey_configurations,

            "chunk_mapping": self.chunk_key_mapping,

            "total_master_keys": 1,

            "total_subkeys": len(self.subkey_configurations),

            "total_chunks": self.total_chunks,

            "hierarchy_depth": 2,

            "message_characters": self.message_characters,

            "ascii_bytes": self.ascii_bytes,

            "binary_bits": self.binary_bits,

            "binary_file": self.binary_file,

            "binary_sha256": self.binary_sha256,

            "qrng": self.qrng_information,

            "statistics": self.statistics,

            "embedding_readiness": self.embedding_readiness,

            "security": self.security,

            "dependencies": self.dependencies,

            "status": "READY FOR QKD"

        }

        print("Hierarchy Built Successfully.")
        print()

    def prepare_manifest(self):

        print("Preparing Hierarchy Manifest...\n")

        self.manifest = {

            "hierarchy_id": self.hierarchy["hierarchy_id"],

            "version": self.hierarchy["hierarchy_version"],

            "created_on": self.hierarchy["created_on"],

            "master_keys": 1,

            "subkeys": len(self.subkey_configurations),

            "chunks": self.total_chunks,

            "mapping_count": len(self.chunk_key_mapping),

            "key_generation": "PENDING",

            "qkd_synchronization": "PENDING",

            "fingerprints": "PENDING",

            "master_embedding": "PENDING",

            "subkey_embedding": "PENDING",

            "chunk_embedding": "PENDING",

            "receiver_status": "PENDING",

            "overall_status": "READY FOR QKD"

        }

        print("Manifest Prepared.")
        print()


    def prepare_qkd_package(self):

        print("Preparing QKD Synchronization Package...\n")

        self.qkd_package = {

            "hierarchy": self.hierarchy,

            "master_key_configuration": self.master_key_configuration,

            "subkey_configurations": self.subkey_configurations,

            "chunk_mapping": self.chunk_key_mapping,

            "manifest": self.manifest,

            "statistics": self.statistics,

            "embedding_readiness": self.embedding_readiness,

            "security": self.security,

            "dependencies": self.dependencies,

            "qkd_plan": self.qkd_plan,

            "status": "READY",

            "next_module": "QKD Synchronization"

        }

        print("QKD Package Ready.")
        print()


    def save_outputs(self):

        print("Saving Output Files...\n")

        with open(self.configuration_file, "w") as file:
            json.dump(
                {
                    "master_key": self.master_key_configuration,
                    "subkeys": self.subkey_configurations
                },
                file,
                indent=4
            )

        with open(self.mapping_file, "w") as file:
            json.dump(
                self.chunk_key_mapping,
                file,
                indent=4
            )

        with open(self.hierarchy_file, "w") as file:
            json.dump(
                self.hierarchy,
                file,
                indent=4
            )

        with open(self.manifest_file, "w") as file:
            json.dump(
                self.manifest,
                file,
                indent=4
            )

        with open(self.qkd_package_file, "w") as file:
            json.dump(
                self.qkd_package,
                file,
                indent=4
            )

        with open(self.output_directory / "statistics.json", "w") as file:
            json.dump(self.statistics, file, indent=4)

        with open(self.output_directory / "embedding_readiness.json", "w") as file:
            json.dump(self.embedding_readiness, file, indent=4)

        with open(self.output_directory / "security_analysis.json", "w") as file:
            json.dump(self.security, file, indent=4)

        with open(self.output_directory / "dependency_report.json", "w") as file:
            json.dump(self.dependencies, file, indent=4)

        with open(self.output_directory / "qkd_plan.json", "w") as file:
            json.dump(self.qkd_plan, file, indent=4)

        with open(self.report_file, "w") as file:

            file.write("MULTI-KEY QUANTUM HIERARCHY REPORT\n")
            file.write("=" * 60 + "\n\n")

            file.write(f"Hierarchy ID        : {self.hierarchy['hierarchy_id']}\n")
            file.write(f"Master Keys         : 1\n")
            file.write(f"Subkeys             : {len(self.subkey_configurations)}\n")
            file.write(f"Chunks              : {self.total_chunks}\n")
            file.write(f"Hierarchy Depth     : 2\n")
            file.write(f"Generation Status   : PENDING\n")
            file.write(f"QKD Status          : PENDING\n")
            file.write(f"Fingerprint Status  : PENDING\n")
            file.write(f"Embedding Status    : PENDING\n")
            file.write(f"Overall Status      : READY FOR QKD\n")

        print("Files Saved Successfully.")
        print()


    def print_summary(self):

        print("=" * 80)
        print("                     MODULE SUMMARY")
        print("=" * 80)
        print()
        print(f"Total Chunks             : {self.total_chunks}")
        print(f"Master Keys              : 1")
        print(f"Subkeys                  : {len(self.subkey_configurations)}")
        print(f"Hierarchy Depth          : 2")
        print(f"Mappings                 : {len(self.chunk_key_mapping)}")
        print(f"QKD Package              : READY")
        print(f"Key Generation           : PENDING")
        print(f"QKD Synchronization      : PENDING")
        print(f"Fingerprints             : PENDING")
        print(f"Embedding                : PENDING")
        print()
        print("Module Completed Successfully.")
        print("=" * 80)
        print()


    def hierarchy_statistics(self):

        print("Calculating Hierarchy Statistics...\n")

        sizes = []

        for subkey in self.subkey_configurations:
            sizes.append(subkey["reserved_key_bits"])

        self.total_reserved_key_bits = sum(sizes)

        self.minimum_key_size = min(sizes)

        self.maximum_key_size = max(sizes)

        self.average_key_size = round(
            self.total_reserved_key_bits / len(sizes),
            2
        )

        self.statistics = {

            "total_chunks": self.total_chunks,

            "master_keys": 1,

            "subkeys": len(self.subkey_configurations),

            "hierarchy_depth": 2,

            "master_key_size": self.master_key_size,

            "largest_chunk_bits":max(subkey["chunk_bits"]for subkey in self.subkey_configurations),

            "smallest_chunk_bits":min(subkey["chunk_bits"]for subkey in self.subkey_configurations),

            "average_chunk_bits":round(sum(subkey["chunk_bits"]for subkey in self.subkey_configurations)/len(self.subkey_configurations),2),

            "subkey_size_formula":"ceil(chunk_bits/128)*128",

            "master_key_size_formula":"ceil(max(subkey_bits)*1.25/128)*128",

            "minimum_key_size": self.minimum_key_size,

            "maximum_key_size": self.maximum_key_size,

            "average_key_size": self.average_key_size,

            "total_reserved_key_bits": self.total_reserved_key_bits,

            "keys_per_chunk": 1,

            "chunks_per_master": self.total_chunks,

            "quantum_keys_required": len(self.subkey_configurations) + 1,

            "estimated_qrng_bits":
                self.total_reserved_key_bits + self.master_key_size,

            "estimated_sha_operations":
                len(self.subkey_configurations) + 1,

            "qkd_sessions_required":
                len(self.subkey_configurations) + 1

        }

        print(f"Quantum Keys Required : {self.statistics['quantum_keys_required']}")

        print(f"Master Key Size       : {self.master_key_size}")

        print(f"Minimum Key Size      : {self.minimum_key_size}")

        print(f"Maximum Key Size      : {self.maximum_key_size}")

        print(f"Average Key Size      : {self.average_key_size}")

        print(f"Reserved Key Bits     : {self.total_reserved_key_bits}")

        print(f"Estimated QRNG Bits   : {self.statistics['estimated_qrng_bits']}")

        print(f"SHA Operations        : {self.statistics['estimated_sha_operations']}")

        print(f"QKD Sessions          : {self.statistics['qkd_sessions_required']}")
        print()


    def prepare_embedding_readiness(self):

        print("Preparing Embedding Readiness...\n")

        self.embedding_readiness = {

            "master_key": {

                "status": "PENDING",

                "embedding": "PENDING",

                "receiver": "PENDING"

            },

            "subkeys": []

        }

        for mapping in self.chunk_key_mapping:

            self.embedding_readiness["subkeys"].append({

                "subkey_id": mapping["subkey_id"],

                "chunk_id": mapping["chunk_id"],

                "status": "PENDING",

                "embedding": "PENDING",

                "receiver": "PENDING"

            })

        print("Embedding Readiness Prepared.\n")


    def security_analysis(self):

        print("Preparing Security Analysis...\n")

        self.security = {

            "single_chunk_compromise": {

                "affected_chunks": 1,

                "affected_subkeys": 1,

                "remaining_chunks": self.total_chunks - 1,

                "master_key_safe": True,

                "message_recoverable": False

            }

        }

        print("Security Analysis Ready.\n")


    def hierarchy_visualization(self):

        print("Hierarchy Structure\n")

        print("MASTER KEY")

        for mapping in self.chunk_key_mapping:

            print(f"├── {mapping['subkey_id']}")
            print(f"│     └── Chunk {mapping['chunk_id']}")

        print()


    def dependency_report(self):

        print("Preparing Dependency Report...\n")

        self.dependencies = {

            "previous_module": "Message Preparation",

            "current_module": "Quantum Key Hierarchy",

            "next_module": "QKD Synchronization",

            "future_modules": [

                "Adaptive Embedding",

                "Receiver",

                "Extraction"

            ]

        }

        print("Dependency Report Ready.\n")


    def prepare_qkd_plan(self):

        print("Preparing QKD Synchronization Plan...\n")

        self.qkd_plan = [

            "Connect Sender",

            "Connect Receiver",

            "Generate Master Quantum Key",

            "Generate Quantum Subkeys",

            "Synchronize Master Key",

            "Synchronize Subkeys",

            "Generate SHA Fingerprints",

            "Return Ready Package"

        ]

        print("QKD Plan Prepared.\n")

    def calculate_readiness(self):

        print("Calculating Module Readiness...\n")

        readiness = {}

        readiness["Hierarchy"] = bool(self.hierarchy)

        readiness["Manifest"] = bool(self.manifest)

        readiness["Statistics"] = bool(self.statistics)

        readiness["Dependencies"] = bool(self.dependencies)

        readiness["Embedding Readiness"] = bool(self.embedding_readiness)

        readiness["Security Analysis"] = bool(self.security)

        readiness["QKD Package"] = bool(self.qkd_package)

        readiness["QKD Plan"] = bool(self.qkd_plan)

        completed = sum(readiness.values())

        total = len(readiness)

        percentage = round((completed / total) * 100, 2)

        self.readiness = {

            "checks": readiness,

            "completed": completed,

            "total": total,

            "overall_readiness": f"{percentage}%"

        }

        with open(

            self.output_directory / "readiness.json",

            "w"

        ) as file:

            json.dump(

                self.readiness,

                file,

                indent=4

            )

        print(f"Completed Checks : {completed}")

        print(f"Total Checks     : {total}")

        print(f"Overall Readiness: {percentage}%")

        print()


    def run(self):

        self.banner()

        self.load_message_preparation()

        self.analyze_chunk_requirements()

        self.assign_subkeys_to_chunks()

        self.hierarchy_statistics()

        self.prepare_embedding_readiness()

        self.security_analysis()

        self.hierarchy_visualization()

        self.dependency_report()

        self.build_hierarchy()

        self.prepare_qkd_plan()

        self.prepare_manifest()

        self.prepare_qkd_package()

        self.calculate_readiness()

        self.save_outputs()

        self.print_summary()


if __name__ == "__main__":

    hierarchy = QuantumKeyHierarchy()

    hierarchy.run()