import os
import math
import json
import hashlib
import io

from PIL import Image


class MessagePreparation:

    def __init__(self):

        self.capacity_profile = {}

        self.secret_message = ""

        self.verified_seed = ""
        self.qrng_usage = []

        self.output_folder = os.path.join(
            "output",
            "message_preparation"
        )

        self.pointer_file = os.path.join(
            self.output_folder,
            "pointer.txt"
        )

        self.capacity_file = os.path.join(
            "output",
            "capacity_profile.txt"
        )

        self.seed_file = os.path.join(
            "qrng",
            "output",
            "history_check",
            "final_verified_seeds",
            "chunk_seed_verified.txt"
        )

        os.makedirs(self.output_folder, exist_ok=True)

    def banner(self):

        print("\n" + "=" * 70)
        print("                 STEGAQENTROPY")
        print("             MESSAGE PREPARATION")
        print("=" * 70)

    def load_capacity_profile(self):

        print("\nLoading Capacity Profile...")

        if not os.path.exists(self.capacity_file):
            raise FileNotFoundError(
                "Capacity Profile Not Found."
            )

        with open(self.capacity_file, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if "=" in line:

                    key, value = line.split("=", 1)

                    self.capacity_profile[key.strip()] = value.strip()

        print("Capacity Profile Loaded Successfully.")

    def load_pointer(self):

        if not os.path.exists(self.pointer_file):

            self.seed_pointer = 0

            with open(self.pointer_file, "w") as file:
                file.write("0")

            return

        with open(self.pointer_file, "r") as file:

            value = file.read().strip()

            if value == "":
                self.seed_pointer = 0
            else:
                self.seed_pointer = int(value)

        print(f"QRNG Pointer Loaded : {self.seed_pointer}")

    def save_pointer(self):

        with open(self.pointer_file, "w") as file:
            file.write(str(self.seed_pointer))

        print(f"QRNG Pointer Saved : {self.seed_pointer}")

    def load_verified_seed(self):
        
        print("\nLoading Verified QRNG Seed...")

        if not os.path.exists(self.seed_file):
            raise FileNotFoundError(
                "Verified Chunk Seed Not Found."
            )

        with open(
            self.seed_file,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

        self.verified_seed = ""

        for index, line in enumerate(lines):

            if line.strip() == "Binary":

                if index + 2 < len(lines):

                    binary = lines[index + 2].strip()

                    if (
                        binary
                        and all(
                            bit in "01"
                            for bit in binary
                        )
                    ):

                        self.verified_seed = binary

                break

        if self.verified_seed == "":

            raise ValueError(
                "Verified QRNG Binary Not Found."
            )

        if len(self.verified_seed) != 16384:

            raise ValueError(
                f"Invalid Verified QRNG Seed Length: "
                f"{len(self.verified_seed)} bits. "
                f"Expected 16384 bits."
            )

        self.load_pointer()

        print(
            "Verified QRNG Seed Loaded Successfully."
        )

        print(
            f"Verified QRNG Seed Length : "
            f"{len(self.verified_seed)} Bits"
        )

        print(
            f"Current QRNG Pointer      : "
            f"{self.seed_pointer}"
        )

    def qrng_random(self, minimum, maximum):

        if minimum > maximum:
            raise ValueError("Invalid Range")

        if minimum == maximum:
            return minimum

        range_size = maximum - minimum + 1

        bits_required = (range_size - 1).bit_length()

        max_value = (1 << bits_required) - 1

        rejection_limit = (
            ((max_value + 1) // range_size)
            * range_size
        )

        while True:

            if self.seed_pointer + bits_required > len(self.verified_seed):

                raise ValueError(
                    "Verified QRNG Seed Fully Consumed."
                )

            pointer_before = self.seed_pointer

            random_bits = self.verified_seed[
                self.seed_pointer:
                self.seed_pointer + bits_required
            ]

            self.seed_pointer += bits_required

            pointer_after = self.seed_pointer

            raw_value = int(random_bits, 2)

            accepted = raw_value < rejection_limit

            generated = (
                minimum + (raw_value % range_size)
                if accepted else None
            )

            self.qrng_usage.append({

                "operation": f"{minimum}-{maximum}",

                "range": range_size,

                "bits_used": bits_required,

                "pointer_before": pointer_before,

                "pointer_after": pointer_after,

                "random_bits": random_bits,

                "raw_value": raw_value,

                "generated_value": generated,

                "accepted": accepted

            })

            if accepted:

                return generated

    def display_capacity_summary(self):

        print("\n" + "-" * 70)
        print("CAPACITY SUMMARY")
        print("-" * 70)

        print(
            f"Recommended Capacity      : "
            f"{self.capacity_profile['recommended_characters']} Characters"
        )

        print(
            f"Safe Capacity             : "
            f"{self.capacity_profile['safe_characters']} Characters"
        )

        print(
            f"Minimum Capacity          : "
            f"{self.capacity_profile['minimum_characters']} Characters"
        )

        print(
            f"Over-Embedding Threshold  : "
            f"{self.capacity_profile['threshold_characters']} Characters"
        )

        print(
            f"\nImage Suitability         : "
            f"{self.capacity_profile['suitability_level']}"
        )

        print(
            f"Image Complexity          : "
            f"{self.capacity_profile['complexity_level']}"
        )

        print(
            f"Embedding Readiness       : "
            f"{self.capacity_profile['embedding_readiness']}"
        )

        print(
            f"Payload Density           : "
            f"{self.capacity_profile['payload_density_level']}"
        )

    def get_secret_message(self):

        print("\n" + "-" * 70)
        print("SECRET MESSAGE INPUT")
        print("-" * 70)

        while True:

            self.secret_message = input(
                "\nEnter Secret Message : "
            )

            if len(self.secret_message.strip()) == 0:

                print("\nMessage cannot be empty.")

                continue

            if self.validate_message():

                break

    def validate_message(self):

        recommended = int(
            self.capacity_profile["recommended_characters"]
        )

        safe = int(
            self.capacity_profile["safe_characters"]
        )

        threshold = int(
            self.capacity_profile["threshold_characters"]
        )

        length = len(self.secret_message)

        print("\n" + "-" * 70)
        print("MESSAGE VALIDATION")
        print("-" * 70)

        print(f"Message Length : {length} Characters")

        if length <= recommended:

            print("Validation     : PASSED")

            print("Status         : Recommended Capacity")

            return True

        elif length <= safe:

            print("Validation     : WARNING")

            print("Status         : Above Recommended Capacity")

            choice = input(
                "\nContinue? (Y/N) : "
            ).upper()

            return choice == "Y"

        elif length <= threshold:

            print("Validation     : STRONG WARNING")

            print("Status         : Near Over-Embedding Limit")

            choice = input(
                "\nContinue? (Y/N) : "
            ).upper()

            return choice == "Y"

        else:

            print("Validation     : FAILED")

            print("Status         : Over-Embedding Threshold Exceeded")

            print("\nPlease Enter a Smaller Message.")

            return False

    def get_input_type(self):

        print("\n" + "-" * 70)
        print("SECRET DATA TYPE")
        print("-" * 70)

        while True:

            print("\n1. Text")
            print("2. Image")

            choice = input("\nSelect input type (1/2) : ").strip()

            if choice == "1":

                self.input_type = "text"

                print("\nInput Type : TEXT")

                return

            elif choice == "2":

                self.input_type = "image"

                print("\nInput Type : IMAGE")

                return

            else:

                print("\nInvalid choice. Please select 1 or 2.")

    def get_secret_image(self):

        print("\n" + "-" * 70)
        print("SECRET IMAGE INPUT")
        print("-" * 70)

        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        image_path = filedialog.askopenfilename(
            title="Select Secret Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*")
            ]
        )

        root.destroy()

        if not image_path:

            raise ValueError(
                "No secret image selected."
            )

        try:

            image = Image.open(
                image_path
            ).convert("L")

            self.secret_image_path = image_path
            self.secret_image = image

            print(
                "\nImage Loaded Successfully."
            )

            print(
                f"Selected Image : {image_path}"
            )

            print(
                f"Original Resolution : "
                f"{image.width} × {image.height}"
            )

        except Exception as e:

            raise ValueError(
                f"Unable to load image: {e}"
            )
        
    def compress_secret_image(self):

        print("\n" + "-" * 70)
        print("SECRET IMAGE COMPRESSION")
        print("-" * 70)

        target_bits = 25000

        image = self.secret_image.resize(
            (128, 128),
            Image.Resampling.LANCZOS
        )

        best_data = None
        best_quality = None

        for quality in range(100, 0, -1):

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=False
            )

            data = buffer.getvalue()

            if len(data) * 8 <= target_bits:

                best_data = data
                best_quality = quality

                break

        if best_data is None:

            raise ValueError(
                "Unable to compress image below 25,000 bits."
            )

        compressed_folder = os.path.join(
            self.output_folder,
            "secret_image"
        )

        os.makedirs(
            compressed_folder,
            exist_ok=True
        )

        self.secret_image_compressed_path = os.path.join(
            compressed_folder,
            "compressed_secret_image.jpg"
        )

        with open(
            self.secret_image_compressed_path,
            "wb"
        ) as file:

            file.write(best_data)

        self.compressed_image_bytes = best_data
        self.compressed_image_bits = len(best_data) * 8
        self.image_compression_quality = best_quality

        print(
            f"\nCompression Resolution : 128 × 128"
        )

        print(
            f"JPEG Quality           : "
            f"{best_quality}"
        )

        print(
            f"Compressed Size        : "
            f"{len(best_data):,} bytes"
        )

        print(
            f"Compressed Bits        : "
            f"{self.compressed_image_bits:,} bits"
        )

        print(
            "Status                 : "
            "UNDER 25,000 BITS"
        )

    def image_binary_conversion(self):

        print("\n" + "-" * 70)
        print("IMAGE BINARY CONVERSION")
        print("-" * 70)

        self.binary_message = "".join(
            format(byte, "08b")
            for byte in self.compressed_image_bytes
        )

        self.binary_length = len(
            self.binary_message
        )

        self.ascii_values = []

        binary_path = os.path.join(
            self.output_folder,
            "binary_message.txt"
        )

        with open(
            binary_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                self.binary_message
            )

        print(
            f"Image Binary Bits : "
            f"{self.binary_length:,}"
        )

        print(
            f"Binary File        : "
            f"{binary_path}"
        )

        print("\nFirst 128 Bits:")

        print(
            self.binary_message[:128]
        )

    def save_input_type_profile(self):

        path = os.path.join(
            self.output_folder,
            "input_type_profile.txt"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "STEGAQENTROPY INPUT TYPE PROFILE\n"
            )

            file.write(
                "====================================\n\n"
            )

            file.write(
                f"input_type={self.input_type}\n"
            )

            if self.input_type == "image":

                file.write(
                    f"source_image={self.secret_image_path}\n"
                )

                file.write(
                    "original_format=image\n"
                )

                file.write(
                    "compression_format=JPEG\n"
                )

                file.write(
                    "compression_resolution=128x128\n"
                )

                file.write(
                    f"jpeg_quality={self.image_compression_quality}\n"
                )

                file.write(
                    f"compressed_bits={self.compressed_image_bits}\n"
                )

                file.write(
                    "receiver_restoration=CNN\n"
                )

            else:

                file.write(
                    "source_format=text\n"
                )

                file.write(
                    "receiver_restoration=NONE\n"
                )

        print(
            "\nInput Type Profile Saved."
        )

        print(
            f"Location : {path}"
        )
        
    def ascii_conversion(self):

        print("\n" + "-" * 70)
        print("ASCII CONVERSION")
        print("-" * 70)

        self.ascii_values = [ord(ch) for ch in self.secret_message]

        print(f"Characters          : {len(self.secret_message)}")
        print(f"ASCII Bytes         : {len(self.ascii_values)}")

    def binary_conversion(self):

        print("\n" + "-" * 70)
        print("BINARY CONVERSION")
        print("-" * 70)

        self.binary_message = ""

        for value in self.ascii_values:

            self.binary_message += format(value, "08b")

        self.binary_length = len(self.binary_message)

        print(f"Binary Bits         : {self.binary_length}")

    def save_message_profile(self):

        profile_path = os.path.join(
            self.output_folder,
            "message_profile.txt"
        )

        with open(profile_path, "w", encoding="utf-8") as file:

            file.write("=" * 60 + "\n")
            file.write("STEGAQENTROPY MESSAGE PROFILE\n")
            file.write("=" * 60 + "\n\n")

            file.write(f"message_characters={len(self.secret_message)}\n")
            file.write(f"ascii_bytes={len(self.ascii_values)}\n")
            file.write(f"binary_bits={self.binary_length}\n")
            file.write(f"verified_seed_length={len(self.verified_seed)}\n")
            file.write(f"seed_pointer_used={self.seed_pointer}\n")
            file.write(
                f"seed_bits_remaining="
                f"{len(self.verified_seed)-self.seed_pointer}\n"
            )
            file.write(
                f"seed_utilization="
                f"{(self.seed_pointer/len(self.verified_seed))*100:.2f}%\n"
            )

        print("\nMessage Profile Saved.")
        print(f"Location : {profile_path}")

    def save_binary_message(self):

        binary_path = os.path.join(
            self.output_folder,
            "binary_message.txt"
        )

        with open(binary_path, "w", encoding="utf-8") as file:

            file.write(self.binary_message)

        print("Binary Message Saved.")
        print(f"Location : {binary_path}")

    def summary(self):

        print("\n" + "=" * 70)
        print("MESSAGE PREPARATION SUMMARY")
        print("=" * 70)

        print(f"Input Type             : {self.input_type}")

        if self.input_type == "text":

            print(
                f"Message Characters     : "
                f"{len(self.secret_message)}"
            )

            print(
                f"ASCII Bytes            : "
                f"{len(self.ascii_values)}"
            )

        else:

            print(
                f"Compressed Image Bits  : "
                f"{self.compressed_image_bits}"
            )

        print(
            f"Binary Bits            : "
            f"{self.binary_length}"
        )
        print(f"Verified Seed Length   : {len(self.verified_seed)} Bits")
        print(f"QRNG Bits Used         : {self.seed_pointer}")

        print(
            f"QRNG Bits Remaining    : "
            f"{len(self.verified_seed)-self.seed_pointer}"
        )

        print(
            f"QRNG Utilization       : "
            f"{(self.seed_pointer/len(self.verified_seed))*100:.2f}%"
        )


    def load_binary_message(self):

        print("\n" + "-" * 70)
        print("LOADING BINARY MESSAGE")
        print("-" * 70)

        binary_path = os.path.join(
            self.output_folder,
            "binary_message.txt"
        )

        if not os.path.exists(binary_path):
            raise FileNotFoundError("binary_message.txt not found.")

        with open(binary_path, "r") as file:
            self.binary_message = file.read().strip()

        self.binary_length = len(self.binary_message)

        print(f"Binary Length : {self.binary_length} Bits")

    def generate_chunk_count(self):

        print("\n" + "-" * 70)
        print("DYNAMIC CHUNK COUNT GENERATION")
        print("-" * 70)

        minimum_chunk_size = 8

        if self.binary_length < minimum_chunk_size:
            self.chunk_count = 1

        else:
            maximum_possible_chunks = (
                self.binary_length // minimum_chunk_size
            )

            self.chunk_count = min(
                32,
                maximum_possible_chunks
            )

        print(
            f"Chunk Count        : "
            f"{self.chunk_count}"
        )

    def generate_chunk_sizes(self):

        print("\n" + "-" * 70)
        print("DYNAMIC CHUNK SIZE GENERATION")
        print("-" * 70)

        remaining_bits = self.binary_length
        remaining_chunks = self.chunk_count

        self.chunk_sizes = []

        if self.chunk_count == 1:

            self.chunk_sizes = [
                self.binary_length
            ]

        else:

            remaining_bytes = (
                self.binary_length // 8
            )

            base_bytes = (
                remaining_bytes //
                self.chunk_count
            )

            extra_bytes = (
                remaining_bytes %
                self.chunk_count
            )

            for index in range(
                self.chunk_count
            ):

                chunk_bytes = base_bytes

                if index < extra_bytes:
                    chunk_bytes += 1

                chunk_bits = (
                    chunk_bytes * 8
                )

                self.chunk_sizes.append(
                    chunk_bits
                )

        print(
            f"Generated "
            f"{len(self.chunk_sizes)} "
            f"Chunk Sizes"
        )

        print(
            "\nChunk Size Preview"
        )

        for i, size in enumerate(
            self.chunk_sizes[:10],
            start=1
        ):

            print(
                f"C{i:03d} : "
                f"{size} Bits"
            )

        print(
            f"\nTotal Bits : "
            f"{sum(self.chunk_sizes)}"
        )

    def split_binary_message(self):

        print("\n" + "-" * 70)
        print("BINARY MESSAGE CHUNKING")
        print("-" * 70)

        self.chunks = []

        start = 0

        for size in self.chunk_sizes:

            end = start + size

            chunk = self.binary_message[start:end]

            self.chunks.append(chunk)

            start = end

        print(f"Chunks Created : {len(self.chunks)}")

        print("\nChunk Preview")

        for i, chunk in enumerate(self.chunks[:5], start=1):

            print(f"C{i:03d} : {len(chunk)} Bits")

    def validate_chunk_integrity(self):

        print("\n" + "-" * 70)
        print("CHUNK INTEGRITY VALIDATION")
        print("-" * 70)

        reconstructed = "".join(self.chunks)

        assert len(self.chunk_sizes) == self.chunk_count, \
            "Chunk Count Mismatch"

        assert sum(self.chunk_sizes) == self.binary_length, \
            "Chunk Size Sum Mismatch"

        assert len(reconstructed) == self.binary_length, \
            "Binary Length Mismatch"

        assert reconstructed == self.binary_message, \
            "Binary Reconstruction Failed"

        print("Chunk Count          : PASSED")
        print("Chunk Size Sum       : PASSED")
        print("Binary Length        : PASSED")
        print("Binary Reconstruction: PASSED")

        print("\nIntegrity Validation Successful.")

    def save_chunks(self):

        print("\n" + "-" * 70)
        print("SAVING CHUNKS")
        print("-" * 70)

        chunk_folder = os.path.join(
            self.output_folder,
            "chunks"
        )

        os.makedirs(chunk_folder, exist_ok=True)
        for file in os.listdir(chunk_folder):

            if file.endswith(".bin"):

                os.remove(os.path.join(chunk_folder, file))

        for index, chunk in enumerate(self.chunks, start=1):

            filename = os.path.join(
                chunk_folder,
                f"C{index:03d}.bin"
            )

            with open(filename, "w") as file:
                file.write(chunk)

        print(f"Saved {len(self.chunks)} Chunk Files")
        print(f"Location : {chunk_folder}")

    def save_chunk_profile(self):

        profile_path = os.path.join(
            self.output_folder,
            "chunk_profile.txt"
        )

        with open(profile_path, "w") as file:

            file.write("=" * 60 + "\n")
            file.write("STEGAQENTROPY CHUNK PROFILE\n")
            file.write("=" * 60 + "\n\n")

            file.write(f"chunk_count={self.chunk_count}\n")
            file.write(f"total_bits={self.binary_length}\n\n")

            file.write("-" * 60 + "\n")
            file.write("CHUNK INFORMATION\n")
            file.write("-" * 60 + "\n\n")

            start_bit = 0

            for index, chunk in enumerate(self.chunks, start=1):

                size = len(chunk)

                end_bit = start_bit + size - 1

                sha256 = hashlib.sha256(
                    chunk.encode()
                ).hexdigest()

                file.write(f"chunk_id=C{index:03d}\n")
                file.write(f"size={size}\n")
                file.write(f"start_bit={start_bit}\n")
                file.write(f"end_bit={end_bit}\n")
                file.write(f"sha256={sha256}\n\n")

                start_bit = end_bit + 1

        print("\nChunk Profile Saved.")
        print(f"Location : {profile_path}")
    def save_binary_hash(self):

        print("\nGenerating Binary SHA-256...")

        sha256 = hashlib.sha256(
            self.binary_message.encode()
        ).hexdigest()

        path = os.path.join(
            self.output_folder,
            "binary_sha256.txt"
        )

        with open(path, "w") as file:

            file.write("=" * 60 + "\n")
            file.write("STEGAQENTROPY BINARY SHA256\n")
            file.write("=" * 60 + "\n\n")

            file.write(
                f"Binary Length : "
                f"{self.binary_length} Bits\n\n"
            )

            file.write(
                f"SHA256 : {sha256}\n"
            )

        self.binary_sha256 = sha256

        print("Binary SHA-256 Generated.")
        print(f"Location : {path}")

    def load_chunks(self):

        print("\n" + "-" * 70)
        print("LOADING CHUNKS")
        print("-" * 70)

        chunk_folder = os.path.join(
            self.output_folder,
            "chunks"
        )

        self.chunk_data = []

        files = sorted(os.listdir(chunk_folder))

        for file in files:

            if file.endswith(".bin"):

                path = os.path.join(chunk_folder, file)

                with open(path, "r") as f:

                    data = f.read().strip()

                self.chunk_data.append(
                    (file.replace(".bin", ""), data)
                )

        print(f"Chunks Loaded : {len(self.chunk_data)}")

    def calculate_chunk_statistics(self):

        print("\n" + "-" * 70)
        print("CHUNK STATISTICS")
        print("-" * 70)

        self.chunk_statistics = []

        for chunk_id, bits in self.chunk_data:

            size = len(bits)

            ones = bits.count("1")

            zeros = bits.count("0")

            ratio = ones / size if size else 0

            density = (ones / size) * 100 if size else 0

            self.chunk_statistics.append({

                "chunk_id": chunk_id,

                "size": size,

                "ones": ones,

                "zeros": zeros,

                "ratio": ratio,

                "density": density

            })

        print(f"Statistics Generated : {len(self.chunk_statistics)}")

    def calculate_bit_transitions(self):

        print("\n" + "-" * 70)
        print("BIT TRANSITION ANALYSIS")
        print("-" * 70)

        self.chunk_transitions = []

        for chunk_id, bits in self.chunk_data:

            t00 = 0
            t01 = 0
            t10 = 0
            t11 = 0

            for i in range(len(bits) - 1):

                pair = bits[i:i + 2]

                if pair == "00":
                    t00 += 1

                elif pair == "01":
                    t01 += 1

                elif pair == "10":
                    t10 += 1

                elif pair == "11":
                    t11 += 1

            total = t00 + t01 + t10 + t11

            transition_rate = (
                (t01 + t10) / total
                if total else 0
            )

            self.chunk_transitions.append({

                "chunk_id": chunk_id,

                "00": t00,

                "01": t01,

                "10": t10,

                "11": t11,

                "transition_rate": transition_rate

            })

        print(
            f"Transition Analysis Generated : "
            f"{len(self.chunk_transitions)} Chunks"
        )

    def save_bit_transitions(self):

        path = os.path.join(
            self.output_folder,
            "bit_transitions.txt"
        )

        with open(path, "w") as file:

            file.write("=" * 60 + "\n")
            file.write("STEGAQENTROPY BIT TRANSITIONS\n")
            file.write("=" * 60 + "\n\n")

            for item in self.chunk_transitions:

                file.write(f"chunk_id={item['chunk_id']}\n")
                file.write(f"00={item['00']}\n")
                file.write(f"01={item['01']}\n")
                file.write(f"10={item['10']}\n")
                file.write(f"11={item['11']}\n")
                file.write(
                    f"transition_rate="
                    f"{item['transition_rate']:.6f}\n\n"
                )

        print("\nBit Transition Report Saved.")
        print(f"Location : {path}")

    def save_chunk_statistics(self):

        path = os.path.join(
            self.output_folder,
            "chunk_statistics.txt"
        )

        with open(path, "w") as file:

            file.write("=" * 60 + "\n")
            file.write("STEGAQENTROPY CHUNK STATISTICS\n")
            file.write("=" * 60 + "\n\n")

            for stat in self.chunk_statistics:

                file.write(f"chunk_id={stat['chunk_id']}\n")
                file.write(f"size={stat['size']}\n")
                file.write(f"ones={stat['ones']}\n")
                file.write(f"zeros={stat['zeros']}\n")
                file.write(f"one_ratio={stat['ratio']:.4f}\n")
                file.write(f"bit_density={stat['density']:.2f}\n\n")

        print("\nChunk Statistics Saved.")
        print(f"Location : {path}")

    def calculate_chunk_entropy(self):

        print("\n" + "-" * 70)
        print("SHANNON ENTROPY ANALYSIS")
        print("-" * 70)

        self.chunk_entropy = []

        for stat in self.chunk_statistics:

            ones = stat["ones"]
            zeros = stat["zeros"]
            total = stat["size"]

            p0 = zeros / total if total else 0
            p1 = ones / total if total else 0

            entropy = 0

            if p0 > 0:
                entropy -= p0 * math.log2(p0)

            if p1 > 0:
                entropy -= p1 * math.log2(p1)

            if entropy >= 0.95:
                level = "High"

            elif entropy >= 0.75:
                level = "Medium"

            else:
                level = "Low"

            self.chunk_entropy.append({

                "chunk_id": stat["chunk_id"],

                "entropy": entropy,

                "level": level

            })

        print(f"Entropy Calculated : {len(self.chunk_entropy)} Chunks")

    def save_chunk_entropy(self):

        path = os.path.join(
            self.output_folder,
            "chunk_entropy.txt"
        )

        with open(path, "w") as file:

            file.write("=" * 60 + "\n")
            file.write("STEGAQENTROPY CHUNK ENTROPY\n")
            file.write("=" * 60 + "\n\n")

            for item in self.chunk_entropy:

                file.write(f"chunk_id={item['chunk_id']}\n")
                file.write(f"entropy={item['entropy']:.6f}\n")
                file.write(f"level={item['level']}\n\n")

        print("\nChunk Entropy Saved.")
        print(f"Location : {path}")

    def analyze_binary_randomness(self):

        print("\n" + "-" * 70)
        print("BINARY RANDOMNESS ANALYSIS")
        print("-" * 70)

        bits = self.binary_message

        zeros = bits.count("0")
        ones = bits.count("1")
        total = len(bits)

        zero_ratio = zeros / total if total else 0
        one_ratio = ones / total if total else 0

        entropy = 0

        if zero_ratio > 0:
            entropy -= zero_ratio * math.log2(zero_ratio)

        if one_ratio > 0:
            entropy -= one_ratio * math.log2(one_ratio)

        transitions = 0

        longest_zero = 0
        longest_one = 0

        current_zero = 0
        current_one = 0

        for i, bit in enumerate(bits):

            if bit == "0":
                current_zero += 1
                current_one = 0
                longest_zero = max(longest_zero, current_zero)
            else:
                current_one += 1
                current_zero = 0
                longest_one = max(longest_one, current_one)

            if i > 0 and bits[i] != bits[i - 1]:
                transitions += 1

        transition_rate = (
            transitions / (total - 1)
            if total > 1 else 0
        )

        path = os.path.join(
            self.output_folder,
            "binary_randomness.txt"
        )

        with open(path, "w") as file:

            file.write("=" * 60 + "\n")
            file.write("STEGAQENTROPY BINARY RANDOMNESS\n")
            file.write("=" * 60 + "\n\n")

            file.write(f"Total Bits          : {total}\n")
            file.write(f"Zeros               : {zeros}\n")
            file.write(f"Ones                : {ones}\n")
            file.write(f"Zero Ratio          : {zero_ratio:.6f}\n")
            file.write(f"One Ratio           : {one_ratio:.6f}\n")
            file.write(f"Binary Entropy      : {entropy:.6f}\n")
            file.write(f"Transition Rate     : {transition_rate:.6f}\n")
            file.write(f"Longest Zero Run    : {longest_zero}\n")
            file.write(f"Longest One Run     : {longest_one}\n")

        print("Binary Randomness Analysis Saved.")
        print(f"Location : {path}")

    def generate_chunk_priority(self):

        print("\n" + "-" * 70)
        print("CHUNK PRIORITY RANKING")
        print("-" * 70)

        self.chunk_priority = []

        transition_lookup = {
            item["chunk_id"]: item["transition_rate"]
            for item in self.chunk_transitions
        }

        entropy_lookup = {
            item["chunk_id"]: item["entropy"]
            for item in self.chunk_entropy
        }

        max_size = max(self.chunk_sizes)

        for stat in self.chunk_statistics:

            chunk_id = stat["chunk_id"]

            entropy = entropy_lookup.get(chunk_id, 0)

            density = stat["density"] / 100

            size_score = stat["size"] / max_size

            transition = transition_lookup.get(chunk_id, 0)

            priority = (
                0.35 * entropy +
                0.25 * density +
                0.20 * size_score +
                0.20 * transition
            )

            self.chunk_priority.append({

                "chunk_id": chunk_id,

                "priority_score": priority,

                "entropy": entropy,

                "density": stat["density"],

                "transition_rate": transition,

                "size": stat["size"]

            })

        self.chunk_priority.sort(
            key=lambda x: x["priority_score"],
            reverse=True
        )

        for rank, item in enumerate(self.chunk_priority, start=1):
            item["priority"] = rank

        print(f"Priority Generated : {len(self.chunk_priority)} Chunks")

    def save_chunk_priority(self):

        path = os.path.join(
            self.output_folder,
            "chunk_priority.txt"
        )

        with open(path, "w") as file:

            file.write("=" * 70 + "\n")
            file.write("STEGAQENTROPY CHUNK PRIORITY\n")
            file.write("=" * 70 + "\n\n")

            file.write(
                "Priority Score = "
                "0.35×Entropy + "
                "0.25×Density + "
                "0.20×Size + "
                "0.20×Transition\n\n"
            )

            file.write(
                "Rank\tChunk\tPriority\tEntropy\tDensity\tTransition\tSize\n"
            )

            for rank, item in enumerate(self.chunk_priority, start=1):

                file.write(
                    f"{rank}\t"
                    f"{item['chunk_id']}\t"
                    f"{item['priority_score']:.6f}\t"
                    f"{item['entropy']:.4f}\t"
                    f"{item['density']:.4f}\t"
                    f"{item['transition_rate']:.4f}\t"
                    f"{item['size']}\n"
                )

        print("\nChunk Priority Report Saved.")
        print(f"Location : {path}")

    def generate_embedding_ready_profile(self):

        print("\n" + "-" * 70)
        print("GENERATING EMBEDDING READY PROFILE")
        print("-" * 70)

        path = os.path.join(
            self.output_folder,
            "embedding_ready_profile.txt"
        )

        with open(path, "w") as file:

            file.write("=" * 60 + "\n")
            file.write("STEGAQENTROPY EMBEDDING READY PROFILE\n")
            file.write("=" * 60 + "\n\n")

            file.write(f"total_chunks={len(self.chunk_priority)}\n")
            file.write(f"total_bits={self.binary_length}\n\n")

            for item in self.chunk_priority:

                file.write(f"chunk_id={item['chunk_id']}\n")
                file.write(f"priority={item['priority']}\n")
                file.write(f"priority_score={item['priority_score']:.4f}\n")
                file.write(f"entropy={item['entropy']:.6f}\n")
                file.write(f"density={item['density']:.2f}\n")
                file.write(f"size={item['size']}\n")
                file.write("status=READY\n\n")

        print("Embedding Ready Profile Generated.")
        print(f"Location : {path}")

    def version3_summary(self):

        print("\n" + "=" * 70)
        print("VERSION 3 SUMMARY")
        print("=" * 70)

        print(f"Chunks Analysed        : {len(self.chunk_priority)}")
        print(f"Total Binary Bits      : {self.binary_length}")

        highest = self.chunk_priority[0]

        print(f"Highest Priority Chunk : {highest['chunk_id']}")
        print(f"Priority Score         : {highest['priority_score']:.4f}")

        print("\nEmbedding Metadata Ready.")

    def save_qrng_usage(self):

        path = os.path.join(
            self.output_folder,
            "qrng_usage.txt"
        )

        with open(path, "w") as file:

            file.write("=" * 80 + "\n")
            file.write("STEGAQENTROPY QRNG USAGE REPORT\n")
            file.write("=" * 80 + "\n\n")

            total = 0

            for index, item in enumerate(self.qrng_usage, start=1):

                file.write(f"Operation #{index}\n")
                file.write("-" * 80 + "\n")
                file.write(f"Range             : {item['operation']}\n")
                file.write(f"Possible Values   : {item['range']}\n")
                file.write(f"QRNG Bits Used    : {item['bits_used']}\n")
                file.write(f"Pointer Before    : {item['pointer_before']}\n")
                file.write(f"Pointer After     : {item['pointer_after']}\n")
                file.write(f"QRNG Bits         : {item['random_bits']}\n")
                file.write(f"Raw Value         : {item['raw_value']}\n")
                file.write(f"Accepted          : {item['accepted']}\n")
                file.write(f"Generated Value   : {item['generated_value']}\n\n")

                total += item["bits_used"]

            file.write("=" * 80 + "\n")
            file.write("SUMMARY\n")
            file.write("=" * 80 + "\n\n")

            file.write(f"Total Operations      : {len(self.qrng_usage)}\n")
            file.write(f"Total Bits Used       : {total}\n")
            file.write(f"Current Pointer       : {self.seed_pointer}\n")
            file.write(
                f"Remaining Bits        : "
                f"{len(self.verified_seed)-self.seed_pointer}\n"
            )
            file.write(
                f"Seed Utilization      : "
                f"{(self.seed_pointer/len(self.verified_seed))*100:.2f}%\n"
            )

        print("QRNG Usage Report Saved.")
        print(f"Location : {path}")

    def save_message_preparation_json(self):

        print("\nSaving Message Preparation JSON...")

        path = os.path.join(
            self.output_folder,
            "message_preparation.json"
        )

        chunk_lookup = {
            item["chunk_id"]: item
            for item in self.chunk_priority
        }

        start_bit = 0

        chunks = []

        for index, chunk in enumerate(self.chunks, start=1):

            chunk_id = f"C{index:03d}"

            size = len(chunk)

            end_bit = start_bit + size - 1

            priority = chunk_lookup[chunk_id]

            chunks.append({

                "chunk_id": chunk_id,

                "chunk_file":
                    f"chunks/{chunk_id}.bin",

                "size": size,

                "start_bit": start_bit,

                "end_bit": end_bit,

                "priority":
                    priority["priority"],

                "priority_score":
                    round(priority["priority_score"], 6),

                "entropy":
                    round(priority["entropy"], 6),

                "density":
                    round(priority["density"], 2),

                "transition_rate":
                    round(priority["transition_rate"], 6),

                "sha256":
                    hashlib.sha256(
                        chunk.encode()
                    ).hexdigest()

            })

            start_bit = end_bit + 1

        data = {

            "input_type": self.input_type,

            "message_characters": len(self.secret_message) if self.input_type == "text" else 0,

            "ascii_bytes": len(self.ascii_values) if self.input_type == "text" else 0,

            "binary_bits": self.binary_length,
            "payload_format": (
                "text"
                if self.input_type == "text"
                else "compressed_jpeg_image"
            ),

            "binary_file": "binary_message.txt",

            "binary_sha256": self.binary_sha256,

            "chunk_count": self.chunk_count,

            "qrng": {

                "pointer": self.seed_pointer,

                "remaining_bits":
                    len(self.verified_seed) - self.seed_pointer,

                "seed_length":
                    len(self.verified_seed)

            },

            "chunks": chunks

        }

        with open(path, "w") as file:

            json.dump(
                data,
                file,
                indent=4
            )

        print("Message Preparation JSON Saved.")
        print(f"Location : {path}")
        
    def run(self):

        self.banner()

        self.load_capacity_profile()

        self.load_verified_seed()

        self.display_capacity_summary()

        self.get_input_type()

        if self.input_type == "text":

            self.get_secret_message()

            self.ascii_conversion()

            self.binary_conversion()

        else:

            self.get_secret_image()

            self.compress_secret_image()

            self.image_binary_conversion()

        self.save_input_type_profile()

        self.save_binary_message()

        self.generate_chunk_count()

        self.generate_chunk_sizes()

        self.split_binary_message()

        self.validate_chunk_integrity()

        self.save_binary_hash()

        self.analyze_binary_randomness()

        self.save_chunks()

        self.save_chunk_profile()

        self.load_chunks()

        self.calculate_chunk_statistics()

        self.save_chunk_statistics()

        self.calculate_bit_transitions()

        self.save_bit_transitions()

        self.calculate_chunk_entropy()

        self.save_chunk_entropy()

        self.generate_chunk_priority()

        self.save_chunk_priority()

        self.generate_embedding_ready_profile()

        self.save_message_preparation_json()

        self.save_qrng_usage()

        self.save_pointer()

        self.version3_summary()

        self.save_message_profile()

        self.summary()

if __name__ == "__main__":

    try:

        obj = MessagePreparation()

        obj.run()

        print("\nMessage Preparation Module Executed Successfully.")

    except Exception as e:

        print("\nERROR :", e)