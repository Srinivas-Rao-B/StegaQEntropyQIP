import os
import time

from qiskit_ibm_runtime import SamplerV2


class QuantumExecution:

    def __init__(self, backend, transpiled_circuit):

        self.backend = backend

        self.circuit = transpiled_circuit

        self.sampler = None

        self.job = None

        self.result = None

        self.bitarray = None

        self.random_pool = []

        self.block_number = 0

        self.total_bits = 0

        self.total_shots = 0

        self.output_folder = "qrng/output"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

        self.pool_file = os.path.join(
            self.output_folder,
            "qrng_pool.txt"
        )

        self.binary_file = os.path.join(
            self.output_folder,
            "qrng_pool.bin"
        )

        self.statistics_file = os.path.join(
            self.output_folder,
            "qrng_statistics.txt"
        )


    def banner(self):

        print("\n" + "=" * 70)
        print("IBM QUANTUM EXECUTION")
        print("=" * 70)


    def initialize_sampler(self):

        print("\nInitializing IBM Runtime Sampler...")

        self.sampler = SamplerV2(
            mode=self.backend
        )

        print("Sampler Initialized Successfully.")


    def print_execution_configuration(self):

        print("\n" + "-" * 70)
        print("EXECUTION CONFIGURATION")
        print("-" * 70)

        bits_per_block = self.circuit.num_qubits * 16384

        total_blocks = 6

        total_bits = bits_per_block * total_blocks

        cycles = total_bits // (26 * 16384)

        print(f"Backend                 : {self.backend.name}")

        print(f"Quantum Qubits          : {self.circuit.num_qubits}")

        print(f"Circuit Depth           : {self.circuit.depth()}")

        print(f"Circuit Width           : {self.circuit.width()}")

        print(f"Measurement Shots       : 16,384")

        print(f"QRNG Blocks             : {total_blocks}")

        print()

        print("QRNG Pool Configuration")

        print("----------------------------")

        print(f"Bits Per Block          : {bits_per_block:,}")

        print(f"Total QRNG Bits         : {total_bits:,}")

        print(f"26-Seed Cycles          : {cycles}")

        print(f"Recommended Retry Limit : 6")

        print()

        print("QRNG Pool Generation Mode")

        print("----------------------------")

        print("Multi Block")

        print("Real IBM Quantum Hardware")

        print("PrimitiveResult")

        print("BitArray Extraction")


    def execute_quantum_block(self, shots=16384):

        self.block_number += 1

        print("\n" + "=" * 70)
        print(f"QUANTUM BLOCK {self.block_number}")
        print("=" * 70)

        print(f"\nShots Requested        : {shots}")
        print(f"Bits Per Shot          : {self.circuit.num_clbits}")
        print(f"Expected Bits          : {shots * self.circuit.num_clbits:,}")

        print("\nSubmitting Job to IBM Quantum...")

        start = time.time()

        self.job = self.sampler.run(
            [self.circuit],
            shots=shots
        )

        print("Job Submitted Successfully.")

        print(f"Job ID                : {self.job.job_id()}")

        print("\nWaiting For IBM Quantum Execution...")

        self.result = self.job.result()

        end = time.time()

        print("Execution Completed Successfully.")

        print(f"Execution Time        : {end-start:.2f} seconds")

        self.total_shots += shots


    def extract_bitarray(self):

        print("\n" + "-" * 70)
        print("BITARRAY EXTRACTION")
        print("-" * 70)

        publication = self.result[0]

        databin = publication.data

        self.bitarray = databin.meas

        print("PrimitiveResult Extracted.")

        print("SamplerPubResult Extracted.")

        print("DataBin Extracted.")

        print("BitArray Extracted Successfully.")

        print()

        print(f"Shots                : {self.bitarray.num_shots}")

        print(f"Bits Per Shot        : {self.bitarray.num_bits}")

        print(f"Expected Total Bits  : {self.bitarray.num_shots * self.bitarray.num_bits:,}")


    def convert_to_bitstrings(self):

        print("\n" + "-" * 70)
        print("QUANTUM BITSTRING GENERATION")
        print("-" * 70)

        bit_strings = self.bitarray.get_bitstrings()

        print(f"Quantum Samples       : {len(bit_strings):,}")

        generated_bits = 0

        for bits in bit_strings:

            self.random_pool.append(bits)

            generated_bits += len(bits)

        self.total_bits += generated_bits

        print(f"Generated Bits        : {generated_bits:,}")

        print(f"Pool Size             : {self.total_bits:,}")

        print()

        print("First 10 Quantum Bitstrings")

        print("-" * 40)

        preview = min(10, len(bit_strings))

        for i in range(preview):

            print(bit_strings[i])

        self.current_block = bit_strings

        return bit_strings


    def save_block(self):

        print("\nSaving Quantum Pool...")

        with open(

            self.pool_file,

            "a"

        ) as file:

            for bits in self.current_block:

                file.write(bits + "\n")

        print("qrng_pool.txt Updated.")

    def measurement_statistics(self):

        print("\n" + "-" * 70)
        print("MEASUREMENT STATISTICS")
        print("-" * 70)

        counts = self.bitarray.get_counts()

        int_counts = self.bitarray.get_int_counts()

        print("\nMeasurement Counts")
        print("----------------------------")

        print(f"Unique Measurement Outcomes : {len(counts)}")

        top_counts = sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True
        )[:10]

        print("\nTop 10 Measurement Outcomes\n")

        for key, value in top_counts:

            if len(key) > 64:

                print(f"{key[:64]}...  {value}")

            else:

                print(f"{key:<35}{value}")

        print()

        print("Integer Counts")
        print("----------------------------")

        top_int = sorted(
            int_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )[:10]

        print(f"Unique Integer Outcomes : {len(int_counts)}")

        print("\nTop 10 Integer Outcomes\n")

        for key, value in top_int:

            print(f"{key:<20}{value}")

        return counts, int_counts


    def bit_balance(self):

        print("\n" + "-" * 70)
        print("BIT BALANCE")
        print("-" * 70)

        bool_array = self.bitarray.to_bool_array()

        ones = int(bool_array.sum())

        zeros = int(bool_array.size - ones)

        total = ones + zeros

        print(f"Total Bits           : {total:,}")

        print(f"Zeros                : {zeros:,}")

        print(f"Ones                 : {ones:,}")

        print(f"Zero Percentage      : {(zeros/total)*100:.4f}%")

        print(f"One Percentage       : {(ones/total)*100:.4f}%")

        print()

        if abs(zeros-ones) <= total*0.02:

            print("Balance Status       : Excellent")

        elif abs(zeros-ones) <= total*0.05:

            print("Balance Status       : Good")

        else:

            print("Balance Status       : Slight Bias")


    def save_binary_pool(self):

        print("\nSaving Binary Pool...")

        with open(

            self.binary_file,

            "wb"

        ) as file:

            for bits in self.random_pool:

                file.write(bits.encode())

        print("qrng_pool.bin Saved Successfully.")


    def save_statistics(self):

        print("\nSaving Statistics...")

        with open(

            self.statistics_file,

            "w"

        ) as file:

            file.write("IBM QUANTUM QRNG STATISTICS\n")

            file.write("="*50+"\n\n")

            file.write(f"Backend : {self.backend.name}\n")

            file.write(f"Blocks Generated : {self.block_number}\n")

            file.write(f"Total Shots : {self.total_shots}\n")

            file.write(f"Total Bits : {self.total_bits}\n")

            file.write(f"Bits Per Shot : {self.circuit.num_qubits}\n")

        print("Statistics Saved Successfully.")


    def generate_pool(self,

                      required_bits=1000000,

                      block_shots=16384):

        print("\n" + "=" * 70)

        print("QUANTUM RANDOM POOL GENERATION")

        print("=" * 70)

        print(f"\nRequested Pool Size : {required_bits:,} bits")

        bits_per_block = block_shots * self.circuit.num_qubits

        expected_blocks = (
            required_bits + bits_per_block - 1
        ) // bits_per_block

        print(f"Bits Per Block      : {bits_per_block:,}")

        print(f"Expected Blocks     : {expected_blocks}")

        print(f"Shots Per Block     : {block_shots:,}")

        print(f"Qubits Per Shot     : {self.circuit.num_qubits}")

        print()

        if self.block_number == 0:

            open(self.pool_file, "w").close()

            open(self.binary_file, "wb").close()

            with open("qrng/output/qrng_pointer.txt", "w") as file:

                file.write("0")

        while self.total_bits < required_bits:

            self.execute_quantum_block(

                shots=block_shots

            )

            self.extract_bitarray()

            self.convert_to_bitstrings()

            self.measurement_statistics()

            self.bit_balance()

            self.save_block()

            print()

            print(f"Current Pool : {self.total_bits:,} bits")

            print("-"*60)

        self.save_binary_pool()

        self.save_statistics()

        print("\nQuantum Pool Generation Completed Successfully.")

    def request_bits(self, number_of_bits):

        print("\n" + "-" * 70)
        print("QRNG BIT REQUEST")
        print("-" * 70)

        available = sum(len(bits) for bits in self.random_pool)

        print(f"Requested Bits        : {number_of_bits:,}")
        print(f"Available Bits        : {available:,}")

        while available < number_of_bits:

            print("\nPool Low.")
            print("Generating Additional Quantum Block...\n")

            self.generate_pool(
                required_bits=available + 500000
            )

            available = sum(len(bits) for bits in self.random_pool)

        collected = ""

        while len(collected) < number_of_bits:

            if len(self.random_pool) == 0:
                break

            current = self.random_pool.pop(0)

            needed = number_of_bits - len(collected)

            if len(current) <= needed:

                collected += current

            else:

                collected += current[:needed]

                remaining = current[needed:]

                self.random_pool.insert(0, remaining)

        print(f"Bits Returned         : {len(collected):,}")

        remaining = sum(len(bits) for bits in self.random_pool)

        print(f"Remaining Pool        : {remaining:,}")

        return collected


    def request_seed(self, seed_name, seed_length=256):

        print("\n" + "-" * 70)
        print(f"{seed_name.upper()} SEED")
        print("-" * 70)

        seed = self.request_bits(seed_length)

        print(f"Seed Length           : {len(seed)} bits")

        return seed


    def pool_information(self):

        print("\n" + "=" * 70)
        print("QRNG POOL INFORMATION")
        print("=" * 70)

        remaining = sum(len(bits) for bits in self.random_pool)

        print(f"\nBlocks Generated      : {self.block_number}")

        print(f"Total Quantum Shots   : {self.total_shots:,}")

        print(f"Total Quantum Bits    : {self.total_bits:,}")

        print(f"Remaining Pool        : {remaining:,}")

        print(f"Backend               : {self.backend.name}")

        print(f"Bits Per Shot         : {self.circuit.num_qubits}")

        print(f"Pool File             : {self.pool_file}")

        print(f"Binary File           : {self.binary_file}")


    def summary(self):

        print("\n" + "=" * 70)
        print("IBM QUANTUM EXECUTION COMPLETED")
        print("=" * 70)

        print("\nGenerated Files")
        print("----------------------------")

        print("qrng_pool.txt")
        print("qrng_pool.bin")
        print("qrng_statistics.txt")

        print("\nIBM Quantum Runtime")
        print("----------------------------")

        print("SamplerV2            : Completed")
        print("PrimitiveResult      : Extracted")
        print("BitArray             : Extracted")
        print("Quantum Pool         : Ready")

        print("\nNext Module")
        print("----------------------------")

        print("Randomness Tests")


    def run(self,
            required_bits=3145728):

        self.banner()

        self.initialize_sampler()

        self.print_execution_configuration()

        self.generate_pool(
            required_bits=required_bits
        )

        self.pool_information()

        self.summary()

        return self
    
    