import os
import math

class QRNGPool:

    def __init__(self, quantum_execution):

        self.execution = quantum_execution

        self.pool = quantum_execution.random_pool

        self.pointer = 0

        self.total_bits = sum(

            len(bits)

            for bits in self.pool

        )

        self.bits_consumed = 0

        self.output_folder = "qrng/output"

        os.makedirs(

            self.output_folder,

            exist_ok=True

        )


    def banner(self):

        print("\n" + "=" * 70)

        print("QRNG POOL MANAGER")

        print("=" * 70)


    def pool_information(self):

        print("\n" + "-" * 70)

        print("POOL INFORMATION")

        print("-" * 70)

        print(f"Quantum Blocks        : {len(self.pool)}")

        print(f"Total Pool Bits       : {self.total_bits:,}")

        print(f"Consumed Bits         : {self.bits_consumed:,}")

        print(f"Remaining Bits        : {self.total_bits-self.bits_consumed:,}")

        print(f"Pool Pointer          : {self.pointer:,}")


    def available_bits(self):

        return self.total_bits - self.bits_consumed


    def request_bits(self, number_of_bits):

        print("\n" + "-" * 70)

        print("QRNG BIT REQUEST")

        print("-" * 70)

        print(f"Requested Bits        : {number_of_bits:,}")

        if self.available_bits() < number_of_bits:

            raise RuntimeError(

                "QRNG Pool Exhausted."

            )

        collected = ""

        while len(collected) < number_of_bits:

            current = self.pool[self.pointer]

            remaining = number_of_bits - len(collected)

            if len(current) <= remaining:

                collected += current

                self.pointer += 1

            else:

                collected += current[:remaining]

                self.pool[self.pointer] = current[remaining:]

        self.bits_consumed += number_of_bits

        print(f"Returned Bits         : {len(collected):,}")

        print(f"Remaining Pool        : {self.available_bits():,}")

        return collected


    def summary(self):

        print("\n" + "=" * 70)

        print("QRNG POOL READY")

        print("=" * 70)

        print(f"\nAvailable Bits : {self.available_bits():,}")

    import random
    import math


    def request_integer(self, bit_length=32):

        print("\nGenerating Quantum Integer...")

        binary = self.request_bits(bit_length)

        value = int(binary, 2)

        print(f"Bit Length            : {bit_length}")

        print(f"Quantum Integer       : {value}")

        return value


    def request_hex(self, bit_length=128):

        print("\nGenerating Quantum Hexadecimal...")

        binary = self.request_bits(bit_length)

        hexadecimal = format(

            int(binary, 2),

            f"0{math.ceil(bit_length/4)}X"

        )

        print(f"Hex Length            : {len(hexadecimal)}")

        print(f"Hex Preview           : {hexadecimal[:64]}...")

        return hexadecimal


    def request_float(self):

        print("\nGenerating Quantum Floating Point...")

        integer = self.request_integer(53)

        value = integer / ((1 << 53) - 1)

        print(f"Quantum Float         : {value:.12f}")

        return value


    def request_probability(self):

        probability = self.request_float()

        print(f"Probability           : {probability:.8f}")

        return probability


    def random_index(self, upper_limit):

        bits = max(

            1,

            math.ceil(

                math.log2(upper_limit)

            )

        )

        while True:

            index = self.request_integer(bits)

            if index < upper_limit:

                print(f"Random Index          : {index}")

                return index


    def random_permutation(self, length):

        print("\nGenerating Quantum Permutation...")

        permutation = list(range(length))

        for i in range(length - 1, 0, -1):

            j = self.random_index(i + 1)

            permutation[i], permutation[j] = (

                permutation[j],

                permutation[i]

            )

        print(f"Permutation Length    : {length}")

        return permutation


    def random_sample(self, population_size, sample_size):

        print("\nGenerating Quantum Sample...")

        population = list(range(population_size))

        permutation = self.random_permutation(

            population_size

        )

        sample = [

            population[i]

            for i in permutation[:sample_size]

        ]

        print(f"Population            : {population_size}")

        print(f"Sample Size           : {sample_size}")

        return sample


    def random_boolean(self):

        bit = self.request_bits(1)

        return bit == "1"


    def random_choice(self, choices):

        index = self.random_index(

            len(choices)

        )

        return choices[index]


    def remaining_percentage(self):

        percentage = (

            self.available_bits()

            /

            self.total_bits

        ) * 100

        print(f"Remaining Pool        : {percentage:.2f}%")

        return percentage
    
    def allocation_history(self):

        if not hasattr(self, "history"):

            self.history = []

        return self.history


    def log_allocation(self, allocation_type, bits_used):

        if not hasattr(self, "history"):

            self.history = []

        self.history.append({

            "type": allocation_type,

            "bits": bits_used

        })


    def auto_refill(self, minimum_bits=100000):

        remaining = self.available_bits()

        if remaining > minimum_bits:

            return

        print("\n" + "=" * 70)
        print("QRNG AUTO REFILL")
        print("=" * 70)

        print("\nQuantum Pool Running Low.")
        print("Generating Additional Quantum Block...")

        self.execution.generate_pool(
            required_bits=self.execution.total_bits + 1000000
        )

        self.pool = self.execution.random_pool

        self.total_bits = sum(

            len(bits)

            for bits in self.pool

        )

        print("\nQuantum Pool Expanded Successfully.")


    def usage_statistics(self):

        print("\n" + "=" * 70)
        print("POOL USAGE STATISTICS")
        print("=" * 70)

        remaining = self.available_bits()

        utilization = (

            self.bits_consumed /

            self.total_bits

        ) * 100

        print(f"\nTotal Pool Bits      : {self.total_bits:,}")

        print(f"Consumed Bits        : {self.bits_consumed:,}")

        print(f"Remaining Bits       : {remaining:,}")

        print(f"Utilization          : {utilization:.4f}%")

        print(f"Remaining            : {100-utilization:.4f}%")


    def export_history(self):

        filename = os.path.join(

            self.output_folder,

            "qrng_pool_history.txt"

        )

        with open(filename, "w") as file:

            file.write("=" * 70 + "\n")

            file.write("QRNG POOL ALLOCATION HISTORY\n")

            file.write("=" * 70 + "\n\n")

            if hasattr(self, "history"):

                for item in self.history:

                    file.write(

                        f"{item['type']:<30}"

                        f"{item['bits']:>10} bits\n"

                    )

        print("Saved : qrng_pool_history.txt")


    def export_pool_summary(self):

        filename = os.path.join(

            self.output_folder,

            "qrng_pool_summary.txt"

        )

        with open(filename, "w") as file:

            file.write("=" * 70 + "\n")

            file.write("QRNG POOL SUMMARY\n")

            file.write("=" * 70 + "\n\n")

            file.write(f"Total Pool Bits : {self.total_bits}\n")

            file.write(f"Consumed Bits   : {self.bits_consumed}\n")

            file.write(f"Remaining Bits  : {self.available_bits()}\n")

            file.write(f"Pointer         : {self.pointer}\n")

            file.write(f"Blocks          : {len(self.pool)}\n")

        print("Saved : qrng_pool_summary.txt")


    def final_summary(self):

        print("\n" + "=" * 70)
        print("QRNG POOL COMPLETED")
        print("=" * 70)

        print()

        print("Generated Files")

        print("-" * 70)

        print("qrng_pool_summary.txt")

        print("qrng_pool_history.txt")

        print()

        print("Randomness Services Available")

        print("-" * 70)

        print("Binary Allocation")

        print("Integer Allocation")

        print("Hex Allocation")

        print("Floating Point")

        print("Probability")

        print("Permutation")

        print("Sampling")

        print("Random Index")

        print("Random Choice")

        print("Boolean Generator")

        print()

        print("Pool Ready For Entire Project.")


    def run(self):

        self.banner()

        self.pool_information()

        self.usage_statistics()

        self.export_history()

        self.export_pool_summary()

        self.final_summary()

        return self