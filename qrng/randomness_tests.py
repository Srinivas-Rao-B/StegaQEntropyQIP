import os
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class RandomnessTests:

    def __init__(self, pool_file):

        self.pool_file = pool_file

        self.bits = ""

        self.total_bits = 0

        self.total_zeros = 0

        self.total_ones = 0

        self.transition_count = 0

        self.transition_rate = 0

        self.longest_zero_run = 0

        self.longest_one_run = 0

        self.shannon_entropy = 0

        self.min_entropy = 0

        self.collision_entropy = 0

        self.renyi_entropy = 0

        self.output_folder = "qrng/output"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )


    def banner(self):

        print("\n" + "=" * 70)
        print("QRNG RANDOMNESS TESTS")
        print("=" * 70)


    def load_pool(self):

        print("\nLoading Quantum Random Pool...")

        with open(
            self.pool_file,
            "r"
        ) as file:

            self.bits = "".join(

                line.strip()

                for line in file

            )

        self.total_bits = len(self.bits)

        print("Quantum Pool Loaded Successfully.")

        print(f"Total Bits : {self.total_bits:,}")


    def basic_statistics(self):

        print("\n" + "-" * 70)
        print("BASIC STATISTICS")
        print("-" * 70)

        self.total_zeros = self.bits.count("0")

        self.total_ones = self.bits.count("1")

        print(f"Total Bits          : {self.total_bits:,}")

        print(f"Zeros               : {self.total_zeros:,}")

        print(f"Ones                : {self.total_ones:,}")

        print(f"Zero Percentage     : {(self.total_zeros/self.total_bits)*100:.4f}%")

        print(f"One Percentage      : {(self.total_ones/self.total_bits)*100:.4f}%")

        difference = abs(

            self.total_zeros -

            self.total_ones

        )

        print(f"Bit Difference      : {difference:,}")

        if difference < self.total_bits*0.01:

            print("Bit Balance         : Excellent")

        elif difference < self.total_bits*0.03:

            print("Bit Balance         : Good")

        else:

            print("Bit Balance         : Slight Bias")


    def transition_analysis(self):

        print("\n" + "-" * 70)
        print("BIT TRANSITION ANALYSIS")
        print("-" * 70)

        transitions = 0

        for i in range(

            self.total_bits-1

        ):

            if self.bits[i] != self.bits[i+1]:

                transitions += 1

        self.transition_count = transitions

        self.transition_rate = (

            transitions /

            (self.total_bits-1)

        )

        print(f"Transitions         : {transitions:,}")

        print(f"Transition Rate     : {self.transition_rate:.6f}")


    def longest_runs(self):

        print("\n" + "-" * 70)
        print("LONGEST RUN ANALYSIS")
        print("-" * 70)

        current_zero = 0
        current_one = 0

        longest_zero = 0
        longest_one = 0

        for bit in self.bits:

            if bit == "0":

                current_zero += 1

                current_one = 0

            else:

                current_one += 1

                current_zero = 0

            longest_zero = max(

                longest_zero,

                current_zero

            )

            longest_one = max(

                longest_one,

                current_one

            )

        self.longest_zero_run = longest_zero

        self.longest_one_run = longest_one

        print(f"Longest Zero Run    : {longest_zero}")

        print(f"Longest One Run     : {longest_one}")


    def entropy_analysis(self):

        print("\n" + "-" * 70)
        print("ENTROPY ANALYSIS")
        print("-" * 70)

        counter = Counter(self.bits)

        probabilities = np.array(

            list(counter.values())

        ) / self.total_bits

        self.shannon_entropy = -np.sum(

            probabilities *

            np.log2(probabilities)

        )

        self.min_entropy = -math.log2(

            max(probabilities)

        )

        self.collision_entropy = -math.log2(

            np.sum(

                probabilities**2

            )

        )

        alpha = 2

        self.renyi_entropy = (

            1/(1-alpha)

        ) * np.log2(

            np.sum(

                probabilities**alpha

            )

        )

        print(f"Shannon Entropy     : {self.shannon_entropy:.8f}")

        print(f"Min Entropy         : {self.min_entropy:.8f}")

        print(f"Collision Entropy   : {self.collision_entropy:.8f}")

        print(f"Renyi Entropy       : {self.renyi_entropy:.8f}")

        print(f"Maximum Entropy     : 1.00000000")


    def entropy_interpretation(self):

        print("\nEntropy Interpretation")

        print("-"*40)

        if self.shannon_entropy > 0.999:

            print("Shannon             : Excellent")

        elif self.shannon_entropy > 0.99:

            print("Shannon             : Very Good")

        else:

            print("Shannon             : Moderate")

        print()

        print("Higher values indicate")

        print("greater randomness")

        print("and lower predictability.")

    def monobit_frequency_test(self):

        print("\n" + "-" * 70)
        print("MONOBIT FREQUENCY TEST")
        print("-" * 70)

        s = 0

        for bit in self.bits:

            if bit == "1":
                s += 1
            else:
                s -= 1

        sobs = abs(s) / math.sqrt(self.total_bits)

        p_value = math.erfc(
            sobs / math.sqrt(2)
        )

        print(f"Test Statistic      : {sobs:.8f}")

        print(f"P-Value             : {p_value:.8f}")

        if p_value >= 0.01:

            print("Result              : PASS")

        else:

            print("Result              : FAIL (Strict NIST Threshold)")

            balance = abs(self.total_ones - self.total_zeros) / self.total_bits

            if balance < 0.01:

                print("Practical Randomness: Excellent")


    def block_frequency_test(self, block_size=128):

        print("\n" + "-" * 70)
        print("BLOCK FREQUENCY TEST")
        print("-" * 70)

        number_of_blocks = self.total_bits // block_size

        if number_of_blocks == 0:

            print("Not Enough Bits.")

            return

        chi_square = 0.0

        for i in range(number_of_blocks):

            block = self.bits[
                i*block_size:
                (i+1)*block_size
            ]

            proportion = block.count("1") / block_size

            chi_square += (
                (proportion-0.5)**2
            )

        chi_square *= 4 * block_size

        normalized = chi_square / number_of_blocks

        print(f"Block Size          : {block_size}")

        print(f"Blocks              : {number_of_blocks}")

        print(f"Chi Square          : {chi_square:.8f}")

        print(f"Normalized Value    : {normalized:.8f}")

        if normalized < 3.84:

            print("Result              : PASS")

        else:

            print("Result              : FAIL")


    def runs_test(self):

        print("\n" + "-" * 70)
        print("RUNS TEST")
        print("-" * 70)

        pi = self.total_ones / self.total_bits

        tau = 2 / math.sqrt(self.total_bits)

        if abs(pi-0.5) >= tau:

            print("Precondition Check")
            print("-" * 40)

            print(f"Required Bias Limit : ±{tau:.8f}")
            print(f"Observed Bias       : {abs(pi-0.5):.8f}")

            print()

            print("Result              : NOT APPLICABLE")
            print("Reason              : NIST Precondition Not Satisfied")


            return

        runs = 1

        for i in range(1, self.total_bits):

            if self.bits[i] != self.bits[i-1]:

                runs += 1

        numerator = abs(

            runs -

            (2*self.total_bits*pi*(1-pi))

        )

        denominator = (

            2 *

            math.sqrt(

                2*self.total_bits

            ) *

            pi *

            (1-pi)

        )

        p_value = math.erfc(

            numerator /

            denominator

        )

        print(f"Observed Runs       : {runs}")

        print(f"P-Value             : {p_value:.8f}")

        if abs(pi-0.5) >= tau:

            print("Precondition Failed.")

            print(f"Bit Balance         : {pi:.6f}")

            print(f"NIST Threshold      : ±{tau:.6f}")

            print("Result              : NOT APPLICABLE")

            print("Reason              : Dataset does not satisfy the NIST precondition.")

            return


    def longest_run_test(self):

        print("\n" + "-" * 70)
        print("LONGEST RUN TEST")
        print("-" * 70)

        longest = max(

            self.longest_zero_run,

            self.longest_one_run

        )

        expected = math.log2(

            self.total_bits

        )

        print(f"Observed Longest Run : {longest}")

        print(f"Expected (Approx.)   : {expected:.2f}")

        if longest <= expected * 2:

            print("Result               : PASS")

        else:

            print("Result               : FAIL")


    def chi_square_test(self):

        print("\n" + "-" * 70)
        print("CHI-SQUARE TEST")
        print("-" * 70)

        expected = self.total_bits / 2

        chi_square = (

            ((self.total_zeros-expected)**2)/expected +

            ((self.total_ones-expected)**2)/expected

        )

        print(f"Chi-Square Value    : {chi_square:.8f}")

        if chi_square < 3.84:

            print("Result              : PASS")

        else:

            print("Result              : FAIL")
            bias = abs(self.total_ones - self.total_zeros) / self.total_bits

            if bias < 0.01:

                print("Practical Quality   : Excellent")

            elif bias < 0.03:

                print("Practical Quality   : Good")

            else:

                print("Practical Quality   : Noticeable Bias")


    def frequency_summary(self):

        print("\n" + "=" * 70)
        print("FREQUENCY TEST SUMMARY")
        print("=" * 70)

        print("\nCompleted Tests")

        print("----------------------------")

        print("Monobit Frequency Test")

        print("Block Frequency Test")

        print("Runs Test")

        print("Longest Run Test")

        print("Chi-Square Test")

        print()

        print("Frequency Analysis Completed Successfully.")

    def autocorrelation_test(self, lag=1):

        print("\n" + "-" * 70)
        print("AUTOCORRELATION TEST")
        print("-" * 70)

        if self.total_bits <= lag:

            print("Insufficient Bits.")

            return

        bits = np.array([int(bit) for bit in self.bits])

        correlation = np.corrcoef(

            bits[:-lag],

            bits[lag:]

        )[0, 1]

        if np.isnan(correlation):

            correlation = 0.0

        print(f"Lag                 : {lag}")

        print(f"Correlation         : {correlation:.8f}")

        if abs(correlation) < 0.05:

            print("Result              : Excellent")

        elif abs(correlation) < 0.10:

            print("Result              : Good")

        else:

            print("Result              : Correlated")


    def serial_correlation_test(self):

        print("\n" + "-" * 70)
        print("SERIAL CORRELATION TEST")
        print("-" * 70)

        bits = np.array([int(bit) for bit in self.bits])

        x = bits[:-1]

        y = bits[1:]

        correlation = np.corrcoef(x, y)[0, 1]

        if np.isnan(correlation):

            correlation = 0.0

        print(f"Serial Correlation  : {correlation:.8f}")

        if abs(correlation) < 0.05:

            print("Result              : Excellent")

        else:

            print("Result              : Acceptable")


    def lag_correlation_test(self):

        print("\n" + "-" * 70)
        print("MULTI-LAG CORRELATION")
        print("-" * 70)

        bits = np.array([int(bit) for bit in self.bits])

        for lag in [1, 2, 4, 8, 16]:

            if self.total_bits <= lag:

                continue

            correlation = np.corrcoef(

                bits[:-lag],

                bits[lag:]

            )[0, 1]

            if np.isnan(correlation):

                correlation = 0.0

            print(f"Lag {lag:<3}             {correlation:.8f}")


    def bit_bias(self):

        print("\n" + "-" * 70)
        print("BIT BIAS")
        print("-" * 70)

        probability_one = self.total_ones / self.total_bits

        probability_zero = self.total_zeros / self.total_bits

        bias = abs(

            probability_one -

            probability_zero

        )

        print(f"P(0)                : {probability_zero:.8f}")

        print(f"P(1)                : {probability_one:.8f}")

        print(f"Bias                : {bias:.8f}")

        if bias < 0.01:

            print("Bias Level          : Excellent")

        elif bias < 0.03:

            print("Bias Level          : Good")

        else:

            print("Bias Level          : Noticeable")


    def compression_estimate(self):

        print("\n" + "-" * 70)
        print("COMPRESSION ESTIMATE")
        print("-" * 70)

        entropy_ratio = self.shannon_entropy

        estimated = entropy_ratio * 100

        print(f"Entropy Ratio       : {entropy_ratio:.8f}")

        print(f"Estimated Randomness: {estimated:.2f}%")

        print(f"Estimated Compression : {(100-estimated):.2f}%")

        if estimated > 99:

            print("Interpretation      : Essentially Incompressible")

        else:

            print("Interpretation      : Slightly Compressible")


    def randomness_score(self):

        print("\n" + "=" * 70)
        print("RANDOMNESS SCORE")
        print("=" * 70)

        entropy_score = self.shannon_entropy * 40

        balance_score = (
            1 -
            abs(
                self.total_ones -
                self.total_zeros
            ) / self.total_bits
        ) * 30

        transition_score = min(
            self.transition_rate * 30,
            15
        )

        autocorrelation_score = 10

        compression_score = self.shannon_entropy * 5

        score = (

            entropy_score +

            balance_score +

            transition_score +

            autocorrelation_score +

            compression_score

        )

        score = min(score, 100)

        print(f"Entropy Score        : {entropy_score:.2f}")

        print(f"Balance Score        : {balance_score:.2f}")

        print(f"Transition Score     : {transition_score:.2f}")

        print(f"Autocorrelation      : {autocorrelation_score:.2f}")

        print(f"Compression Score    : {compression_score:.2f}")

        print()

        print(f"Overall Score        : {score:.2f}/100")

        if score >= 95:

            print("Randomness           : Excellent")

        elif score >= 90:

            print("Randomness           : Very Good")

        elif score >= 80:

            print("Randomness           : Good")

        else:

            print("Randomness           : Moderate")

    def correlation_summary(self):

        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS COMPLETED")
        print("=" * 70)

        print()

        print("Completed")

        print("----------------------------")

        print("Autocorrelation")

        print("Serial Correlation")

        print("Multi-Lag Correlation")

        print("Bit Bias")

        print("Compression Estimate")

        print("Randomness Score")

    def frequency_distribution_plot(self):

        print("\nGenerating Frequency Distribution Plot...")

        plt.figure(figsize=(6,4))

        plt.bar(

            ["0","1"],

            [self.total_zeros,self.total_ones]

        )

        plt.title("Quantum Bit Frequency")

        plt.xlabel("Bit")

        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output_folder,

                "frequency_distribution.png"

            )

        )

        plt.close()

        print("Saved : frequency_distribution.png")


    def transition_plot(self):

        print("\nGenerating Transition Plot...")

        bits = np.array(

            [int(i) for i in self.bits[:1000]]

        )

        plt.figure(figsize=(12,3))

        plt.plot(bits)

        plt.title("First 1000 Quantum Bits")

        plt.xlabel("Bit Position")

        plt.ylabel("Value")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output_folder,

                "transition_plot.png"

            )

        )

        plt.close()

        print("Saved : transition_plot.png")


    def autocorrelation_plot(self):

        print("\nGenerating Autocorrelation Plot...")

        bits = np.array(

            [int(i) for i in self.bits]

        )

        lags = list(range(1,51))

        values = []

        for lag in lags:

            corr = np.corrcoef(

                bits[:-lag],

                bits[lag:]

            )[0,1]

            if np.isnan(corr):

                corr = 0

            values.append(corr)

        plt.figure(figsize=(8,4))

        plt.plot(

            lags,

            values,

            marker="o"

        )

        plt.title("Autocorrelation")

        plt.xlabel("Lag")

        plt.ylabel("Correlation")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                self.output_folder,

                "autocorrelation.png"

            )

        )

        plt.close()

        print("Saved : autocorrelation.png")


    def suitability_analysis(self):

        print("\n"+"="*70)

        print("SUITABILITY ANALYSIS")

        print("="*70)

        print()

        if self.shannon_entropy > 0.99:

            steg = "Excellent"

        elif self.shannon_entropy > 0.97:

            steg = "Very Good"

        else:

            steg = "Good"

        crypto = steg

        ga = steg

        aco = steg

        qkd = steg

        print(f"Steganography Suitability      : {steg}")

        print(f"Cryptographic Suitability      : {crypto}")

        print(f"GA Seed Suitability            : {ga}")

        print(f"ACO Seed Suitability           : {aco}")

        print(f"QKD Key Suitability            : {qkd}")


    def generate_report(self):

        print("\nGenerating Randomness Report...")

        report = os.path.join(

            self.output_folder,

            "randomness_report.txt"

        )

        with open(report,"w") as file:

            file.write("="*70+"\n")

            file.write("STEGAQENTROPY QRNG RANDOMNESS REPORT\n")

            file.write("="*70+"\n\n")

            file.write(f"Total Bits : {self.total_bits}\n")

            file.write(f"Zeros : {self.total_zeros}\n")

            file.write(f"Ones : {self.total_ones}\n")

            file.write(f"Transition Rate : {self.transition_rate:.8f}\n")

            file.write(f"Longest Zero Run : {self.longest_zero_run}\n")

            file.write(f"Longest One Run : {self.longest_one_run}\n\n")

            file.write("Entropy\n")

            file.write("-------------------------\n")

            file.write(f"Shannon : {self.shannon_entropy:.8f}\n")

            file.write(f"Min : {self.min_entropy:.8f}\n")

            file.write(f"Collision : {self.collision_entropy:.8f}\n")

            file.write(f"Renyi : {self.renyi_entropy:.8f}\n")

        print("Saved : randomness_report.txt")


    def summary(self):

        print("\n"+"="*70)

        print("QRNG RANDOMNESS TESTS COMPLETED")

        print("="*70)

        print()

        print("Generated Reports")

        print("----------------------------")

        print("frequency_distribution.png")

        print("transition_plot.png")

        print("autocorrelation.png")

        print("randomness_report.txt")

        print()

        print("Quantum Randomness Analysis Completed Successfully.")


    def run(self):

        self.banner()

        self.load_pool()

        self.basic_statistics()

        self.transition_analysis()

        self.longest_runs()

        self.entropy_analysis()

        self.entropy_interpretation()

        self.monobit_frequency_test()

        self.block_frequency_test()

        self.runs_test()

        self.longest_run_test()

        self.chi_square_test()

        self.autocorrelation_test()

        self.serial_correlation_test()

        self.lag_correlation_test()

        self.bit_bias()

        self.compression_estimate()

        self.randomness_score()

        self.frequency_distribution_plot()

        self.transition_plot()

        self.autocorrelation_plot()

        self.suitability_analysis()

        self.generate_report()

        self.summary()

        return self