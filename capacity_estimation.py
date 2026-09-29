import math
import os
import numpy as np


class CapacityEstimation:

    def __init__(self):

        self.profile = None
        self.capacity = {}

    def banner(self):

        print("\n" + "=" * 70)
        print("                 STEGAQENTROPY")
        print("             CAPACITY ESTIMATION")
        print("=" * 70)

    def load_profile(self, image_profile):

        self.profile = image_profile

        print("\nImage Profile Received Successfully.")

    def extract_features(self):

        print("\n" + "-" * 70)
        print("IMAGE FEATURES")
        print("-" * 70)

        self.width = self.profile["width"]
        self.height = self.profile["height"]
        self.channels = self.profile["channels"]

        self.entropy = self.profile["entropy"]
        self.variance = self.profile["variance"]

        self.gradient = self.profile["average_gradient"]
        self.edge_density = self.profile["edge_density"]

        self.contrast = self.profile["contrast"]
        self.correlation = self.profile["correlation"]
        self.energy = self.profile["energy"]
        self.homogeneity = self.profile["homogeneity"]
        self.asm = self.profile["asm"]

        print(f"Image Width              : {self.width}")
        print(f"Image Height             : {self.height}")
        print(f"Channels                 : {self.channels}")

        print(f"Entropy                  : {self.entropy:.6f}")
        print(f"Variance                 : {self.variance:.6f}")

        print(f"Gradient                 : {self.gradient:.6f}")
        print(f"Edge Density             : {self.edge_density:.4f} %")

        print(f"Contrast                 : {self.contrast:.6f}")
        print(f"Correlation              : {self.correlation:.6f}")
        print(f"Energy                   : {self.energy:.6f}")
        print(f"Homogeneity              : {self.homogeneity:.6f}")
        print(f"ASM                      : {self.asm:.6f}")

    def normalize(self, value, minimum, maximum):

        if value < minimum:
            value = minimum

        if value > maximum:
            value = maximum

        return (value - minimum) / (maximum - minimum)

    def feature_analysis(self):

        print("\n" + "-" * 70)
        print("FEATURE ANALYSIS")
        print("-" * 70)

        self.entropy_score = self.normalize(
            self.entropy,
            0,
            8
        )

        self.variance_score = self.normalize(
            self.variance,
            0,
            5000
        )

        self.gradient_score = self.normalize(
            self.gradient,
            0,
            300
        )

        self.edge_score = self.normalize(
            self.edge_density,
            0,
            100
        )

        self.contrast_score = self.normalize(
            self.contrast,
            0,
            1000
        )

        self.correlation_score = self.normalize(
            self.correlation,
            -1,
            1
        )

        self.energy_score = self.normalize(
            self.energy,
            0,
            1
        )

        self.homogeneity_score = self.normalize(
            self.homogeneity,
            0,
            1
        )

        self.asm_score = self.normalize(
            self.asm,
            0,
            1
        )

        print(f"Entropy Score            : {self.entropy_score:.4f}")
        print(f"Variance Score           : {self.variance_score:.4f}")
        print(f"Gradient Score           : {self.gradient_score:.4f}")
        print(f"Edge Density Score       : {self.edge_score:.4f}")
        print(f"Contrast Score           : {self.contrast_score:.4f}")
        print(f"Correlation Score        : {self.correlation_score:.4f}")
        print(f"Energy Score             : {self.energy_score:.4f}")
        print(f"Homogeneity Score        : {self.homogeneity_score:.4f}")
        print(f"ASM Score                : {self.asm_score:.4f}")

    def image_suitability(self):

        print("\n" + "-" * 70)
        print("IMAGE SUITABILITY")
        print("-" * 70)

        suitability = (

            self.entropy_score * 0.20 +

            self.variance_score * 0.15 +

            self.gradient_score * 0.15 +

            self.edge_score * 0.10 +

            self.contrast_score * 0.10 +

            self.correlation_score * 0.10 +

            self.energy_score * 0.05 +

            self.homogeneity_score * 0.05 +

            self.asm_score * 0.10

        )

        self.capacity["image_suitability"] = suitability

        print(f"Image Suitability Score  : {suitability:.6f}")

        if suitability >= 0.80:

            level = "Excellent"

        elif suitability >= 0.60:

            level = "Good"

        elif suitability >= 0.40:

            level = "Moderate"

        else:

            level = "Poor"

        self.capacity["suitability_level"] = level

        print(f"Suitability Level        : {level}")

    def image_complexity(self):

        print("\n" + "-" * 70)
        print("IMAGE COMPLEXITY")
        print("-" * 70)

        complexity = (

            self.entropy_score +

            self.gradient_score +

            self.contrast_score +

            self.edge_score

        ) / 4

        self.capacity["complexity_index"] = complexity

        print(f"Complexity Index         : {complexity:.6f}")

        if complexity >= 0.70:

            text = "Highly Complex"

        elif complexity >= 0.35:

            text = "Moderately Complex"

        else:

            text = "Simple"

        self.capacity["complexity_level"] = text

        print(f"Complexity Level         : {text}")

    def embedding_readiness(self):

        print("\n" + "-" * 70)
        print("EMBEDDING READINESS")
        print("-" * 70)

        readiness = (

            self.capacity["image_suitability"] +

            self.capacity["complexity_index"]

        ) / 2

        self.capacity["embedding_readiness"] = readiness

        print(f"Embedding Readiness      : {readiness:.6f}")

        if readiness >= 0.70:

            print("Status                   : READY")

        elif readiness >= 0.35:

            print("Status                   : ACCEPTABLE")

        else:

            print("Status                   : NOT RECOMMENDED")


    def payload_density(self):

        print("\n" + "-" * 70)
        print("PAYLOAD DENSITY ANALYSIS")
        print("-" * 70)

        payload_density = (

            self.capacity["embedding_readiness"] *

            self.entropy_score

        )

        self.capacity["payload_density"] = payload_density

        print(f"Payload Density          : {payload_density:.6f}")

        if payload_density >= 0.70:

            density = "High"

        elif payload_density >= 0.35:

            density = "Medium"

        else:

            density = "Low"

        self.capacity["payload_density_level"] = density

        print(f"Density Level            : {density}")


    def maximum_theoretical_capacity(self):

        print("\n" + "-" * 70)
        print("MAXIMUM THEORETICAL CAPACITY")
        print("-" * 70)

        total_pixels = self.width * self.height

        total_values = total_pixels * self.channels

        total_bits = total_values

        total_bytes = total_bits / 8

        total_characters = int(total_bytes)

        self.capacity["total_pixels"] = total_pixels

        self.capacity["maximum_theoretical_bits"] = total_bits

        self.capacity["maximum_theoretical_bytes"] = total_bytes

        self.capacity["maximum_theoretical_characters"] = total_characters

        print(f"Image Pixels             : {total_pixels:,}")

        print(f"Pixel Values             : {total_values:,}")

        print("\nAssumption")

        print("1 Bit / Pixel Value")

        print("\nMaximum Capacity")

        print(f"Bits                     : {total_bits:,}")

        print(f"Bytes                    : {total_bytes:,.2f}")

        print(f"Characters               : {total_characters:,}")


    def maximum_dwt_capacity(self):

        print("\n" + "-" * 70)
        print("MAXIMUM DWT CAPACITY")
        print("-" * 70)

        total_pixels = self.capacity["total_pixels"]

        coeff_per_band = total_pixels // 4

        ll = coeff_per_band

        lh = coeff_per_band

        hl = coeff_per_band

        hh = coeff_per_band

        usable = lh + hl + hh

        bits = usable

        bytes_ = bits / 8

        chars = int(bytes_)

        self.capacity["ll_coefficients"] = ll

        self.capacity["lh_coefficients"] = lh

        self.capacity["hl_coefficients"] = hl

        self.capacity["hh_coefficients"] = hh

        self.capacity["usable_coefficients"] = usable

        self.capacity["maximum_dwt_bits"] = bits

        self.capacity["maximum_dwt_bytes"] = bytes_

        self.capacity["maximum_dwt_characters"] = chars

        print(f"LL Coefficients          : {ll:,}")

        print(f"LH Coefficients          : {lh:,}")

        print(f"HL Coefficients          : {hl:,}")

        print(f"HH Coefficients          : {hh:,}")

        print()

        print("Embedding Bands")

        print("LL : Ignored")

        print("LH : Used")

        print("HL : Used")

        print("HH : Used")

        print()

        print(f"Usable Coefficients      : {usable:,}")

        print()

        print(f"Maximum DWT Bits         : {bits:,}")

        print(f"Maximum DWT Bytes        : {bytes_:,.2f}")

        print(f"Maximum DWT Characters   : {chars:,}")


    def estimated_safe_capacity(self):

        print("\n" + "-" * 70)
        print("ESTIMATED SAFE CAPACITY")
        print("Heuristic Estimate")
        print("-" * 70)

        score = (

            self.capacity["embedding_readiness"] +

            self.capacity["payload_density"]

        ) / 2

        self.capacity["safe_score"] = score

        safe_bits = int(

            self.capacity["maximum_dwt_bits"]

            *

            score

        )

        safe_bytes = safe_bits / 8

        safe_characters = int(safe_bytes)

        self.capacity["estimated_safe_bits"] = safe_bits

        self.capacity["estimated_safe_bytes"] = safe_bytes

        self.capacity["estimated_safe_characters"] = safe_characters

        print("Method")

        print("Global Statistical Estimation")

        print()

        print(f"Estimated Safe Bits         : {safe_bits:,}")

        print(f"Estimated Safe Bytes        : {safe_bytes:,.2f}")

        print(f"Estimated Safe Characters   : {safe_characters:,}")


    def estimated_recommended_capacity(self):

        print("\n" + "-" * 70)
        print("ESTIMATED RECOMMENDED CAPACITY")
        print("Heuristic Estimate")
        print("-" * 70)

        recommended_bits = int(

            self.capacity["estimated_safe_bits"]

            * 0.90

        )

        recommended_bytes = recommended_bits / 8

        recommended_characters = int(recommended_bytes)

        self.capacity["estimated_recommended_bits"] = recommended_bits

        self.capacity["estimated_recommended_bytes"] = recommended_bytes

        self.capacity["estimated_recommended_characters"] = recommended_characters

        print("Method")

        print("Global Statistical Estimation")

        print()

        print(f"Estimated Recommended Bits         : {recommended_bits:,}")

        print(f"Estimated Recommended Bytes        : {recommended_bytes:,.2f}")

        print(f"Estimated Recommended Characters   : {recommended_characters:,}")


    def estimated_minimum_capacity(self):

        print("\n" + "-" * 70)
        print("ESTIMATED MINIMUM CAPACITY")
        print("Heuristic Estimate")
        print("-" * 70)

        minimum_bits = int(

            self.capacity["estimated_safe_bits"]

            * 0.70

        )

        minimum_bytes = minimum_bits / 8

        minimum_characters = int(minimum_bytes)

        self.capacity["estimated_minimum_bits"] = minimum_bits

        self.capacity["estimated_minimum_bytes"] = minimum_bytes

        self.capacity["estimated_minimum_characters"] = minimum_characters

        print("Method")

        print("Global Statistical Estimation")

        print()

        print(f"Estimated Minimum Bits         : {minimum_bits:,}")

        print(f"Estimated Minimum Bytes        : {minimum_bytes:,.2f}")

        print(f"Estimated Minimum Characters   : {minimum_characters:,}")

    def estimated_over_embedding_threshold(self):

        print("\n" + "-" * 70)
        print("ESTIMATED OVER-EMBEDDING THRESHOLD")
        print("Heuristic Estimate")
        print("-" * 70)

        threshold_bits = int(
            self.capacity["estimated_safe_bits"] * 1.10
        )

        threshold_bytes = threshold_bits / 8

        threshold_characters = int(threshold_bytes)

        self.capacity["estimated_threshold_bits"] = threshold_bits
        self.capacity["estimated_threshold_bytes"] = threshold_bytes
        self.capacity["estimated_threshold_characters"] = threshold_characters

        print("Method")
        print("Global Statistical Estimation\n")

        print(f"Estimated Threshold Bits        : {threshold_bits:,}")
        print(f"Estimated Threshold Bytes       : {threshold_bytes:,.2f}")
        print(f"Estimated Threshold Characters  : {threshold_characters:,}")


    def capacity_utilization(self):

        print("\n" + "-" * 70)
        print("ESTIMATED CAPACITY UTILIZATION")
        print("Heuristic Estimate")
        print("-" * 70)

        utilization = (

            self.capacity["estimated_recommended_bits"]

            /

            self.capacity["maximum_dwt_bits"]

        ) * 100

        self.capacity["capacity_utilization"] = utilization

        print(f"Estimated Utilization      : {utilization:.2f} %")


    def capacity_summary(self):

        print("\n" + "=" * 70)
        print("CAPACITY SUMMARY")
        print("=" * 70)

        print(f"Maximum Theoretical Capacity      : {self.capacity['maximum_theoretical_characters']:,} Characters")

        print(f"Maximum DWT Capacity              : {self.capacity['maximum_dwt_characters']:,} Characters")

        print()

        print(f"Estimated Safe Capacity           : {self.capacity['estimated_safe_characters']:,} Characters")

        print(f"Estimated Recommended Capacity    : {self.capacity['estimated_recommended_characters']:,} Characters")

        print(f"Estimated Minimum Capacity        : {self.capacity['estimated_minimum_characters']:,} Characters")

        print(f"Estimated Over-Embedding Limit    : {self.capacity['estimated_threshold_characters']:,} Characters")

        print()

        print(f"Image Suitability                 : {self.capacity['suitability_level']}")

        print(f"Image Complexity                  : {self.capacity['complexity_level']}")

        print(f"Payload Density                   : {self.capacity['payload_density_level']}")

        print(f"Embedding Readiness               : {self.capacity['embedding_readiness']:.4f}")

        print(f"Capacity Utilization              : {self.capacity['capacity_utilization']:.2f}%")

        print("\nNOTE")
        print("-" * 70)
        print("The above safe/recommended capacities are heuristic")
        print("pre-estimates based on global image statistics.")
        print("Actual adaptive capacity will be calculated")
        print("after DWT + Region Analysis + LR + ACO + GA +")
        print("Gradient Optimization.")



    def update_profile(self):

        self.profile["capacity"] = self.capacity

        return self.profile
    
    def save_capacity_profile(self):

        os.makedirs("output", exist_ok=True)

        file_path = os.path.join("output", "capacity_profile.txt")

        with open(file_path, "w", encoding="utf-8") as file:

            file.write("============================================================\n")
            file.write("STEGAQENTROPY CAPACITY PROFILE\n")
            file.write("============================================================\n\n")

            file.write("[IMAGE INFORMATION]\n")
            file.write(f"width={self.width}\n")
            file.write(f"height={self.height}\n")
            file.write(f"channels={self.channels}\n\n")

            file.write("[IMAGE FEATURES]\n")
            file.write(f"entropy={self.entropy:.6f}\n")
            file.write(f"variance={self.variance:.6f}\n")
            file.write(f"gradient={self.gradient:.6f}\n")
            file.write(f"edge_density={self.edge_density:.4f}\n")
            file.write(f"contrast={self.contrast:.6f}\n")
            file.write(f"correlation={self.correlation:.6f}\n")
            file.write(f"energy={self.energy:.6f}\n")
            file.write(f"homogeneity={self.homogeneity:.6f}\n")
            file.write(f"asm={self.asm:.6f}\n\n")

            file.write("[IMAGE ANALYSIS]\n")
            file.write(f"image_suitability={self.capacity['image_suitability']:.6f}\n")
            file.write(f"suitability_level={self.capacity['suitability_level']}\n")
            file.write(f"complexity_index={self.capacity['complexity_index']:.6f}\n")
            file.write(f"complexity_level={self.capacity['complexity_level']}\n")
            file.write(f"embedding_readiness={self.capacity['embedding_readiness']:.6f}\n")
            file.write(f"payload_density={self.capacity['payload_density']:.6f}\n")
            file.write(f"payload_density_level={self.capacity['payload_density_level']}\n\n")

            file.write("[CAPACITY]\n")
            file.write(f"maximum_theoretical_characters={self.capacity['maximum_theoretical_characters']}\n")
            file.write(f"maximum_dwt_characters={self.capacity['maximum_dwt_characters']}\n")
            file.write(f"recommended_characters={self.capacity['estimated_recommended_characters']}\n")
            file.write(f"safe_characters={self.capacity['estimated_safe_characters']}\n")
            file.write(f"minimum_characters={self.capacity['estimated_minimum_characters']}\n")
            file.write(f"threshold_characters={self.capacity['estimated_threshold_characters']}\n")
            file.write(f"capacity_utilization={self.capacity['capacity_utilization']:.2f}\n")

        print("\nCapacity Profile Saved Successfully.")
        print(f"Location : {file_path}")


    def run(self, image_profile):

        self.banner()

        self.load_profile(image_profile)

        self.extract_features()

        self.feature_analysis()

        self.image_suitability()

        self.image_complexity()

        self.embedding_readiness()

        self.payload_density()

        self.maximum_theoretical_capacity()

        self.maximum_dwt_capacity()

        self.estimated_safe_capacity()

        self.estimated_recommended_capacity()

        self.estimated_minimum_capacity()

        self.estimated_over_embedding_threshold()

        self.capacity_utilization()

        self.capacity_summary()

        self.save_capacity_profile()

        return self.update_profile()