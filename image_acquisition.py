import os
import cv2
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from skimage.feature import local_binary_pattern


class ImageAcquisition:

    def __init__(self):

        self.image_path = None
        self.image = None
        self.gray = None
        self.profile = {}

    def banner(self):

        print("\n" + "=" * 70)
        print("                 STEGAQENTROPY")
        print("          IMAGE ACQUISITION MODULE")
        print("=" * 70)

    def load_image(self):

        print("\n[1] Loading Cover Image...")

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"\nImage not found : {self.image_path}")

        self.image = cv2.imread(self.image_path)

        if self.image is None:
            raise ValueError("\nInvalid image.")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        print("Image Loaded Successfully")

    def basic_information(self):

        print("\n" + "-" * 70)
        print("BASIC IMAGE INFORMATION")
        print("-" * 70)

        filename = os.path.basename(self.image_path)

        filesize = os.path.getsize(self.image_path) / 1024

        height, width, channels = self.image.shape

        bit_depth = self.image.dtype.itemsize * 8

        aspect_ratio = round(width / height, 2)

        self.profile["filename"] = filename
        self.profile["width"] = width
        self.profile["height"] = height
        self.profile["channels"] = channels
        self.profile["bit_depth"] = bit_depth
        self.profile["resolution"] = f"{width} x {height}"
        self.profile["file_size_kb"] = round(filesize, 2)
        self.profile["aspect_ratio"] = aspect_ratio

        print(f"File Name           : {filename}")
        print(f"Image Width         : {width} pixels")
        print(f"Image Height        : {height} pixels")
        print(f"Color Space         : RGB")
        print(f"Channels            : {channels}")
        print(f"Bit Depth           : {bit_depth} bits/channel")
        print(f"Resolution          : {width} x {height}")
        print(f"File Size           : {filesize:.2f} KB")
        print(f"Aspect Ratio        : {aspect_ratio}")

    def statistical_analysis(self):

        print("\n" + "-" * 70)
        print("STATISTICAL ANALYSIS")
        print("-" * 70)

        mean = np.mean(self.gray)
        std = np.std(self.gray)
        var = np.var(self.gray)

        minimum = np.min(self.gray)
        maximum = np.max(self.gray)

        self.profile["mean"] = float(mean)
        self.profile["std_dev"] = float(std)
        self.profile["variance"] = float(var)
        self.profile["minimum"] = int(minimum)
        self.profile["maximum"] = int(maximum)

        print(f"Mean Intensity      : {mean:.4f}")
        print(f"Standard Deviation  : {std:.4f}")
        print(f"Variance            : {var:.4f}")
        print(f"Minimum Pixel       : {minimum}")
        print(f"Maximum Pixel       : {maximum}")

    def histogram_analysis(self):

        print("\n" + "-" * 70)
        print("HISTOGRAM ANALYSIS")
        print("-" * 70)

        histogram = cv2.calcHist([self.gray], [0], None, [256], [0, 256])

        histogram = histogram.flatten()

        normalized_hist = histogram / histogram.sum()

        dominant_peak = np.argmax(histogram)

        self.profile["histogram"] = histogram
        self.profile["normalized_histogram"] = normalized_hist
        self.profile["dominant_peak"] = int(dominant_peak)

        print("Histogram Generated Successfully")
        print("Histogram Bins      : 256")
        print("Normalized          : YES")
        print(f"Dominant Peak       : {dominant_peak}")

        os.makedirs("output", exist_ok=True)

        plt.figure(figsize=(8, 4))
        plt.plot(histogram, color="black")
        plt.title("Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output/image_acquisition/histogram.png")
        plt.close()

        print("Histogram Saved     : output/histogram.png")

    def shannon_entropy(self):

        print("\n" + "-" * 70)
        print("SHANNON ENTROPY")
        print("-" * 70)

        histogram = self.profile["normalized_histogram"]

        shannon = entropy(histogram, base=2)

        self.profile["entropy"] = float(shannon)
        utilization = (shannon / 8) * 100

        print(f"Maximum Entropy     : 8.000000 bits")
        print(f"Current Entropy     : {shannon:.6f} bits")
        print(f"Entropy Utilization : {utilization:.2f} %")

    def save_grayscale(self):

        cv2.imwrite("output/image_acquisition/grayscale.png", self.gray)

    def image_profile(self):

        print("\n" + "-" * 70)
        print("IMAGE PROFILE GENERATED")
        print("-" * 70)

        for key, value in self.profile.items():

            if isinstance(value, np.ndarray):
                print(f"{key:<22}: Stored")
                continue

            print(f"{key:<22}: {value}")

    def run(self):
        Tk().withdraw()

        self.image_path = askopenfilename(
            title="Select Cover Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")
            ]
        )

        if not self.image_path:
            print("No image selected.")
            exit()


        self.load_image()

        self.basic_information()

        self.statistical_analysis()

        self.histogram_analysis()

        self.shannon_entropy()

        self.save_grayscale()

        self.gradient_analysis()

        self.edge_analysis()

        self.texture_analysis()

        self.image_profile()

        self.final_summary()

        self.save_image_profile_json()

        self.profile["image_path"] = self.image_path

        print("\nImage Acquisition Completed Successfully")


        return self.profile

    def gradient_analysis(self):

        print("\n" + "-" * 70)
        print("GRADIENT ANALYSIS")
        print("-" * 70)

        sobel_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)

        gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        min_gradient = np.min(gradient)
        std_gradient = np.std(gradient)

        avg_gradient = np.mean(gradient)
        max_gradient = np.max(gradient)

        self.profile["gradient_map"] = gradient
        self.profile["average_gradient"] = float(avg_gradient)
        self.profile["maximum_gradient"] = float(max_gradient)
        self.profile["minimum_gradient"] = float(min_gradient)
        self.profile["gradient_std"] = float(std_gradient)

        gradient_display = cv2.normalize(
            gradient,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        cv2.imwrite("output/image_acquisition/gradient_map.png", gradient_display)


        print(f"Average Gradient    : {avg_gradient:.4f}")
        print(f"Maximum Gradient    : {max_gradient:.4f}")
        print(f"Minimum Gradient    : {min_gradient:.4f}")
        print(f"Gradient Std Dev    : {std_gradient:.4f}")
        print("Gradient Map Saved  : output/gradient_map.png")


    def edge_analysis(self):

        print("\n" + "-" * 70)
        print("EDGE ANALYSIS")
        print("-" * 70)

        edges = cv2.Canny(self.gray, 100, 200)

        edge_pixels = np.count_nonzero(edges)

        total_pixels = self.gray.shape[0] * self.gray.shape[1]

        edge_density = (edge_pixels / total_pixels) * 100

        self.profile["edge_pixels"] = int(edge_pixels)
        self.profile["edge_density"] = float(edge_density)

        cv2.imwrite("output/image_acquisition/edge_map.png", edges)

        print(f"Edge Pixels         : {edge_pixels}")
        print(f"Edge Density        : {edge_density:.4f} %")
        print("Edge Map Saved      : output/edge_map.png")


    def texture_analysis(self):

        print("\n" + "-" * 70)
        print("TEXTURE ANALYSIS (GLCM)")
        print("-" * 70)

        print("\nGenerating Gray-Level Co-occurrence Matrix (GLCM)...")

        distances = [1, 2]

        angles = [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ]

        glcm = graycomatrix(
            self.gray,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        print("GLCM Generated Successfully")

        print("\nExtracting Haralick Texture Features...\n")

        contrast = np.mean(graycoprops(glcm, "contrast"))

        correlation = np.mean(graycoprops(glcm, "correlation"))

        energy = np.mean(graycoprops(glcm, "energy"))

        homogeneity = np.mean(graycoprops(glcm, "homogeneity"))

        asm = np.mean(graycoprops(glcm, "ASM"))

        self.profile["contrast"] = float(contrast)
        self.profile["correlation"] = float(correlation)
        self.profile["energy"] = float(energy)
        self.profile["homogeneity"] = float(homogeneity)
        self.profile["asm"] = float(asm)

        self.profile["glcm"] = glcm

        print(f"Distances           : {distances}")
        print("Angles              : 0°, 45°, 90°, 135°")
        print(f"Gray Levels         : 256")
        print(f"Distances           : {len(distances)}")
        print(f"Angles              : {len(angles)}")
        print(f"Matrix Shape        : {glcm.shape}")

        print("\nHaralick Features")
        print("-" * 35)

        print(f"Contrast            : {contrast:.6f}")
        print(f"Correlation         : {correlation:.6f}")
        print(f"Energy              : {energy:.6f}")
        print(f"Homogeneity         : {homogeneity:.6f}")
        print(f"ASM                 : {asm:.6f}")

        print("\nFeature Interpretation")
        print("-" * 35)

        print("\nTexture Interpretation")
        print("-" * 45)

        print(
            f"Contrast ({contrast:.4f}) : "
            "Measures local intensity variation. "
            "Higher values indicate stronger edges and richer texture."
        )

        print(
            f"Correlation ({correlation:.4f}) : "
            "Measures the similarity between neighbouring pixels. "
            "Higher values indicate stronger structural continuity."
        )

        print(
            f"Energy ({energy:.4f}) : "
            "Measures texture uniformity. "
            "Higher values indicate more regular and homogeneous regions."
        )

        print(
            f"Homogeneity ({homogeneity:.4f}) : "
            "Measures the closeness of neighbouring gray levels. "
            "Higher values indicate smoother image regions."
        )

        print(
            f"ASM ({asm:.6f}) : "
            "Angular Second Moment represents textural uniformity and image order."
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(glcm[:, :, 0, 0], cmap="gray")
        plt.title("GLCM Matrix")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("output/image_acquisition/glcm_matrix.png")
        plt.close()

        print("\nGLCM Matrix Saved   : output/glcm_matrix.png")
        print("\nGenerating LBP Texture Visualization...")

        radius = 1
        points = 8 * radius

        lbp = local_binary_pattern(
            self.gray,
            points,
            radius,
            method="uniform"
        )

        lbp_display = cv2.normalize(
            lbp,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        cv2.imwrite("output/image_acquisition/texture_map.png", lbp_display)

        self.profile["texture_map"] = lbp_display

        print("LBP Texture Map Saved : output/texture_map.png")

        print("\nTexture Analysis Completed Successfully")

    def save_image_profile_json(self):

        print("\nSaving Image Profile JSON...")

        os.makedirs("output/image_acquisition", exist_ok=True)

        profile = {
            "module": "Image Acquisition",
            "module_version": "1.0",
            "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "SUCCESS",

            "filename": self.profile.get("filename"),
            "image_path": self.image_path,
            "width": self.profile.get("width"),
            "height": self.profile.get("height"),
            "channels": self.profile.get("channels"),
            "bit_depth": self.profile.get("bit_depth"),
            "resolution": self.profile.get("resolution"),
            "file_size_kb": self.profile.get("file_size_kb"),
            "aspect_ratio": self.profile.get("aspect_ratio"),

            "mean": self.profile.get("mean"),
            "std_dev": self.profile.get("std_dev"),
            "variance": self.profile.get("variance"),
            "minimum": self.profile.get("minimum"),
            "maximum": self.profile.get("maximum"),

            "dominant_peak": self.profile.get("dominant_peak"),
            "histogram": self.profile.get("histogram").tolist()
            if isinstance(self.profile.get("histogram"), np.ndarray)
            else self.profile.get("histogram"),

            "normalized_histogram": self.profile.get("normalized_histogram").tolist()
            if isinstance(self.profile.get("normalized_histogram"), np.ndarray)
            else self.profile.get("normalized_histogram"),

            "entropy": self.profile.get("entropy"),

            "average_gradient": self.profile.get("average_gradient"),
            "maximum_gradient": self.profile.get("maximum_gradient"),
            "minimum_gradient": self.profile.get("minimum_gradient"),
            "gradient_std": self.profile.get("gradient_std"),

            "edge_pixels": self.profile.get("edge_pixels"),
            "edge_density": self.profile.get("edge_density"),

            "contrast": self.profile.get("contrast"),
            "correlation": self.profile.get("correlation"),
            "energy": self.profile.get("energy"),
            "homogeneity": self.profile.get("homogeneity"),
            "asm": self.profile.get("asm"),

           "generated_files": {
                "grayscale_image": "output/image_acquisition/grayscale.png",
                "histogram_image": "output/image_acquisition/histogram.png",
                "gradient_image": "output/image_acquisition/gradient_map.png",
                "edge_image": "output/image_acquisition/edge_map.png",
                "texture_image": "output/image_acquisition/texture_map.png",
                "glcm_image": "output/image_acquisition/glcm_matrix.png"
            }
        }

        output_path = os.path.join(
            "output",
            "image_acquisition",
            "image_profile.json"
        )

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(profile, file, indent=4)

        print(f"Image Profile Saved : {output_path}")

    

    def final_summary(self):

        print("\n" + "=" * 70)
        print("IMAGE ACQUISITION COMPLETED")
        print("=" * 70)

        print("\nGenerated Files")

        print("----------------------------")

        print("output/image_acquisition/grayscale.png")
        print("output/image_acquisition/histogram.png")
        print("output/image_acquisition/gradient_map.png")
        print("output/image_acquisition/edge_map.png")
        print("output/image_acquisition/texture_map.png")
        print("output/image_acquisition/glcm_matrix.png")

        print("\nImage successfully profiled.")
        print("Ready for Capacity Estimation Module.")

