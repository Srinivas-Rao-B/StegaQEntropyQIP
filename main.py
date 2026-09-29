import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import shutil

from image_acquisition import ImageAcquisition
from capacity_estimation import CapacityEstimation
from qrng.qrng_randomness_engine import QRNGRandomnessEngine
from history_checker import HistoryChecker
from message_preparation import MessagePreparation
from quantum_key_hierarchy import QuantumKeyHierarchy
from dwt_decomposition import DWTDecomposition
from entropy_intelligence_engine import EntropyIntelligenceEngine
from ml_module.dataset_builder import DatasetBuilder
from ml_module.post_embedding_analysis_fixed import PostEmbeddingAnalysis
from ml_module.gaussian_probability_plot import main as gaussian_probability_main
from ml_module.surrogate_model import main as surrogate_model_main


class Tee:

    def __init__(self, *files):

        self.files = files

        self.encoding = getattr(sys.__stdout__, "encoding", "utf-8")

        self.errors = getattr(sys.__stdout__, "errors", "strict")

    def write(self, data):

        for file in self.files:

            file.write(data)

            file.flush()

    def flush(self):

        for file in self.files:

            file.flush()

    def isatty(self):

        return False

    def fileno(self):

        return sys.__stdout__.fileno()


def next_module(module_name):

    print("\n" + "=" * 70)

    input(f"Press ENTER to continue to {module_name}...")

    print("=" * 70 + "\n")


def run_module(module_name, module_function, execution_log):

    module_folder = os.path.join(
        "output",
        "output_modules"
    )

    os.makedirs(
        module_folder,
        exist_ok=True
    )

    module_log_path = os.path.join(
        module_folder,
        f"{module_name}_Execution.txt"
    )

    module_log = open(
        module_log_path,
        "w",
        encoding="utf-8"
    )

    original_stdout = sys.__stdout__

    sys.stdout = Tee(
        original_stdout,
        execution_log,
        module_log
    )

    try:

        result = module_function()

    finally:

        sys.stdout = original_stdout

        module_log.close()

    return result


def remove_path(path):

    if os.path.isdir(path):

        shutil.rmtree(
            path,
            ignore_errors=True
        )

    elif os.path.isfile(path):

        try:

            os.remove(path)

        except:

            pass

def select_start_module():

    print("\n" + "=" * 70)

    print("SELECT EXECUTION MODE")

    print("=" * 70)

    print("0. Run Complete Pipeline")
    print("1. Image Acquisition & Capacity Estimation")
    print("2. QRNG Randomness Engine")
    print("3. History Checker")
    print("4. Message Preparation")
    print("5. Quantum Key Hierarchy")
    print("6. DWT Decomposition")
    print("7. Entropy Intelligence Engine")
    print("8. Dataset Builder")
    print("9. Post Embedding Analysis")
    print("10. Gaussian Probability Plot")
    print("11. Surrogate Model")     
   

    while True:

        try:

            choice = int(
                input("\nEnter Choice : ")
            )

            if 0 <= choice <= 11:

                return choice

        except:

            pass

        print("Invalid Choice.")

def check_previous_data(start_module):

    if start_module <= 2:

        return start_module

    required = {

        3: [
            "qrng/output/qrng_pool.txt",
            "qrng/output/seeds"
        ],

        4: [
            "output/capacity_profile.txt",
            "qrng/output/history_check/final_verified_seeds"
        ],

        5: "output/message_preparation/message_preparation.json",

        6: [
            "output/image_acquisition/image_profile.json",
            "qrng/output/history_check/final_verified_seeds"
        ],

        7: [
            "output/dwt_decomposition/manifest.json",
            "qrng/output/history_check/final_verified_seeds"
        ],
        8: [
            "output/entropy_intelligence/packages/primary_regions.json",
            "output/entropy_intelligence/packages/secondary_regions.json",
            "output/entropy_intelligence/packages/tertiary_regions.json",
            "output/entropy_intelligence/packages/rejected_regions.json"
        ],
        9: [
            "ml_module/output/dataset_builder/master_dataset.csv"
        ],

        10: [
            "ml_module/output/dataset_builder/master_dataset.csv",
            "ml_module/output/post_embedding_analysis/actual_post_embedding_final_dataset.csv"
        ],

        11: [
            "ml_module/output/probability_analysis/probability_intelligence_final_dataset.csv",
            "output/image_acquisition/image_profile.json",
            "output/dwt_decomposition"
        ]

    }

    paths = required[start_module]

    if not isinstance(paths, list):

        paths = [paths]

    exists = True

    for path in paths:

        if os.path.isdir(path):

            current = (
                os.path.exists(path)
                and
                len(os.listdir(path)) > 0
            )

        else:

            current = os.path.exists(path)

        if not current:

            exists = False

            break

    if exists:

        return start_module

    print("\n" + "=" * 70)

    print("PREVIOUS MODULE DATA NOT FOUND")

    print("=" * 70)

    print("\nNo previous module output found.")

    print("Data might have been deleted.")

    print("Please start execution from Module 1.")

    while True:

        choice = input(
            "\n1 - Start from Module 1\n0 - Exit\n\nChoice : "
        )

        if choice == "1":

            return 1

        if choice == "0":

            sys.exit()

        print("Invalid Choice.")


def main():

    os.makedirs(
        "output",
        exist_ok=True
    )

    execution_log_path = os.path.join(
        "output",
        "StegaQEntropy_Execution.txt"
    )

    execution_log = open(
        execution_log_path,
        "w",
        encoding="utf-8"
    )

    original_stdout = sys.stdout

    sys.stdout = Tee(
        original_stdout,
        execution_log
    )

    try:

        print("\n" + "=" * 70)

        print("                 STEGAQENTROPY")

        print("     HYBRID QUANTUM STEGANOGRAPHY SYSTEM")

        print("=" * 70)

        while True:

            start_module = select_start_module()

            full_pipeline = False

            if start_module == 0:

                full_pipeline = True

                end_module = 11

            else:

                try:

                    end_module = int(
                        input("\nEnter End Module : ")
                    )

                except:

                    print("\nInvalid Choice.")
                    print("Please select Start and End modules again.\n")
                    continue

                if start_module > end_module or end_module > 11:

                    print("\nInvalid Range.")
                    print("Please select Start and End modules again.\n")
                    continue

            if full_pipeline:

                start_module = 1

            else:

                result = check_previous_data(start_module)

                if result == 1:

                    start_module = 1

                else:

                    start_module = result

            break

        image_profile = None

        if start_module <= 1 <= end_module:

            print("\nStarting from Image Acquisition Module.")

            print("Please Upload the cover image proceeding.")

            image_module = ImageAcquisition()

            image_profile = run_module(
                "image_acquisition",
                image_module.run,
                execution_log
            )

            next_module("Capacity Estimation Module")

            capacity_module = CapacityEstimation()

            project_profile = run_module(
                "capacity_estimation",
                lambda: capacity_module.run(image_profile),
                execution_log
            )

            next_module("QRNG Randomness Engine")


        if start_module <= 2 <= end_module:

            qrng_engine = QRNGRandomnessEngine()

            qrng_profile = run_module(
                "qrng_randomness_engine",
                qrng_engine.run,
                execution_log
            )

            next_module("History Checker")



        if start_module <= 3 <= end_module:

            history_checker = HistoryChecker()

            run_module(
                "history_checker",
                history_checker.run,
                execution_log
            )

            next_module("Message Preparation")


        if start_module <= 4 <= end_module:

            message_preparation = MessagePreparation()

            run_module(
                "message_preparation",
                message_preparation.run,
                execution_log
            )

        if start_module <= 5 <= end_module:

            quantum_key_hierarchy = QuantumKeyHierarchy()

            run_module(
                "quantum_key_hierarchy",
                quantum_key_hierarchy.run,
                execution_log
            )

            next_module("DWT Analysis")

        if start_module <= 6 <= end_module:

            dwt = DWTDecomposition()

            run_module(
                "dwt_decomposition",
                dwt.run,
                execution_log
            )

            next_module("Entropy Intelligence")

        if start_module <= 7 <= end_module:

            entropy_engine = EntropyIntelligenceEngine()

            run_module(
                "entropy_intelligence_engine",
                entropy_engine.run,
                execution_log
            )

            next_module("Machine Learning/CNN Analysis")

        if start_module <= 8 <= end_module:

            dataset_builder = DatasetBuilder()

            run_module(
                "dataset_builder",
                dataset_builder.run,
                execution_log
            )

            next_module("Post Embedding Analysis")


        if start_module <= 9 <= end_module:

            post_embedding_analysis = PostEmbeddingAnalysis()

            def run_post_embedding_analysis_module():

                post_embedding_analysis.initialize_post_embedding_pipeline()

                print(
                    "\nPre-Embedding Dataset Ready."
                )

                print(
                    f"Rows        : "
                    f"{len(post_embedding_analysis.dataset)}"
                )

                print(
                    f"Columns     : "
                    f"{len(post_embedding_analysis.dataset.columns)}"
                )

                print(
                    f"Experiments : "
                    f"{len(post_embedding_analysis.experiment_plan)}"
                )

                print(
                    "\n"
                    + "=" * 80
                )

                print(
                    "STARTING ACTUAL POST-EMBEDDING EXPERIMENTS"
                )

                print(
                    "=" * 80
                )

                result = (
                    post_embedding_analysis
                    .run_complete_post_embedding_dataset_generation(
                        post_embedding_analysis.actual_embedding_function
                    )
                )

                print(
                    "\n"
                    + "=" * 80
                )

                print(
                    "POST-EMBEDDING DATASET GENERATION COMPLETED"
                )

                print(
                    "=" * 80
                )

                print(
                    f"Final Dataset : "
                    f"{result['final_dataset']}"
                )

                print(
                    f"ML Dataset    : "
                    f"{result['ml_dataset']}"
                )

                print(
                    f"Targets       : "
                    f"{result['targets']}"
                )

                print(
                    "=" * 80
                )

                return result

            run_module(
                "post_embedding_analysis_fixed",
                run_post_embedding_analysis_module,
                execution_log
            )

            next_module("Gaussian Probability Plot")


        if start_module <= 10 <= end_module:

            run_module(
                "gaussian_probability_plot",
                gaussian_probability_main,
                execution_log
            )

            next_module("Surrogate Model")


        if start_module <= 11 <= end_module:

            run_module(
                "surrogate_model",
                surrogate_model_main,
                execution_log
            )

        print("\nCurrent Execution Finished.")

        print("\n" + "=" * 70)

        print("STEGAQENTROPY EXECUTION COMPLETED")

        print("=" * 70)

    finally:

        sys.stdout = original_stdout

        execution_log.close()

    print("\nComplete Execution Log Saved")

    print(execution_log_path)

    print("\nIndividual Module Logs Saved")

    print(os.path.join("output", "output_modules"))


if __name__ == "__main__":

    main()