import os
import json
import numpy as np
import pandas as pd

from .ml_configuration import MLConfiguration
from .ml_random_manager import MLRandomManager


class DatasetBuilder:

    def __init__(self):

        self.configuration = MLConfiguration()

        self.random_manager = MLRandomManager()

        self.loaded_data = {}

        self.master_dataset = []

        self.dataset_manifest = {}

        self.dataset_statistics = {}

        self.feature_names = []

        self.neighbor_feature_keys = [
            "entropy",
            "statistics_entropy",
            "statistics_variance",
            "statistics_std",
            "statistics_energy",
            "statistics_minimum",
            "statistics_maximum",
            "gradient_intelligence_average_gradient",
            "gradient_intelligence_maximum_gradient",
            "gradient_intelligence_edge_density",
            "noise_intelligence_noise_mean",
            "noise_intelligence_noise_variance",
            "noise_intelligence_noise_std",
            "noise_intelligence_noise_density",
            "texture_complexity_gradient_strength",
            "coefficient_density",
            "complexity_score"
        ]

        self.spatial_directions = {
            "N1": (-1, -1),
            "N2": (-1, 0),
            "N3": (-1, 1),
            "N4": (0, -1),
            "N5": (0, 1),
            "N6": (1, -1),
            "N7": (1, 0),
            "N8": (1, 1)
        }

        self.coordinate_map = {}

        self.required_files = {

            "candidate_regions":
            "output/dwt_decomposition/candidate_regions.json",

            "adaptive_regions":
            "output/dwt_decomposition/adaptive_regions.json",

            "region_analysis":
            "output/dwt_decomposition/region_analysis.json",

            "fitness_weights":
            "output/dwt_decomposition/adaptive_fitness_weights.json",

            "entropy_database":
            "output/entropy_intelligence/entropy_database.json",

            "feature_dataset":
            "output/entropy_intelligence/feature_dataset.json",

            "normalized_dataset":
            "output/entropy_intelligence/normalized_dataset.json",

            "region_statistics":
            "output/entropy_intelligence/region_statistics.json",

            "region_combination_database":
            "output/entropy_intelligence/region_combination_database.json",

            "distortion_database":
            "output/entropy_intelligence/distortion_database.json",

            "embedding_scenarios":
            "output/embedding/embedding_scenarios.json",

            "embedding_results":
            "output/embedding/embedding_results.json",

            "prediction_features":
            "output/entropy_intelligence/prediction_features.json",

            "chunk_region_candidates":
            "output/entropy_intelligence/chunk_region_candidates.json",

            "primary_regions":
            "output/entropy_intelligence/packages/primary_regions.json",

            "secondary_regions":
            "output/entropy_intelligence/packages/secondary_regions.json",

            "tertiary_regions":
            "output/entropy_intelligence/packages/tertiary_regions.json",

            "rejected_regions":
            "output/entropy_intelligence/packages/rejected_regions.json",

        }


    def log(

        self,

        message

    ):

        print(message)

    def convert_region_list_to_dictionary(self, data):

        if isinstance(data, dict):
            return data

        if not isinstance(data, list):
            return {}

        converted = {}

        for item in data:

            if isinstance(item, dict):

                region_id = item.get("region_id")

                if region_id:

                    converted[region_id] = item

        return converted


    def verify_input_files(self):

        self.log(

            "\nChecking Required Files...\n"

        )

        missing_files = []

        optional_files = {
            "embedding_scenarios",
            "embedding_results"
        }

        for name, path in self.required_files.items():

            if not os.path.exists(path):

                if name in optional_files:

                    self.log(
                        f"[OPTIONAL] {name} not available"
                    )

                else:

                    missing_files.append(path)

            else:

                self.log(
                    f"[OK] {name}"
                )

        if len(missing_files) > 0:

            print()

            for file in missing_files:

                print(file)

            raise FileNotFoundError(

                "\nRequired files missing."

            )


    def load_input_files(self):

        self.log(

            "\nLoading Previous Module Outputs...\n"

        )

        optional_files = {
            "embedding_scenarios",
            "embedding_results"
        }

        for name, path in self.required_files.items():

            if not os.path.exists(path):

                if name in optional_files:

                    self.loaded_data[name] = {}

                    continue

                raise FileNotFoundError(path)

            if path.endswith(".json"):

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    self.loaded_data[name] = json.load(
                        file
                    )

            else:

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    self.loaded_data[name] = file.read()


        self.log(

            "Input Files Loaded Successfully\n"

        )

    def add_embedding_targets(self, row, result):

        if not isinstance(result, dict):
            result = {}

        target_fields = {
            "delta_entropy": "target_delta_entropy",
            "delta_noise": "target_delta_noise",
            "mse": "target_mse",
            "psnr": "target_psnr",
            "ssim": "target_ssim",
            "neighbor_impact": "target_neighbor_impact",
            "post_embedding_quality": "target_post_embedding_quality",
            "post_entropy": "target_post_entropy",
            "post_noise": "target_post_noise",
            "post_variance": "target_post_variance",
            "post_gradient": "target_post_gradient"
        }

        for source_key, target_key in target_fields.items():
            row[target_key] = result.get(
                source_key,
                np.nan
            )

    def flatten_dictionary(
        self,
        dictionary,
        prefix=""
    ):

        flattened = {}

        if isinstance(dictionary, dict):

            for key, value in dictionary.items():

                new_key = (
                    f"{prefix}_{key}"
                    if prefix
                    else str(key)
                )

                if isinstance(value, dict):

                    flattened.update(
                        self.flatten_dictionary(
                            value,
                            new_key
                        )
                    )

                elif isinstance(value, list):

                    for index, item in enumerate(value):

                        list_key = (
                            f"{new_key}_{index}"
                        )

                        if isinstance(item, dict):

                            flattened.update(
                                self.flatten_dictionary(
                                    item,
                                    list_key
                                )
                            )

                        else:

                            flattened[list_key] = item

                else:

                    flattened[new_key] = value

        else:

            if prefix:

                flattened[prefix] = dictionary

        return flattened

    def build_coordinate_map(self, region_analysis):

        self.coordinate_map = {}

        for region_id, data in region_analysis.items():

            row = (
                data.get("row_start")
                if isinstance(data, dict)
                else None
            )

            column = (
                data.get("column_start")
                if isinstance(data, dict)
                else None
            )

            if row is None and isinstance(data, dict):
                row = data.get("row")

            if column is None and isinstance(data, dict):
                column = data.get("column")

            if row is None or column is None:
                continue

            self.coordinate_map[
                (int(row), int(column))
            ] = region_id

    def get_spatial_neighbors(self, region_id, region_data):

        row = region_data.get("row_start")
        column = region_data.get("column_start")

        if row is None:
            row = region_data.get("row")

        if column is None:
            column = region_data.get("column")

        if row is None or column is None:
            return {
                name: None
                for name in self.spatial_directions
            }

        row = int(row)
        column = int(column)

        neighbors = {}

        for name, (dr, dc) in self.spatial_directions.items():

            neighbors[name] = self.coordinate_map.get(
                (row + dr, column + dc)
            )

        return neighbors

    def add_neighbor_features(
        self,
        row,
        neighbors,
        region_database
    ):

        for neighbor_name, neighbor_id in neighbors.items():

            for feature_name in self.neighbor_feature_keys:

                column_name = (
                    f"{neighbor_name}_{feature_name}"
                )

                if neighbor_id is None:
                    row[column_name] = np.nan
                    continue

                neighbor_data = region_database.get(
                    neighbor_id,
                    {}
                )

                value = neighbor_data.get(
                    feature_name,
                    np.nan
                )

                if isinstance(value, (dict, list)):
                    value = np.nan

                row[column_name] = value

    def build_region_feature_lookup(
        self,
        region_analysis,
        candidate_regions,
        adaptive_regions,
        entropy_database,
        region_database,
        prediction_features,
        fitness_weights,
        feature_dataset,
        normalized_dataset,
        region_statistics,
        distortion_database,
        chunk_region_candidates
    ):

        lookup = {}

        region_ids = set()

        region_ids.update(region_analysis.keys())
        region_ids.update(candidate_regions.keys())
        region_ids.update(adaptive_regions.keys())
        region_ids.update(entropy_database.keys())
        region_ids.update(region_database.keys())
        region_ids.update(prediction_features.keys())
        region_ids.update(fitness_weights.keys())
        region_ids.update(feature_dataset.keys())
        region_ids.update(normalized_dataset.keys())
        region_ids.update(region_statistics.keys())
        region_ids.update(distortion_database.keys())
        region_ids.update(chunk_region_candidates.keys())

        for region_id in region_ids:

            combined = {}

            sources = [
                candidate_regions,
                adaptive_regions,
                region_analysis,
                entropy_database,
                region_database,
                fitness_weights,
                prediction_features,
                feature_dataset,
                normalized_dataset,
                region_statistics,
                distortion_database,
                chunk_region_candidates
            ]

            for source in sources:

                if region_id not in source:
                    continue

                values = self.flatten_dictionary(
                    source[region_id]
                )

                for key, value in values.items():

                    if key not in combined:

                        combined[key] = value
            lookup[region_id] = combined

        return lookup

    def add_neighbor_derived_features(
        self,
        row
    ):

        for feature_name in self.neighbor_feature_keys:

            values = []

            for neighbor_name in self.spatial_directions:

                value = row.get(
                    f"{neighbor_name}_{feature_name}"
                )

                if value is None:
                    continue

                if pd.isna(value):
                    continue

                try:
                    values.append(float(value))
                except:
                    continue

            if not values:
                continue

            row[
                f"neighbor_{feature_name}_mean"
            ] = float(np.mean(values))

            row[
                f"neighbor_{feature_name}_std"
            ] = float(np.std(values))

            row[
                f"neighbor_{feature_name}_minimum"
            ] = float(np.min(values))

            row[
                f"neighbor_{feature_name}_maximum"
            ] = float(np.max(values))

            center_value = row.get(feature_name)

            if (
                center_value is not None
                and not pd.isna(center_value)
            ):

                try:

                    row[
                        f"center_neighbor_{feature_name}_difference"
                    ] = float(
                        center_value -
                        np.mean(values)
                    )

                except:
                    pass

    def add_embedding_scenario_features(
        self,
        row,
        scenario
    ):

        if not scenario:
            scenario = {}

        scenario_fields = [
            "payload_size",
            "chunk_size",
            "embedding_strength",
            "embedding_density",
            "bit_plane",
            "dwt_subband",
            "coefficient_type",
            "embedding_method",
            "embedding_position",
            "threshold"
        ]

        for field in scenario_fields:

            row[
                f"embedding_{field}"
            ] = scenario.get(
                field,
                np.nan
            )

    def get_embedding_scenarios(
        self,
        region_id
    ):

        scenarios = self.loaded_data.get(
            "embedding_scenarios",
            {}
        )

        if isinstance(scenarios, dict):

            value = scenarios.get(
                region_id,
                []
            )

            if isinstance(value, list):
                return value

            if isinstance(value, dict):
                return [value]

        if isinstance(scenarios, list):

            output = []

            for item in scenarios:

                if not isinstance(item, dict):
                    continue

                item_region = item.get(
                    "region_id"
                )

                if str(item_region) == str(region_id):
                    output.append(item)

            return output

        return []

    def get_embedding_result(
        self,
        region_id,
        scenario
    ):

        results = self.loaded_data.get(
            "embedding_results",
            {}
        )

        scenario_id = scenario.get(
            "scenario_id"
        )

        if isinstance(results, list):

            for result in results:

                if not isinstance(result, dict):
                    continue

                if str(
                    result.get("region_id")
                ) != str(region_id):

                    continue

                if scenario_id is not None:

                    if str(
                        result.get("scenario_id")
                    ) == str(scenario_id):

                        return result

                else:

                    return result

        elif isinstance(results, dict):

            region_results = results.get(
                region_id,
                []
            )

            if isinstance(
                region_results,
                dict
            ):

                if scenario_id:

                    return region_results.get(
                        scenario_id,
                        {}
                    )

                return region_results

            if isinstance(
                region_results,
                list
            ):

                for result in region_results:

                    if str(
                        result.get("scenario_id")
                    ) == str(scenario_id):

                        return result

        return {}

    def build_master_dataset(self):

        self.log(
            "\nBuilding Master Dataset...\n"
        )

        region_analysis = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "region_analysis",
                []
            )
        )

        candidate_regions = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "candidate_regions",
                []
            )
        )

        adaptive_regions = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "adaptive_regions",
                []
            )
        )

        entropy_database = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "entropy_database",
                []
            )
        )

        region_database = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "region_combination_database",
                []
            )
        )

        prediction_features = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "prediction_features",
                []
            )
        )

        fitness_weights = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "fitness_weights",
                []
            )
        )

        feature_dataset = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "feature_dataset",
                []
            )
        )

        normalized_dataset = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "normalized_dataset",
                []
            )
        )

        region_statistics = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "region_statistics",
                []
            )
        )

        distortion_database = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "distortion_database",
                []
            )
        )

        chunk_region_candidates = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "chunk_region_candidates",
                {}
            )
        )

        primary_regions = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "primary_regions",
                []
            )
        )

        secondary_regions = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "secondary_regions",
                []
            )
        )

        tertiary_regions = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "tertiary_regions",
                []
            )
        )

        rejected_regions = self.convert_region_list_to_dictionary(
            self.loaded_data.get(
                "rejected_regions",
                []
            )
        )

        self.build_coordinate_map(
            region_analysis
        )

        region_feature_lookup = (
            self.build_region_feature_lookup(
                region_analysis,
                candidate_regions,
                adaptive_regions,
                entropy_database,
                region_database,
                prediction_features,
                fitness_weights,
                feature_dataset,
                normalized_dataset,
                region_statistics,
                distortion_database,
                chunk_region_candidates
            )
        )

        self.master_dataset = []

        for region_id in region_analysis.keys():

            base_row = {}

            base_row["dataset_version"] = "3.0"

            base_row["dataset_source"] = "StegaQEntropy"

            base_row["dataset_row_id"] = (
                len(self.master_dataset) + 1
            )

            base_row["region_id"] = region_id

            if region_id in primary_regions:

                base_row["target_class"] = "PRIMARY"
                base_row["target_class_id"] = 3

            elif region_id in secondary_regions:

                base_row["target_class"] = "SECONDARY"
                base_row["target_class_id"] = 2

            elif region_id in tertiary_regions:

                base_row["target_class"] = "TERTIARY"
                base_row["target_class_id"] = 1

            elif region_id in rejected_regions:

                base_row["target_class"] = "REJECT"
                base_row["target_class_id"] = 0

            else:

                base_row["target_class"] = "UNKNOWN"
                base_row["target_class_id"] = -1

            base_row["candidate_region"] = int(
                region_id in candidate_regions
            )

            base_row["adaptive_region"] = int(
                region_id in adaptive_regions
            )

            region_data = region_analysis.get(
                region_id,
                {}
            )

            row = dict(base_row)

            combined_features = (
                region_feature_lookup.get(
                    region_id,
                    {}
                )
            )

            row.update(
                combined_features
            )

            recommended_payload = combined_features.get(
                "embedding_recommended_payload"
            )

            if (
                recommended_payload is not None
                and not pd.isna(
                    recommended_payload
                )
            ):

                row[
                    "estimated_capacity"
                ] = int(
                    round(
                        float(
                            recommended_payload
                        )
                    )
                )

            neighbors = self.get_spatial_neighbors(
                region_id,
                region_data
            )

            for neighbor_name, neighbor_id in neighbors.items():

                row[
                    f"{neighbor_name}_region_id"
                ] = neighbor_id

            self.add_neighbor_features(
                row,
                neighbors,
                region_feature_lookup
            )

            self.add_neighbor_derived_features(
                row
            )

            scenarios = self.get_embedding_scenarios(
                region_id
            )

            if not scenarios:

                scenarios = [{}]

            for scenario in scenarios:

                scenario_row = dict(row)

                scenario_row["scenario_id"] = scenario.get(
                    "scenario_id",
                    "BASELINE"
                )

                self.add_embedding_scenario_features(
                    scenario_row,
                    scenario
                )

                result = self.get_embedding_result(
                    region_id,
                    scenario
                )

                self.add_embedding_targets(
                    scenario_row,
                    result
                )

                scenario_row[
                    "scenario_available"
                ] = int(bool(scenario))

                scenario_row[
                    "embedding_result_available"
                ] = int(bool(result))

                scenario_row[
                    "feature_count"
                ] = len(scenario_row)

                scenario_row[
                    "dataset_ready"
                ] = True

                scenario_row[
                    "ml_ready"
                ] = True

                self.master_dataset.append(
                    scenario_row
                )

        self.master_dataset = pd.DataFrame(
            self.master_dataset
        )

        self.feature_names = list(
            self.master_dataset.columns
        )

        self.log(
            f"Rows Generated : "
            f"{len(self.master_dataset)}"
        )

        self.log(
            f"Features Found : "
            f"{len(self.feature_names)}"
        )



    def save_dataset(self):

        self.log(

            "\nSaving Dataset...\n"

        )

        folder = (

            self.configuration.dataset_directory

        )

        self.master_dataset.to_csv(

            os.path.join(

                folder,

                "master_dataset.csv"

            ),

            index=False

        )


        self.master_dataset.to_json(

            os.path.join(

                folder,

                "master_dataset.json"

            ),

            orient="records",

            indent=4

        )

        np.save(

            os.path.join(

                folder,

                "master_dataset.npy"

            ),

            self.master_dataset.to_numpy()

        )

        with open(

            os.path.join(

                folder,

                "dataset_statistics.json"

            ),

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                self.dataset_statistics,

                file,

                indent=4

            )

        with open(

            os.path.join(

                folder,

                "dataset_manifest.json"

            ),

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                self.dataset_manifest,

                file,

                indent=4

            )

        self.log(

            "Dataset Saved Successfully."

        )

    def run(self):

        self.log("\n" + "=" * 80)

        self.log("ML DATASET BUILDER")

        self.log("=" * 80)

        self.verify_input_files()

        self.load_input_files()

        self.build_master_dataset()

        self.save_dataset()

        self.log("\n" + "=" * 80)

        self.log("DATASET BUILDER COMPLETED")

        self.log("=" * 80)

        self.log("\nMaster Dataset Shape")

        self.log(

            str(

                self.master_dataset.shape

            )

        )

        self.log("\nDataset Successfully Generated.\n")


if __name__ == "__main__":

    builder = DatasetBuilder()

    builder.run()