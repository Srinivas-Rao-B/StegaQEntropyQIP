import os
import json
import copy
import pywt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


class PostEmbeddingAnalysis:

    def __init__(self):

        self.base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.output_dir = os.path.join(
            self.base_dir,
            "ml_module",
            "output",
            "post_embedding_analysis"
        )

        self.master_dataset_path = os.path.join(
            self.base_dir,
            "ml_module",
            "output",
            "dataset_builder",
            "master_dataset.csv"
        )
        self.experiment_dataset_path = os.path.join(
            self.output_dir,
            "post_embedding_experiments.csv"
        )

        self.post_dataset_path = os.path.join(
            self.output_dir,
            "post_embedding_dataset.csv"
        )

        self.scenario_path = os.path.join(
            self.output_dir,
            "embedding_scenarios.json"
        )

        self.dataset = None
        self.experiment_dataset = None
        self.post_dataset = None

        self.region_map = {}

        self.neighbor_directions = {
            "N1": (-1, -1),
            "N2": (-1, 0),
            "N3": (-1, 1),
            "N4": (0, -1),
            "N5": (0, 1),
            "N6": (1, -1),
            "N7": (1, 0),
            "N8": (1, 1)
        }

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    def prepare_post_embedding_experiment_dataset(
        self
    ):

        if self.dataset is None:

            raise RuntimeError(
                "Master dataset has not been loaded."
            )

        if self.dataset.empty:

            raise ValueError(
                "Master dataset is empty."
            )

        rows = []

        for index, source_row in (
            self.dataset.iterrows()
        ):

            region_id = source_row[
                "region_id"
            ]

            base_row = {
                "region_id":
                    region_id,

                "source_row_index":
                    int(index)
            }

            for column in self.dataset.columns:

                if column == "region_id":
                    continue

                base_row[
                    f"pre_{column}"
                ] = source_row[
                    column
                ]

            # -------------------------------------------------
            # USE THE SAME CAPACITY RESOLUTION AS THE
            # ACTIVE REGION EXPERIMENT PIPELINE
            # -------------------------------------------------

            capacity_value = (
                self.find_region_capacity(
                    source_row
                )
            )

            capacity_value = max(
                1,
                int(
                    round(
                        capacity_value
                    )
                )
            )

            # -------------------------------------------------
            # EXACTLY 1% -> 100%
            # -------------------------------------------------

            for percentage in range(
                1,
                101
            ):

                payload_size = int(
                    round(
                        capacity_value
                        *
                        percentage
                        /
                        100.0
                    )
                )

                payload_size = max(
                    1,
                    min(
                        payload_size,
                        capacity_value
                    )
                )

                scenario_id = (
                    f"region_{region_id}_"
                    f"payload_{percentage:03d}"
                )

                row = base_row.copy()

                row[
                    "scenario_id"
                ] = scenario_id

                row[
                    "payload_percentage"
                ] = float(
                    percentage
                )

                row[
                    "capacity_ratio"
                ] = float(
                    percentage
                    /
                    100.0
                )

                row[
                    "scenario_capacity"
                ] = capacity_value

                row[
                    "scenario_payload_size"
                ] = payload_size

                rows.append(
                    row
                )

        experiment_dataset = pd.DataFrame(
            rows
        )

        if experiment_dataset.empty:

            raise ValueError(
                "Post-embedding experiment dataset is empty."
            )

        experiment_dataset.reset_index(
            drop=True,
            inplace=True
        )

        self.experiment_dataset = (
            experiment_dataset
        )

        experiment_dataset.to_csv(
            self.experiment_dataset_path,
            index=False
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "POST-EMBEDDING EXPERIMENT DATASET PREPARED"
        )

        print(
            "=" * 80
        )

        print(
            f"Rows       : "
            f"{len(experiment_dataset)}"
        )

        print(
            f"Features   : "
            f"{len(experiment_dataset.columns)}"
        )

        print(
            f"Regions    : "
            f"{experiment_dataset['region_id'].nunique()}"
        )

        print(
            f"Scenarios  : "
            f"{experiment_dataset['scenario_id'].nunique()}"
        )

        print(
            f"Saved      : "
            f"{self.experiment_dataset_path}"
        )

        print(
            "=" * 80
        )

        return experiment_dataset

    def initialize_post_embedding_analysis(self):

        self.load_master_dataset()

        self.build_region_map()

        self.load_dwt_coefficients()

        print(
            f"Embedding Subbands : "
            f"{', '.join(self.get_embedding_subbands())}"
        )

        print(
            "LL Subband          : IGNORED"
        )

        print(
            f"DWT Region Shape    : "
            f"{self.dwt_shape}"
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "POST-EMBEDDING ANALYSIS INITIALIZED"
        )

        print(
            "=" * 80
        )

        print(
            f"Rows     : {len(self.dataset)}"
        )

        print(
            f"Features : {len(self.dataset.columns)}"
        )

        print(
            f"Regions  : "
            f"{self.dataset['region_id'].nunique()}"
        )

        print(
            f"Source   : "
            f"{self.master_dataset_path}"
        )

        print(
            "=" * 80
        )

        return self.dataset
    
    def load_master_dataset(self):

        if not os.path.exists(
            self.master_dataset_path
        ):
            raise FileNotFoundError(
                self.master_dataset_path
            )

        self.dataset = pd.read_csv(
            self.master_dataset_path
        )

        if self.dataset.empty:
            raise ValueError(
                "Master dataset is empty."
            )

        if "region_id" not in self.dataset.columns:
            raise ValueError(
                "region_id column not found in master dataset."
            )

        self.dataset.reset_index(
            drop=True,
            inplace=True
        )

        self.pre_embedding_master_dataset = (
            self.dataset.copy()
        )

        print(
            f"Master Dataset Rows     : {len(self.dataset)}"
        )

        print(
            f"Master Dataset Features : {len(self.dataset.columns)}"
        )

        print(
            f"Region Count             : "
            f"{self.dataset['region_id'].nunique()}"
        )

        return self.dataset

    def find_coordinate_columns(self):

        row_candidates = [
            "row_start",
            "row",
            "region_row"
        ]

        column_candidates = [
            "column_start",
            "col_start",
            "column",
            "col",
            "region_column"
        ]

        row_column = next(
            (
                column
                for column in row_candidates
                if column in self.dataset.columns
            ),
            None
        )

        column_column = next(
            (
                column
                for column in column_candidates
                if column in self.dataset.columns
            ),
            None
        )

        if row_column is None:
            raise ValueError(
                "Region row coordinate not found."
            )

        if column_column is None:
            raise ValueError(
                "Region column coordinate not found."
            )

        return row_column, column_column

    def build_region_map(self):

        row_column, column_column = (
            self.find_coordinate_columns()
        )

        valid_rows = (
            self.dataset[
                row_column
            ]
            .dropna()
            .astype(int)
            .unique()
        )

        valid_columns = (
            self.dataset[
                column_column
            ]
            .dropna()
            .astype(int)
            .unique()
        )

        valid_rows = sorted(
            valid_rows
        )

        valid_columns = sorted(
            valid_columns
        )

        self.region_map = {}

        for row_index, row_value in enumerate(
            valid_rows
        ):

            for column_index, column_value in enumerate(
                valid_columns
            ):

                source = self.dataset[
                    (
                        self.dataset[
                            row_column
                        ].astype(float)
                        ==
                        float(row_value)
                    )
                    &
                    (
                        self.dataset[
                            column_column
                        ].astype(float)
                        ==
                        float(column_value)
                    )
                ]

                if source.empty:
                    continue

                region_id = source.iloc[0][
                    "region_id"
                ]

                self.region_map[
                    (
                        row_index,
                        column_index
                    )
                ] = region_id

        self.region_row_positions = valid_rows

        self.region_column_positions = valid_columns

    def get_neighbors(
        self,
        region_id
    ):

        source = self.dataset[
            self.dataset[
                "region_id"
            ] == region_id
        ]

        if source.empty:

            return {
                name: None
                for name in self.neighbor_directions
            }

        row_column, column_column = (
            self.find_coordinate_columns()
        )

        row_value = int(
            source.iloc[0][
                row_column
            ]
        )

        column_value = int(
            source.iloc[0][
                column_column
            ]
        )

        try:

            row_index = (
                self.region_row_positions.index(
                    row_value
                )
            )

            column_index = (
                self.region_column_positions.index(
                    column_value
                )
            )

        except ValueError:

            return {
                name: None
                for name in self.neighbor_directions
            }

        neighbors = {}

        for name, (
            dr,
            dc
        ) in self.neighbor_directions.items():

            neighbor_row_index = (
                row_index + dr
            )

            neighbor_column_index = (
                column_index + dc
            )

            neighbors[
                name
            ] = self.region_map.get(
                (
                    neighbor_row_index,
                    neighbor_column_index
                )
            )

        return neighbors

    def find_region_capacity(self, row):

        capacity_columns = [
            "estimated_capacity",
            "capacity",
            "region_capacity",
            "max_capacity",
            "maximum_capacity",
            "statistics_capacity",
            "adaptive_capacity",
            "max_payload",
            "payload_capacity"
        ]

        for column in capacity_columns:

            if column not in row.index:
                continue

            value = row[column]

            if pd.isna(value):
                continue

            try:
                value = float(value)

            except (ValueError, TypeError):
                continue

            if np.isfinite(value) and value > 0:
                return value

        raise ValueError(
            f"No valid capacity found for region "
            f"{row.get('region_id', 'UNKNOWN')}. "
            f"Expected 'estimated_capacity' or another valid capacity column."
        )


    def get_region_row(self, region_id):

        result = self.dataset[
            self.dataset["region_id"] == region_id
        ]

        if result.empty:
            raise ValueError(
                f"Region {region_id} not found."
            )

        return result.iloc[0]


    def get_normalized_region_features(
        self,
        region_id
    ):

        row = self.get_region_row(
            region_id
        )

        numeric_values = {}

        excluded = {
            "region_id",
            "target_class",
            "target_class_id"
        }

        for column in self.dataset.columns:

            if column in excluded:
                continue

            value = row[column]

            if pd.isna(value):
                continue

            try:

                numeric_values[column] = float(
                    value
                )

            except (ValueError, TypeError):

                continue

        return numeric_values


    def calculate_region_sensitivity(
        self,
        region_id
    ):

        features = (
            self.get_normalized_region_features(
                region_id
            )
        )

        sensitivity_groups = {

            "entropy": [
                key
                for key in features
                if "entropy" in key.lower()
            ],

            "noise": [
                key
                for key in features
                if "noise" in key.lower()
            ],

            "gradient": [
                key
                for key in features
                if "gradient" in key.lower()
            ],

            "variance": [
                key
                for key in features
                if "variance" in key.lower()
            ],

            "correlation": [
                key
                for key in features
                if "correlation" in key.lower()
            ],

            "risk": [
                key
                for key in features
                if "risk" in key.lower()
            ],

            "distortion": [
                key
                for key in features
                if "distortion" in key.lower()
            ]
        }

        group_values = {}

        for group, keys in sensitivity_groups.items():

            if not keys:
                group_values[group] = 0.0
                continue

            values = [
                abs(features[key])
                for key in keys
                if np.isfinite(features[key])
            ]

            if not values:

                group_values[group] = 0.0

            else:

                group_values[group] = float(
                    np.mean(values)
                )

        return group_values


    def calculate_initial_payload_range(
        self,
        region_id
    ):

        row = self.get_region_row(
            region_id
        )

        capacity = (
            self.find_region_capacity(
                row
            )
        )

        return {
            "minimum_payload": 1,
            "maximum_payload": int(
                max(1, capacity)
            ),
            "initial_payload" : int(
                max(
                    1,
                    capacity
                )
            )
        }

    def calculate_feature_changes(
        self,
        pre_features,
        post_features
    ):

        delta = {}
        relative = {}

        all_features = set(
            pre_features.keys()
        ) | set(
            post_features.keys()
        )

        for feature in all_features:

            pre_value = pre_features.get(
                feature
            )

            post_value = post_features.get(
                feature
            )

            try:

                pre_value = float(
                    pre_value
                )

                post_value = float(
                    post_value
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            if not (
                np.isfinite(pre_value)
                and
                np.isfinite(post_value)
            ):
                continue

            difference = (
                post_value - pre_value
            )

            delta[
                feature
            ] = difference

            scale = max(
                abs(pre_value),
                abs(post_value),
                1e-9
            )

            relative[feature] = min(
                1.0,
                abs(difference) / scale
            )

        return delta, relative


    def calculate_normalized_disturbance(
        self,
        pre_features,
        post_features
    ):

        delta, relative = (
            self.calculate_feature_changes(
                pre_features,
                post_features
            )
        )

        disturbance = {}

        for feature, value in delta.items():

            try:

                pre_value = float(
                    pre_features.get(
                        feature,
                        0.0
                    )
                )

            except (
                ValueError,
                TypeError
            ):

                pre_value = 0.0

            scale = max(
                abs(pre_value),
                abs(
                    pre_value + value
                ),
                1e-9
            )

            disturbance[feature] = min(
                1.0,
                abs(value) / scale
            )

        return disturbance, delta, relative


    def calculate_group_disturbance(
        self,
        disturbance
    ):

        groups = {

            "entropy": [
                key for key in disturbance
                if "entropy" in key.lower()
            ],

            "noise": [
                key for key in disturbance
                if "noise" in key.lower()
            ],

            "gradient": [
                key for key in disturbance
                if "gradient" in key.lower()
            ],

            "variance": [
                key for key in disturbance
                if "variance" in key.lower()
            ],

            "energy": [
                key for key in disturbance
                if "energy" in key.lower()
            ],

            "texture": [
                key for key in disturbance
                if "texture" in key.lower()
                or "histogram" in key.lower()
                or "spectral" in key.lower()
            ],

            "correlation": [
                key for key in disturbance
                if "correlation" in key.lower()
            ],

            "coefficient": [
                key for key in disturbance
                if "coefficient" in key.lower()
            ],

            "risk": [
                key for key in disturbance
                if "risk" in key.lower()
                or "detectability" in key.lower()
            ]
        }

        result = {}

        for group, features in groups.items():

            values = [
                disturbance[feature]
                for feature in features
                if np.isfinite(
                    disturbance[feature]
                )
            ]

            if values:

                result[group] = float(
                    np.mean(values)
                )

            else:

                result[group] = 0.0

        return result


    # def calculate_consequence_score(
    #     self,
    #     center_disturbance,
    #     neighbor_disturbance,
    #     global_disturbance
    # ):

    #     center_values = list(
    #         center_disturbance.values()
    #     )

    #     neighbor_values = list(
    #         neighbor_disturbance.values()
    #     )

    #     global_values = list(
    #         global_disturbance.values()
    #     )

    #     center_score = (
    #         float(np.mean(center_values))
    #         if center_values
    #         else 0.0
    #     )

    #     neighbor_score = (
    #         float(np.mean(neighbor_values))
    #         if neighbor_values
    #         else 0.0
    #     )

    #     global_score = (
    #         float(np.mean(global_values))
    #         if global_values
    #         else 0.0
    #     )

    #     consequence = (
    #         0.45 * center_score
    #         +
    #         0.35 * neighbor_score
    #         +
    #         0.20 * global_score
    #     )

    #     consequence = float(
    #         np.clip(
    #             consequence,
    #             0.0,
    #             1.0
    #         )
    #     )

    #     safety_score = (
    #         1.0 - consequence
    #     )

    #     return {
    #         "center_consequence": center_score,
    #         "neighbor_consequence": neighbor_score,
    #         "global_consequence": global_score,
    #         "embedding_consequence_score":
    #             consequence,
    #         "embedding_safety_score":
    #             safety_score
    #     }

    def calculate_payload_gradient(
        self,
        previous_result,
        current_result
    ):

        if (
            previous_result is None
            or
            current_result is None
        ):
            return None

        previous_percentage = float(
            previous_result.get(
                "payload_percentage",
                np.nan
            )
        )

        current_percentage = float(
            current_result.get(
                "payload_percentage",
                np.nan
            )
        )

        previous_consequence = float(
            previous_result.get(
                "total_consequence",
                np.nan
            )
        )

        current_consequence = float(
            current_result.get(
                "total_consequence",
                np.nan
            )
        )

        if not all(
            np.isfinite(value)
            for value in [
                previous_percentage,
                current_percentage,
                previous_consequence,
                current_consequence
            ]
        ):

            return None

        percentage_delta = (
            current_percentage
            -
            previous_percentage
        )

        if abs(
            percentage_delta
        ) < 1e-12:

            return 0.0

        return float(
            (
                current_consequence
                -
                previous_consequence
            )
            /
            percentage_delta
        )

    def calculate_quality_gradient(
        self,
        previous_result,
        current_result
    ):

        if (
            previous_result is None
            or
            current_result is None
        ):
            return None

        previous_percentage = float(
            previous_result.get(
                "payload_percentage",
                np.nan
            )
        )

        current_percentage = float(
            current_result.get(
                "payload_percentage",
                np.nan
            )
        )

        previous_quality = float(
            previous_result.get(
                "total_safety",
                np.nan
            )
        )

        current_quality = float(
            current_result.get(
                "total_safety",
                np.nan
            )
        )

        if not all(
            np.isfinite(value)
            for value in [
                previous_percentage,
                current_percentage,
                previous_quality,
                current_quality
            ]
        ):

            return None

        percentage_delta = (
            current_percentage
            -
            previous_percentage
        )

        if abs(
            percentage_delta
        ) < 1e-12:

            return 0.0

        return float(
            (
                current_quality
                -
                previous_quality
            )
            /
            percentage_delta
        )

    def calculate_safe_capacity(
        self,
        results,
        consequence_threshold=0.30
    ):

        if not results:

            return {
                "safe_capacity": 0,
                "maximum_tested_payload": 0,
                "capacity_status": "NO_DATA"
            }

        valid_results = [

            result
            for result in results

            if (
                "payload_size" in result
                and
                "total_consequence" in result
            )
        ]

        if not valid_results:

            return {
                "safe_capacity": 0,
                "maximum_tested_payload": 0,
                "capacity_status": "NO_VALID_DATA"
            }

        valid_results = sorted(
            valid_results,
            key=lambda item:
                int(
                    item[
                        "payload_size"
                    ]
                )
        )

        safe_results = [

            result
            for result in valid_results

            if float(
                result[
                    "total_consequence"
                ]
            ) <= consequence_threshold
        ]

        if not safe_results:

            return {
                "safe_capacity": 0,

                "maximum_tested_payload":
                    int(
                        valid_results[-1][
                            "payload_size"
                        ]
                    ),

                "capacity_status":
                    "NO_SAFE_PAYLOAD_FOUND"
            }

        best = max(
            safe_results,
            key=lambda result:
                int(
                    result[
                        "payload_size"
                    ]
                )
        )

        return {

            "safe_capacity":
                int(
                    best[
                        "payload_size"
                    ]
                ),

            "maximum_tested_payload":
                int(
                    valid_results[-1][
                        "payload_size"
                    ]
                ),

            "safety_confidence_score":
                float(
                    best.get(
                        "safety_confidence_score",
                        np.nan
                    )
                ),

            "risk_confidence_score":
                float(
                    best.get(
                        "risk_confidence_score",
                        np.nan
                    )
                ),

            "total_consequence":
                float(
                    best[
                        "total_consequence"
                    ]
                ),

            "total_safety":
                float(
                    best[
                        "total_safety"
                    ]
                ),

            "predicted_class":
                best.get(
                    "predicted_class"
                ),

            "capacity_status":
                "SAFE_CAPACITY_FOUND"
        }

    def build_gradient_payload_schedule(
        self,
        capacity,
        step_percent=1
    ):

        capacity = max(
            1,
            int(
                round(
                    capacity
                )
            )
        )

        if step_percent <= 0:
            raise ValueError(
                "step_percent must be greater than zero."
            )

        if step_percent > 100:
            raise ValueError(
                "step_percent cannot exceed 100."
            )

        payloads = []

        for percentage in range(
            step_percent,
            101,
            step_percent
        ):

            payload_size = int(
                round(
                    capacity
                    *
                    percentage
                    /
                    100.0
                )
            )

            payload_size = max(
                1,
                min(
                    payload_size,
                    capacity
                )
            )

            payloads.append({
                "percentage":
                    percentage,

                "payload_size":
                    payload_size
            })

        return payloads
  
    def identify_sensitive_payload_range(
        self,
        results,
        gradient_threshold=0.01
    ):

        if len(results) < 2:

            return []

        ordered = sorted(
            results,
            key=lambda item:
                item.get(
                    "payload_percentage",
                    0.0
                )
        )

        sensitive_ranges = []

        for index in range(
            1,
            len(ordered)
        ):

            previous = ordered[
                index - 1
            ]

            current = ordered[
                index
            ]

            gradient = (
                current.get(
                    "payload_gradient"
                )
            )

            if gradient is None:
                continue

            if abs(
                gradient
            ) >= gradient_threshold:

                sensitive_ranges.append({

                    "lower_percentage":
                        previous[
                            "payload_percentage"
                        ],

                    "upper_percentage":
                        current[
                            "payload_percentage"
                        ],

                    "gradient":
                        gradient
                })

        return sensitive_ranges

    def build_consequence_result(
        self,
        region_id,
        payload_size,
        pre_features,
        post_features,
        neighbor_pre_features=None,
        neighbor_post_features=None
    ):

        if neighbor_pre_features is None:
            neighbor_pre_features = {}

        if neighbor_post_features is None:
            neighbor_post_features = {}

        center_disturbance, center_delta, center_relative = (
            self.calculate_normalized_disturbance(
                pre_features,
                post_features
            )
        )

        center_groups = (
            self.calculate_group_disturbance(
                center_disturbance
            )
        )

        neighbor_results = {}

        for neighbor_name in self.neighbor_directions:

            pre_neighbor = (
                neighbor_pre_features.get(
                    neighbor_name,
                    {}
                )
            )

            post_neighbor = (
                neighbor_post_features.get(
                    neighbor_name,
                    {}
                )
            )

            if not pre_neighbor or not post_neighbor:

                neighbor_results[
                    neighbor_name
                ] = {
                    "disturbance": {},
                    "delta": {},
                    "relative": {},
                    "group_disturbance": {}
                }

                continue

            disturbance, delta, relative = (
                self.calculate_normalized_disturbance(
                    pre_neighbor,
                    post_neighbor
                )
            )

            groups = (
                self.calculate_group_disturbance(
                    disturbance
                )
            )

            neighbor_results[
                neighbor_name
            ] = {

                "disturbance":
                    disturbance,

                "delta":
                    delta,

                "relative":
                    relative,

                "group_disturbance":
                    groups
            }

        neighbor_impacts = []

        for result in neighbor_results.values():

            if not result.get(
                "disturbance"
            ):
                continue

            values = [
                float(value)
                for value in result[
                    "disturbance"
                ].values()
                if np.isfinite(value)
            ]

            if values:

                neighbor_impacts.append(
                    float(
                        np.mean(values)
                    )
                )

        neighbor_mean = (
            float(
                np.mean(
                    neighbor_impacts
                )
            )
            if neighbor_impacts
            else 0.0
        )

        neighbor_max = (
            float(
                np.max(
                    neighbor_impacts
                )
            )
            if neighbor_impacts
            else 0.0
        )

        neighbor_std = (
            float(
                np.std(
                    neighbor_impacts
                )
            )
            if neighbor_impacts
            else 0.0
        )

        neighbor_consequence = {

            "mean":
                neighbor_mean,

            "maximum":
                neighbor_max,

            "standard_deviation":
                neighbor_std,

            "total":
                float(
                    np.sum(
                        neighbor_impacts
                    )
                )
        }

        global_disturbance = (
            self.calculate_group_disturbance(
                {
                    **center_disturbance,
                    **{
                        f"{name}_{feature}":
                            value

                        for name, result
                        in neighbor_results.items()

                        for feature, value
                        in result[
                            "disturbance"
                        ].items()
                    }
                }
            )
        )

        consequence = (
            self.calculate_consequence_score(
                center_groups,
                {
                    "neighbor_mean":
                        neighbor_mean,

                    "neighbor_max":
                        neighbor_max,

                    "neighbor_std":
                        neighbor_std,

                    "neighbor_total":
                        min(
                            1.0,
                            neighbor_consequence[
                                "total"
                            ]
                        )
                },
                global_disturbance
            )
        )

        result = {

            "region_id":
                region_id,

            "payload_size":
                payload_size,

            "center_feature_count":
                len(pre_features),

            "center_changed_feature_count":
                len(center_delta),

            "center_consequence":
                consequence[
                    "center_consequence"
                ],

            "neighbor_consequence":
                consequence[
                    "neighbor_consequence"
                ],

            "global_consequence":
                consequence[
                    "global_consequence"
                ],

            "total_consequence":
                consequence,

            "neighbor_mean_impact":
                neighbor_mean,

            "neighbor_max_impact":
                neighbor_max,

            "neighbor_std_impact":
                neighbor_std,

            "neighbor_total_impact":
                neighbor_consequence[
                    "total"
                ],

            "center_delta_entropy":
                center_delta.get(
                    "entropy",
                    0.0
                ),

            "center_delta_noise":
                self._sum_matching_deltas(
                    center_delta,
                    "noise"
                ),

            "center_delta_gradient":
                self._sum_matching_deltas(
                    center_delta,
                    "gradient"
                ),

            "center_delta_variance":
                self._sum_matching_deltas(
                    center_delta,
                    "variance"
                ),

            "center_delta_energy":
                self._sum_matching_deltas(
                    center_delta,
                    "energy"
                ),

            "center_group_disturbance":
                center_groups,

            "neighbor_results":
                neighbor_results
        }

        return result


    def _sum_matching_deltas(
        self,
        delta,
        keyword
    ):

        values = []

        for feature, value in delta.items():

            if keyword.lower() in feature.lower():

                try:

                    value = float(value)

                    if np.isfinite(value):

                        values.append(
                            abs(value)
                        )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    def calculate_probability_from_consequence(
        self,
        consequence_score,
        threshold=0.30,
        sharpness=12.0
    ):

        consequence_score = float(
            np.clip(
                consequence_score,
                0.0,
                1.0
            )
        )

        safety_score = float(
            1.0 - consequence_score
        )

        probability_score = float(
            1.0
            /
            (
                1.0
                +
                np.exp(
                    -sharpness
                    *
                    (
                        safety_score
                        -
                        (1.0 - threshold)
                    )
                )
            )
        )

        probability_score = float(
            np.clip(
                probability_score,
                0.0,
                1.0
            )
        )

        return {
            "safety_confidence_score":
                probability_score,

            "risk_confidence_score":
                float(
                    1.0 - probability_score
                )
        }

    def classify_consequence(
        self,
        safety_score,
        primary_threshold,
        secondary_threshold,
        tertiary_threshold
    ):

        safety_score = float(
            np.clip(
                safety_score,
                0.0,
                1.0
            )
        )

        if safety_score >= primary_threshold:
            return "PRIMARY"

        elif safety_score >= secondary_threshold:
            return "SECONDARY"

        elif safety_score >= tertiary_threshold:
            return "TERTIARY"

        return "REJECT"


    def finalize_consequence_result(
        self,
        result
    ):

        probabilities = (
            self.calculate_probability_from_consequence(
                result[
                   "total_consequence"
                ]
            )
        )

        result.update(
            probabilities
        )

        return result

    def load_region_pre_features(
        self,
        region_id
    ):

        row = self.get_region_row(
            region_id
        )

        pre_features = {}

        for column in self.dataset.columns:

            if column.startswith(
                "target_"
            ):
                continue

            if column in {
                "region_id",
                "target_class",
                "target_class_id"
            }:
                continue

            value = row[column]

            try:

                value = float(value)

            except (
                ValueError,
                TypeError
            ):

                continue

            if np.isfinite(value):

                pre_features[
                    column
                ] = value

        return pre_features


    def load_neighbor_pre_features(
        self,
        region_id
    ):

        neighbors = self.get_neighbors(
            region_id
        )

        neighbor_features = {}

        for neighbor_name, neighbor_id in (
            neighbors.items()
        ):

            if neighbor_id is None:

                neighbor_features[
                    neighbor_name
                ] = {}

                continue

            neighbor_features[
                neighbor_name
            ] = self.load_region_pre_features(
                neighbor_id
            )

        return neighbor_features

    def prepare_region_experiment(
        self,
        region_id
    ):

        row = self.get_region_row(
            region_id
        )

        selected_band = str(
            row["band"]
        ).upper()

        if selected_band not in {
            "LH",
            "HL",
            "HH"
        }:
            raise ValueError(
                f"Invalid embedding band "
                f"{selected_band} for {region_id}"
            )

        capacity = self.find_region_capacity(
            row
        )

        capacity = max(
            1,
            int(
                round(
                    capacity
                )
            )
        )

        payload_schedule = []

        for percentage in range(
            1,
            101
        ):

            payload_size = int(
                round(
                    capacity
                    *
                    percentage
                    /
                    100.0
                )
            )

            payload_size = max(
                1,
                min(
                    payload_size,
                    capacity
                )
            )

            payload_schedule.append({

                "percentage":
                    percentage,

                "payload_percentage":
                    percentage,

                "payload_size":
                    payload_size,

                "capacity":
                    capacity,

                "capacity_ratio":
                    percentage / 100.0
            })

        neighbors = self.get_neighbors(
            region_id
        )

        pre_features = (
            self.load_region_pre_features(
                region_id
            )
        )

        neighbor_pre_features = (
            self.load_neighbor_pre_features(
                region_id
            )
        )

        return {

            "region_id":
                region_id,

            "band": selected_band,

            "capacity":
                capacity,

            "payload_schedule":
                payload_schedule,

            "neighbors":
                neighbors,

            "pre_features":
                pre_features,

            "neighbor_pre_features":
                neighbor_pre_features,

            "experiments":
                []
        }

    def build_all_region_experiments(self):

        experiments = []

        region_ids = (
            self.dataset[
                "region_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        for region_id in region_ids:

            experiment = (
                self.prepare_region_experiment(
                    region_id
                )
            )

            experiments.append(
                experiment
            )

        experiment_path = os.path.join(
            self.output_dir,
            "region_experiment_plan.json"
        )

        with open(
            experiment_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                experiments,
                file,
                indent=4,
                default=str
            )

        print(
            f"Region Experiment Plans : "
            f"{len(experiments)}"
        )

        print(
            f"Experiment Plan Saved : "
            f"{experiment_path}"
        )

        return experiments

    def calculate_region_quality_metrics(
        self,
        pre_array,
        post_array
    ):

        if pre_array is None:

            raise ValueError(
                "pre_array is required."
            )

        if post_array is None:

            raise ValueError(
                "post_array is required."
            )

        pre_array = np.asarray(
            pre_array,
            dtype=np.float64
        )

        post_array = np.asarray(
            post_array,
            dtype=np.float64
        )

        if pre_array.shape != post_array.shape:

            raise ValueError(
                "Pre and post arrays must have "
                "the same shape."
            )

        if pre_array.size == 0:

            raise ValueError(
                "Quality metric arrays are empty."
            )

        if not (
            np.all(
                np.isfinite(
                    pre_array
                )
            )
            and
            np.all(
                np.isfinite(
                    post_array
                )
            )
        ):

            raise ValueError(
                "Pre/post arrays contain "
                "NaN or infinite values."
            )

        difference = (
            post_array
            -
            pre_array
        )

        absolute_difference = np.abs(
            difference
        )

        squared_difference = (
            difference ** 2
        )

        mse = float(
            np.mean(
                squared_difference
            )
        )

        mae = float(
            np.mean(
                absolute_difference
            )
        )

        maximum_change = float(
            np.max(
                absolute_difference
            )
        )

        changed_mask = (
            absolute_difference
            >
            1e-12
        )

        changed_count = int(
            np.count_nonzero(
                changed_mask
            )
        )

        total_count = int(
            pre_array.size
        )

        changed_density = float(
            changed_count
            /
            total_count
        )

        pre_mean = float(
            np.mean(
                pre_array
            )
        )

        post_mean = float(
            np.mean(
                post_array
            )
        )

        pre_variance = float(
            np.var(
                pre_array
            )
        )

        post_variance = float(
            np.var(
                post_array
            )
        )

        mean_change = float(
            abs(
                post_mean
                -
                pre_mean
            )
        )

        variance_change = float(
            abs(
                post_variance
                -
                pre_variance
            )
        )

        pre_std = float(
            np.std(
                pre_array
            )
        )

        post_std = float(
            np.std(
                post_array
            )
        )

        # -------------------------------------------------
        # CORRELATION
        # -------------------------------------------------

        denominator = (
            pre_std
            *
            post_std
        )

        if denominator <= 1e-12:

            correlation = 1.0

        else:

            covariance = float(
                np.mean(
                    (
                        pre_array
                        -
                        pre_mean
                    )
                    *
                    (
                        post_array
                        -
                        post_mean
                    )
                )
            )

            correlation = float(
                np.clip(
                    covariance
                    /
                    denominator,
                    -1.0,
                    1.0
                )
            )

        # -------------------------------------------------
        # NORMALIZED MSE
        # -------------------------------------------------

        data_range = float(
            np.max(
                pre_array
            )
            -
            np.min(
                pre_array
            )
        )

        if data_range <= 1e-12:

            data_range = 1.0

        normalized_mse = float(
            mse
            /
            (
                data_range ** 2
                +
                1e-12
            )
        )

        normalized_mse = float(
            np.clip(
                normalized_mse,
                0.0,
                1.0
            )
        )

        # -------------------------------------------------
        # PSNR
        # -------------------------------------------------

        if mse <= 1e-12:

            psnr = float(
                "inf"
            )

        else:

            psnr = float(
                10.0
                *
                np.log10(
                    (
                        data_range ** 2
                    )
                    /
                    mse
                )
            )

        # -------------------------------------------------
        # REAL SSIM
        # -------------------------------------------------

        # SSIM expects image-like 2D arrays.
        # If the input is RGB/multi-channel, convert
        # to grayscale for a stable single SSIM value.

        if pre_array.ndim == 3:

            pre_ssim_array = np.mean(
                pre_array,
                axis=2
            )

            post_ssim_array = np.mean(
                post_array,
                axis=2
            )

        elif pre_array.ndim == 2:

            pre_ssim_array = pre_array
            post_ssim_array = post_array

        else:

            raise ValueError(
                "SSIM requires 2D grayscale or "
                "3D multi-channel image arrays."
            )

        ssim_data_range = float(
            max(
                np.max(
                    pre_ssim_array
                ),
                np.max(
                    post_ssim_array
                )
            )
            -
            min(
                np.min(
                    pre_ssim_array
                ),
                np.min(
                    post_ssim_array
                )
            )
        )

        if ssim_data_range <= 1e-12:

            ssim_data_range = 1.0

        # Small regions can be smaller than the default
        # SSIM window size of 7.
        minimum_dimension = min(
            pre_ssim_array.shape
        )

        if minimum_dimension < 3:

            # SSIM cannot be computed reliably on a
            # 1xN / Nx1 mapped region.
            # Keep the region and all other metrics.
            ssim_score = 1.0

        else:

            if minimum_dimension < 7:

                win_size = (
                    minimum_dimension
                    if minimum_dimension % 2 == 1
                    else minimum_dimension - 1
                )

            else:

                win_size = 7

            ssim_score = float(
                structural_similarity(
                    pre_ssim_array,
                    post_ssim_array,
                    data_range=ssim_data_range,
                    win_size=win_size
                )
            )

        ssim_score = float(
            np.clip(
                ssim_score,
                -1.0,
                1.0
            )
        )

        # -------------------------------------------------
        # RETURN ALL QUALITY METRICS
        # -------------------------------------------------

        return {

            "mse":
                mse,

            "mae":
                mae,

            "psnr":
                psnr,

            "correlation":
                correlation,

            "ssim":
                ssim_score,

            "mean_change":
                mean_change,

            "variance_change":
                variance_change,

            "maximum_change":
                maximum_change,

            "changed_count":
                changed_count,

            "total_count":
                total_count,

            "changed_density":
                changed_density,

            "normalized_mse":
                normalized_mse
        }

    def calculate_neighbor_quality_metrics(
        self,
        pre_arrays,
        post_arrays
    ):

        results = {}

        for neighbor_name in (
            self.neighbor_directions
        ):

            pre_array = (
                pre_arrays.get(
                    neighbor_name
                )
            )

            post_array = (
                post_arrays.get(
                    neighbor_name
                )
            )

            if (
                pre_array is None
                or
                post_array is None
            ):

                results[
                    neighbor_name
                ] = {

                    "available":
                        False,

                    "mse":
                        np.nan,

                    "psnr":
                        np.nan,

                    "ssim":
                        np.nan,

                    "impact_score":
                        np.nan
                }

                continue

            metrics = (
                self.calculate_region_quality_metrics(
                    pre_array,
                    post_array
                )
            )

            impact = float(
                np.clip(
                    metrics[
                        "normalized_mse"
                    ]
                    +
                    abs(
                        metrics[
                            "variance_change"
                        ]
                    )
                    /
                    (
                        abs(
                            np.var(
                                pre_array
                            )
                        )
                        +
                        1e-9
                    ),
                    0.0,
                    1.0
                )
            )

            results[
                neighbor_name
            ] = {

                "available":
                    True,

                **metrics,

                "impact_score":
                    impact
            }

        return results

    def extract_post_features(
        self,
        post_region_data
    ):

        if not isinstance(
            post_region_data,
            dict
        ):
            raise ValueError(
                "post_region_data must be a dictionary."
            )

        post_features = {}

        for key, value in (
            post_region_data.items()
        ):

            if isinstance(
                value,
                (int, float, np.integer, np.floating)
            ):

                value = float(value)

                if np.isfinite(value):

                    post_features[
                        key
                    ] = value

        return post_features

    def calculate_center_post_metrics(
        self,
        pre_features,
        post_features
    ):

        disturbance, delta, relative = (
            self.calculate_normalized_disturbance(
                pre_features,
                post_features
            )
        )

        groups = (
            self.calculate_group_disturbance(
                disturbance
            )
        )

        group_values = np.asarray(
            [
                float(value)
                for value in groups.values()
                if np.isfinite(value)
            ],
            dtype=np.float64
        )

        if group_values.size:

            center_mean = float(
                np.mean(group_values)
            )

            center_median = float(
                np.median(group_values)
            )

            center_variance = float(
                np.var(group_values)
            )

            center_std = float(
                np.std(group_values)
            )

            center_min = float(
                np.min(group_values)
            )

            center_max = float(
                np.max(group_values)
            )

        else:

            center_mean = 0.0
            center_median = 0.0
            center_variance = 0.0
            center_std = 0.0
            center_min = 0.0
            center_max = 0.0

        return {

            "delta_features":
                delta,

            "relative_changes":
                relative,

            "disturbance":
                disturbance,

            "group_disturbance":
                groups,

            "center_mean_disturbance":
                center_mean,

            "center_median_disturbance":
                center_median,

            "center_variance_disturbance":
                center_variance,

            "center_std_disturbance":
                center_std,

            "center_min_disturbance":
                center_min,

            "center_max_disturbance":
                center_max,

            "center_consequence":
                center_mean,

            "center_safety":
                1.0 - center_mean
        }
    
    def calculate_neighbor_feature_changes(
        self,
        neighbor_pre_features,
        neighbor_post_features
    ):

        neighbor_changes = {}

        for neighbor_name in self.neighbor_directions:

            pre_features = neighbor_pre_features.get(
                neighbor_name,
                {}
            )

            post_features = neighbor_post_features.get(
                neighbor_name,
                {}
            )

            if not pre_features or not post_features:

                neighbor_changes[neighbor_name] = {
                    "available": False,
                    "delta": {},
                    "relative": {},
                    "disturbance": {},

                    "raw_mean": np.nan,
                    "raw_median": np.nan,
                    "raw_variance": np.nan,
                    "raw_std": np.nan,
                    "raw_min": np.nan,
                    "raw_max": np.nan,

                    "relative_mean": np.nan,
                    "relative_median": np.nan,
                    "relative_variance": np.nan,
                    "relative_std": np.nan,
                    "relative_min": np.nan,
                    "relative_max": np.nan,

                    "changed_count": 0,
                    "feature_count": 0,
                    "changed_ratio": np.nan,

                    "correlation": np.nan
                }

                continue

            disturbance, delta, relative = (
                self.calculate_normalized_disturbance(
                    pre_features,
                    post_features
                )
            )

            raw_values = np.asarray(
                [
                    float(value)
                    for value in delta.values()
                    if np.isfinite(value)
                ],
                dtype=np.float64
            )

            relative_values = np.asarray(
                [
                    float(value)
                    for value in relative.values()
                    if np.isfinite(value)
                ],
                dtype=np.float64
            )

            if raw_values.size:

                raw_mean = float(
                    np.mean(raw_values)
                )

                raw_median = float(
                    np.median(raw_values)
                )

                raw_variance = float(
                    np.var(raw_values)
                )

                raw_std = float(
                    np.std(raw_values)
                )

                raw_min = float(
                    np.min(raw_values)
                )

                raw_max = float(
                    np.max(raw_values)
                )

            else:

                raw_mean = 0.0
                raw_median = 0.0
                raw_variance = 0.0
                raw_std = 0.0
                raw_min = 0.0
                raw_max = 0.0

            if relative_values.size:

                relative_mean = float(
                    np.mean(relative_values)
                )

                relative_median = float(
                    np.median(relative_values)
                )

                relative_variance = float(
                    np.var(relative_values)
                )

                relative_std = float(
                    np.std(relative_values)
                )

                relative_min = float(
                    np.min(relative_values)
                )

                relative_max = float(
                    np.max(relative_values)
                )

            else:

                relative_mean = 0.0
                relative_median = 0.0
                relative_variance = 0.0
                relative_std = 0.0
                relative_min = 0.0
                relative_max = 0.0

            changed_mask = (
                np.abs(relative_values)
                >
                1e-12
            )

            changed_count = int(
                np.count_nonzero(
                    changed_mask
                )
            )

            feature_count = int(
                relative_values.size
            )

            changed_ratio = (
                float(
                    changed_count
                    /
                    feature_count
                )
                if feature_count > 0
                else 0.0
            )

            common_features = (
                set(pre_features.keys())
                &
                set(post_features.keys())
            )

            pre_values = []
            post_values = []

            for feature in common_features:

                try:

                    pre_value = float(
                        pre_features[feature]
                    )

                    post_value = float(
                        post_features[feature]
                    )

                    if (
                        np.isfinite(pre_value)
                        and
                        np.isfinite(post_value)
                    ):

                        pre_values.append(
                            pre_value
                        )

                        post_values.append(
                            post_value
                        )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

            if (
                len(pre_values) >= 2
                and
                len(post_values) >= 2
                and
                np.std(pre_values) > 1e-12
                and
                np.std(post_values) > 1e-12
            ):

                correlation = float(
                    np.corrcoef(
                        pre_values,
                        post_values
                    )[0, 1]
                )

                if not np.isfinite(
                    correlation
                ):

                    correlation = np.nan

            else:

                correlation = np.nan

            neighbor_changes[neighbor_name] = {

                "available": True,

                "delta":
                    delta,

                "relative":
                    relative,

                "disturbance":
                    disturbance,

                "raw_mean":
                    raw_mean,

                "raw_median":
                    raw_median,

                "raw_variance":
                    raw_variance,

                "raw_std":
                    raw_std,

                "raw_min":
                    raw_min,

                "raw_max":
                    raw_max,

                "relative_mean":
                    relative_mean,

                "relative_median":
                    relative_median,

                "relative_variance":
                    relative_variance,

                "relative_std":
                    relative_std,

                "relative_min":
                    relative_min,

                "relative_max":
                    relative_max,

                "changed_count":
                    changed_count,

                "feature_count":
                    feature_count,

                "changed_ratio":
                    changed_ratio,

                "correlation":
                    correlation
            }

        return neighbor_changes

    def aggregate_neighbor_impact(
        self,
        neighbor_changes
    ):

        available = [
            result
            for result in neighbor_changes.values()
            if result.get("available", False)
        ]

        if not available:

            return {

                "mean":
                    0.0,

                "median":
                    0.0,

                "maximum":
                    0.0,

                "minimum":
                    0.0,

                "variance":
                    0.0,

                "standard_deviation":
                    0.0,

                "total":
                    0.0,

                "changed_ratio_mean":
                    0.0,

                "changed_ratio_maximum":
                    0.0,

                "correlation_mean":
                    np.nan,

                "raw_mean":
                    0.0,

                "raw_median":
                    0.0,

                "raw_variance":
                    0.0,

                "raw_std":
                    0.0,

                "raw_min":
                    0.0,

                "raw_max":
                    0.0,

                "relative_mean":
                    0.0,

                "relative_median":
                    0.0,

                "relative_variance":
                    0.0,

                "relative_std":
                    0.0,

                "relative_min":
                    0.0,

                "relative_max":
                    0.0
            }

        def collect(key):

            values = []

            for result in available:

                value = result.get(key)

                try:

                    value = float(value)

                except (
                    ValueError,
                    TypeError
                ):

                    continue

                if np.isfinite(value):

                    values.append(value)

            return np.asarray(
                values,
                dtype=np.float64
            )

        relative_means = collect(
            "relative_mean"
        )

        relative_medians = collect(
            "relative_median"
        )

        relative_variances = collect(
            "relative_variance"
        )

        relative_stds = collect(
            "relative_std"
        )

        relative_mins = collect(
            "relative_min"
        )

        relative_maxs = collect(
            "relative_max"
        )

        raw_means = collect(
            "raw_mean"
        )

        raw_medians = collect(
            "raw_median"
        )

        raw_variances = collect(
            "raw_variance"
        )

        raw_stds = collect(
            "raw_std"
        )

        raw_mins = collect(
            "raw_min"
        )

        raw_maxs = collect(
            "raw_max"
        )

        changed_ratios = collect(
            "changed_ratio"
        )

        correlations = collect(
            "correlation"
        )

        relative_sequences = [
            list(
                result.get(
                    "relative",
                    {}
                ).values()
            )
            for result in available
            if result.get(
                "relative",
                {}
            )
        ]

        if relative_sequences:

            all_relative_values = np.concatenate(
                relative_sequences
            )

        else:

            all_relative_values = np.asarray(
                [],
                dtype=np.float64
            )
        all_relative_values = np.asarray(
            [
                float(value)
                for value in all_relative_values
                if np.isfinite(value)
            ],
            dtype=np.float64
        )

        if all_relative_values.size:

            mean_value = float(
                np.mean(
                    all_relative_values
                )
            )

            median_value = float(
                np.median(
                    all_relative_values
                )
            )

            variance_value = float(
                np.var(
                    all_relative_values
                )
            )

            std_value = float(
                np.std(
                    all_relative_values
                )
            )

            minimum_value = float(
                np.min(
                    all_relative_values
                )
            )

            maximum_value = float(
                np.max(
                    all_relative_values
                )
            )

            total_value = float(
                np.sum(
                    all_relative_values
                )
            )

        else:

            mean_value = 0.0
            median_value = 0.0
            variance_value = 0.0
            std_value = 0.0
            minimum_value = 0.0
            maximum_value = 0.0
            total_value = 0.0

        return {

            "mean":
                mean_value,

            "median":
                median_value,

            "maximum":
                maximum_value,

            "minimum":
                minimum_value,

            "variance":
                variance_value,

            "standard_deviation":
                std_value,

            "total":
                total_value,

            "changed_ratio_mean":
                float(
                    np.mean(
                        changed_ratios
                    )
                )
                if changed_ratios.size
                else 0.0,

            "changed_ratio_maximum":
                float(
                    np.max(
                        changed_ratios
                    )
                )
                if changed_ratios.size
                else 0.0,

            "correlation_mean":
                float(
                    np.mean(
                        correlations
                    )
                )
                if correlations.size
                else np.nan,

            "raw_mean":
                float(
                    np.mean(
                        raw_means
                    )
                )
                if raw_means.size
                else 0.0,

            "raw_median":
                float(
                    np.mean(
                        raw_medians
                    )
                )
                if raw_medians.size
                else 0.0,

            "raw_variance":
                float(
                    np.mean(
                        raw_variances
                    )
                )
                if raw_variances.size
                else 0.0,

            "raw_std":
                float(
                    np.mean(
                        raw_stds
                    )
                )
                if raw_stds.size
                else 0.0,

            "raw_min":
                float(
                    np.min(
                        raw_mins
                    )
                )
                if raw_mins.size
                else 0.0,

            "raw_max":
                float(
                    np.max(
                        raw_maxs
                    )
                )
                if raw_maxs.size
                else 0.0,

            "relative_mean":
                float(
                    np.mean(
                        relative_means
                    )
                )
                if relative_means.size
                else 0.0,

            "relative_median":
                float(
                    np.mean(
                        relative_medians
                    )
                )
                if relative_medians.size
                else 0.0,

            "relative_variance":
                float(
                    np.mean(
                        relative_variances
                    )
                )
                if relative_variances.size
                else 0.0,

            "relative_std":
                float(
                    np.mean(
                        relative_stds
                    )
                )
                if relative_stds.size
                else 0.0,

            "relative_min":
                float(
                    np.min(
                        relative_mins
                    )
                )
                if relative_mins.size
                else 0.0,

            "relative_max":
                float(
                    np.max(
                        relative_maxs
                    )
                )
                if relative_maxs.size
                else 0.0
        }
     
    def calculate_spatial_neighbor_impact(
            self,
            pre_image,
            post_image,
            region_id
        ):
            neighbors = self.get_neighbors(region_id)

            pre_image = np.asarray(
                pre_image,
                dtype=np.float64
            )

            post_image = np.asarray(
                post_image,
                dtype=np.float64
            )

            height = min(
                pre_image.shape[0],
                post_image.shape[0]
            )

            width = min(
                pre_image.shape[1],
                post_image.shape[1]
            )

            pre_image = pre_image[:height, :width]
            post_image = post_image[:height, :width]

            dwt_height = float(
                self.dwt_shape[0]
            )

            dwt_width = float(
                self.dwt_shape[1]
            )

            image_height = float(height)
            image_width = float(width)

            results = {}

            for direction, neighbor_id in neighbors.items():

                if neighbor_id is None:

                    results[direction] = {
                        "available": False,
                        "neighbor_id": None,
                        "mse": np.nan,
                        "mae": np.nan,
                        "variance_change": np.nan,
                        "maximum_pixel_change": np.nan,
                        "changed_pixel_count": np.nan,
                        "changed_pixel_density": np.nan,
                        "correlation": np.nan,
                        "normalized_mse": np.nan
                    }

                    continue

                bounds = self.get_dwt_region_bounds(
                    neighbor_id
                )

                # Use round() instead of floor/ceil combination to prevent window overlapping
                row_start = int(
                    round(
                        bounds["row_start"]
                        * image_height
                        / dwt_height
                    )
                )

                row_end = int(
                    round(
                        bounds["row_end"]
                        * image_height
                        / dwt_height
                    )
                )

                column_start = int(
                    round(
                        bounds["column_start"]
                        * image_width
                        / dwt_width
                    )
                )

                column_end = int(
                    round(
                        bounds["column_end"]
                        * image_width
                        / dwt_width
                    )
                )

                row_start = max(
                    0,
                    min(row_start, height - 1)
                )

                row_end = max(
                    row_start + 1,
                    min(row_end, height)
                )

                column_start = max(
                    0,
                    min(column_start, width - 1)
                )

                column_end = max(
                    column_start + 1,
                    min(column_end, width)
                )

                pre_region = pre_image[
                    row_start:row_end,
                    column_start:column_end
                ]

                post_region = post_image[
                    row_start:row_end,
                    column_start:column_end
                ]

                difference = (
                    post_region
                    -
                    pre_region
                )

                absolute_difference = np.abs(
                    difference
                )

                mse = float(
                    np.mean(
                        difference ** 2
                    )
                )

                mae = float(
                    np.mean(
                        absolute_difference
                    )
                )

                variance_change = float(
                    abs(
                        np.var(post_region)
                        -
                        np.var(pre_region)
                    )
                )

                maximum_change = float(
                    np.max(
                        absolute_difference
                    )
                )

                changed_count = int(
                    np.count_nonzero(
                        absolute_difference > 1e-9
                    )
                )

                total_count = int(
                    difference.size
                )

                changed_density = float(
                    changed_count / total_count
                ) if total_count else 0.0

                pre_flat = np.asarray(
                    pre_region,
                    dtype=np.float64
                ).ravel()

                post_flat = np.asarray(
                    post_region,
                    dtype=np.float64
                ).ravel()

                pre_std = float(
                    np.std(pre_flat)
                )

                post_std = float(
                    np.std(post_flat)
                )

                if (
                    pre_flat.size < 2
                    or
                    post_flat.size < 2
                    or
                    pre_std <= 1e-12
                    or
                    post_std <= 1e-12
                ):

                    correlation = 1.0

                else:

                    with np.errstate(
                        divide="ignore",
                        invalid="ignore"
                    ):

                        correlation_matrix = np.corrcoef(
                            pre_flat,
                            post_flat
                        )

                        correlation = float(
                            correlation_matrix[0, 1]
                        )

                    if not np.isfinite(
                        correlation
                    ):

                        correlation = 1.0

                data_range = float(
                    np.max(pre_region)
                    -
                    np.min(pre_region)
                )

                data_range = max(
                    data_range,
                    1.0
                )

                normalized_mse = float(
                    mse
                    /
                    (
                        data_range ** 2
                        +
                        1e-12
                    )
                )

                impact_score = float(
                    np.clip(
                        0.45 * normalized_mse
                        +
                        0.35 * changed_density
                        +
                        0.20 * (
                            1.0
                            -
                            abs(correlation)
                        ),
                        0.0,
                        1.0
                    )
                )

                results[direction] = {
                    "available": True,
                    "neighbor_id": neighbor_id,
                    "row_start": row_start,
                    "row_end": row_end,
                    "column_start": column_start,
                    "column_end": column_end,
                    "mse": mse,
                    "mae": mae,
                    "variance_change": variance_change,
                    "maximum_pixel_change": maximum_change,
                    "changed_pixel_count": changed_count,
                    "changed_pixel_density": changed_density,
                    "correlation": correlation,
                    "normalized_mse": normalized_mse,
                    "impact_score": impact_score
                }

                # print(
                #     f"Spatial Neighbor | "
                #     f"{region_id} -> {direction} "
                #     f"({neighbor_id}) | "
                #     f"MSE={mse:.6e} | "
                #     f"MAE={mae:.6e} | "
                #     f"Density={changed_density:.6f} | "
                #     f"Correlation={correlation:.8f} | "
                #     f"Impact={impact_score:.6f}"
                # )

            return results

    def calculate_total_consequence(
        self,
        center_consequence,
        neighbor_impact,
        quality_metrics,
        spatial_neighbor_impact=0.0
    ):

        if not isinstance(
            quality_metrics,
            dict
        ):

            raise ValueError(
                "quality_metrics must be a dictionary."
            )

        center = float(
            center_consequence
        )

        neighbor = float(
            neighbor_impact
        )

        spatial_neighbor = float(
            spatial_neighbor_impact
        )

        normalized_mse = quality_metrics.get(
            "normalized_mse"
        )

        ssim_score = quality_metrics.get(
            "ssim"
        )

        if normalized_mse is None:
            raise ValueError(
                "normalized_mse is missing."
            )

        if ssim_score is None:
            raise ValueError(
                "SSIM is missing."
            )

        normalized_mse = float(
            normalized_mse
        )

        ssim_score = float(
            ssim_score
        )

        values = [
            center,
            neighbor,
            spatial_neighbor,
            normalized_mse,
            ssim_score
        ]

        if not all(
            np.isfinite(value)
            for value in values
        ):

            raise ValueError(
                "Consequence inputs contain invalid values."
            )

        center = float(
            np.clip(
                center,
                0.0,
                1.0
            )
        )

        neighbor = float(
            np.clip(
                neighbor,
                0.0,
                1.0
            )
        )

        spatial_neighbor = float(
            np.clip(
                spatial_neighbor,
                0.0,
                1.0
            )
        )

        normalized_mse = float(
            np.clip(
                normalized_mse,
                0.0,
                1.0
            )
        )

        ssim_score = float(
            np.clip(
                ssim_score,
                -1.0,
                1.0
            )
        )

        structural_loss = float(
            np.clip(
                1.0 - ssim_score,
                0.0,
                1.0
            )
        )

        total_consequence = float(
            np.clip(
                0.40 * center
                +
                0.25 * neighbor
                +
                0.20 * spatial_neighbor
                +
                0.10 * normalized_mse
                +
                0.05 * structural_loss,
                0.0,
                1.0
            )
        )

        total_safety = float(
            1.0 - total_consequence
        )

        return {

            "total_consequence":
                total_consequence,

            "total_safety":
                total_safety,

            "center_consequence":
                center,

            "neighbor_feature_consequence":
                neighbor,

            "spatial_neighbor_consequence":
                spatial_neighbor,

            "normalized_mse":
                normalized_mse,

            "ssim":
                ssim_score,

            "structural_loss":
                structural_loss,

            "center_weight":
                0.40,

            "neighbor_feature_weight":
                0.25,

            "spatial_neighbor_weight":
                0.20,

            "normalized_mse_weight":
                0.10,

            "structural_loss_weight":
                0.05
        }

    def calculate_embedding_scenario_metrics(
        self,
        region_id,
        payload_size,
        pre_features,
        post_features,
        neighbor_pre_features,
        neighbor_post_features,
        payload_percentage=None,
        pre_array=None,
        post_array=None,
        neighbor_pre_arrays=None,
        neighbor_post_arrays=None,
        spatial_neighbor_metrics=None
    ):

        center_metrics = (
            self.calculate_center_post_metrics(
                pre_features,
                post_features
            )
        )

        neighbor_changes = (
            self.calculate_neighbor_feature_changes(
                neighbor_pre_features,
                neighbor_post_features
            )
        )

        neighbor_aggregate = (
            self.aggregate_neighbor_impact(
                neighbor_changes
            )
        )

        if (
            pre_array is None
            or
            post_array is None
        ):

            raise ValueError(
                "pre_array and post_array are required "
                "for post-embedding quality analysis."
            )

        quality_metrics = (
            self.calculate_region_quality_metrics(
                pre_array,
                post_array
            )
        )

        if spatial_neighbor_metrics is not None:

            neighbor_quality = (
                self.calculate_neighbor_quality_metrics(
                    neighbor_pre_arrays,
                    neighbor_post_arrays
                )
                if (
                    neighbor_pre_arrays is not None
                    and
                    neighbor_post_arrays is not None
                )
                else {}
            )

            spatial_impacts = [

                float(
                    value.get(
                        "impact_score",
                        np.nan
                    )
                )

                for value
                in spatial_neighbor_metrics.values()

                if value.get(
                    "available",
                    False
                )
                and
                np.isfinite(
                    value.get(
                        "impact_score",
                        np.nan
                    )
                )
            ]

            spatial_neighbor_mean = (
                float(
                    np.mean(
                        spatial_impacts
                    )
                )
                if spatial_impacts
                else 0.0
            )

            spatial_neighbor_max = (
                float(
                    np.max(
                        spatial_impacts
                    )
                )
                if spatial_impacts
                else 0.0
            )

            spatial_neighbor_std = (
                float(
                    np.std(
                        spatial_impacts
                    )
                )
                if spatial_impacts
                else 0.0
            )

            spatial_neighbor_total = (
                float(
                    np.sum(
                        spatial_impacts
                    )
                )
                if spatial_impacts
                else 0.0
            )

        elif (
            neighbor_pre_arrays is not None
            and
            neighbor_post_arrays is not None
        ):

            neighbor_quality = (
                self.calculate_neighbor_quality_metrics(
                    neighbor_pre_arrays,
                    neighbor_post_arrays
                )
            )

            spatial_impacts = [

                float(
                    result[
                        "impact_score"
                    ]
                )

                for result
                in neighbor_quality.values()

                if result.get(
                    "available",
                    False
                )
                and
                np.isfinite(
                    result.get(
                        "impact_score",
                        np.nan
                    )
                )
            ]

            spatial_neighbor_mean = (
                float(
                    np.mean(
                        spatial_impacts
                    )
                )
                if spatial_impacts
                else 0.0
            )

            spatial_neighbor_max = (
                float(
                    np.max(
                        spatial_impacts
                    )
                )
                if spatial_impacts
                else 0.0
            )

            spatial_neighbor_std = (
                float(
                    np.std(
                        spatial_impacts
                    )
                )
                if spatial_impacts
                else 0.0
            )

            spatial_neighbor_total = (
                float(
                    np.sum(
                        spatial_impacts
                    )
                )
                if spatial_impacts
                else 0.0
            )

        else:

            neighbor_quality = {}

            spatial_impacts = []

            spatial_neighbor_mean = 0.0
            spatial_neighbor_max = 0.0
            spatial_neighbor_std = 0.0
            spatial_neighbor_total = 0.0

        neighbor_feature_consequence = float(
            np.clip(
                0.40 * neighbor_aggregate["mean"]
                +
                0.20 * neighbor_aggregate["median"]
                +
                0.15 * neighbor_aggregate["maximum"]
                +
                0.10 * neighbor_aggregate["standard_deviation"]
                +
                0.15 * neighbor_aggregate["changed_ratio_mean"],
                0.0,
                1.0
            )
        )

        spatial_feature_consequence = float(
            np.clip(
                0.40 * spatial_neighbor_mean
                +
                0.25 * spatial_neighbor_max
                +
                0.15 * spatial_neighbor_std
                +
                0.20 * (
                    spatial_neighbor_total
                    /
                    max(
                        len(spatial_impacts),
                        1
                    )
                ),
                0.0,
                1.0
            )
        )

        total_consequence = (
            self.calculate_total_consequence(
                center_metrics[
                    "center_consequence"
                ],
                neighbor_feature_consequence,
                quality_metrics,
                spatial_neighbor_impact=
                    spatial_feature_consequence
            )
        )

        probabilities = (
            self.calculate_probability_from_consequence(
                total_consequence[
                    "total_consequence"
                ]
            )
        )

        classification = (
            self.classify_consequence(
                total_consequence[
                    "total_safety"
                ],
                primary_threshold=0.85,
                secondary_threshold=0.55,
                tertiary_threshold=0.25
            )
        )

        return {

            "region_id":
                region_id,

            "payload_size":
                payload_size,

            "payload_percentage":
                (
                    float(
                        payload_percentage
                    )
                    if payload_percentage is not None
                    else np.nan
                ),

            "center_consequence":
                center_metrics[
                    "center_consequence"
                ],

            "center_safety":
                center_metrics[
                    "center_safety"
                ],

            "center_mean_disturbance":
                center_metrics[
                    "center_mean_disturbance"
                ],

            "center_median_disturbance":
                center_metrics[
                    "center_median_disturbance"
                ],

            "center_variance_disturbance":
                center_metrics[
                    "center_variance_disturbance"
                ],

            "center_std_disturbance":
                center_metrics[
                    "center_std_disturbance"
                ],

            "center_min_disturbance":
                center_metrics[
                    "center_min_disturbance"
                ],

            "center_max_disturbance":
                center_metrics[
                    "center_max_disturbance"
                ],

            "neighbor_mean_impact":
                neighbor_aggregate[
                    "mean"
                ],

            "neighbor_median_impact":
                neighbor_aggregate[
                    "median"
                ],

            "neighbor_variance_impact":
                neighbor_aggregate[
                    "variance"
                ],

            "neighbor_std_impact":
                neighbor_aggregate[
                    "standard_deviation"
                ],

            "neighbor_min_impact":
                neighbor_aggregate[
                    "minimum"
                ],

            "neighbor_max_impact":
                neighbor_aggregate[
                    "maximum"
                ],

            "neighbor_total_impact":
                neighbor_aggregate[
                    "total"
                ],

            "neighbor_changed_ratio_mean":
                neighbor_aggregate[
                    "changed_ratio_mean"
                ],

            "neighbor_changed_ratio_maximum":
                neighbor_aggregate[
                    "changed_ratio_maximum"
                ],

            "neighbor_correlation_mean":
                neighbor_aggregate[
                    "correlation_mean"
                ],

            "neighbor_raw_mean":
                neighbor_aggregate[
                    "raw_mean"
                ],

            "neighbor_raw_median":
                neighbor_aggregate[
                    "raw_median"
                ],

            "neighbor_raw_variance":
                neighbor_aggregate[
                    "raw_variance"
                ],

            "neighbor_raw_std":
                neighbor_aggregate[
                    "raw_std"
                ],

            "neighbor_raw_min":
                neighbor_aggregate[
                    "raw_min"
                ],

            "neighbor_raw_max":
                neighbor_aggregate[
                    "raw_max"
                ],

            "neighbor_relative_mean":
                neighbor_aggregate[
                    "relative_mean"
                ],

            "neighbor_relative_median":
                neighbor_aggregate[
                    "relative_median"
                ],

            "neighbor_relative_variance":
                neighbor_aggregate[
                    "relative_variance"
                ],

            "neighbor_relative_std":
                neighbor_aggregate[
                    "relative_std"
                ],

            "neighbor_relative_min":
                neighbor_aggregate[
                    "relative_min"
                ],

            "neighbor_relative_max":
                neighbor_aggregate[
                    "relative_max"
                ],

            "spatial_neighbor_mean_impact":
                spatial_neighbor_mean,

            "spatial_neighbor_max_impact":
                spatial_neighbor_max,

            "spatial_neighbor_std_impact":
                spatial_neighbor_std,

            "spatial_neighbor_total_impact":
                spatial_neighbor_total,

            "mse":
                quality_metrics[
                    "mse"
                ],

            "mae":
                quality_metrics[
                    "mae"
                ],

            "psnr":
                quality_metrics[
                    "psnr"
                ],

            "ssim":
                quality_metrics[
                    "ssim"
                ],

            "normalized_mse":
                quality_metrics[
                    "normalized_mse"
                ],

            "mean_change":
                quality_metrics[
                    "mean_change"
                ],

            "variance_change":
                quality_metrics[
                    "variance_change"
                ],

            "maximum_change":
                quality_metrics[
                    "maximum_change"
                ],

            "changed_count":
                quality_metrics[
                    "changed_count"
                ],

            "changed_density":
                quality_metrics[
                    "changed_density"
                ],

            "total_consequence":
                total_consequence[
                    "total_consequence"
                ],

            "total_safety":
                total_consequence[
                    "total_safety"
                ],

            "safety_confidence_score":
                probabilities[
                    "safety_confidence_score"
                ],

            "risk_confidence_score":
                probabilities[
                    "risk_confidence_score"
                ],

            "predicted_class":
                classification,

            "neighbor_changes":
                neighbor_changes,

            "neighbor_quality":
                neighbor_quality,

            "neighbor_quality_details":
            {
                name: {
                    key: value
                    for key, value
                    in metrics.items()
                    if key != "available"
                }
                for name, metrics
                in neighbor_quality.items()
                if metrics.get(
                    "available",
                    False
                )
            },

            "center_delta_features":
                center_metrics[
                    "delta_features"
                ],

            "center_relative_changes":
                center_metrics[
                    "relative_changes"
                ],

            "all_consequence_metrics":
                total_consequence,
            "consequence_components":
            {
                "center":
                    total_consequence[
                        "center_consequence"
                    ],

                "neighbor_feature":
                    total_consequence[
                        "neighbor_feature_consequence"
                    ],

                "spatial_neighbor":
                    total_consequence[
                        "spatial_neighbor_consequence"
                    ],

                "normalized_mse":
                    total_consequence[
                        "normalized_mse"
                    ],

                "structural_loss":
                    total_consequence[
                        "structural_loss"
                    ]
            },
        }

    def update_payload_gradient(
        self,
        previous_result,
        current_result
    ):

        gradient = (
            self.calculate_payload_gradient(
                previous_result,
                current_result
            )
        )

        current_result[
            "payload_gradient"
        ] = gradient

        return current_result

    def calculate_consequence_gradient(
        self,
        previous_result,
        current_result
    ):

        if (
            previous_result is None
            or
            current_result is None
        ):
            return None

        previous_percentage = float(
            previous_result.get(
                "payload_percentage",
                np.nan
            )
        )

        current_percentage = float(
            current_result.get(
                "payload_percentage",
                np.nan
            )
        )

        if not (
            np.isfinite(
                previous_percentage
            )
            and
            np.isfinite(
                current_percentage
            )
        ):
            return None

        percentage_delta = (
            current_percentage
            -
            previous_percentage
        )

        if abs(
            percentage_delta
        ) < 1e-12:

            return 0.0

        consequence_delta = (
            float(
                current_result[
                    "total_consequence"
                ]
            )
            -
            float(
                previous_result[
                    "total_consequence"
                ]
            )
        )

        return float(
            consequence_delta
            /
            percentage_delta
        )

    def find_payload_limit(
        self,
        results,
        consequence_threshold=0.30,
        safety_threshold=0.70,
        gradient_threshold=0.01
    ):

        if not results:

            return {

                "maximum_safe_payload":
                    0,

                "maximum_tested_payload":
                    0,

                "payload_limit_status":
                    "NO_RESULTS"
            }

        ordered = sorted(
            results,
            key=lambda result:
                result["payload_size"]
        )

        safe_results = []

        for result in ordered:

            consequence = float(
                result[
                    "total_consequence"
                ]
            )

            safety = float(
                result[
                    "total_safety"
                ]
            )

            if (
                consequence <=
                consequence_threshold
                and
                safety >=
                safety_threshold
            ):

                safe_results.append(
                    result
                )

        if not safe_results:

            return {

                "maximum_safe_payload":
                    0,

                "maximum_tested_payload":
                    ordered[-1][
                        "payload_size"
                    ],

                "payload_limit_status":
                    "NO_SAFE_PAYLOAD"
            }

        maximum_safe = max(
            result[
                "payload_size"
            ]
            for result in safe_results
        )

        gradients = []

        for index in range(
            1,
            len(ordered)
        ):

            gradient = (
                self.calculate_consequence_gradient(
                    ordered[index - 1],
                    ordered[index]
                )
            )

            if gradient is not None:

                gradients.append(
                    gradient
                )

        maximum_gradient = (
            max(
                abs(value)
                for value in gradients
            )
            if gradients
            else 0.0
        )

        if maximum_gradient >= gradient_threshold:

            status = (
                "SAFE_LIMIT_REACHED_SENSITIVITY_DETECTED"
            )

        else:

            status = (
                "SAFE_LIMIT_REACHED"
            )

        return {

            "maximum_safe_payload":
                maximum_safe,

            "maximum_tested_payload":
                ordered[-1][
                    "payload_size"
                ],

            "maximum_consequence_gradient":
                maximum_gradient,

            "payload_limit_status":
                status
        }

    def build_post_embedding_record(
        self,
        scenario,
        post_features,
        neighbor_post_features,
        pre_arrays=None,
        post_arrays=None,
        neighbor_pre_arrays=None,
        neighbor_post_arrays=None,
        spatial_neighbor_metrics=None
    ):

        region_id = scenario[
            "region_id"
        ]

        payload_size = scenario[
            "payload_size"
        ]

        pre_features = self.load_region_pre_features(
            region_id
        )

        neighbor_pre_features = (
            self.load_neighbor_pre_features(
                region_id
            )
        )

        result = (
            self.calculate_embedding_scenario_metrics(
                region_id=region_id,
                payload_size=payload_size,
                pre_features=pre_features,
                post_features=post_features,
                neighbor_pre_features=neighbor_pre_features,
                neighbor_post_features=neighbor_post_features,
                payload_percentage=scenario.get(
                    "payload_percentage"
                ),
                pre_array=(
                    pre_arrays.get(
                        "center"
                    )
                    if pre_arrays
                    else None
                ),
                post_array=(
                    post_arrays.get(
                        "center"
                    )
                    if post_arrays
                    else None
                ),
                neighbor_pre_arrays=(
                    neighbor_pre_arrays
                    if neighbor_pre_arrays
                    else None
                ),
                neighbor_post_arrays=(
                    neighbor_post_arrays
                    if neighbor_post_arrays
                    else None
                ),
                spatial_neighbor_metrics=(
                    spatial_neighbor_metrics
                    if spatial_neighbor_metrics
                    else None
                )
            )
        )

        result = (
            self.finalize_consequence_result(
                result
            )
        )

        result[
            "scenario_id"
        ] = scenario[
            "scenario_id"
        ]

        result[
            "embedding_capacity"
        ] = scenario[
            "capacity"
        ]

        result[
            "embedding_capacity_ratio"
        ] = scenario[
            "capacity_ratio"
        ]

        result[
            "embedding_status"
        ] = "COMPLETED"

        for feature, value in (
            pre_features.items()
        ):

            result[
                f"pre_{feature}"
            ] = value

        for feature, value in (
            post_features.items()
        ):

            result[
                f"post_{feature}"
            ] = value

        for feature, value in (
            result.get(
                "center_delta_features",
                {}
            ).items()
        ):

            result[
                f"delta_{feature}"
            ] = value

        for neighbor_name, changes in (
            result.get(
                "neighbor_changes",
                {}
            ).items()
        ):

            result[
                f"{neighbor_name}_available"
            ] = changes.get(
                "available",
                False
            )

            result[
                f"{neighbor_name}_raw_mean"
            ] = changes.get(
                "raw_mean",
                np.nan
            )

            result[
                f"{neighbor_name}_raw_median"
            ] = changes.get(
                "raw_median",
                np.nan
            )

            result[
                f"{neighbor_name}_raw_variance"
            ] = changes.get(
                "raw_variance",
                np.nan
            )

            result[
                f"{neighbor_name}_raw_std"
            ] = changes.get(
                "raw_std",
                np.nan
            )

            result[
                f"{neighbor_name}_raw_min"
            ] = changes.get(
                "raw_min",
                np.nan
            )

            result[
                f"{neighbor_name}_raw_max"
            ] = changes.get(
                "raw_max",
                np.nan
            )

            result[
                f"{neighbor_name}_relative_mean"
            ] = changes.get(
                "relative_mean",
                np.nan
            )

            result[
                f"{neighbor_name}_relative_median"
            ] = changes.get(
                "relative_median",
                np.nan
            )

            result[
                f"{neighbor_name}_relative_variance"
            ] = changes.get(
                "relative_variance",
                np.nan
            )

            result[
                f"{neighbor_name}_relative_std"
            ] = changes.get(
                "relative_std",
                np.nan
            )

            result[
                f"{neighbor_name}_relative_min"
            ] = changes.get(
                "relative_min",
                np.nan
            )

            result[
                f"{neighbor_name}_relative_max"
            ] = changes.get(
                "relative_max",
                np.nan
            )

            result[
                f"{neighbor_name}_changed_count"
            ] = changes.get(
                "changed_count",
                0
            )

            result[
                f"{neighbor_name}_feature_count"
            ] = changes.get(
                "feature_count",
                0
            )

            result[
                f"{neighbor_name}_changed_ratio"
            ] = changes.get(
                "changed_ratio",
                np.nan
            )

            result[
                f"{neighbor_name}_correlation"
            ] = changes.get(
                "correlation",
                np.nan
            )

            for feature, value in (
                changes.get(
                    "delta",
                    {}
                ).items()
            ):

                result[
                    f"{neighbor_name}_delta_{feature}"
                ] = value

            for feature, value in (
                changes.get(
                    "relative",
                    {}
                ).items()
            ):

                result[
                    f"{neighbor_name}_relative_{feature}"
                ] = value

        return result


    def append_post_embedding_result(
        self,
        record
    ):

        if self.post_dataset is None:

            self.post_dataset = pd.DataFrame()

        new_record = pd.DataFrame(
            [record]
        )

        self.post_dataset = pd.concat(
            [
                self.post_dataset,
                new_record
            ],
            ignore_index=True
        )


    def save_post_embedding_dataset(self):

        if (
            self.post_dataset is None
            or
            self.post_dataset.empty
        ):

            raise ValueError(
                "No post-embedding results available."
            )

        self.post_dataset.to_csv(
            self.post_dataset_path,
            index=False
        )

        print(
            f"Post-Embedding Rows : "
            f"{len(self.post_dataset)}"
        )

        print(
            f"Post-Embedding Features : "
            f"{len(self.post_dataset.columns)}"
        )

        print(
            f"Post-Embedding Dataset Saved : "
            f"{self.post_dataset_path}"
        )

    def load_experiment_plan(self):

        if not os.path.exists(
            self.scenario_path
        ):

            raise FileNotFoundError(
                self.scenario_path
            )

        with open(
            self.scenario_path,
            "r",
            encoding="utf-8"
        ) as file:

            scenarios = json.load(
                file
            )

        if not isinstance(
            scenarios,
            list
        ):

            raise ValueError(
                "Invalid experiment scenario format."
            )

        print(
            f"Experiment Scenarios Loaded : "
            f"{len(scenarios)}"
        )

        return scenarios


    def prepare_embedding_context(
        self,
        scenario
    ):

        region_id = scenario[
            "region_id"
        ]

        region_row = (
            self.get_region_row(
                region_id
            )
        )

        neighbors = (
            self.get_neighbors(
                region_id
            )
        )

        context = {

            "scenario_id":
                scenario[
                    "scenario_id"
                ],

            "region_id":
                region_id,

            "payload_size":
                int(
                    scenario[
                        "payload_size"
                    ]
                ),

            "payload_percentage":
                float(
                    scenario[
                        "payload_percentage"
                    ]
                ),

            "capacity":
                float(
                    scenario[
                        "capacity"
                    ]
                ),

            "capacity_ratio":
                float(
                    scenario[
                        "capacity_ratio"
                    ]
                ),

            "region_row":
                region_row.to_dict(),

            "neighbors":
                neighbors,

            "center_pre_features":
                self.load_region_pre_features(
                    region_id
                ),

            "neighbor_pre_features":
                self.load_neighbor_pre_features(
                    region_id
                )
        }

        return context


    def validate_embedding_context(
        self,
        context
    ):

        required = [
            "scenario_id",
            "region_id",
            "payload_size",
            "capacity",
            "center_pre_features",
            "neighbor_pre_features"
        ]

        missing = [
            key
            for key in required
            if key not in context
        ]

        if missing:

            raise ValueError(
                f"Embedding context missing: "
                f"{missing}"
            )

        payload = int(
            context[
                "payload_size"
            ]
        )

        capacity = float(
            context[
                "capacity"
            ]
        )

        if payload <= 0:

            raise ValueError(
                "Payload must be greater than zero."
            )

        if payload > capacity:

            raise ValueError(
                f"Payload {payload} exceeds "
                f"region capacity {capacity}."
            )

        if not context[
            "center_pre_features"
        ]:

            raise ValueError(
                "Center pre-embedding features are empty."
            )

        return True


    def create_embedding_request(
        self,
        context
    ):

        self.validate_embedding_context(
            context
        )

        request = {

            "scenario_id":
                context[
                    "scenario_id"
                ],

            "region_id":
                context[
                    "region_id"
                ],

            "payload_size":
                context[
                    "payload_size"
                ],

            "payload_percentage":
                context[
                    "payload_percentage"
                ],

            "capacity":
                context[
                    "capacity"
                ],

            "capacity_ratio":
                context[
                    "capacity_ratio"
                ],

            "neighbors":
                context[
                    "neighbors"
                ],

            "status":
                "READY_FOR_ACTUAL_EMBEDDING"
        }

        return request


    def prepare_actual_embedding_batch(
        self,
        scenarios
    ):

        requests = []

        for scenario in scenarios:

            try:

                context = (
                    self.prepare_embedding_context(
                        scenario
                    )
                )

                request = (
                    self.create_embedding_request(
                        context
                    )
                )

                requests.append(
                    request
                )

            except Exception as error:

                requests.append({

                    "scenario_id":
                        scenario.get(
                            "scenario_id"
                        ),

                    "region_id":
                        scenario.get(
                            "region_id"
                        ),

                    "status":
                        "PREPARATION_FAILED",

                    "error":
                        str(error)
                })

        batch_path = os.path.join(
            self.output_dir,
            "actual_embedding_requests.json"
        )

        with open(
            batch_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                requests,
                file,
                indent=4,
                default=str
            )

        ready = sum(
            1
            for request in requests
            if request.get(
                "status"
            )
            ==
            "READY_FOR_ACTUAL_EMBEDDING"
        )

        failed = len(
            requests
        ) - ready

        print(
            f"Embedding Requests : "
            f"{len(requests)}"
        )

        print(
            f"Ready : {ready}"
        )

        print(
            f"Failed : {failed}"
        )

        print(
            f"Requests Saved : "
            f"{batch_path}"
        )

        return requests

    def register_actual_embedding_result(
        self,
        scenario,
        post_features,
        neighbor_post_features,
        pre_array=None,
        post_array=None,
        neighbor_pre_arrays=None,
        neighbor_post_arrays=None
    ):

        if not isinstance(
            post_features,
            dict
        ):
            raise ValueError(
                "post_features must be a dictionary."
            )

        if neighbor_post_features is None:

            neighbor_post_features = {
                name: {}
                for name in self.neighbor_directions
            }

        record = (
            self.build_post_embedding_record(
                scenario=scenario,
                post_features=post_features,
                neighbor_post_features=neighbor_post_features,
                pre_arrays=pre_array,
                post_arrays=post_array,
                neighbor_pre_arrays=neighbor_pre_arrays,
                neighbor_post_arrays=neighbor_post_arrays
            )
        )

        self.append_post_embedding_result(
            record
        )

        return record


    def process_actual_embedding_result(
        self,
        scenario,
        embedding_result
    ):

        if not isinstance(
            embedding_result,
            dict
        ):
            raise ValueError(
                "Embedding result must be a dictionary."
            )

        post_features = (
            embedding_result.get(
                "post_features"
            )
        )

        neighbor_post_features = (
            embedding_result.get(
                "neighbor_post_features",
                {}
            )
        )

        pre_array = (
            embedding_result.get(
                "pre_array"
            )
        )

        post_array = (
            embedding_result.get(
                "post_array"
            )
        )

        neighbor_pre_arrays = (
            embedding_result.get(
                "neighbor_pre_arrays",
                {}
            )
        )

        neighbor_post_arrays = (
            embedding_result.get(
                "neighbor_post_arrays",
                {}
            )
        )

        if post_features is None:

            raise ValueError(
                "Actual embedding result does not "
                "contain post_features."
            )

        return (
            self.register_actual_embedding_result(
                scenario=scenario,
                post_features=post_features,
                neighbor_post_features=neighbor_post_features,
                pre_arrays=pre_array,
                post_arrays=post_array,
                neighbor_pre_arrays=neighbor_pre_arrays,
                neighbor_post_arrays=neighbor_post_arrays
            )
        )


    def calculate_payload_sensitivity_curve(
        self,
        results
    ):

        if not results:

            return []

        ordered = sorted(
            results,
            key=lambda item:
                item.get(
                    "payload_percentage",
                    0.0
                )
        )

        curve = []

        previous = None

        for current in ordered:

            payload = float(
                current.get(
                    "payload_percentage",
                    np.nan
                )
            )

            consequence = float(
                current[
                    "total_consequence"
                ]
            )

            safety = float(
                current[
                    "total_safety"
                ]
            )

            if previous is None:

                consequence_gradient = np.nan
                safety_gradient = np.nan

            else:

                previous_payload = float(
                    previous.get(
                        "payload_percentage",
                        np.nan
                    )
                )

                payload_delta = (
                    payload
                    -
                    previous_payload
                )

                if abs(
                    payload_delta
                ) < 1e-12:

                    consequence_gradient = 0.0
                    safety_gradient = 0.0

                else:

                    consequence_gradient = (
                        consequence
                        -
                        previous[
                            "total_consequence"
                        ]
                    ) / payload_delta

                    safety_gradient = (
                        safety
                        -
                        previous[
                            "total_safety"
                        ]
                    ) / payload_delta

            curve.append({

                "payload_percentage":
                    payload,

                "total_consequence":
                    consequence,

                "total_safety":
                    safety,

                "consequence_gradient":
                    consequence_gradient,

                "safety_gradient":
                    safety_gradient
            })

            previous = current

        return curve


    def optimize_payload_from_results(
        self,
        results,
        consequence_threshold=0.30,
        safety_threshold=0.70,
        gradient_threshold=0.01
    ):

        if not results:

            return {

                "maximum_safe_payload":
                    0,

                "optimal_payload":
                    0,

                "status":
                    "NO_RESULTS",

                "sensitivity_detected":
                    False
            }

        ordered = sorted(
            results,
            key=lambda item:
                item["payload_size"]
        )

        curve = (
            self.calculate_payload_sensitivity_curve(
                ordered
            )
        )

        safe_results = [

            result

            for result in ordered

            if (
                float(
                    result[
                        "total_consequence"
                    ]
                )
                <= consequence_threshold
                and
                float(
                    result[
                        "total_safety"
                    ]
                )
                >= safety_threshold
            )
        ]

        if not safe_results:

            return {

                "maximum_safe_payload":
                    0,

                "optimal_payload":
                    0,

                "status":
                    "NO_SAFE_PAYLOAD",

                "sensitivity_detected":
                    any(
                        not np.isnan(
                            point[
                                "consequence_gradient"
                            ]
                        )
                        and
                        abs(
                            point[
                                "consequence_gradient"
                            ]
                        )
                        >= gradient_threshold
                        for point in curve
                    )
            }

        maximum_safe_payload = max(
            result[
                "payload_size"
            ]
            for result in safe_results
        )

        sensitivity_detected = any(

            not np.isnan(
                point[
                    "consequence_gradient"
                ]
            )

            and

            abs(
                point[
                    "consequence_gradient"
                ]
            ) >= gradient_threshold

            for point in curve
        )

        optimal_candidates = sorted(
            safe_results,
            key=lambda result: (
                -float(
                    result[
                        "safe_probability"
                    ]
                ),
                float(
                    result[
                        "total_consequence"
                    ]
                ),
                -float(
                    result[
                        "payload_size"
                    ]
                )
            )
        )

        optimal_payload = int(
            optimal_candidates[0][
                "payload_size"
            ]
        )

        return {

            "maximum_safe_payload":
                int(
                    maximum_safe_payload
                ),

            "optimal_payload":
                optimal_payload,

            "status":
                "OPTIMIZED",

            "sensitivity_detected":
                sensitivity_detected,

            "sensitivity_curve":
                curve
        }

    def validate_actual_embedding_output(
        self,
        embedding_result
    ):

        required = [
            "post_features",
            "neighbor_post_features"
        ]

        missing = [
            key
            for key in required
            if key not in embedding_result
        ]

        if missing:

            raise ValueError(
                "Actual embedding output missing: "
                + ", ".join(missing)
            )

        if not isinstance(
            embedding_result["post_features"],
            dict
        ):

            raise ValueError(
                "post_features must be a dictionary."
            )

        if not isinstance(
            embedding_result[
                "neighbor_post_features"
            ],
            dict
        ):

            raise ValueError(
                "neighbor_post_features must be a dictionary."
            )

        return True


    def create_embedding_execution_record(
        self,
        scenario
    ):

        return {

            "scenario_id":
                scenario[
                    "scenario_id"
                ],

            "region_id":
                scenario[
                    "region_id"
                ],

            "payload_size":
                int(
                    scenario[
                        "payload_size"
                    ]
                ),

            "capacity":
                float(
                    scenario[
                        "capacity"
                    ]
                ),

            "capacity_ratio":
                float(
                    scenario[
                        "capacity_ratio"
                    ]
                ),

            "embedding_status":
                "READY",

            "post_analysis_status":
                "PENDING",

            "ml_status":
                "PENDING",

            "ga_status":
                "PENDING",

            "aco_status":
                "PENDING"
        }


    def save_execution_status(
        self,
        execution_records
    ):

        status_path = os.path.join(
            self.output_dir,
            "embedding_execution_status.json"
        )

        with open(
            status_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                execution_records,
                file,
                indent=4,
                default=str
            )

        print(
            f"Execution Status Saved : "
            f"{status_path}"
        )


    def update_execution_status(
        self,
        execution_record,
        embedding_result
    ):

        self.validate_actual_embedding_output(
            embedding_result
        )

        execution_record[
            "embedding_status"
        ] = "COMPLETED"

        execution_record[
            "post_analysis_status"
        ] = "COMPLETED"

        execution_record[
            "ml_status"
        ] = "READY_AFTER_FEATURE_ENGINEERING"

        return execution_record


    def process_completed_embedding(
        self,
        scenario,
        embedding_result
    ):

        self.validate_actual_embedding_output(
            embedding_result
        )

        record = (
            self.process_actual_embedding_result(
                scenario,
                embedding_result
            )
        )

        execution_record = (
            self.create_embedding_execution_record(
                scenario
            )
        )

        execution_record = (
            self.update_execution_status(
                execution_record,
                embedding_result
            )
        )

        record[
            "execution_status"
        ] = execution_record

        return record


    def save_complete_post_result(
        self,
        record
    ):

        result_path = os.path.join(
            self.output_dir,
            f"result_"
            f"{record['scenario_id']}.json"
        )

        with open(
            result_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                record,
                file,
                indent=4,
                default=str
            )

        return result_path


    def finalize_region_payload_analysis(
        self,
        region_id,
        results
    ):

        optimization = (
            self.optimize_payload_from_results(
                results
            )
        )

        region_summary = {

            "region_id":
                region_id,

            "tested_scenarios":
                len(results),

            "maximum_safe_payload":
                optimization[
                    "maximum_safe_payload"
                ],

            "optimal_payload":
                optimization[
                    "optimal_payload"
                ],

            "status":
                optimization[
                    "status"
                ],

            "sensitivity_detected":
                optimization[
                    "sensitivity_detected"
                ],

            "sensitivity_curve":
                optimization.get(
                    "sensitivity_curve",
                    []
                )
        }

        return region_summary


    def save_region_payload_analysis(
        self,
        region_summary
    ):

        region_id = region_summary[
            "region_id"
        ]

        summary_path = os.path.join(
            self.output_dir,
            f"region_{region_id}_"
            f"payload_analysis.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                region_summary,
                file,
                indent=4,
                default=str
            )

        return summary_path

    def build_post_embedding_summary(
        self,
        records
    ):

        if not records:

            return {

                "total_experiments":
                    0,

                "completed_experiments":
                    0,

                "safe_experiments":
                    0,

                "unsafe_experiments":
                    0,

                "regions_analyzed":
                    0
            }

        completed = [
            record
            for record in records
            if record.get(
                "embedding_status"
            ) == "COMPLETED"
        ]

        safe = [
            record
            for record in completed
            if record.get(
                "predicted_class"
            ) in {
                "PRIMARY",
                "SECONDARY",
                "TERTIARY"
            }
        ]

        unsafe = [
            record
            for record in completed
            if record.get(
                "predicted_class"
            ) == "REJECT"
        ]

        regions = set(
            record.get(
                "region_id"
            )
            for record in completed
        )

        consequence_values = [

            float(
                record[
                    "total_consequence"
                ]
            )

            for record in completed

            if record.get(
                "total_consequence"
            ) is not None
        ]

        safety_values = [

            float(
                record[
                    "total_safety"
                ]
            )

            for record in completed

            if record.get(
                "total_safety"
            ) is not None
        ]

        summary = {

            "total_experiments":
                len(records),

            "completed_experiments":
                len(completed),

            "safe_experiments":
                len(safe),

            "unsafe_experiments":
                len(unsafe),

            "regions_analyzed":
                len(regions),

            "average_consequence":
                float(
                    np.mean(
                        consequence_values
                    )
                )
                if consequence_values
                else 0.0,

            "maximum_consequence":
                float(
                    np.max(
                        consequence_values
                    )
                )
                if consequence_values
                else 0.0,

            "average_safety":
                float(
                    np.mean(
                        safety_values
                    )
                )
                if safety_values
                else 0.0,

            "minimum_safety":
                float(
                    np.min(
                        safety_values
                    )
                )
                if safety_values
                else 0.0
        }

        return summary


    def save_post_embedding_summary(
        self,
        summary
    ):

        summary_path = os.path.join(
            self.output_dir,
            "post_embedding_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4
            )

        print(
            f"Post-Embedding Summary Saved : "
            f"{summary_path}"
        )


    def export_payload_optimization_table(
        self,
        region_summaries
    ):

        if not region_summaries:

            return

        rows = []

        for summary in region_summaries:

            rows.append({

                "region_id":
                    summary.get(
                        "region_id"
                    ),

                "tested_scenarios":
                    summary.get(
                        "tested_scenarios",
                        0
                    ),

                "maximum_safe_payload":
                    summary.get(
                        "maximum_safe_payload",
                        0
                    ),

                "optimal_payload":
                    summary.get(
                        "optimal_payload",
                        0
                    ),

                "status":
                    summary.get(
                        "status"
                    ),

                "sensitivity_detected":
                    summary.get(
                        "sensitivity_detected",
                        False
                    )
            })

        dataframe = pd.DataFrame(
            rows
        )

        output_path = os.path.join(
            self.output_dir,
            "region_payload_optimization.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

        print(
            f"Payload Optimization Table Saved : "
            f"{output_path}"
        )


    def export_completed_results(
        self,
        records
    ):

        if not records:

            return

        flattened = []

        for record in records:

            row = {}

            for key, value in (
                record.items()
            ):

                if isinstance(
                    value,
                    dict
                ):
                    continue

                if isinstance(
                    value,
                    (list, tuple)
                ):
                    continue

                row[key] = value

            flattened.append(
                row
            )

        dataframe = pd.DataFrame(
            flattened
        )

        output_path = os.path.join(
            self.output_dir,
            "post_embedding_results.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

        print(
            f"Completed Results Saved : "
            f"{output_path}"
        )

    def load_dwt_coefficients(self):

        dwt_dir = os.path.join(
            self.base_dir,
            "output",
            "dwt_decomposition"
        )

        self.dwt_paths = {
            "LH": os.path.join(
                dwt_dir,
                "LH.npy"
            ),
            "HL": os.path.join(
                dwt_dir,
                "HL.npy"
            ),
            "HH": os.path.join(
                dwt_dir,
                "HH.npy"
            )
        }

        self.dwt_coefficients = {}

        for subband, path in (
            self.dwt_paths.items()
        ):

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"DWT coefficient file missing: "
                    f"{path}"
                )

            coefficients = np.load(
                path
            )

            if coefficients.ndim != 2:

                raise ValueError(
                    f"{subband} must be a 2D coefficient matrix."
                )

            if not np.all(
                np.isfinite(coefficients)
            ):

                raise ValueError(
                    f"{subband} contains invalid coefficient values."
                )

            self.dwt_coefficients[
                subband
            ] = coefficients.astype(
                np.float64
            )

        shapes = {
            subband: matrix.shape
            for subband, matrix
            in self.dwt_coefficients.items()
        }

        if len(
            set(
                shapes.values()
            )
        ) != 1:

            raise ValueError(
                "LH, HL and HH shapes do not match."
            )

        self.dwt_shape = next(
            iter(
                shapes.values()
            )
        )

        print(
            f"LH Shape : "
            f"{self.dwt_coefficients['LH'].shape}"
        )

        print(
            f"HL Shape : "
            f"{self.dwt_coefficients['HL'].shape}"
        )

        print(
            f"HH Shape : "
            f"{self.dwt_coefficients['HH'].shape}"
        )

    def get_dwt_region_bounds(
        self,
        region_id
    ):

        row = self.get_region_row(
            region_id
        )

        row_column, column_column = (
            self.find_coordinate_columns()
        )

        row_start = int(
            row[row_column]
        )

        column_start = int(
            row[column_column]
        )

        row_end_column = (
            "row_end"
            if "row_end" in row.index
            else None
        )

        column_end_column = (
            "column_end"
            if "column_end" in row.index
            else None
        )

        if (
            row_end_column is not None
            and
            column_end_column is not None
        ):

            row_end = min(
                int(row[row_end_column]),
                self.dwt_shape[0]
            )

            column_end = min(
                int(row[column_end_column]),
                self.dwt_shape[1]
            )

        else:

            region_size = int(
                row.get(
                    "block_size",
                    row.get(
                        "rows",
                        16
                    )
                )
            )

            row_end = min(
                row_start + region_size,
                self.dwt_shape[0]
            )

            column_end = min(
                column_start + region_size,
                self.dwt_shape[1]
            )
        if (
            row_start < 0
            or
            column_start < 0
            or
            row_start >= self.dwt_shape[0]
            or
            column_start >= self.dwt_shape[1]
        ):

            raise ValueError(
                f"Invalid DWT region coordinates: "
                f"{region_id}"
            )

        if (
            row_end <= row_start
            or
            column_end <= column_start
        ):

            raise ValueError(
                f"Invalid DWT region bounds: "
                f"{region_id}"
            )

        return {
            "row_start": row_start,
            "row_end": row_end,
            "column_start": column_start,
            "column_end": column_end,
            "size": row_end - row_start
        }

    def extract_dwt_region(
        self,
        subband,
        region_id
    ):

        if subband not in self.dwt_coefficients:

            raise ValueError(
                f"Unsupported DWT subband: "
                f"{subband}"
            )

        bounds = (
            self.get_dwt_region_bounds(
                region_id
            )
        )

        matrix = self.dwt_coefficients[
            subband
        ]

        region = matrix[
            bounds["row_start"]:
            bounds["row_end"],

            bounds["column_start"]:
            bounds["column_end"]
        ]

        if region.size == 0:

            raise ValueError(
                f"Empty {subband} region: "
                f"{region_id}"
            )

        return region.copy()


    def extract_dwt_region_all_subbands(
        self,
        region_id
    ):

        return {

            "LH":
                self.extract_dwt_region(
                    "LH",
                    region_id
                ),

            "HL":
                self.extract_dwt_region(
                    "HL",
                    region_id
                ),

            "HH":
                self.extract_dwt_region(
                    "HH",
                    region_id
                )
        }


    def extract_neighbor_dwt_regions(
        self,
        region_id
    ):

        neighbors = self.get_neighbors(
            region_id
        )

        result = {}

        for neighbor_name, neighbor_id in (
            neighbors.items()
        ):

            if neighbor_id is None:

                result[
                    neighbor_name
                ] = None

                continue

            result[
                neighbor_name
            ] = (
                self.extract_dwt_region_all_subbands(
                    neighbor_id
                )
            )

        return result

    def create_dwt_embedding_workspaces(
        self,
        region_id
    ):

        if not hasattr(
            self,
            "dwt_coefficients"
        ):

            self.load_dwt_coefficients()

        bounds = (
            self.get_dwt_region_bounds(
                region_id
            )
        )

        workspace = {

            "region_id":
                region_id,

            "original": {

                "LH":
                    self.dwt_coefficients[
                        "LH"
                    ].copy(),

                "HL":
                    self.dwt_coefficients[
                        "HL"
                    ].copy(),

                "HH":
                    self.dwt_coefficients[
                        "HH"
                    ].copy()
            },

            "modified": {

                "LH":
                    self.dwt_coefficients[
                        "LH"
                    ].copy(),

                "HL":
                    self.dwt_coefficients[
                        "HL"
                    ].copy(),

                "HH":
                    self.dwt_coefficients[
                        "HH"
                    ].copy()
            },

            "embedding_mask": {

                "LH":
                    np.zeros(
                        self.dwt_coefficients[
                            "LH"
                        ].shape,
                        dtype=bool
                    ),

                "HL":
                    np.zeros(
                        self.dwt_coefficients[
                            "HL"
                        ].shape,
                        dtype=bool
                    ),

                "HH":
                    np.zeros(
                        self.dwt_coefficients[
                            "HH"
                        ].shape,
                        dtype=bool
                    )
            },

            "embedding_count":
                0,

            "payload_size":
                0,

            "status":
                "READY"
        }

        for subband in [
            "LH",
            "HL",
            "HH"
        ]:

            workspace[
                "embedding_mask"
            ][subband][
                bounds["row_start"]:
                bounds["row_end"],

                bounds["column_start"]:
                bounds["column_end"]
            ] = True

        return workspace

    

    def validate_dwt_workspace(
        self,
        workspace
    ):

        required_subbands = [
            "LH",
            "HL",
            "HH"
        ]

        for section in [
            "original",
            "modified",
            "embedding_mask"
        ]:

            if section not in workspace:

                raise ValueError(
                    f"Missing workspace section: "
                    f"{section}"
                )

            for subband in required_subbands:

                if subband not in workspace[
                    section
                ]:

                    raise ValueError(
                        f"Missing {subband} "
                        f"in workspace {section}."
                    )

        for subband in required_subbands:

            original = workspace[
                "original"
            ][subband]

            modified = workspace[
                "modified"
            ][subband]

            mask = workspace[
                "embedding_mask"
            ][subband]

            if original.shape != modified.shape:

                raise ValueError(
                    f"{subband} original/modified "
                    f"shape mismatch."
                )

            if original.shape != mask.shape:

                raise ValueError(
                    f"{subband} coefficient/mask "
                    f"shape mismatch."
                )

            if not np.all(
                np.isfinite(original)
            ):

                raise ValueError(
                    f"{subband} contains invalid "
                    f"original coefficients."
                )

            if not np.all(
                np.isfinite(modified)
            ):

                raise ValueError(
                    f"{subband} contains invalid "
                    f"modified coefficients."
                )

        return True


    def get_embedding_candidate_coefficients(
        self,
        workspace
    ):

        candidates = []

        bounds = self.get_dwt_region_bounds(
            workspace["region_id"]
        )

        for subband in [
            "LH",
            "HL",
            "HH"
        ]:

            coefficients = workspace[
                "original"
            ][subband]

            mask = workspace[
                "embedding_mask"
            ][subband]

            row_start = bounds[
                "row_start"
            ]

            row_end = min(
                bounds["row_end"],
                coefficients.shape[0]
            )

            column_start = bounds[
                "column_start"
            ]

            column_end = min(
                bounds["column_end"],
                coefficients.shape[1]
            )

            for row in range(
                row_start,
                row_end
            ):

                for column in range(
                    column_start,
                    column_end
                ):

                    if not mask[
                        row,
                        column
                    ]:

                        continue

                    value = float(
                        coefficients[
                            row,
                            column
                        ]
                    )

                    candidates.append({
                        "subband": subband,
                        "row": row,
                        "column": column,
                        "coefficient": value,
                        "absolute_coefficient": abs(
                            value
                        )
                    })

        candidates.sort(
            key=lambda item:
                item["absolute_coefficient"],
            reverse=True
        )

        return candidates

    def calculate_workspace_distortion(
        self,
        workspace
    ):

        original_values = []
        modified_values = []

        for subband in [
            "LH",
            "HL",
            "HH"
        ]:

            original = workspace[
                "original"
            ][subband]

            modified = workspace[
                "modified"
            ][subband]

            original_values.extend(
                original.flatten()
            )

            modified_values.extend(
                modified.flatten()
            )

        original_values = np.asarray(
            original_values,
            dtype=np.float64
        )

        modified_values = np.asarray(
            modified_values,
            dtype=np.float64
        )

        difference = (
            modified_values
            -
            original_values
        )

        mse = float(
            np.mean(
                difference ** 2
            )
        )

        signal_power = float(
            np.mean(
                original_values ** 2
            )
        )

        if mse <= 1e-12:

            snr = float("inf")

        elif signal_power <= 1e-12:

            snr = 0.0

        else:

            snr = float(
                10.0
                *
                np.log10(
                    signal_power
                    /
                    mse
                )
            )

        return {

            "coefficient_mse":
                mse,

            "coefficient_snr":
                snr,

            "modified_coefficients":
                int(
                    np.count_nonzero(
                        difference
                    )
                ),

            "total_coefficients":
                int(
                    original_values.size
                )
        }

    def prepare_payload_bits(
        self,
        payload_size,
        seed=None
    ):

        payload_size = int(
            payload_size
        )

        if payload_size <= 0:

            raise ValueError(
                "Payload size must be greater than zero."
            )

        if seed is not None:

            rng = np.random.default_rng(
                int(seed)
            )

            bits = rng.integers(
                0,
                2,
                size=payload_size,
                dtype=np.uint8
            )

        else:

            bits = np.random.randint(
                0,
                2,
                size=payload_size,
                dtype=np.uint8
            )

        return bits.tolist()


    def get_candidate_coefficients_for_embedding(
        self,
        workspace,
        payload_size
    ):

        self.validate_dwt_workspace(
            workspace
        )

        candidates = (
            self.get_embedding_candidate_coefficients(
                workspace
            )
        )

        if len(candidates) < payload_size:

            raise ValueError(
                f"Insufficient coefficients. "
                f"Required: {payload_size}, "
                f"Available: {len(candidates)}."
            )

        return candidates[
            :payload_size
        ]

    def create_embedding_request_data(
        self,
        workspace,
        payload_size,
        payload_bits,
        embedding_subbands=None
    ):

        if embedding_subbands is None:

            embedding_subbands = [
                "LH",
                "HL",
                "HH"
            ]

        for subband in embedding_subbands:

            if subband not in [
                "LH",
                "HL",
                "HH"
            ]:

                raise ValueError(
                    f"Invalid embedding subband: "
                    f"{subband}"
                )

        candidates = []

        bounds = self.get_dwt_region_bounds(
            workspace["region_id"]
        )

        selected_subbands = set(
            embedding_subbands
        )

        for subband in selected_subbands:

            coefficients = workspace[
                "original"
            ][subband]

            mask = workspace[
                "embedding_mask"
            ][subband]

            row_start = bounds[
                "row_start"
            ]

            row_end = min(
                bounds["row_end"],
                coefficients.shape[0]
            )

            column_start = bounds[
                "column_start"
            ]

            column_end = min(
                bounds["column_end"],
                coefficients.shape[1]
            )

            for row in range(
                row_start,
                row_end
            ):

                for column in range(
                    column_start,
                    column_end
                ):

                    if not mask[
                        row,
                        column
                    ]:

                        continue

                    value = float(
                        coefficients[
                            row,
                            column
                        ]
                    )

                    candidates.append({

                        "subband":
                            subband,

                        "row":
                            row,

                        "column":
                            column,

                        "coefficient":
                            value,

                        "absolute_coefficient":
                            abs(value)
                    })

        candidates.sort(
            key=lambda item:
                item["absolute_coefficient"],
            reverse=True
        )

        if len(candidates) < payload_size:

            raise ValueError(
                f"Not enough embedding positions. "
                f"Required: {payload_size}, "
                f"Available: {len(candidates)}."
            )

        embedding_positions = (
            candidates[:payload_size]
        )

        request = {
            "payload_size": int(
                payload_size
            ),
            "payload_bits": [
                int(bit)
                for bit in payload_bits
            ],
            "embedding_subbands":
                embedding_subbands,
            "embedding_positions":
                embedding_positions,
            "status":
                "READY_FOR_EXISTING_EMBEDDER"
        }

        return request


    def apply_existing_embedding(
        self,
        workspace,
        embedding_request,
        embedding_function
    ):

        if embedding_function is None:

            raise ValueError(
                "The existing embedding function "
                "must be supplied."
            )

        if not callable(
            embedding_function
        ):

            raise TypeError(
                "embedding_function must be callable."
            )

        result = embedding_function(
            workspace=workspace,
            embedding_request=embedding_request
        )

        if not isinstance(
            result,
            dict
        ):

            raise ValueError(
                "Existing embedding function "
                "must return a dictionary."
            )

        if "modified" not in result:

            raise ValueError(
                "Embedding result must contain "
                "'modified' DWT coefficients."
            )

        for subband in [
            "LH",
            "HL",
            "HH"
        ]:

            if subband not in result[
                "modified"
            ]:

                raise ValueError(
                    f"Modified {subband} coefficients "
                    f"not returned."
                )

            workspace[
                "modified"
            ][subband] = np.asarray(
                result[
                    "modified"
                ][subband],
                dtype=np.float64
            )

        workspace[
            "embedding_mask"
        ] = {

            "LH":
                np.zeros(
                    workspace[
                        "original"
                    ]["LH"].shape,
                    dtype=bool
                ),

            "HL":
                np.zeros(
                    workspace[
                        "original"
                    ]["HL"].shape,
                    dtype=bool
                ),

            "HH":
                np.zeros(
                    workspace[
                        "original"
                    ]["HH"].shape,
                    dtype=bool
                )
        }

        for position in embedding_request[
            "embedding_positions"
        ]:

            subband = position[
                "subband"
            ]

            row = int(
                position["row"]
            )

            column = int(
                position["column"]
            )

            workspace[
                "embedding_mask"
            ][subband][
                row,
                column
            ] = True

        workspace[
            "payload_size"
        ] = int(
            embedding_request[
                "payload_size"
            ]
        )

        workspace[
            "embedding_count"
        ] = int(
            embedding_request[
                "payload_size"
            ]
        )

        workspace[
            "status"
        ] = "EMBEDDED"

        return workspace

    def verify_embedding_workspace(
        self,
        workspace
    ):

        self.validate_dwt_workspace(
            workspace
        )

        total_changed = 0
        total_masked = 0
        total_unmasked_changed = 0

        for subband in [
            "LH",
            "HL",
            "HH"
        ]:

            original = workspace[
                "original"
            ][subband]

            modified = workspace[
                "modified"
            ][subband]

            mask = workspace[
                "embedding_mask"
            ][subband]

            difference = np.abs(
                modified
                -
                original
            )

            changed = (
                difference > 1e-12
            )

            changed_count = int(
                np.count_nonzero(
                    changed
                )
            )

            masked_changed = int(
                np.count_nonzero(
                    changed & mask
                )
            )

            unmasked_changed = int(
                np.count_nonzero(
                    changed & ~mask
                )
            )

            mask_count = int(
                np.count_nonzero(
                    mask
                )
            )

            total_changed += changed_count
            total_masked += mask_count
            total_unmasked_changed += (
                unmasked_changed
            )

            # print(
            #     f"Embedding Verification | "
            #     f"{subband} | "
            #     f"Changed={changed_count} | "
            #     f"Masked={masked_changed} | "
            #     f"OutsideMask={unmasked_changed}"
            # )

        if total_changed == 0:

            raise ValueError(
                "Embedding produced no coefficient changes."
            )

        if total_unmasked_changed > 0:

            raise ValueError(
                "Embedding modified coefficients "
                "outside the embedding mask."
            )

        workspace[
            "embedding_count"
        ] = total_changed

        workspace[
            "masked_embedding_count"
        ] = total_masked

        workspace[
            "status"
        ] = "EMBEDDING_VERIFIED"

        return workspace


    def calculate_dwt_embedding_distortion(
        self,
        workspace
    ):

        self.validate_dwt_workspace(
            workspace
        )

        result = (
            self.calculate_workspace_distortion(
                workspace
            )
        )

        result[
            "payload_size"
        ] = int(
            workspace.get(
                "payload_size",
                0
            )
        )

        result[
            "embedding_count"
        ] = int(
            workspace.get(
                "embedding_count",
                0
            )
        )

        return result

    def load_ll_coefficients(self):

        dwt_dir = os.path.join(
            self.base_dir,
            "output",
            "dwt_decomposition"
        )

        ll_path = os.path.join(
            dwt_dir,
            "LL.npy"
        )

        if not os.path.exists(
            ll_path
        ):

            raise FileNotFoundError(
                ll_path
            )

        self.ll_coefficients = np.load(
            ll_path
        ).astype(
            np.float64
        )

        if self.ll_coefficients.ndim != 2:

            raise ValueError(
                "LL coefficient matrix must be 2D."
            )

        if not np.all(
            np.isfinite(
                self.ll_coefficients
            )
        ):

            raise ValueError(
                "LL contains invalid coefficient values."
            )

        if self.ll_coefficients.shape != self.dwt_shape:

            raise ValueError(
                "LL shape does not match LH/HL/HH."
            )

        print(
            f"LL Shape : "
            f"{self.ll_coefficients.shape}"
        )


    def build_complete_dwt(
        self,
        workspace
    ):

        self.validate_dwt_workspace(
            workspace
        )

        if not hasattr(
            self,
            "ll_coefficients"
        ):

            self.load_ll_coefficients()

        return {

            "LL":
                self.ll_coefficients.copy(),

            "LH":
                workspace[
                    "modified"
                ][
                    "LH"
                ].copy(),

            "HL":
                workspace[
                    "modified"
                ][
                    "HL"
                ].copy(),

            "HH":
                workspace[
                    "modified"
                ][
                    "HH"
                ].copy()
        }


    def inverse_dwt_reconstruction(
        self,
        workspace,
        wavelet="db2"
    ):

        try:

            import pywt

        except ImportError:

            raise ImportError(
                "PyWavelets is required for "
                "inverse DWT reconstruction."
            )

        dwt = self.build_complete_dwt(
            workspace
        )

        reconstructed = pywt.idwt2(
            (
                dwt["LL"],
                (
                    dwt["LH"],
                    dwt["HL"],
                    dwt["HH"]
                )
            ),
            wavelet
        )

        reconstructed = np.asarray(
            reconstructed,
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(
                reconstructed
            )
        ):

            raise ValueError(
                "Inverse DWT produced invalid values."
            )

        return reconstructed


    def clip_reconstructed_image(
        self,
        image,
        minimum=0.0,
        maximum=255.0
    ):

        image = np.asarray(
            image,
            dtype=np.float64
        )

        return np.clip(
            image,
            minimum,
            maximum
        )


    def save_reconstructed_image(
        self,
        image,
        scenario_id
    ):

        from PIL import Image

        image = self.clip_reconstructed_image(
            image
        )

        image_uint8 = np.rint(
            image
        ).astype(
            np.uint8
        )

        image_path = os.path.join(
            self.output_dir,
            f"{scenario_id}_post_embedding.png"
        )

        Image.fromarray(
            image_uint8
        ).save(
            image_path
        )

        return image_path


    def reconstruct_post_embedding_image(
        self,
        workspace,
        scenario_id
    ):

        reconstructed = (
            self.inverse_dwt_reconstruction(
                workspace
            )
        )

        image_path = (
            self.save_reconstructed_image(
                reconstructed,
                scenario_id
            )
        )

        return {

            "image":
                reconstructed,

            "image_path":
                image_path,

            "shape":
                reconstructed.shape
        }

    def validate_embedding_subbands(
        self,
        subbands=None
    ):

        if subbands is None:

            subbands = [
                "LH",
                "HL",
                "HH"
            ]

        allowed = {
            "LH",
            "HL",
            "HH"
        }

        invalid = [
            subband
            for subband in subbands
            if subband not in allowed
        ]

        if invalid:

            raise ValueError(
                f"Invalid embedding subbands: {invalid}"
            )

        return list(
            dict.fromkeys(
                subbands
            )
        )


    def get_embedding_subbands(self):

        return [
            "LH",
            "HL",
            "HH"
        ]


    def create_embedding_coefficient_view(
        self,
        region_id
    ):

        subbands = (
            self.get_embedding_subbands()
        )

        region_data = {}

        for subband in subbands:

            region_data[
                subband
            ] = self.extract_dwt_region(
                subband,
                region_id
            )

        return region_data


    def calculate_subband_statistics(
        self,
        coefficient_data
    ):

        statistics = {}

        for subband, coefficients in (
            coefficient_data.items()
        ):

            coefficients = np.asarray(
                coefficients,
                dtype=np.float64
            )

            statistics[
                subband
            ] = {

                "mean":
                    float(
                        np.mean(
                            coefficients
                        )
                    ),

                "std":
                    float(
                        np.std(
                            coefficients
                        )
                    ),

                "variance":
                    float(
                        np.var(
                            coefficients
                        )
                    ),

                "minimum":
                    float(
                        np.min(
                            coefficients
                        )
                    ),

                "maximum":
                    float(
                        np.max(
                            coefficients
                        )
                    ),

                "energy":
                    float(
                        np.sum(
                            coefficients ** 2
                        )
                    ),

                "absolute_mean":
                    float(
                        np.mean(
                            np.abs(
                                coefficients
                            )
                        )
                    ),

                "nonzero_count":
                    int(
                        np.count_nonzero(
                            coefficients
                        )
                    ),

                "coefficient_count":
                    int(
                        coefficients.size
                    )
            }

        return statistics

    def calculate_embedding_subband_changes(
        self,
        original,
        modified
    ):

        changes = {}

        for subband in [
            "LH",
            "HL",
            "HH"
        ]:

            if subband not in original:
                continue

            if subband not in modified:
                continue

            pre = np.asarray(
                original[subband],
                dtype=np.float64
            )

            post = np.asarray(
                modified[subband],
                dtype=np.float64
            )

            if pre.shape != post.shape:

                raise ValueError(
                    f"{subband} shape changed during embedding."
                )

            delta = (
                post - pre
            )

            absolute_delta = np.abs(
                delta
            )

            pre_energy = float(
                np.sum(
                    pre ** 2
                )
            )

            post_energy = float(
                np.sum(
                    post ** 2
                )
            )

            delta_energy = float(
                np.sum(
                    delta ** 2
                )
            )

            mse = float(
                np.mean(
                    delta ** 2
                )
            )

            mean_absolute_change = float(
                np.mean(
                    absolute_delta
                )
            )

            maximum_change = float(
                np.max(
                    absolute_delta
                )
            )

            changed_coefficients = int(
                np.count_nonzero(
                    absolute_delta > 1e-12
                )
            )

            coefficient_count = int(
                pre.size
            )

            change_density = (
                changed_coefficients
                /
                coefficient_count
                if coefficient_count > 0
                else 0.0
            )

            energy_change = (
                post_energy
                -
                pre_energy
            )

            relative_energy_change = (
                abs(energy_change)
                /
                (
                    abs(pre_energy)
                    +
                    1e-12
                )
            )

            changes[subband] = {

                "mse":
                    mse,

                "mean_absolute_change":
                    mean_absolute_change,

                "maximum_change":
                    maximum_change,

                "changed_coefficients":
                    changed_coefficients,

                "coefficient_count":
                    coefficient_count,

                "change_density":
                    float(
                        change_density
                    ),

                "pre_energy":
                    pre_energy,

                "post_energy":
                    post_energy,

                "energy_change":
                    float(
                        energy_change
                    ),

                "relative_energy_change":
                    float(
                        relative_energy_change
                    ),

                "delta_energy":
                    delta_energy
            }

        return changes


    def calculate_combined_embedding_distortion(
        self,
        subband_changes
    ):

        if not subband_changes:

            return {

                "combined_mse":
                    0.0,

                "combined_mean_change":
                    0.0,

                "combined_maximum_change":
                    0.0,

                "combined_change_density":
                    0.0,

                "combined_energy_change":
                    0.0,

                "combined_relative_energy_change":
                    0.0
            }

        mse_values = []
        mean_values = []
        maximum_values = []
        density_values = []
        energy_values = []
        relative_energy_values = []

        total_coefficients = 0
        total_changed = 0

        total_pre_energy = 0.0
        total_post_energy = 0.0

        for metrics in (
            subband_changes.values()
        ):

            mse_values.append(
                metrics["mse"]
            )

            mean_values.append(
                metrics[
                    "mean_absolute_change"
                ]
            )

            maximum_values.append(
                metrics[
                    "maximum_change"
                ]
            )

            density_values.append(
                metrics[
                    "change_density"
                ]
            )

            energy_values.append(
                metrics[
                    "energy_change"
                ]
            )

            relative_energy_values.append(
                metrics[
                    "relative_energy_change"
                ]
            )

            total_coefficients += (
                metrics[
                    "coefficient_count"
                ]
            )

            total_changed += (
                metrics[
                    "changed_coefficients"
                ]
            )

            total_pre_energy += (
                metrics[
                    "pre_energy"
                ]
            )

            total_post_energy += (
                metrics[
                    "post_energy"
                ]
            )

        combined_density = (
            total_changed
            /
            total_coefficients
            if total_coefficients > 0
            else 0.0
        )

        combined_energy_change = (
            total_post_energy
            -
            total_pre_energy
        )

        combined_relative_energy_change = (
            abs(
                combined_energy_change
            )
            /
            (
                abs(
                    total_pre_energy
                )
                +
                1e-12
            )
        )

        return {

            "combined_mse":
                float(
                    np.mean(
                        mse_values
                    )
                ),

            "combined_mean_change":
                float(
                    np.mean(
                        mean_values
                    )
                ),

            "combined_maximum_change":
                float(
                    np.max(
                        maximum_values
                    )
                ),

            "combined_change_density":
                float(
                    combined_density
                ),

            "combined_energy_change":
                float(
                    combined_energy_change
                ),

            "combined_relative_energy_change":
                float(
                    combined_relative_energy_change
                ),

            "total_changed_coefficients":
                total_changed,

            "total_coefficients":
                total_coefficients
        }


    def calculate_payload_distortion_rate(
        self,
        payload_size,
        distortion
    ):

        payload_size = float(
            payload_size
        )

        if payload_size <= 0:

            return {

                "distortion_per_bit":
                    0.0,

                "change_per_bit":
                    0.0,

                "energy_change_per_bit":
                    0.0
            }

        return {

            "distortion_per_bit":
                distortion[
                    "combined_mse"
                ]
                /
                payload_size,

            "change_per_bit":
                distortion[
                    "combined_mean_change"
                ]
                /
                payload_size,

            "energy_change_per_bit":
                abs(
                    distortion[
                        "combined_energy_change"
                    ]
                )
                /
                payload_size
        }


    def build_dwt_post_embedding_metrics(
        self,
        workspace,
        payload_size
    ):

        self.validate_dwt_workspace(
            workspace
        )

        original = workspace[
            "original"
        ]

        modified = workspace[
            "modified"
        ]

        subband_changes = (
            self.calculate_embedding_subband_changes(
                original,
                modified
            )
        )

        combined_distortion = (
            self.calculate_combined_embedding_distortion(
                subband_changes
            )
        )

        payload_metrics = (
            self.calculate_payload_distortion_rate(
                payload_size,
                combined_distortion
            )
        )

        return {

            "payload_size":
                int(payload_size),

            "subband_changes":
                subband_changes,

            "combined_distortion":
                combined_distortion,

            "payload_distortion":
                payload_metrics
        }

    def calculate_neighbor_dwt_distortion(
        self,
        neighbor_workspaces
    ):

        neighbor_results = {}

        for neighbor_name, workspace in (
            neighbor_workspaces.items()
        ):

            if workspace is None:

                neighbor_results[
                    neighbor_name
                ] = {

                    "available":
                        False,

                    "distortion":
                        None
                }

                continue

            self.validate_dwt_workspace(
                workspace
            )

            original = workspace[
                "original"
            ]

            modified = workspace[
                "modified"
            ]

            subband_changes = (
                self.calculate_embedding_subband_changes(
                    original,
                    modified
                )
            )

            combined = (
                self.calculate_combined_embedding_distortion(
                    subband_changes
                )
            )

            neighbor_results[
                neighbor_name
            ] = {

                "available":
                    True,

                "subband_changes":
                    subband_changes,

                "distortion":
                    combined
            }

        return neighbor_results


    def aggregate_neighbor_dwt_distortion(
        self,
        neighbor_distortion
    ):

        mse_values = []
        mean_changes = []
        maximum_changes = []
        densities = []
        energy_changes = []

        for result in (
            neighbor_distortion.values()
        ):

            if not result.get(
                "available",
                False
            ):
                continue

            distortion = result[
                "distortion"
            ]

            mse_values.append(
                distortion[
                    "combined_mse"
                ]
            )

            mean_changes.append(
                distortion[
                    "combined_mean_change"
                ]
            )

            maximum_changes.append(
                distortion[
                    "combined_maximum_change"
                ]
            )

            densities.append(
                distortion[
                    "combined_change_density"
                ]
            )

            energy_changes.append(
                distortion[
                    "combined_relative_energy_change"
                ]
            )

        if not mse_values:

            return {

                "neighbor_count":
                    0,

                "mean_mse":
                    0.0,

                "maximum_mse":
                    0.0,

                "mean_change":
                    0.0,

                "maximum_change":
                    0.0,

                "mean_change_density":
                    0.0,

                "maximum_change_density":
                    0.0,

                "mean_energy_change":
                    0.0,

                "maximum_energy_change":
                    0.0,

                "total_neighbor_impact":
                    0.0
            }

        total_neighbor_impact = (
            0.35 * float(
                np.mean(
                    mse_values
                )
            )
            +
            0.25 * float(
                np.mean(
                    mean_changes
                )
            )
            +
            0.20 * float(
                np.mean(
                    densities
                )
            )
            +
            0.20 * float(
                np.mean(
                    energy_changes
                )
            )
        )

        return {

            "neighbor_count":
                len(mse_values),

            "mean_mse":
                float(
                    np.mean(
                        mse_values
                    )
                ),

            "maximum_mse":
                float(
                    np.max(
                        mse_values
                    )
                ),

            "mean_change":
                float(
                    np.mean(
                        mean_changes
                    )
                ),

            "maximum_change":
                float(
                    np.max(
                        maximum_changes
                    )
                ),

            "mean_change_density":
                float(
                    np.mean(
                        densities
                    )
                ),

            "maximum_change_density":
                float(
                    np.max(
                        densities
                    )
                ),

            "mean_energy_change":
                float(
                    np.mean(
                        energy_changes
                    )
                ),

            "maximum_energy_change":
                float(
                    np.max(
                        energy_changes
                    )
                ),

            "total_neighbor_impact":
                float(
                    np.clip(
                        total_neighbor_impact,
                        0.0,
                        1.0
                    )
                )
        }


    def calculate_local_embedding_consequence(
        self,
        center_distortion,
        neighbor_aggregate
    ):

        center_mse = float(
            center_distortion[
                "combined_mse"
            ]
        )

        center_change = float(
            center_distortion[
                "combined_mean_change"
            ]
        )

        center_density = float(
            center_distortion[
                "combined_change_density"
            ]
        )

        center_energy = float(
            center_distortion[
                "combined_relative_energy_change"
            ]
        )

        neighbor_impact = float(
            neighbor_aggregate[
                "total_neighbor_impact"
            ]
        )

        center_impact = (
            0.35 * min(
                center_mse,
                1.0
            )
            +
            0.25 * min(
                center_change,
                1.0
            )
            +
            0.20 * min(
                center_density,
                1.0
            )
            +
            0.20 * min(
                center_energy,
                1.0
            )
        )

        total_consequence = (
            0.60 * center_impact
            +
            0.40 * neighbor_impact
        )

        total_consequence = float(
            np.clip(
                total_consequence,
                0.0,
                1.0
            )
        )

        return {

            "center_impact":
                float(
                    np.clip(
                        center_impact,
                        0.0,
                        1.0
                    )
                ),

            "neighbor_impact":
                float(
                    np.clip(
                        neighbor_impact,
                        0.0,
                        1.0
                    )
                ),

            "local_consequence":
                total_consequence,

            "local_safety":
                1.0 -
                total_consequence
        }


    def build_complete_dwt_consequence(
        self,
        center_workspace,
        neighbor_workspaces,
        payload_size
    ):

        center_metrics = (
            self.build_dwt_post_embedding_metrics(
                center_workspace,
                payload_size
            )
        )

        neighbor_distortion = (
            self.calculate_neighbor_dwt_distortion(
                neighbor_workspaces
            )
        )

        neighbor_aggregate = (
            self.aggregate_neighbor_dwt_distortion(
                neighbor_distortion
            )
        )

        local_consequence = (
            self.calculate_local_embedding_consequence(
                center_metrics[
                    "combined_distortion"
                ],
                neighbor_aggregate
            )
        )

        return {

            "payload_size":
                int(payload_size),

            "center":
                center_metrics,

            "neighbors":
                neighbor_distortion,

            "neighbor_aggregate":
                neighbor_aggregate,

            "local_consequence":
                local_consequence
        }

    def calculate_local_consequence_probability(
        self,
        local_consequence
    ):

        local_consequence = float(
            np.clip(
                local_consequence,
                0.0,
                1.0
            )
        )

        safe_probability = float(
            1.0
            /
            (
                1.0
                +
                np.exp(
                    12.0
                    *
                    (
                        local_consequence
                        -
                        0.30
                    )
                )
            )
        )

        risk_probability = float(
            1.0
            -
            safe_probability
        )

        return {

            "safety_confidence_score":
                safe_probability,

            "risk_confidence_score":
                risk_probability
        }

    def classify_local_embedding(
        self,
        local_safety,
        safe_probability
    ):

        if (
            local_safety >= 0.85
            and
            safe_probability >= 0.85
        ):

            return "PRIMARY"

        elif (
            local_safety >= 0.55
            and
            safe_probability >= 0.55
        ):

            return "SECONDARY"

        elif (
            local_safety >= 0.25
            and
            safe_probability >= 0.25
        ):

            return "TERTIARY"

        return "REJECT"

    def build_local_consequence_record(
        self,
        region_id,
        scenario_id,
        payload_size,
        dwt_consequence
    ):

        local = dwt_consequence[
            "local_consequence"
        ]

        local_consequence = float(
            local[
                "local_consequence"
            ]
        )

        local_safety = float(
            local[
                "local_safety"
            ]
        )

        probabilities = (
            self.calculate_local_consequence_probability(
                local_consequence
            )
        )

        classification = (
            self.classify_local_embedding(
                local_safety,
                probabilities[
                    "safety_confidence_score"
                ]
            )
        )

        center = dwt_consequence[
            "center"
        ]

        center_distortion = center[
            "combined_distortion"
        ]

        neighbor = dwt_consequence[
            "neighbor_aggregate"
        ]

        record = {

            "region_id":
                region_id,

            "scenario_id":
                scenario_id,

            "payload_size":
                int(payload_size),

            "center_impact":
                local[
                    "center_impact"
                ],

            "neighbor_impact":
                local[
                    "neighbor_impact"
                ],

            "local_consequence":
                local_consequence,

            "local_safety":
                local_safety,

            "safety_confidence_score":
                probabilities[
                    "safety_confidence_score"
                ],

            "risk_confidence_score":
                probabilities[
                    "risk_confidence_score"
                ],

            "classification":
                classification,

            "center_mse":
                center_distortion[
                    "combined_mse"
                ],

            "center_mean_change":
                center_distortion[
                    "combined_mean_change"
                ],

            "center_maximum_change":
                center_distortion[
                    "combined_maximum_change"
                ],

            "center_change_density":
                center_distortion[
                    "combined_change_density"
                ],

            "center_energy_change":
                center_distortion[
                    "combined_energy_change"
                ],

            "center_relative_energy_change":
                center_distortion[
                    "combined_relative_energy_change"
                ],

            "neighbor_count":
                neighbor[
                    "neighbor_count"
                ],

            "neighbor_mean_mse":
                neighbor[
                    "mean_mse"
                ],

            "neighbor_maximum_mse":
                neighbor[
                    "maximum_mse"
                ],

            "neighbor_mean_change":
                neighbor[
                    "mean_change"
                ],

            "neighbor_maximum_change":
                neighbor[
                    "maximum_change"
                ],

            "neighbor_mean_change_density":
                neighbor[
                    "mean_change_density"
                ],

            "neighbor_maximum_change_density":
                neighbor[
                    "maximum_change_density"
                ],

            "neighbor_mean_energy_change":
                neighbor[
                    "mean_energy_change"
                ],

            "neighbor_maximum_energy_change":
                neighbor[
                    "maximum_energy_change"
                ],

            "neighbor_total_impact":
                neighbor[
                    "total_neighbor_impact"
                ]
        }

        return record


    def calculate_scenario_gradient(
        self,
        previous_record,
        current_record
    ):

        if (
            previous_record is None
            or
            current_record is None
        ):

            return {

                "consequence_gradient":
                    np.nan,

                "safety_gradient":
                    np.nan,

                "center_gradient":
                    np.nan,

                "neighbor_gradient":
                    np.nan
            }

        previous_payload = float(
            previous_record[
                "payload_size"
            ]
        )

        current_payload = float(
            current_record[
                "payload_size"
            ]
        )

        payload_difference = (
            current_payload
            -
            previous_payload
        )

        if abs(
            payload_difference
        ) < 1e-12:

            return {

                "consequence_gradient":
                    0.0,

                "safety_gradient":
                    0.0,

                "center_gradient":
                    0.0,

                "neighbor_gradient":
                    0.0
            }

        return {

            "consequence_gradient":
                (
                    current_record[
                        "local_consequence"
                    ]
                    -
                    previous_record[
                        "local_consequence"
                    ]
                )
                /
                payload_difference,

            "safety_gradient":
                (
                    current_record[
                        "local_safety"
                    ]
                    -
                    previous_record[
                        "local_safety"
                    ]
                )
                /
                payload_difference,

            "center_gradient":
                (
                    current_record[
                        "center_impact"
                    ]
                    -
                    previous_record[
                        "center_impact"
                    ]
                )
                /
                payload_difference,

            "neighbor_gradient":
                (
                    current_record[
                        "neighbor_impact"
                    ]
                    -
                    previous_record[
                        "neighbor_impact"
                    ]
                )
                /
                payload_difference
        }


    def attach_gradient_to_record(
        self,
        record,
        previous_record
    ):

        gradients = (
            self.calculate_scenario_gradient(
                previous_record,
                record
            )
        )

        record.update(
            gradients
        )

        return record


    def calculate_region_safe_capacity_from_records(
        self,
        records,
        consequence_threshold=0.30,
        safety_threshold=0.70
    ):

        if not records:

            return {

                "maximum_safe_payload":
                    0,

                "safe_capacity_probability":
                    0.0,

                "status":
                    "NO_DATA"
            }

        ordered = sorted(
            records,
            key=lambda record:
                record[
                    "payload_size"
                ]
        )

        safe_records = [

            record

            for record in ordered

            if (
                record[
                    "local_consequence"
                ]
                <= consequence_threshold
                and
                record[
                    "local_safety"
                ]
                >= safety_threshold
            )
        ]

        if not safe_records:

            return {

                "maximum_safe_payload":
                    0,

                "safe_capacity_probability":
                    float(
                        max(
                            record[
                                "safe_probability"
                            ]
                            for record in ordered
                        )
                    ),

                "status":
                    "NO_SAFE_CAPACITY"
            }

        best = max(
            safe_records,
            key=lambda record:
                (
                    record[
                        "payload_size"
                    ],
                    record[
                        "safe_probability"
                    ]
                )
        )

        return {

            "maximum_safe_payload":
                int(
                    best[
                        "payload_size"
                    ]
                ),

            "safe_capacity_probability":
                float(
                    best[
                        "safe_probability"
                    ]
                ),

            "status":
                "SAFE_CAPACITY_FOUND"
        }

    def build_region_payload_curve(
        self,
        region_id,
        scenario_records
    ):

        ordered = sorted(
            scenario_records,
            key=lambda record:
                record["payload_size"]
        )

        curve = []

        previous_record = None

        for record in ordered:

            record = dict(record)

            gradients = (
                self.calculate_scenario_gradient(
                    previous_record,
                    record
                )
            )

            record.update(
                gradients
            )

            curve.append(
                record
            )

            previous_record = record

        return curve


    def optimize_region_payload(
        self,
        region_id,
        scenario_records,
        consequence_threshold=0.30,
        safety_threshold=0.70,
        gradient_threshold=0.01
    ):

        curve = (
            self.build_region_payload_curve(
                region_id,
                scenario_records
            )
        )

        if not curve:

            return {

                "region_id":
                    region_id,

                "maximum_safe_payload":
                    0,

                "optimal_payload":
                    0,

                "status":
                    "NO_DATA",

                "sensitivity_detected":
                    False,

                "payload_curve":
                    []
            }

        safe_records = [

            record

            for record in curve

            if (
                record[
                    "local_consequence"
                ]
                <= consequence_threshold

                and

                record[
                    "local_safety"
                ]
                >= safety_threshold
            )
        ]

        if not safe_records:

            return {

                "region_id":
                    region_id,

                "maximum_safe_payload":
                    0,

                "optimal_payload":
                    0,

                "status":
                    "NO_SAFE_PAYLOAD",

                "sensitivity_detected":
                    any(
                        (
                            not np.isnan(
                                record[
                                    "consequence_gradient"
                                ]
                            )
                            and
                            abs(
                                record[
                                    "consequence_gradient"
                                ]
                            )
                            >= gradient_threshold
                        )
                        for record in curve
                    ),

                "payload_curve":
                    curve
            }

        maximum_safe = max(
            safe_records,
            key=lambda record:
                record[
                    "payload_size"
                ]
        )

        sensitivity_detected = any(

            (
                not np.isnan(
                    record[
                        "consequence_gradient"
                    ]
                )

                and

                abs(
                    record[
                        "consequence_gradient"
                    ]
                )
                >= gradient_threshold
            )

            for record in curve
        )

        optimal = max(
            safe_records,
            key=lambda record:
                (
                    record[
                        "safe_probability"
                    ],
                    record[
                        "payload_size"
                    ]
                )
        )

        return {

            "region_id":
                region_id,

            "maximum_safe_payload":
                int(
                    maximum_safe[
                        "payload_size"
                    ]
                ),

            "optimal_payload":
                int(
                    optimal[
                        "payload_size"
                    ]
                ),

            "maximum_safe_probability":
                float(
                    maximum_safe[
                        "safe_probability"
                    ]
                ),

            "optimal_safe_probability":
                float(
                    optimal[
                        "safe_probability"
                    ]
                ),

            "maximum_safe_consequence":
                float(
                    maximum_safe[
                        "local_consequence"
                    ]
                ),

            "optimal_consequence":
                float(
                    optimal[
                        "local_consequence"
                    ]
                ),

            "sensitivity_detected":
                sensitivity_detected,

            "status":
                "OPTIMIZED",

            "payload_curve":
                curve
        }


    def build_all_region_payload_optimizations(
        self,
        records
    ):

        if not records:

            return []

        grouped = {}

        for record in records:

            region_id = record[
                "region_id"
            ]

            grouped.setdefault(
                region_id,
                []
            ).append(
                record
            )

        summaries = []

        for region_id, region_records in (
            grouped.items()
        ):

            summary = (
                self.optimize_region_payload(
                    region_id,
                    region_records
                )
            )

            summaries.append(
                summary
            )

        return summaries


    def save_region_optimization_results(
        self,
        summaries
    ):

        output_path = os.path.join(
            self.output_dir,
            "region_safe_capacity_results.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summaries,
                file,
                indent=4,
                default=str
            )

        print(
            f"Region Optimization Results Saved : "
            f"{output_path}"
        )

        return output_path

    def generate_embedding_scenarios(
        self,
        region_id,
        capacity=None
    ):

        row = self.get_region_row(
            region_id
        )

        if capacity is None:

            capacity = (
                self.find_region_capacity(
                    row
                )
            )

        capacity = max(
            1,
            int(
                round(
                    capacity
                )
            )
        )

        payloads = (
            self.build_gradient_payload_schedule(
                capacity,
                step_percent=1
            )
        )

        scenarios = []

        for index, payload_info in enumerate(
            payloads,
            start=1
        ):

            percentage = (
                payload_info["percentage"]
            )

            payload_size = (
                payload_info["payload_size"]
            )

            scenarios.append({

                "scenario_id":
                    (
                        f"{region_id}_"
                        f"P{percentage:03d}"
                    ),

                "region_id":
                    region_id,

                "payload_size":
                    int(
                        payload_size
                    ),

                "capacity":
                    int(
                        capacity
                    ),

                "capacity_ratio":
                    float(
                        percentage / 100.0
                    ),

                "payload_percentage":
                    int(
                        percentage
                    ),

                "scenario_index":
                    index,

                "scenario_status":
                    "PENDING"
            })

        return scenarios

    def build_all_embedding_scenarios(
        self
    ):

        region_ids = (
            self.dataset[
                "region_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        all_scenarios = []

        for region_id in region_ids:

            scenarios = (
                self.generate_embedding_scenarios(
                    region_id
                )
            )

            all_scenarios.extend(
                scenarios
            )

        self.scenario_path = os.path.join(
            self.output_dir,
            "embedding_scenarios.json"
        )

        with open(
            self.scenario_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                all_scenarios,
                file,
                indent=4
            )

        print(
            f"Regions Scheduled : "
            f"{len(region_ids)}"
        )

        print(
            f"Embedding Scenarios : "
            f"{len(all_scenarios)}"
        )

        print(
            f"Embedding Scenarios Saved : "
            f"{self.scenario_path}"
        )

        return all_scenarios


    def select_next_payload_by_gradient(
        self,
        results,
        capacity,
        minimum_step=1
    ):

        if not results:

            return {
                "next_payload":
                    max(
                        1,
                        int(
                            capacity * 0.10
                        )
                    ),

                "reason":
                    "INITIAL_PAYLOAD"
            }

        ordered = sorted(
            results,
            key=lambda result:
                result[
                    "payload_size"
                ]
        )

        last = ordered[-1]

        last_payload = int(
            last[
                "payload_size"
            ]
        )

        consequence = float(
            last[
                "local_consequence"
            ]
        )

        safety = float(
            last[
                "local_safety"
            ]
        )

        gradient = last.get(
            "consequence_gradient"
        )

        if (
            consequence >= 0.30
            or
            safety <= 0.70
        ):

            if len(ordered) >= 2:

                previous = ordered[-2]

                previous_payload = int(
                    previous[
                        "payload_size"
                    ]
                )

                lower = (
                    previous_payload
                )

                upper = (
                    last_payload
                )

                if upper - lower > minimum_step:

                    next_payload = (
                        lower
                        +
                        (
                            upper - lower
                        )
                        // 2
                    )

                    return {

                        "next_payload":
                            int(
                                next_payload
                            ),

                        "reason":
                            "GRADIENT_REFINEMENT"
                    }

            return {

                "next_payload":
                    last_payload,

                "reason":
                    "SAFE_LIMIT_REACHED"
            }

        if (
            gradient is not None
            and
            np.isfinite(
                gradient
            )
            and
            gradient > 0
        ):

            remaining = (
                capacity
                -
                last_payload
            )

            if remaining <= 0:

                return {

                    "next_payload":
                        last_payload,

                    "reason":
                        "CAPACITY_REACHED"
                }

            estimated_step = int(
                max(
                    minimum_step,
                    round(
                        min(
                            remaining,
                            max(
                                1,
                                0.10
                                *
                                capacity
                                /
                                max(
                                    gradient,
                                    1e-9
                                )
                            )
                        )
                    )
                )
            )

            next_payload = min(
                capacity,
                last_payload
                +
                estimated_step
            )

        else:

            next_payload = min(
                capacity,
                last_payload
                +
                max(
                    minimum_step,
                    int(
                        capacity * 0.10
                    )
                )
            )

        if next_payload <= last_payload:

            return {

                "next_payload":
                    last_payload,

                "reason":
                    "NO_FURTHER_PAYLOAD"
            }

        return {

            "next_payload":
                int(
                    next_payload
                ),

            "reason":
                "ADAPTIVE_GRADIENT_STEP"
        }


    def should_stop_payload_search(
        self,
        results,
        capacity,
        consequence_threshold=0.30,
        safety_threshold=0.70
    ):

        if not results:

            return False

        ordered = sorted(
            results,
            key=lambda result:
                result[
                    "payload_size"
                ]
        )

        latest = ordered[-1]

        payload = int(
            latest[
                "payload_size"
            ]
        )

        consequence = float(
            latest[
                "local_consequence"
            ]
        )

        safety = float(
            latest[
                "local_safety"
            ]
        )

        if payload >= int(
            capacity
        ):

            return True

        if (
            consequence >
            consequence_threshold
            and
            safety <
            safety_threshold
        ):

            if len(ordered) >= 2:

                previous = ordered[-2]

                previous_consequence = float(
                    previous[
                        "local_consequence"
                    ]
                )

                if (
                    consequence
                    >
                    previous_consequence
                ):

                    return True

        return False

    def prepare_single_scenario(
        self,
        scenario
    ):

        region_id = scenario[
            "region_id"
        ]

        payload_size = int(
            scenario[
                "payload_size"
            ]
        )

        center_workspace = (
            self.create_dwt_embedding_workspaces(
                region_id
            )
        )

        neighbors = self.get_neighbors(
            region_id
        )

        neighbor_workspaces = {}

        for neighbor_name, neighbor_id in (
            neighbors.items()
        ):

            if neighbor_id is None:

                neighbor_workspaces[
                    neighbor_name
                ] = None

                continue

            neighbor_workspaces[
                neighbor_name
            ] = (
                self.create_dwt_embedding_workspaces(
                    neighbor_id
                )
            )

        self.validate_dwt_workspace(
            center_workspace
        )

        for workspace in (
            neighbor_workspaces.values()
        ):

            if workspace is not None:

                self.validate_dwt_workspace(
                    workspace
                )

        capacity = float(
            scenario[
                "capacity"
            ]
        )

        if payload_size > capacity:

            raise ValueError(
                f"Payload {payload_size} exceeds "
                f"capacity {capacity} for region "
                f"{region_id}."
            )

        payload_bits = (
            self.prepare_payload_bits(
                payload_size
            )
        )

        selected_band = str(
            scenario["band"]
        ).upper()

        embedding_request = (
            self.create_embedding_request_data(
                center_workspace,
                payload_size,
                payload_bits,
                embedding_subbands=[
                    selected_band
                ]
            )
        )

        return {
            "scenario":
                scenario,

            "center_workspace":
                center_workspace,

            "neighbor_workspaces":
                neighbor_workspaces,

            "payload_bits":
                payload_bits,

            "embedding_request":
                embedding_request,

            "status":
                "READY"
        }

    def get_post_embedding_feature_input(
        self,
        workspace
    ):

        self.validate_dwt_workspace(
            workspace
        )

        post_coefficients = {

            "LH":
                workspace[
                    "modified"
                ][
                    "LH"
                ].copy(),

            "HL":
                workspace[
                    "modified"
                ][
                    "HL"
                ].copy(),

            "HH":
                workspace[
                    "modified"
                ][
                    "HH"
                ].copy()
        }

        post_statistics = (
            self.calculate_subband_statistics(
                post_coefficients
            )
        )

        distortion = (
            self.calculate_dwt_embedding_distortion(
                workspace
            )
        )

        return {

            "coefficients":
                post_coefficients,

            "statistics":
                post_statistics,

            "distortion":
                distortion
        }


    def get_neighbor_post_embedding_feature_input(
        self,
        neighbor_workspaces
    ):

        result = {}

        for neighbor_name, workspace in (
            neighbor_workspaces.items()
        ):

            if workspace is None:

                result[
                    neighbor_name
                ] = None

                continue

            result[
                neighbor_name
            ] = (
                self.get_post_embedding_feature_input(
                    workspace
                )
            )

        return result


    def build_actual_post_embedding_record(
        self,
        prepared_scenario
    ):

        scenario = prepared_scenario[
            "scenario"
        ]

        center_workspace = (
            prepared_scenario[
                "center_workspace"
            ]
        )

        neighbor_workspaces = (
            prepared_scenario[
                "neighbor_workspaces"
            ]
        )

        center_post = (
            self.get_post_embedding_feature_input(
                center_workspace
            )
        )

        neighbor_post = (
            self.get_neighbor_post_embedding_feature_input(
                neighbor_workspaces
            )
        )

        center_distortion = (
            center_post[
                "distortion"
            ]
        )

        neighbor_distortions = {}

        for neighbor_name, data in (
            neighbor_post.items()
        ):

            if data is None:

                neighbor_distortions[
                    neighbor_name
                ] = None

            else:

                neighbor_distortions[
                    neighbor_name
                ] = data[
                    "distortion"
                ]

        neighbor_aggregate = (
            self.aggregate_neighbor_dwt_distortion(
                {
                    name:
                    {
                        "available":
                            distortion is not None,

                        "distortion":
                            distortion
                    }

                    for name, distortion
                    in neighbor_distortions.items()
                }
            )
        )

        local_consequence = (
            self.calculate_local_embedding_consequence(
                center_distortion[
                    "combined_distortion"
                ],
                neighbor_aggregate
            )
        )

        probabilities = (
            self.calculate_local_consequence_probability(
                local_consequence[
                    "local_consequence"
                ]
            )
        )

        classification = (
            self.classify_local_embedding(
                local_consequence[
                    "local_safety"
                ],
                probabilities[
                    "safe_probability"
                ]
            )
        )

        record = {

            "scenario_id":
                scenario[
                    "scenario_id"
                ],

            "region_id":
                scenario[
                    "region_id"
                ],

            "payload_size":
                scenario[
                    "payload_size"
                ],

            "capacity":
                scenario[
                    "capacity"
                ],

            "capacity_ratio":
                scenario[
                    "capacity_ratio"
                ],

            "center_distortion":
                center_distortion,

            "neighbor_distortion":
                neighbor_distortions,

            "neighbor_aggregate":
                neighbor_aggregate,

            "local_consequence":
                local_consequence,

            "safety_confidence_score":
                probabilities[
                    "safety_confidence_score"
                ],

            "risk_confidence_score":
                probabilities[
                    "risk_confidence_score"
                ],

            "classification":
                classification,

            "status":
                "POST_EMBEDDING_ANALYZED"
        }

        return record


    def save_single_scenario_result(
        self,
        record
    ):

        scenario_id = str(
            record[
                "scenario_id"
            ]
        )

        safe_name = (
            scenario_id
            .replace(
                "/",
                "_"
            )
            .replace(
                "\\",
                "_"
            )
        )

        path = os.path.join(
            self.output_dir,
            f"{safe_name}_post_result.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                record,
                file,
                indent=4,
                default=str
            )

        return path

    def build_post_feature_vector(
        self,
        region_id,
        scenario,
        center_post,
        neighbor_post,
        center_pre_features,
        neighbor_pre_features
    ):

        feature_vector = {}

        feature_vector[
            "region_id"
        ] = region_id

        feature_vector[
            "scenario_id"
        ] = scenario[
            "scenario_id"
        ]

        feature_vector[
            "payload_size"
        ] = scenario[
            "payload_size"
        ]

        feature_vector[
            "capacity"
        ] = scenario[
            "capacity"
        ]

        feature_vector[
            "capacity_ratio"
        ] = scenario[
            "capacity_ratio"
        ]

        for feature, value in (
            center_pre_features.items()
        ):

            try:

                value = float(value)

                if np.isfinite(value):

                    feature_vector[
                        f"pre_center_{feature}"
                    ] = value

            except (
                ValueError,
                TypeError
            ):

                continue

        center_statistics = (
            center_post.get(
                "statistics",
                {}
            )
        )

        for subband, statistics in (
            center_statistics.items()
        ):

            for metric, value in (
                statistics.items()
            ):

                try:

                    value = float(value)

                    if np.isfinite(value):

                        feature_vector[
                            f"post_center_{subband}_{metric}"
                        ] = value

                except (
                    ValueError,
                    TypeError
                ):

                    continue

        center_distortion = (
            center_post.get(
                "distortion",
                {}
            )
        )

        for group, values in (
            center_distortion.items()
        ):

            if isinstance(
                values,
                dict
            ):

                for metric, value in (
                    values.items()
                ):

                    try:

                        value = float(value)

                        if np.isfinite(value):

                            feature_vector[
                                f"post_center_{group}_{metric}"
                            ] = value

                    except (
                        ValueError,
                        TypeError
                    ):

                        continue

            else:

                try:

                    value = float(values)

                    if np.isfinite(value):

                        feature_vector[
                            f"post_center_{group}"
                        ] = value

                except (
                    ValueError,
                    TypeError
                ):

                    continue

        for neighbor_name, neighbor_data in (
            neighbor_post.items()
        ):

            if neighbor_data is None:
                continue

            neighbor_statistics = (
                neighbor_data.get(
                    "statistics",
                    {}
                )
            )

            for subband, statistics in (
                neighbor_statistics.items()
            ):

                for metric, value in (
                    statistics.items()
                ):

                    try:

                        value = float(value)

                        if np.isfinite(value):

                            feature_vector[
                                f"post_{neighbor_name}_{subband}_{metric}"
                            ] = value

                    except (
                        ValueError,
                        TypeError
                    ):

                        continue

            neighbor_distortion = (
                neighbor_data.get(
                    "distortion",
                    {}
                )
            )

            for group, values in (
                neighbor_distortion.items()
            ):

                if isinstance(
                    values,
                    dict
                ):

                    for metric, value in (
                        values.items()
                    ):

                        try:

                            value = float(value)

                            if np.isfinite(value):

                                feature_vector[
                                    f"post_{neighbor_name}_{group}_{metric}"
                                ] = value

                        except (
                            ValueError,
                            TypeError
                        ):

                            continue

                else:

                    try:

                        value = float(values)

                        if np.isfinite(value):

                            feature_vector[
                                f"post_{neighbor_name}_{group}"
                            ] = value

                    except (
                        ValueError,
                        TypeError
                    ):

                        continue

        return feature_vector


    def calculate_pre_post_feature_differences(
        self,
        feature_vector
    ):

        difference_features = {}

        pre_features = {}

        post_features = {}

        for feature, value in (
            feature_vector.items()
        ):

            if feature.startswith(
                "pre_center_"
            ):

                original_name = feature[
                    len("pre_center_"):
                ]

                pre_features[
                    original_name
                ] = value

            elif feature.startswith(
                "post_center_"
            ):

                post_name = feature[
                    len("post_center_"):
                ]

                post_features[
                    post_name
                ] = value

        for feature, post_value in (
            post_features.items()
        ):

            if feature not in pre_features:
                continue

            pre_value = pre_features[
                feature
            ]

            try:

                pre_value = float(
                    pre_value
                )

                post_value = float(
                    post_value
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            if not (
                np.isfinite(
                    pre_value
                )
                and
                np.isfinite(
                    post_value
                )
            ):
                continue

            difference_features[
                f"delta_{feature}"
            ] = (
                post_value
                -
                pre_value
            )

            difference_features[
                f"relative_delta_{feature}"
            ] = (
                abs(
                    post_value
                    -
                    pre_value
                )
                /
                (
                    abs(
                        pre_value
                    )
                    +
                    1e-9
                )
            )

        return difference_features


    def merge_post_feature_vector(
        self,
        feature_vector
    ):

        differences = (
            self.calculate_pre_post_feature_differences(
                feature_vector
            )
        )

        feature_vector.update(
            differences
        )

        return feature_vector


    def calculate_post_feature_count(
        self,
        feature_vector
    ):

        numeric_features = 0

        for key, value in (
            feature_vector.items()
        ):

            if key in {
                "region_id",
                "scenario_id"
            }:
                continue

            try:

                value = float(value)

            except (
                ValueError,
                TypeError
            ):

                continue

            if np.isfinite(value):

                numeric_features += 1

        return numeric_features

    def build_post_embedding_dataset_row(
        self,
        scenario,
        center_post,
        neighbor_post,
        center_pre_features,
        neighbor_pre_features
    ):

        region_id = scenario[
            "region_id"
        ]

        feature_vector = (
            self.build_post_feature_vector(
                region_id=region_id,
                scenario=scenario,
                center_post=center_post,
                neighbor_post=neighbor_post,
                center_pre_features=center_pre_features,
                neighbor_pre_features=neighbor_pre_features
            )
        )

        feature_vector = (
            self.merge_post_feature_vector(
                feature_vector
            )
        )

        center_distortion = (
            center_post.get(
                "distortion",
                {}
            )
        )

        neighbor_distortions = {}

        for neighbor_name, neighbor_data in (
            neighbor_post.items()
        ):

            if neighbor_data is None:
                continue

            neighbor_distortions[
                neighbor_name
            ] = neighbor_data.get(
                "distortion",
                {}
            )

        neighbor_aggregate = (
            self.aggregate_neighbor_dwt_distortion(
                {
                    name: {
                        "available": True,
                        "distortion": distortion
                    }
                    for name, distortion
                    in neighbor_distortions.items()
                }
            )
        )

        local_consequence = (
            self.calculate_local_embedding_consequence(
                center_distortion.get(
                    "combined_distortion",
                    {}
                ),
                neighbor_aggregate
            )
        )

        probabilities = (
            self.calculate_local_consequence_probability(
                local_consequence[
                    "local_consequence"
                ]
            )
        )

        classification = (
            self.classify_local_embedding(
                local_consequence[
                    "local_safety"
                ],
                probabilities[
                    "safe_probability"
                ]
            )
        )

        feature_vector.update({

            "center_impact":
                local_consequence[
                    "center_impact"
                ],

            "neighbor_impact":
                local_consequence[
                    "neighbor_impact"
                ],

            "local_consequence":
                local_consequence[
                    "local_consequence"
                ],

            "local_safety":
                local_consequence[
                    "local_safety"
                ],

            "safe_probability":
                probabilities[
                    "safe_probability"
                ],

            "risk_probability":
                probabilities[
                    "risk_probability"
                ],

            "safety_confidence_score":
                probabilities[
                    "safety_confidence_score"
                ],

            "risk_confidence_score":
                probabilities[
                    "risk_confidence_score"
                ],

            "consequence_class":
                classification,

            "neighbor_count":
                neighbor_aggregate[
                    "neighbor_count"
                ],

            "neighbor_mean_mse":
                neighbor_aggregate[
                    "mean_mse"
                ],

            "neighbor_maximum_mse":
                neighbor_aggregate[
                    "maximum_mse"
                ],

            "neighbor_mean_change":
                neighbor_aggregate[
                    "mean_change"
                ],

            "neighbor_maximum_change":
                neighbor_aggregate[
                    "maximum_change"
                ],

            "neighbor_mean_change_density":
                neighbor_aggregate[
                    "mean_change_density"
                ],

            "neighbor_maximum_change_density":
                neighbor_aggregate[
                    "maximum_change_density"
                ],

            "neighbor_mean_energy_change":
                neighbor_aggregate[
                    "mean_energy_change"
                ],

            "neighbor_maximum_energy_change":
                neighbor_aggregate[
                    "maximum_energy_change"
                ],

            "neighbor_total_impact":
                neighbor_aggregate[
                    "total_neighbor_impact"
                ]
        })

        feature_vector[
            "post_feature_count"
        ] = self.calculate_post_feature_count(
            feature_vector
        )

        return feature_vector


    def append_post_embedding_feature_row(
        self,
        row
    ):

        if self.post_dataset is None:

            self.post_dataset = pd.DataFrame()

        dataframe = pd.DataFrame(
            [row]
        )

        self.post_dataset = pd.concat(
            [
                self.post_dataset,
                dataframe
            ],
            ignore_index=True
        )


    def save_post_embedding_feature_dataset(
        self
    ):

        if (
            self.post_dataset is None
            or
            self.post_dataset.empty
        ):

            raise ValueError(
                "Post-embedding feature dataset is empty."
            )

        self.post_dataset.to_csv(
            self.post_dataset_path,
            index=False
        )

        print(
            f"Post Dataset Rows     : "
            f"{len(self.post_dataset)}"
        )

        print(
            f"Post Dataset Features : "
            f"{len(self.post_dataset.columns)}"
        )

        print(
            f"Post Dataset Saved    : "
            f"{self.post_dataset_path}"
        )


    def get_post_dataset_statistics(
        self
    ):

        if (
            self.post_dataset is None
            or
            self.post_dataset.empty
        ):

            return {}

        numeric_columns = (
            self.post_dataset.select_dtypes(
                include=[np.number]
            ).columns
        )

        statistics = {

            "rows":
                int(
                    len(
                        self.post_dataset
                    )
                ),

            "features":
                int(
                    len(
                        self.post_dataset.columns
                    )
                ),

            "numeric_features":
                int(
                    len(
                        numeric_columns
                    )
                ),

            "regions":
                int(
                    self.post_dataset[
                        "region_id"
                    ].nunique()
                )
                if "region_id"
                in self.post_dataset.columns
                else 0,

            "scenarios":
                int(
                    self.post_dataset[
                        "scenario_id"
                    ].nunique()
                )
                if "scenario_id"
                in self.post_dataset.columns
                else 0
        }

        return statistics

    def build_post_dataset_from_result(
        self,
        scenario,
        embedding_result
    ):

        self.validate_actual_embedding_output(
            embedding_result
        )

        center_post_features = (
            embedding_result[
                "post_features"
            ]
        )

        neighbor_post_features = (
            embedding_result.get(
                "neighbor_post_features",
                {}
            )
        )

        center_pre_features = (
            self.load_region_pre_features(
                scenario[
                    "region_id"
                ]
            )
        )

        neighbor_pre_features = (
            self.load_neighbor_pre_features(
                scenario[
                    "region_id"
                ]
            )
        )

        row = (
            self.build_post_embedding_dataset_row(
                scenario=scenario,
                center_post=center_post_features,
                neighbor_post=neighbor_post_features,
                center_pre_features=center_pre_features,
                neighbor_pre_features=neighbor_pre_features
            )
        )

        return row


    def validate_post_dataset(
        self
    ):

        if (
            self.post_dataset is None
            or
            self.post_dataset.empty
        ):

            raise ValueError(
                "Post-embedding dataset is empty."
            )

        required_columns = [

            "region_id",

            "scenario_id",

            "payload_size",

            "capacity",

            "capacity_ratio",

            "center_impact",

            "neighbor_impact",

            "local_consequence",

            "local_safety",

            "safe_probability",

            "risk_probability",

            "consequence_class"
        ]

        missing = [

            column

            for column in required_columns

            if column
            not in self.post_dataset.columns
        ]

        if missing:

            raise ValueError(
                "Post dataset missing columns: "
                + ", ".join(missing)
            )

        numeric_columns = [

            "payload_size",

            "capacity",

            "capacity_ratio",

            "center_impact",

            "neighbor_impact",

            "local_consequence",

            "local_safety",

            "safe_probability",

            "risk_probability"
        ]

        for column in numeric_columns:

            values = pd.to_numeric(
                self.post_dataset[
                    column
                ],
                errors="coerce"
            )

            if values.isna().any():

                raise ValueError(
                    f"Invalid numeric values in "
                    f"{column}."
                )

            if not np.all(
                np.isfinite(
                    values.to_numpy()
                )
            ):

                raise ValueError(
                    f"Non-finite values in "
                    f"{column}."
                )

        probability_sum = (
            self.post_dataset[
                "safe_probability"
            ]
            +
            self.post_dataset[
                "risk_probability"
            ]
        )

        if not np.allclose(
            probability_sum.to_numpy(),
            1.0,
            atol=1e-6
        ):

            raise ValueError(
                "Safe and risk probabilities "
                "do not sum to 1."
            )

        return True


    def save_validated_post_dataset(
        self
    ):

        self.validate_post_dataset()

        validated_path = os.path.join(
            self.output_dir,
            "post_embedding_dataset_validated.csv"
        )

        self.post_dataset.to_csv(
            validated_path,
            index=False
        )

        print(
            f"Validated Dataset Saved : "
            f"{validated_path}"
        )

        return validated_path


    def build_region_training_targets(
        self
    ):

        if (
            self.post_dataset is None
            or
            self.post_dataset.empty
        ):

            raise ValueError(
                "Post dataset is empty."
            )

        targets = self.post_dataset[
            [
                "region_id",
                "scenario_id",
                "payload_size",
                "local_consequence",
                "local_safety",
                "safe_probability",
                "risk_probability",
                "consequence_class"
            ]
        ].copy()

        targets[
            "is_safe"
        ] = (
            targets[
                "consequence_class"
            ]
            !=
            "REJECT"
        ).astype(
            int
        )

        targets[
            "is_rejected"
        ] = (
            targets[
                "consequence_class"
            ]
            ==
            "REJECT"
        ).astype(
            int
        )

        target_path = os.path.join(
            self.output_dir,
            "post_embedding_targets.csv"
        )

        targets.to_csv(
            target_path,
            index=False
        )

        print(
            f"Post-Embedding Targets Saved : "
            f"{target_path}"
        )

        return targets


    def build_region_summary_from_post_dataset(
        self
    ):

        if (
            self.post_dataset is None
            or
            self.post_dataset.empty
        ):

            return pd.DataFrame()

        summary = (
            self.post_dataset
            .groupby(
                "region_id",
                as_index=False
            )
            .agg({

                "payload_size":
                    "max",

                "local_consequence":
                    "min",

                "local_safety":
                    "max",

                "safe_probability":
                    "max",

                "risk_probability":
                    "min"
            })
        )

        summary.rename(
            columns={
                "payload_size":
                    "maximum_tested_payload",

                "local_consequence":
                    "minimum_consequence",

                "local_safety":
                    "maximum_safety",

                "safe_probability":
                    "maximum_safe_probability",

                "risk_probability":
                    "minimum_risk_probability"
            },
            inplace=True
        )

        summary[
            "recommended_for_ga"
        ] = (
            (
                summary[
                    "maximum_safe_probability"
                ] >= 0.25
            )
            &
            (
                summary[
                    "maximum_safety"
                ] >= 0.25
            )
        )

        summary_path = os.path.join(
            self.output_dir,
            "region_post_embedding_summary.csv"
        )

        summary.to_csv(
            summary_path,
            index=False
        )

        print(
            f"Region Summary Saved : "
            f"{summary_path}"
        )

        return summary

    def build_post_embedding_dataset(
        self,
        completed_results
    ):

        if not completed_results:

            raise ValueError(
                "No completed post-embedding results."
            )

        self.post_dataset = pd.DataFrame()

        for result in completed_results:

            scenario = result.get(
                "scenario"
            )

            if scenario is None:

                scenario = {

                    "scenario_id":
                        result.get(
                            "scenario_id"
                        ),

                    "region_id":
                        result.get(
                            "region_id"
                        ),

                    "payload_size":
                        result.get(
                            "payload_size"
                        ),

                    "capacity":
                        result.get(
                            "capacity"
                        ),

                    "capacity_ratio":
                        result.get(
                            "capacity_ratio"
                        )
                }

            embedding_result = result.get(
                "embedding_result"
            )

            if embedding_result is None:

                continue

            try:

                row = (
                    self.build_post_dataset_from_result(
                        scenario,
                        embedding_result
                    )
                )

                self.append_post_embedding_feature_row(
                    row
                )

            except Exception as error:

                print(
                    f"Post Dataset Row Failed | "
                    f"Region {scenario.get('region_id')} | "
                    f"{error}"
                )

        if self.post_dataset.empty:

            raise ValueError(
                "No valid rows were generated "
                "for the post-embedding dataset."
            )

        self.save_post_embedding_feature_dataset()

        self.validate_post_dataset()

        return self.post_dataset


    def calculate_dataset_target_distribution(
        self
    ):

        if (
            self.post_dataset is None
            or
            self.post_dataset.empty
        ):

            return {}

        distribution = {}

        if "consequence_class" in (
            self.post_dataset.columns
        ):

            class_counts = (
                self.post_dataset[
                    "consequence_class"
                ]
                .value_counts()
                .to_dict()
            )

            distribution[
                "consequence_class"
            ] = {
                str(key): int(value)
                for key, value
                in class_counts.items()
            }

        if "safe_probability" in (
            self.post_dataset.columns
        ):

            safe_values = pd.to_numeric(
                self.post_dataset[
                    "safe_probability"
                ],
                errors="coerce"
            )

            distribution[
                "safe_probability"
            ] = {

                "mean":
                    float(
                        safe_values.mean()
                    ),

                "minimum":
                    float(
                        safe_values.min()
                    ),

                "maximum":
                    float(
                        safe_values.max()
                    )
            }

        if "risk_probability" in (
            self.post_dataset.columns
        ):

            risk_values = pd.to_numeric(
                self.post_dataset[
                    "risk_probability"
                ],
                errors="coerce"
            )

            distribution[
                "risk_probability"
            ] = {

                "mean":
                    float(
                        risk_values.mean()
                    ),

                "minimum":
                    float(
                        risk_values.min()
                    ),

                "maximum":
                    float(
                        risk_values.max()
                    )
            }

        return distribution


    def save_dataset_metadata(
        self
    ):

        if (
            self.post_dataset is None
            or
            self.post_dataset.empty
        ):

            raise ValueError(
                "Post dataset is empty."
            )

        numeric_columns = (
            self.post_dataset
            .select_dtypes(
                include=[np.number]
            )
            .columns
            .tolist()
        )

        metadata = {

            "dataset_name":
                "Post Embedding Consequence Dataset",

            "rows":
                int(
                    len(
                        self.post_dataset
                    )
                ),

            "columns":
                int(
                    len(
                        self.post_dataset.columns
                    )
                ),

            "numeric_features":
                len(
                    numeric_columns
                ),

            "regions":
                int(
                    self.post_dataset[
                        "region_id"
                    ].nunique()
                ),

            "scenarios":
                int(
                    self.post_dataset[
                        "scenario_id"
                    ].nunique()
                ),

            "embedding_subbands":
                [
                    "LH",
                    "HL",
                    "HH"
                ],

            "ignored_subbands":
                [
                    "LL"
                ],

            "target_variables":
                [
                    "local_consequence",
                    "local_safety",
                    "safe_probability",
                    "risk_probability",
                    "consequence_class"
                ],

            "dataset_distribution":
                self.calculate_dataset_target_distribution(),

            "numeric_feature_names":
                numeric_columns
        }

        metadata_path = os.path.join(
            self.output_dir,
            "post_embedding_dataset_metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        print(
            f"Dataset Metadata Saved : "
            f"{metadata_path}"
        )

        return metadata


    def finalize_post_embedding_dataset(
        self
    ):

        self.validate_post_dataset()

        self.save_post_embedding_feature_dataset()

        self.build_region_summary_from_post_dataset()

        self.build_region_training_targets()

        metadata = (
            self.save_dataset_metadata()
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "POST-EMBEDDING DATASET PREPARATION COMPLETED"
        )

        print(
            "=" * 80
        )

        print(
            f"Rows                : "
            f"{len(self.post_dataset)}"
        )

        print(
            f"Features            : "
            f"{len(self.post_dataset.columns)}"
        )

        print(
            f"Regions             : "
            f"{metadata['regions']}"
        )

        print(
            f"Scenarios           : "
            f"{metadata['scenarios']}"
        )

        print(
            "Embedding Subbands  : LH, HL, HH"
        )

        print(
            "LL                  : IGNORED"
        )

        print(
            "Targets             : "
            "Consequence + Safety + Probability + Class"
        )

        print(
            "=" * 80
        )

        return self.post_dataset

    def load_embedding_scenarios(
        self
    ):

        if not hasattr(
            self,
            "scenario_path"
        ):

            self.scenario_path = os.path.join(
                self.output_dir,
                "embedding_scenarios.json"
            )

        if not os.path.exists(
            self.scenario_path
        ):

            return self.build_all_embedding_scenarios()

        with open(
            self.scenario_path,
            "r",
            encoding="utf-8"
        ) as file:

            scenarios = json.load(
                file
            )

        if not isinstance(
            scenarios,
            list
        ):

            raise ValueError(
                "embedding_scenarios.json must contain a list."
            )

        print(
            f"Embedding Scenarios Loaded : "
            f"{len(scenarios)}"
        )

        return scenarios


    def group_scenarios_by_region(
        self,
        scenarios
    ):

        grouped = {}

        for scenario in scenarios:

            region_id = scenario[
                "region_id"
            ]

            grouped.setdefault(
                region_id,
                []
            ).append(
                scenario
            )

        for region_id in grouped:

            grouped[
                region_id
            ].sort(
                key=lambda item:
                    item[
                        "payload_size"
                    ]
            )

        return grouped


    def create_post_embedding_experiment_plan(
        self,
        scenarios
    ):

        grouped = (
            self.group_scenarios_by_region(
                scenarios
            )
        )

        plan = []

        for region_id, region_scenarios in (
            grouped.items()
        ):

            for scenario in region_scenarios:

                plan.append({

                    "experiment_id":
                        f"EXP_{scenario['scenario_id']}",

                    "scenario_id":
                        scenario[
                            "scenario_id"
                        ],

                    "region_id":
                        region_id,

                    "payload_size":
                        int(
                            scenario[
                                "payload_size"
                            ]
                        ),

                    "capacity":
                        int(
                            scenario[
                                "capacity"
                            ]
                        ),

                    "capacity_ratio":
                        float(
                            scenario[
                                "capacity_ratio"
                            ]
                        ),

                    "status":
                        "PENDING"
                })

        plan_path = os.path.join(
            self.output_dir,
            "post_embedding_experiment_plan.json"
        )

        with open(
            plan_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                plan,
                file,
                indent=4
            )

        print(
            f"Experiment Plan Generated : "
            f"{len(plan)}"
        )

        print(
            f"Experiment Plan Saved : "
            f"{plan_path}"
        )

        return plan


    def validate_experiment_plan(
        self,
        plan
    ):

        if not plan:

            raise ValueError(
                "Experiment plan is empty."
            )

        required = [

            "experiment_id",

            "scenario_id",

            "region_id",
            "band",

            "payload_size",

            "payload_percentage",

            "capacity",

            "capacity_ratio"
        ]

        for index, experiment in enumerate(
            plan
        ):

            missing = [

                key

                for key in required

                if key not in experiment
            ]

            if missing:

                raise ValueError(
                    f"Experiment {index} missing: "
                    f"{missing}"
                )

            payload = int(
                experiment[
                    "payload_size"
                ]
            )

            capacity = int(
                experiment[
                    "capacity"
                ]
            )

            if payload <= 0:

                raise ValueError(
                    f"Invalid payload in experiment "
                    f"{experiment['experiment_id']}"
                )

            if capacity <= 0:

                raise ValueError(
                    f"Invalid capacity in experiment "
                    f"{experiment['experiment_id']}"
                )

            if payload > capacity:

                raise ValueError(
                    f"Payload exceeds capacity in "
                    f"{experiment['experiment_id']}"
                )
            percentage = float(
                experiment[
                    "payload_percentage"
                ]
            )

            if not (
                1.0
                <=
                percentage
                <=
                100.0
            ):

                raise ValueError(
                    f"Invalid payload percentage "
                    f"{percentage} in "
                    f"{experiment['experiment_id']}"
                )

            expected_payload = int(
                round(
                    capacity
                    *
                    percentage
                    /
                    100.0
                )
            )

            expected_payload = max(
                1,
                min(
                    expected_payload,
                    capacity
                )
            )

            if payload != expected_payload:

                raise ValueError(
                    f"Payload/percentage mismatch in "
                    f"{experiment['experiment_id']}: "
                    f"payload={payload}, "
                    f"expected={expected_payload}, "
                    f"percentage={percentage}"
                )

        return True


    def prepare_post_embedding_experiments(
        self
    ):

        experiments = []

        region_ids = (
            self.dataset[
                "region_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        for region_id in region_ids:

            row = self.get_region_row(
                region_id
            )

            selected_band = str(
                row["band"]
            ).upper()

            if selected_band not in {
                "LH",
                "HL",
                "HH"
            }:
                raise ValueError(
                    f"Invalid embedding band "
                    f"{selected_band} for {region_id}"
                )

            capacity = (
                self.find_region_capacity(
                    row
                )
            )

            capacity = max(
                1,
                int(
                    round(
                        capacity
                    )
                )
            )

            for percentage in range(
                1,
                101
            ):

                payload_size = int(
                    round(
                        capacity
                        *
                        percentage
                        /
                        100.0
                    )
                )

                payload_size = max(
                    1,
                    min(
                        payload_size,
                        capacity
                    )
                )

                experiment_id = (
                    f"{region_id}_"
                    f"P{percentage:03d}"
                )

                experiments.append({

                    "experiment_id":
                        experiment_id,

                    "scenario_id":
                        experiment_id,

                    "region_id":
                        region_id,

                    "band":
                        selected_band,

                    "payload_percentage":
                        percentage,

                    "payload_size":
                        payload_size,

                    "capacity":
                        capacity,

                    "capacity_ratio":
                        percentage / 100.0
                })

        if not experiments:

            raise ValueError(
                "No post-embedding experiments generated."
            )

        # -------------------------------------------------
        # FINAL STRUCTURE VALIDATION
        # -------------------------------------------------

        df = pd.DataFrame(
            experiments
        )

        counts = (
            df.groupby(
                "region_id"
            )
            .size()
        )

        invalid_regions = (
            counts[
                counts != 100
            ]
        )

        if not invalid_regions.empty:

            raise ValueError(
                "Some regions do not have exactly "
                "100 scenarios: "
                f"{invalid_regions.to_dict()}"
            )

        return experiments

    def initialize_post_embedding_pipeline(self):

        self.post_dataset = pd.DataFrame()

        self.post_embedding_results = []

        self.experiment_plan = []

        self.initialize_post_embedding_analysis()

        self.experiment_plan = (
            self.prepare_post_embedding_experiments()
        )

        return self.experiment_plan

    def execute_single_post_embedding_experiment(
        self,
        experiment,
        embedding_function
    ):

        required_fields = [
            "experiment_id",
            "region_id",
            "payload_size",
            "band",
            "payload_percentage",
            "capacity",
            "capacity_ratio"
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in experiment
        ]

        if missing_fields:

            raise ValueError(
                "Experiment is missing required fields: "
                f"{missing_fields}"
            )

        scenario = {

            "experiment_id":
                experiment[
                    "experiment_id"
                ],

            "scenario_id":
                experiment[
                    "scenario_id"
                ],

            "region_id":
                experiment[
                    "region_id"
                ],
            "band":
                experiment[
                    "band"
                ],

            "payload_size":
                int(
                    experiment[
                        "payload_size"
                    ]
                ),

            "payload_percentage":
                float(
                    experiment[
                        "payload_percentage"
                    ]
                ),

            "capacity":
                int(
                    experiment[
                        "capacity"
                    ]
                ),

            "capacity_ratio":
                float(
                    experiment[
                        "capacity_ratio"
                    ]
                )
        }

        expected_payload = int(
            round(
                scenario["capacity"]
                *
                scenario["payload_percentage"]
                /
                100.0
            )
        )

        expected_payload = max(
            1,
            min(
                expected_payload,
                scenario["capacity"]
            )
        )

        if (
            scenario["payload_size"]
            !=
            expected_payload
        ):

            raise ValueError(
                "Payload size does not match "
                "capacity × percentage."
            )

        prepared = (
            self.prepare_single_scenario(
                scenario
            )
        )

        center_workspace = (
            self.apply_existing_embedding(
                prepared[
                    "center_workspace"
                ],
                prepared[
                    "embedding_request"
                ],
                embedding_function
            )
        )

        self.verify_embedding_workspace(
            center_workspace
        )

        actual_analysis = (
            self.calculate_actual_post_embedding_analysis(
                center_workspace,
                scenario[
                    "region_id"
                ]
            )
        )

        if actual_analysis is None:

            raise ValueError(
                "Actual post-embedding analysis "
                "returned None."
            )

        post_center = actual_analysis[
            "center_post_features"
        ]

        post_neighbors = actual_analysis[
            "neighbor_post_features"
        ]

        pre_image = actual_analysis.get(
            "pre_image"
        )

        post_image = actual_analysis.get(
            "reconstructed_image"
        )

        if pre_image is None:

            raise ValueError(
                "Actual analysis did not provide "
                "pre_image."
            )

        if post_image is None:

            raise ValueError(
                "Actual analysis did not provide "
                "reconstructed_image."
            )

        pre_features = (
            self.load_region_pre_features(
                scenario[
                    "region_id"
                ]
            )
        )

        neighbor_pre_features = (
            self.load_neighbor_pre_features(
                scenario[
                    "region_id"
                ]
            )
        )

        record = (
            self.build_post_embedding_record(
                scenario=scenario,
                post_features=post_center,
                neighbor_post_features=post_neighbors,
                pre_arrays={
                    "center":
                        pre_image
                },
                post_arrays={
                    "center":
                        post_image
                },
                spatial_neighbor_metrics=
                    actual_analysis[
                        "spatial_neighbor_impact"
                    ]
            )
        )

        record[
            "scenario"
        ] = scenario

        record[
            "payload_percentage"
        ] = float(
            scenario[
                "payload_percentage"
            ]
        )

        record[
            "safe_probability"
        ] = float(
            record[
                "safety_confidence_score"
            ]
        )

        record[
            "risk_probability"
        ] = float(
            record[
                "risk_confidence_score"
            ]
        )

        record[
            "predicted_class"
        ] = record[
            "predicted_class"
        ]

        record[
            "total_consequence"
        ] = float(
            record[
                "total_consequence"
            ]
        )

        record[
            "total_safety"
        ] = float(
            record[
                "total_safety"
            ]
        )

        record[
            "embedding_result"
        ] = {

            "workspace":
                center_workspace,

            "center_workspace":
                center_workspace,

            "post_features":
                post_center,

            "neighbor_post_features":
                post_neighbors,

            "embedding_request":
                prepared[
                    "embedding_request"
                ],

            "actual_post_analysis":
                actual_analysis
        }

        record[
            "embedding_status"
        ] = "COMPLETED"

        record[
            "post_analysis_status"
        ] = "COMPLETED"

        return record

    def execute_post_embedding_experiments(
        self,
        embedding_function,
        experiments=None
    ):

        if experiments is None:

            experiments = (
                self.experiment_plan
            )

        if not experiments:

            raise ValueError(
                "No post-embedding experiments available."
            )

        self.validate_experiment_plan(
            experiments
        )

        # -------------------------------------------------
        # GROUP SCENARIOS BY REGION
        # -------------------------------------------------

        region_scenarios = {}

        for experiment in experiments:

            region_id = experiment[
                "region_id"
            ]

            region_scenarios.setdefault(
                region_id,
                []
            ).append(
                experiment
            )

        # -------------------------------------------------
        # VALIDATE 1% -> 100%
        # -------------------------------------------------

        for region_id, scenarios in (
            region_scenarios.items()
        ):

            percentages = sorted(
                int(
                    round(
                        scenario[
                            "payload_percentage"
                        ]
                    )
                )
                for scenario in scenarios
            )

            expected_percentages = list(
                range(
                    1,
                    101
                )
            )

            if percentages != expected_percentages:

                raise ValueError(
                    f"{region_id} does not contain "
                    "exactly 1%-100% scenarios."
                )

        results = []

        failed_results = []
        no_safe_results = []

        # -------------------------------------------------
        # REGION-BY-REGION
        # -------------------------------------------------

        total_regions = len(
            region_scenarios
        )

        for region_index, (
            region_id,
            scenarios
        ) in enumerate(
            region_scenarios.items(),
            start=1
        ):

            # -------------------------------------------------
            # START AT 100%
            # GO DOWN TO 1%
            # -------------------------------------------------

            scenarios = sorted(
                scenarios,
                key=lambda scenario:
                    float(
                        scenario[
                            "payload_percentage"
                        ]
                    ),
                reverse=True
            )

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"REGION {region_index}/"
                f"{total_regions} : "
                f"{region_id}"
            )

            print(
                "Searching payload "
                "from 100% -> 1%"
            )

            print(
                "-" * 80
            )

            region_final_result = None

            previous_analysis_result = None

            # -------------------------------------------------
            # DESCENDING SEARCH
            # -------------------------------------------------

            for scenario in scenarios:

                percentage = float(
                    scenario[
                        "payload_percentage"
                    ]
                )

                try:

                    result = (
                        self.execute_single_post_embedding_experiment(
                            scenario,
                            embedding_function
                        )
                    )

                    if previous_analysis_result is not None:

                        result[
                            "payload_gradient"
                        ] = self.calculate_payload_gradient(
                            previous_analysis_result,
                            result
                        )

                        result[
                            "quality_gradient"
                        ] = self.calculate_quality_gradient(
                            previous_analysis_result,
                            result
                        )

                    else:

                        result[
                            "payload_gradient"
                        ] = None

                        result[
                            "quality_gradient"
                        ] = None

                    previous_analysis_result = result

                    # -------------------------------------------------
                    # GET ACTUAL ANALYSIS RESULT
                    # -------------------------------------------------

                    predicted_class = result.get(
                        "predicted_class"
                    )

                    # -------------------------------------------------
                    # SAFE CONDITION
                    # -------------------------------------------------

                    is_safe = (
                        predicted_class
                        in {
                            "PRIMARY",
                            "SECONDARY",
                            "TERTIARY"
                        }
                    )

                    print(
                        f"Region={region_id} | "
                        f"Payload={percentage:.0f}% | "
                        f"Payload={scenario['payload_size']} bits | "
                        f"Safe={is_safe}"
                    )

                    # -------------------------------------------------
                    # FIRST SAFE RESULT = MAXIMUM SAFE PAYLOAD
                    # -------------------------------------------------

                    if is_safe:

                        region_final_result = result

                        results.append(
                            region_final_result
                        )

                        print(
                            f"FINAL SAFE PAYLOAD | "
                            f"Region={region_id} | "
                            f"Percentage={percentage:.0f}% | "
                            f"Payload={scenario['payload_size']} bits"
                        )

                        break

                except Exception as error:

                    failed_results.append({

                        "experiment_id":
                            scenario.get(
                                "experiment_id"
                            ),

                        "scenario_id":
                            scenario.get(
                                "scenario_id"
                            ),

                        "region_id":
                            region_id,

                        "payload_percentage":
                            percentage,

                        "payload_size":
                            scenario.get(
                                "payload_size"
                            ),

                        "error":
                            str(error)
                    })

                    print(
                        f"Region={region_id} | "
                        f"Payload={percentage:.0f}% | "
                        f"FAILED : {error}"
                    )

                    continue

            # -------------------------------------------------
            # NO SAFE PAYLOAD
            # -------------------------------------------------

            if region_final_result is None:

                print(
                    f"NO SAFE PAYLOAD FOUND | "
                    f"Region={region_id}"
                )

                no_safe_results.append({

                    "experiment_id":
                        None,

                    "scenario_id":
                        None,

                    "region_id":
                        region_id,

                    "payload_percentage":
                        0.0,

                    "payload_size":
                        0,

                    "status":
                        "NO_SAFE_PAYLOAD_FOUND"
                })

        # -------------------------------------------------
        # DO NOT ACCEPT PARTIAL FAILURES
        # -------------------------------------------------

        if failed_results:

            failed_path = os.path.join(
                self.output_dir,
                "failed_post_embedding_experiments.csv"
            )

            pd.DataFrame(
                failed_results
            ).to_csv(
                failed_path,
                index=False
            )

            print(
                f"\nFailed experiments : "
                f"{len(failed_results)}"
            )

            print(
                f"Failure CSV : "
                f"{failed_path}"
            )

        # -------------------------------------------------
        # SAVE ONLY EXECUTED RESULTS
        # -------------------------------------------------

        self.post_embedding_results = (
            results
        )

        result_path = os.path.join(
            self.output_dir,
            "post_embedding_experiment_results.json"
        )

        with open(
            result_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                default=str
            )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "POST-EMBEDDING SEARCH COMPLETED"
        )

        print(
            "=" * 80
        )

        print(
            f"Regions processed : "
            f"{total_regions}"
        )

        print(
            f"Final region results : "
            f"{len(results)}"
        )
        print(
            f"Failed scenarios : "
            f"{len(failed_results)}"
        )

        print(
            f"Results saved : "
            f"{result_path}"
        )

        print(
            "=" * 80
        )

        return results

    def build_post_dataset_from_experiments(
        self,
        results
    ):

        if not results:

            raise ValueError(
                "No experiment results available."
            )

        rows = []

        for result in results:

            scenario = result[
                "scenario"
            ]

            embedding_result = result[
                "embedding_result"
            ]

            try:

                row = (
                    self.build_post_dataset_from_result(
                        scenario,
                        embedding_result
                    )
                )

                rows.append(
                    row
                )

            except Exception as error:

                print(
                    f"Dataset row failed | "
                    f"{scenario['scenario_id']} | "
                    f"{error}"
                )

        if not rows:

            raise ValueError(
                "No valid post-embedding dataset rows generated."
            )

        self.post_dataset = pd.DataFrame(
            rows
        )

        self.save_post_embedding_feature_dataset()

        return self.post_dataset

    def save_post_embedding_experiment_results(
        self,
        results
    ):

        if not results:

            raise ValueError(
                "No experiment results available."
            )

        output_path = os.path.join(
            self.output_dir,
            "post_embedding_experiment_results.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                default=str
            )

        return output_path

    def build_post_embedding_region_curves(
        self,
        results
    ):

        if not results:
            return {}

        grouped = {}

        for result in results:

            scenario = result.get(
                "scenario",
                {}
            )

            region_id = scenario.get(
                "region_id"
            )

            if region_id is None:
                continue

            record = {

                "scenario_id":
                    scenario.get(
                        "scenario_id"
                    ),

                "region_id":
                    region_id,

                "payload_size":
                    int(
                        scenario.get(
                            "payload_size",
                            0
                        )
                    ),

                "capacity":
                    int(
                        scenario.get(
                            "capacity",
                            0
                        )
                    ),

                "capacity_ratio":
                    float(
                        scenario.get(
                            "capacity_ratio",
                            0.0
                        )
                    ),

                "center_impact":
                    float(
                        result.get(
                            "center_consequence",
                            0.0
                        )
                    ),

                "neighbor_impact":
                    float(
                        result.get(
                            "neighbor_feature_consequence",
                            0.0
                        )
                    ),

                "local_consequence":
                    float(
                        result.get(
                            "total_consequence",
                            1.0
                        )
                    ),

                "local_safety":
                    float(
                        result.get(
                            "total_safety",
                            0.0
                        )
                    ),

                "safe_probability":
                    float(
                        result.get(
                            "safe_probability",
                            0.0
                        )
                    ),

                "risk_probability":
                    float(
                        result.get(
                            "risk_probability",
                            1.0
                        )
                    ),

                "classification":
                    result.get(
                        "predicted_class"
                    )
            }

            grouped.setdefault(
                region_id,
                []
            ).append(
                record
            )

        for region_id in grouped:

            grouped[
                region_id
            ].sort(
                key=lambda record:
                    record[
                        "payload_size"
                    ]
            )

        return grouped

    def calculate_post_embedding_gradients(
        self,
        region_curves
    ):

        gradient_results = {}

        for region_id, records in (
            region_curves.items()
        ):

            previous = None

            gradient_records = []

            for record in records:

                current = dict(
                    record
                )

                if previous is None:

                    current[
                        "consequence_gradient"
                    ] = np.nan

                    current[
                        "safety_gradient"
                    ] = np.nan

                    current[
                        "center_impact_gradient"
                    ] = np.nan

                    current[
                        "neighbor_impact_gradient"
                    ] = np.nan

                    current[
                        "safe_probability_gradient"
                    ] = np.nan

                else:

                    payload_delta = (
                        current[
                            "payload_size"
                        ]
                        -
                        previous[
                            "payload_size"
                        ]
                    )

                    if payload_delta == 0:

                        payload_delta = 1

                    current[
                        "consequence_gradient"
                    ] = (
                        current[
                            "local_consequence"
                        ]
                        -
                        previous[
                            "local_consequence"
                        ]
                    ) / payload_delta

                    current[
                        "safety_gradient"
                    ] = (
                        current[
                            "local_safety"
                        ]
                        -
                        previous[
                            "local_safety"
                        ]
                    ) / payload_delta

                    current[
                        "center_impact_gradient"
                    ] = (
                        current[
                            "center_impact"
                        ]
                        -
                        previous[
                            "center_impact"
                        ]
                    ) / payload_delta

                    current[
                        "neighbor_impact_gradient"
                    ] = (
                        current[
                            "neighbor_impact"
                        ]
                        -
                        previous[
                            "neighbor_impact"
                        ]
                    ) / payload_delta

                    current[
                        "safe_probability_gradient"
                    ] = (
                        current[
                            "safe_probability"
                        ]
                        -
                        previous[
                            "safe_probability"
                        ]
                    ) / payload_delta

                gradient_records.append(
                    current
                )

                previous = current

            gradient_results[
                region_id
            ] = gradient_records

        return gradient_results


    def calculate_gradient_safe_capacity(
        self,
        gradient_results,
        consequence_threshold=0.30,
        safety_threshold=0.25
    ):

        capacity_results = {}

        for region_id, records in (
            gradient_results.items()
        ):

            if not records:

                capacity_results[
                    region_id
                ] = {

                    "safe_capacity":
                        0,

                    "safe_probability":
                        0.0,

                    "status":
                        "NO_DATA"
                }

                continue

            safe_records = [

                record

                for record in records

                if (
                    record[
                        "local_consequence"
                    ]
                    <= consequence_threshold

                    and

                    record[
                        "local_safety"
                    ]
                    >= safety_threshold

                    and

                    record[
                        "safe_probability"
                    ]
                    >= safety_threshold
                )
            ]

            if not safe_records:

                capacity_results[
                    region_id
                ] = {

                    "safe_capacity":
                        0,

                    "safe_probability":
                        max(
                            record[
                                "safe_probability"
                            ]
                            for record in records
                        ),

                    "status":
                        "REJECT"
                }

                continue

            best = max(
                safe_records,
                key=lambda record:
                    (
                        record[
                            "payload_size"
                        ],
                        record[
                            "safe_probability"
                        ]
                    )
            )

            capacity_results[
                region_id
            ] = {

                "safe_capacity":
                    int(
                        best[
                            "payload_size"
                        ]
                    ),

                "safe_probability":
                    float(
                        best[
                            "safe_probability"
                        ]
                    ),

                "local_consequence":
                    float(
                        best[
                            "local_consequence"
                        ]
                    ),

                "local_safety":
                    float(
                        best[
                            "local_safety"
                        ]
                    ),

                "center_impact":
                    float(
                        best[
                            "center_impact"
                        ]
                    ),

                "neighbor_impact":
                    float(
                        best[
                            "neighbor_impact"
                        ]
                    ),

                "status":
                    "SAFE"
            }

        return capacity_results

    def build_post_embedding_ga_input(
        self,
        results
    ):

        if not results:

            raise ValueError(
                "No post-embedding results available."
            )

        rows = []

        for result in results:

            scenario = result.get(
                "scenario",
                {}
            )

            rows.append({

                "scenario_id":
                    scenario.get(
                        "scenario_id"
                    ),

                "region_id":
                    scenario.get(
                        "region_id"
                    ),

                "payload_size":
                    float(
                        scenario.get(
                            "payload_size",
                            0
                        )
                    ),

                "capacity":
                    float(
                        scenario.get(
                            "capacity",
                            0
                        )
                    ),

                "capacity_ratio":
                    float(
                        scenario.get(
                            "capacity_ratio",
                            0.0
                        )
                    ),

                "center_impact":
                    float(
                        result.get(
                            "center_consequence",
                            0.0
                        )
                    ),

                "neighbor_impact":
                    float(
                        result.get(
                            "neighbor_feature_consequence",
                            0.0
                        )
                    ),

                "local_consequence":
                    float(
                        result.get(
                            "total_consequence",
                            1.0
                        )
                    ),

                "local_safety":
                    float(
                        result.get(
                            "total_safety",
                            0.0
                        )
                    ),

                "safe_probability":
                    float(
                        result.get(
                            "safe_probability",
                            0.0
                        )
                    ),

                "risk_probability":
                    float(
                        result.get(
                            "risk_probability",
                            1.0
                        )
                    ),

                "center_mse":
                    float(
                        result.get(
                            "mse",
                            0.0
                        )
                    ),

                "center_mean_change":
                    float(
                        result.get(
                            "mean_change",
                            0.0
                        )
                    ),

                "center_maximum_change":
                    float(
                        result.get(
                            "maximum_change",
                            0.0
                        )
                    ),

                "center_change_density":
                    float(
                        result.get(
                            "changed_density",
                            0.0
                        )
                    ),

                "neighbor_mean_mse":
                    float(
                        result.get(
                            "spatial_neighbor_mean_impact",
                            0.0
                        )
                    ),

                "neighbor_maximum_mse":
                    float(
                        result.get(
                            "spatial_neighbor_max_impact",
                            0.0
                        )
                    ),

                "neighbor_mean_change":
                    float(
                        result.get(
                            "neighbor_mean_impact",
                            0.0
                        )
                    ),

                "neighbor_maximum_change":
                    float(
                        result.get(
                            "neighbor_max_impact",
                            0.0
                        )
                    ),

                "neighbor_total_impact":
                    float(
                        result.get(
                            "neighbor_total_impact",
                            0.0
                        )
                    )
            })

        dataframe = pd.DataFrame(
            rows
        )

        output_path = os.path.join(
            self.output_dir,
            "post_embedding_ga_input.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

        self.ga_input_dataset = dataframe

        return dataframe

    def calculate_ga_region_fitness(
        self,
        record
    ):

        safe_probability = float(
            record.get(
                "safe_probability",
                0.0
            )
        )

        local_safety = float(
            record.get(
                "local_safety",
                0.0
            )
        )

        consequence = float(
            record.get(
                "local_consequence",
                1.0
            )
        )

        center_impact = float(
            record.get(
                "center_impact",
                1.0
            )
        )

        neighbor_impact = float(
            record.get(
                "neighbor_impact",
                1.0
            )
        )

        capacity_ratio = float(
            record.get(
                "capacity_ratio",
                0.0
            )
        )

        distortion = float(
            record.get(
                "center_mse",
                0.0
            )
        )

        distortion_penalty = min(
            1.0,
            max(
                0.0,
                distortion
            )
        )

        fitness = (

            0.30
            * safe_probability

            +

            0.20
            * local_safety

            +

            0.15
            * (1.0 - consequence)

            +

            0.10
            * (1.0 - center_impact)

            +

            0.10
            * (1.0 - neighbor_impact)

            +

            0.10
            * capacity_ratio

            +

            0.05
            * (1.0 - distortion_penalty)
        )

        return float(
            np.clip(
                fitness,
                0.0,
                1.0
            )
        )

    def rank_post_embedding_regions_for_ga(
        self,
        results
    ):

        if not results:

            return pd.DataFrame()

        rows = []

        for result in results:

            scenario = result.get(
                "scenario",
                {}
            )

            record = {

                "scenario_id":
                    scenario.get(
                        "scenario_id"
                    ),

                "region_id":
                    scenario.get(
                        "region_id"
                    ),

                "payload_size":
                    scenario.get(
                        "payload_size",
                        0
                    ),

                "capacity":
                    scenario.get(
                        "capacity",
                        0
                    ),

                "capacity_ratio":
                    scenario.get(
                        "capacity_ratio",
                        0.0
                    ),

                "safe_probability":
                    result.get(
                        "safe_probability",
                        0.0
                    ),

                "risk_probability":
                    result.get(
                        "risk_probability",
                        1.0
                    ),

                "local_consequence":
                    result.get(
                        "total_consequence",
                        1.0
                    ),

                "local_safety":
                    result.get(
                        "total_safety",
                        0.0
                    ),

                "center_impact":
                    result.get(
                        "center_consequence",
                        1.0
                    ),

                "neighbor_impact":
                    result.get(
                        "neighbor_feature_consequence",
                        1.0
                    )
            }

            record[
                "ga_fitness"
            ] = (
                self.calculate_ga_region_fitness(
                    record
                )
            )

            rows.append(
                record
            )

        dataframe = pd.DataFrame(
            rows
        )

        dataframe.sort_values(
            by=[
                "ga_fitness",
                "safe_probability",
                "local_safety"
            ],
            ascending=[
                False,
                False,
                False
            ],
            inplace=True
        )

        dataframe.reset_index(
            drop=True,
            inplace=True
        )

        dataframe[
            "ga_rank"
        ] = (
            np.arange(
                len(dataframe)
            )
            + 1
        )

        output_path = os.path.join(
            self.output_dir,
            "post_embedding_ga_region_ranking.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

        return dataframe

    def build_final_region_decision_table(
        self,
        ga_ranking
    ):

        if ga_ranking is None or ga_ranking.empty:

            raise ValueError(
                "GA ranking is empty."
            )

        dataframe = ga_ranking.copy()

        dataframe[
            "decision"
        ] = "REJECT"

        primary_mask = (
            (dataframe["safe_probability"] >= 0.85)
            &
            (dataframe["local_safety"] >= 0.85)
            &
            (dataframe["local_consequence"] <= 0.15)
        )

        secondary_mask = (
            (dataframe["safe_probability"] >= 0.55)
            &
            (dataframe["local_safety"] >= 0.55)
            &
            (dataframe["local_consequence"] <= 0.45)
        )

        tertiary_mask = (
            (dataframe["safe_probability"] >= 0.25)
            &
            (dataframe["local_safety"] >= 0.25)
            &
            (dataframe["local_consequence"] <= 0.75)
        )

        dataframe.loc[
            tertiary_mask,
            "decision"
        ] = "TERTIARY"

        dataframe.loc[
            secondary_mask,
            "decision"
        ] = "SECONDARY"

        dataframe.loc[
            primary_mask,
            "decision"
        ] = "PRIMARY"

        dataframe[
            "embedding_allowed"
        ] = (
            dataframe[
                "decision"
            ] != "REJECT"
        )

        dataframe[
            "safe_capacity"
        ] = 0

        dataframe.loc[
            dataframe[
                "embedding_allowed"
            ],
            "safe_capacity"
        ] = dataframe.loc[
            dataframe[
                "embedding_allowed"
            ],
            "payload_size"
        ]

        dataframe.sort_values(
            by=[
                "embedding_allowed",
                "ga_fitness",
                "safe_probability",
                "local_safety"
            ],
            ascending=[
                False,
                False,
                False,
                False
            ],
            inplace=True
        )

        dataframe.reset_index(
            drop=True,
            inplace=True
        )

        dataframe[
            "final_rank"
        ] = (
            np.arange(
                len(dataframe)
            )
            + 1
        )

        return dataframe


    def save_final_region_decisions(
        self,
        decision_table
    ):

        if (
            decision_table is None
            or
            decision_table.empty
        ):

            raise ValueError(
                "Final region decision table is empty."
            )

        output_path = os.path.join(
            self.output_dir,
            "final_region_embedding_decisions.csv"
        )

        decision_table.to_csv(
            output_path,
            index=False
        )

        print(
            f"Final Region Decisions Saved : "
            f"{output_path}"
        )

        return output_path


    def build_primary_secondary_tertiary_lists(
        self,
        decision_table
    ):

        if (
            decision_table is None
            or
            decision_table.empty
        ):

            return {

                "PRIMARY": [],
                "SECONDARY": [],
                "TERTIARY": [],
                "REJECT": []
            }

        result = {

            "PRIMARY": [],
            "SECONDARY": [],
            "TERTIARY": [],
            "REJECT": []
        }

        for decision in result:

            subset = decision_table[
                decision_table[
                    "decision"
                ] == decision
            ]

            for _, row in subset.iterrows():

                result[
                    decision
                ].append({

                    "region_id":
                        row[
                            "region_id"
                        ],

                    "scenario_id":
                        row[
                            "scenario_id"
                        ],

                    "payload_size":
                        int(
                            row[
                                "payload_size"
                            ]
                        ),

                    "safe_capacity":
                        int(
                            row[
                                "safe_capacity"
                            ]
                        ),

                    "safe_probability":
                        float(
                            row[
                                "safe_probability"
                            ]
                        ),

                    "risk_probability":
                        float(
                            row[
                                "risk_probability"
                            ]
                        ),

                    "local_safety":
                        float(
                            row[
                                "local_safety"
                            ]
                        ),

                    "local_consequence":
                        float(
                            row[
                                "local_consequence"
                            ]
                        ),

                    "center_impact":
                        float(
                            row[
                                "center_impact"
                            ]
                        ),

                    "neighbor_impact":
                        float(
                            row[
                                "neighbor_impact"
                            ]
                        ),

                    "ga_fitness":
                        float(
                            row[
                                "ga_fitness"
                            ]
                        ),

                    "ga_rank":
                        int(
                            row[
                                "ga_rank"
                            ]
                        ),

                    "final_rank":
                        int(
                            row[
                                "final_rank"
                            ]
                        )
                })

        output_path = os.path.join(
            self.output_dir,
            "region_priority_groups.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4
            )

        print(
            f"Region Priority Groups Saved : "
            f"{output_path}"
        )

        print(
            f"PRIMARY   : "
            f"{len(result['PRIMARY'])}"
        )

        print(
            f"SECONDARY : "
            f"{len(result['SECONDARY'])}"
        )

        print(
            f"TERTIARY  : "
            f"{len(result['TERTIARY'])}"
        )

        print(
            f"REJECT    : "
            f"{len(result['REJECT'])}"
        )

        return result

    def save_final_embedding_candidates(
        self,
        priority_groups
    ):

        candidates = []

        for priority in [
            "PRIMARY",
            "SECONDARY",
            "TERTIARY"
        ]:

            for item in priority_groups.get(
                priority,
                []
            ):

                candidates.append({

                    "region_id":
                        item[
                            "region_id"
                        ],

                    "scenario_id":
                        item[
                            "scenario_id"
                        ],

                    "priority":
                        priority,

                    "payload_size":
                        item[
                            "payload_size"
                        ],

                    "safe_capacity":
                        item[
                            "safe_capacity"
                        ],

                    "safe_probability":
                        item[
                            "safe_probability"
                        ],

                    "risk_probability":
                        item[
                            "risk_probability"
                        ],

                    "local_safety":
                        item[
                            "local_safety"
                        ],

                    "local_consequence":
                        item[
                            "local_consequence"
                        ],

                    "center_impact":
                        item[
                            "center_impact"
                        ],

                    "neighbor_impact":
                        item[
                            "neighbor_impact"
                        ],

                    "ga_fitness":
                        item[
                            "ga_fitness"
                        ],

                    "ga_rank":
                        item[
                            "ga_rank"
                        ],

                    "final_rank":
                        item[
                            "final_rank"
                        ]
                })

        candidates.sort(
            key=lambda item: (
                {
                    "PRIMARY": 1,
                    "SECONDARY": 2,
                    "TERTIARY": 3
                }.get(
                    item[
                        "priority"
                    ],
                    4
                ),
                -float(
                    item[
                        "safe_probability"
                    ]
                ),
                -float(
                    item[
                        "ga_fitness"
                    ]
                )
            )
        )

        for index, item in enumerate(
            candidates,
            start=1
        ):

            item[
                "embedding_candidate_rank"
            ] = index

        output_path = os.path.join(
            self.output_dir,
            "final_embedding_candidates.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                candidates,
                file,
                indent=4
            )

        print(
            f"Final Embedding Candidates : "
            f"{len(candidates)}"
        )

        print(
            f"Final Candidate File Saved : "
            f"{output_path}"
        )

        return candidates


    def build_ml_ready_dataset(
        self
    ):

        if (
            self.post_dataset is None
            or
            self.post_dataset.empty
        ):

            raise ValueError(
                "Post-embedding dataset is empty."
            )

        dataset = self.post_dataset.copy()

        excluded_columns = [
            "region_id",
            "scenario_id",
            "consequence_class"
        ]

        feature_columns = [
            column
            for column in dataset.columns
            if column not in excluded_columns
        ]

        numeric_features = []

        for column in feature_columns:

            converted = pd.to_numeric(
                dataset[
                    column
                ],
                errors="coerce"
            )

            if converted.notna().all():

                dataset[
                    column
                ] = converted

                numeric_features.append(
                    column
                )

        ml_dataset = dataset[
            numeric_features
        ].copy()

        ml_dataset[
            "region_id"
        ] = dataset[
            "region_id"
        ].values

        ml_dataset[
            "scenario_id"
        ] = dataset[
            "scenario_id"
        ].values

        ml_dataset[
            "consequence_class"
        ] = dataset[
            "consequence_class"
        ].values

        output_path = os.path.join(
            self.output_dir,
            "post_embedding_ml_ready_dataset.csv"
        )

        ml_dataset.to_csv(
            output_path,
            index=False
        )

        print(
            f"ML Ready Rows     : "
            f"{len(ml_dataset)}"
        )

        print(
            f"ML Ready Features : "
            f"{len(numeric_features)}"
        )

        print(
            f"ML Dataset Saved  : "
            f"{output_path}"
        )

        self.ml_ready_dataset = (
            ml_dataset
        )

        return ml_dataset


    def build_post_embedding_pipeline_summary(
        self,
        results,
        ga_ranking=None
    ):

        summary = {

            "experiments":

                int(
                    len(results)
                )
                if results
                else 0,

            "regions":

                int(
                    len(
                        set(
                            result[
                                "scenario"
                            ][
                                "region_id"
                            ]
                            for result in results
                        )
                    )
                )
                if results
                else 0,

            "subbands_used": [
                "LH",
                "HL",
                "HH"
            ],

            "ll_used":
                False,

            "post_embedding_analysis":
                True,

            "neighbor_analysis":
                True,

            "gradient_optimization":
                True,

            "probability_estimation":
                True,

            "ga_ranking":
                ga_ranking is not None
        }

        output_path = os.path.join(
            self.output_dir,
            "post_embedding_pipeline_summary.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4
            )

        return summary

    def finalize_post_embedding_pipeline(
        self,
        results
    ):

        if not results:

            raise ValueError(
                "No post-embedding results available."
            )

        self.post_embedding_results = results

        self.save_post_embedding_experiment_results(
            results
        )

        region_curves = (
            self.build_post_embedding_region_curves(
                results
            )
        )

        gradient_results = (
            self.calculate_post_embedding_gradients(
                region_curves
            )
        )

        gradient_capacity = (
            self.calculate_gradient_safe_capacity(
                gradient_results
            )
        )

        gradient_path = os.path.join(
            self.output_dir,
            "gradient_safe_capacity.json"
        )

        with open(
            gradient_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                gradient_capacity,
                file,
                indent=4,
                default=str
            )

        ga_input = (
            self.build_post_embedding_ga_input(
                results
            )
        )

        ga_ranking = (
            self.rank_post_embedding_regions_for_ga(
                results
            )
        )

        decision_table = (
            self.build_final_region_decision_table(
                ga_ranking
            )
        )

        self.save_final_region_decisions(
            decision_table
        )

        priority_groups = (
            self.build_primary_secondary_tertiary_lists(
                decision_table
            )
        )

        final_candidates = (
            self.save_final_embedding_candidates(
                priority_groups
            )
        )

        ml_dataset = (
            self.build_ml_ready_dataset()
        )

        summary = (
            self.build_post_embedding_pipeline_summary(
                results,
                ga_ranking
            )
        )

        summary.update({

            "gradient_capacity_regions":
                len(
                    gradient_capacity
                ),

            "ga_candidates":
                len(
                    ga_ranking
                ),

            "final_embedding_candidates":
                len(
                    final_candidates
                ),

            "ml_rows":
                len(
                    ml_dataset
                ),

            "ml_features":
                len(
                    ml_dataset.columns
                ),

            "gradient_capacity_file":
                gradient_path
        })

        summary_path = os.path.join(
            self.output_dir,
            "post_embedding_final_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
                default=str
            )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "POST-EMBEDDING PIPELINE FINALIZED"
        )

        print(
            "=" * 80
        )

        print(
            f"Experiments              : "
            f"{len(results)}"
        )

        print(
            f"Regions                  : "
            f"{summary['regions']}"
        )

        print(
            f"GA Candidates             : "
            f"{len(ga_ranking)}"
        )

        print(
            f"Final Embedding Candidates: "
            f"{len(final_candidates)}"
        )

        print(
            f"ML Dataset Rows           : "
            f"{len(ml_dataset)}"
        )

        print(
            f"ML Dataset Features       : "
            f"{len(ml_dataset.columns)}"
        )

        print(
            f"Final Summary Saved       : "
            f"{summary_path}"
        )

        print(
            "=" * 80
        )

        return {

            "results":
                results,

            "gradient_capacity":
                gradient_capacity,

            "ga_input":
                ga_input,

            "ga_ranking":
                ga_ranking,

            "decision_table":
                decision_table,

            "priority_groups":
                priority_groups,

            "final_candidates":
                final_candidates,

            "ml_dataset":
                ml_dataset,

            "summary":
                summary
        }


    def get_top_embedding_candidates(
        self,
        limit=10
    ):

        path = os.path.join(
            self.output_dir,
            "final_embedding_candidates.json"
        )

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                path
            )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            candidates = json.load(
                file
            )

        return candidates[
            :int(limit)
        ]


    def print_top_embedding_candidates(
        self,
        limit=10
    ):

        candidates = (
            self.get_top_embedding_candidates(
                limit
            )
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "TOP EMBEDDING CANDIDATES"
        )

        print(
            "=" * 80
        )

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            print(
                f"{index:03d} | "
                f"Region : "
                f"{candidate['region_id']} | "
                f"Priority : "
                f"{candidate['priority']} | "
                f"Payload : "
                f"{candidate['payload_size']} | "
                f"Safe P : "
                f"{candidate['safe_probability']:.4f} | "
                f"Risk P : "
                f"{candidate['risk_probability']:.4f} | "
                f"GA : "
                f"{candidate['ga_fitness']:.4f}"
            )

        print(
            "=" * 80
        )


    def run_post_embedding_analysis(
        self,
        embedding_function
    ):

        self.initialize_post_embedding_pipeline()

        results = (
            self.execute_post_embedding_experiments(
                embedding_function=embedding_function
            )
        )

        final_output = (
            self.finalize_post_embedding_pipeline(
                results
            )
        )

        self.print_top_embedding_candidates(
            limit=10
        )

        return final_output

    def reconstruct_post_embedding_dwt(
        self,
        workspace
    ):

        try:

            import pywt

        except ImportError:

            raise ImportError(
                "PyWavelets is required."
            )

        dwt_dir = os.path.join(
            self.base_dir,
            "output",
            "dwt_decomposition"
        )

        ll_path = os.path.join(
            dwt_dir,
            "LL.npy"
        )

        if not os.path.exists(
            ll_path
        ):

            raise FileNotFoundError(
                ll_path
            )

        ll = np.load(
            ll_path
        ).astype(
            np.float64
        )

        reconstructed = pywt.idwt2(
            (
                ll,
                (
                    workspace["modified"]["LH"],
                    workspace["modified"]["HL"],
                    workspace["modified"]["HH"]
                )
            ),
            "db2"
        )

        reconstructed = np.asarray(
            reconstructed,
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(
                reconstructed
            )
        ):

            raise ValueError(
                "Post-embedding reconstruction contains "
                "invalid values."
            )

        return reconstructed


    def decompose_post_embedding_image(
        self,
        reconstructed
    ):

        try:

            import pywt

        except ImportError:

            raise ImportError(
                "PyWavelets is required."
            )

        ll, (lh, hl, hh) = pywt.dwt2(
            reconstructed,
            "db2"
        )

        return {

            "LL":
                np.asarray(
                    ll,
                    dtype=np.float64
                ),

            "LH":
                np.asarray(
                    lh,
                    dtype=np.float64
                ),

            "HL":
                np.asarray(
                    hl,
                    dtype=np.float64
                ),

            "HH":
                np.asarray(
                    hh,
                    dtype=np.float64
                )
        }


    def calculate_post_dwt_region_features(
        self,
        post_dwt,
        region_id
    ):
        bounds = self.get_dwt_region_bounds(region_id)
        r_start = bounds["row_start"]
        r_end = bounds["row_end"]
        c_start = bounds["column_start"]
        c_end = bounds["column_end"]

        features = {}
        for subband in ["LH", "HL", "HH"]:
            band_mat = post_dwt[subband]
            max_r, max_c = band_mat.shape
            
            curr_r_end = min(r_end, max_r)
            curr_c_end = min(c_end, max_c)
            curr_r_start = min(r_start, curr_r_end)
            curr_c_start = min(c_start, curr_c_end)

            matrix = band_mat[curr_r_start:curr_r_end, curr_c_start:curr_c_end].astype(np.float64)

            if matrix.size == 0:
                features[f"{subband}_mean"] = 0.0
                features[f"{subband}_std"] = 0.0
                features[f"{subband}_variance"] = 0.0
                features[f"{subband}_energy"] = 0.0
                features[f"{subband}_min"] = 0.0
                features[f"{subband}_max"] = 0.0
                features[f"{subband}_density"] = 0.0
                features[f"{subband}_entropy"] = 0.0
                features[f"{subband}_gradient"] = 0.0
                continue

            features[f"{subband}_mean"] = float(np.mean(matrix))
            features[f"{subband}_std"] = float(np.std(matrix))
            features[f"{subband}_variance"] = float(np.var(matrix))
            features[f"{subband}_energy"] = float(np.sum(matrix ** 2))
            features[f"{subband}_min"] = float(np.min(matrix))
            features[f"{subband}_max"] = float(np.max(matrix))
            features[f"{subband}_density"] = float(np.count_nonzero(matrix) / matrix.size)

            hist, _ = np.histogram(matrix.flatten(), bins=16)
            prob = hist.astype(np.float64)
            prob = prob[prob > 0]
            if prob.size > 0:
                prob /= np.sum(prob)
                features[f"{subband}_entropy"] = float(-np.sum(prob * np.log2(prob)))
            else:
                features[f"{subband}_entropy"] = 0.0

            if matrix.shape[0] > 1 and matrix.shape[1] > 1:
                gy, gx = np.gradient(matrix)
                features[f"{subband}_gradient"] = float(np.mean(np.sqrt(gx**2 + gy**2)))
            else:
                features[f"{subband}_gradient"] = 0.0

        return features

    def calculate_post_neighbor_features(
        self,
        post_dwt,
        region_id
    ):
        neighbors = self.get_neighbors(region_id)
        neighbor_features = {}

        for direction, n_id in neighbors.items():
            if n_id is None:
                neighbor_features[direction] = None
                continue

            neighbor_features[direction] = self.calculate_post_dwt_region_features(
                post_dwt=post_dwt,
                region_id=n_id
            )

        return neighbor_features

    def calculate_baseline_roundtrip_dwt(
        self
    ):

        original_dwt = {
            "LH": self.dwt_coefficients["LH"].copy(),
            "HL": self.dwt_coefficients["HL"].copy(),
            "HH": self.dwt_coefficients["HH"].copy()
        }

        baseline_workspace = {
            "original": {
                "LH": original_dwt["LH"].copy(),
                "HL": original_dwt["HL"].copy(),
                "HH": original_dwt["HH"].copy()
            },

            "modified": {
                "LH": original_dwt["LH"].copy(),
                "HL": original_dwt["HL"].copy(),
                "HH": original_dwt["HH"].copy()
            },

            "embedding_mask": {
                "LH": np.zeros(
                    original_dwt["LH"].shape,
                    dtype=bool
                ),
                "HL": np.zeros(
                    original_dwt["HL"].shape,
                    dtype=bool
                ),
                "HH": np.zeros(
                    original_dwt["HH"].shape,
                    dtype=bool
                )
            },

            "embedding_count": 0,

            "payload_size": 0,

            "status": "BASELINE"
        }

        baseline_image = (
            self.inverse_dwt_reconstruction(
                baseline_workspace
            )
        )

        baseline_dwt = (
            self.decompose_post_embedding_image(
                baseline_image
            )
        )

        return baseline_dwt

    def calculate_actual_neighbor_impact(
        self,
        post_dwt,
        region_id
    ):

        neighbors = self.get_neighbors(
            region_id
        )

        print(
            f"Neighbor Check | "
            f"Region {region_id} | "
            f"{neighbors}"
        )

        baseline_dwt = (
            self.calculate_baseline_roundtrip_dwt()
        )

        results = {}

        for direction, neighbor_id in (
            neighbors.items()
        ):

            if neighbor_id is None:

                results[direction] = {
                    "available": False,
                    "neighbor_id": None,
                    "mean_absolute_change": 0.0,
                    "median_absolute_change": 0.0,
                    "variance_absolute_change": 0.0,
                    "std_absolute_change": 0.0,
                    "maximum_absolute_change": 0.0,
                    "minimum_absolute_change": 0.0,
                    "mean_relative_change": 0.0,
                    "median_relative_change": 0.0,
                    "variance_relative_change": 0.0,
                    "std_relative_change": 0.0,
                    "maximum_relative_change": 0.0,
                    "minimum_relative_change": 0.0,
                    "changed_feature_count": 0,
                    "unchanged_feature_count": 0,
                    "changed_feature_ratio": 0.0,
                    "feature_correlation": 1.0
                }

                continue

            absolute_changes = []
            relative_changes = []

            changes = {}

            for subband in [
                "LH",
                "HL",
                "HH"
            ]:

                bounds = self.get_dwt_region_bounds(
                    neighbor_id
                )

                r_s = bounds[
                    "row_start"
                ]

                r_e = min(
                    bounds["row_end"],
                    baseline_dwt[subband].shape[0],
                    post_dwt[subband].shape[0]
                )

                c_s = bounds[
                    "column_start"
                ]

                c_e = min(
                    bounds["column_end"],
                    baseline_dwt[subband].shape[1],
                    post_dwt[subband].shape[1]
                )

                baseline_region = (
                    baseline_dwt[subband][
                        r_s:r_e,
                        c_s:c_e
                    ]
                )

                post_region = (
                    post_dwt[subband][
                        r_s:r_e,
                        c_s:c_e
                    ]
                )

                difference = (
                    post_region
                    -
                    baseline_region
                )

                absolute_difference = np.abs(
                    difference
                )

                for value in (
                    absolute_difference.flatten()
                ):

                    value = float(value)

                    if np.isfinite(value):

                        absolute_changes.append(
                            value
                        )

                denominator = np.maximum(
                    np.abs(
                        baseline_region
                    ),
                    1e-8
                )

                relative_difference = (
                    absolute_difference
                    /
                    denominator
                )

                for value in (
                    relative_difference.flatten()
                ):

                    value = float(value)

                    if np.isfinite(value):

                        relative_changes.append(
                            value
                        )

                changes[
                    f"{subband}_mse"
                ] = float(
                    np.mean(
                        difference ** 2
                    )
                )

                changes[
                    f"{subband}_mean"
                ] = float(
                    np.mean(
                        difference
                    )
                )

                changes[
                    f"{subband}_max"
                ] = float(
                    np.max(
                        absolute_difference
                    )
                )

                changes[
                    f"{subband}_changed"
                ] = int(
                    np.count_nonzero(
                        absolute_difference
                        >
                        1e-12
                    )
                )

        if absolute_changes:

            absolute_array = np.asarray(
                absolute_changes,
                dtype=np.float64
            )

            mean_abs = float(
                np.mean(
                    absolute_array
                )
            )

            median_abs = float(
                np.median(
                    absolute_array
                )
            )

            variance_abs = float(
                np.var(
                    absolute_array
                )
            )

            std_abs = float(
                np.std(
                    absolute_array
                )
            )

            max_abs = float(
                np.max(
                    absolute_array
                )
            )

            min_abs = float(
                np.min(
                    absolute_array
                )
            )

        else:

            mean_abs = 0.0
            median_abs = 0.0
            variance_abs = 0.0
            std_abs = 0.0
            max_abs = 0.0
            min_abs = 0.0

        if relative_changes:

            relative_array = np.asarray(
                relative_changes,
                dtype=np.float64
            )

            mean_rel = float(
                np.mean(
                    relative_array
                )
            )

            median_rel = float(
                np.median(
                    relative_array
                )
            )

            variance_rel = float(
                np.var(
                    relative_array
                )
            )

            std_rel = float(
                np.std(
                    relative_array
                )
            )

            max_rel = float(
                np.max(
                    relative_array
                )
            )

            min_rel = float(
                np.min(
                    relative_array
                )
            )

            changed_count = int(
                np.count_nonzero(
                    relative_array
                    >
                    1e-12
                )
            )

            total_count = int(
                relative_array.size
            )

            changed_ratio = float(
                changed_count
                /
                total_count
            )

        else:

            mean_rel = 0.0
            median_rel = 0.0
            variance_rel = 0.0
            std_rel = 0.0
            max_rel = 0.0
            min_rel = 0.0
            changed_count = 0
            total_count = 0
            changed_ratio = 0.0

        if (
            absolute_changes
            and
            relative_changes
        ):

            absolute_array = np.asarray(
                absolute_changes,
                dtype=np.float64
            )

            relative_array = np.asarray(
                relative_changes,
                dtype=np.float64
            )

            absolute_std = float(
                np.std(
                    absolute_array
                )
            )

            relative_std = float(
                np.std(
                    relative_array
                )
            )

            if (
                absolute_array.size < 2
                or
                relative_array.size < 2
                or
                absolute_std <= 1e-12
                or
                relative_std <= 1e-12
            ):

                correlation = 1.0

            else:

                with np.errstate(
                    divide="ignore",
                    invalid="ignore"
                ):

                    correlation = float(
                        np.corrcoef(
                            absolute_array,
                            relative_array
                        )[0, 1]
                    )

                if not np.isfinite(
                    correlation
                ):

                    correlation = 1.0

        else:

            correlation = 1.0

        impact_score = float(
            np.clip(
                0.70 * mean_rel
                +
                0.30 * changed_ratio,
                0.0,
                1.0
            )
        )

        results[direction] = {

            "available": True,

            "neighbor_id":
                neighbor_id,

            "changes":
                changes,

            "mean_absolute_change":
                mean_abs,

            "median_absolute_change":
                median_abs,

            "variance_absolute_change":
                variance_abs,

            "std_absolute_change":
                std_abs,

            "maximum_absolute_change":
                max_abs,

            "minimum_absolute_change":
                min_abs,

            "mean_relative_change":
                mean_rel,

            "median_relative_change":
                median_rel,

            "variance_relative_change":
                variance_rel,

            "std_relative_change":
                std_rel,

            "maximum_relative_change":
                max_rel,

            "minimum_relative_change":
                min_rel,

            "changed_feature_count":
                changed_count,

            "unchanged_feature_count":
                total_count - changed_count,

            "changed_feature_ratio":
                changed_ratio,

            "feature_correlation":
                correlation,

            "impact_score":
                impact_score
        }

        # print(
        #     f"Neighbor Statistics | "
        #     f"{region_id} -> {direction} "
        #     f"({neighbor_id}) | "
        #     f"MeanRaw={mean_abs:.6e} | "
        #     f"MaxRaw={max_abs:.6e} | "
        #     f"MeanRelative={mean_rel:.6e} | "
        #     f"Changed={changed_count}/{total_count} | "
        #     f"Ratio={changed_ratio:.4f}"
        # )

        return results



    def reconstruct_pre_embedding_dwt(
        self
    ):

        coefficients = {

            "LL":
                self.dwt_coefficients[
                    "LL"
                ],

            "LH":
                self.dwt_coefficients[
                    "LH"
                ],

            "HL":
                self.dwt_coefficients[
                    "HL"
                ],

            "HH":
                self.dwt_coefficients[
                    "HH"
                ]
        }

        return (
            self.inverse_dwt_reconstruction(
                coefficients
            )
        )

    def calculate_actual_post_embedding_analysis(
        self,
        center_workspace,
        region_id
    ):

        if (
            not hasattr(
                self,
                "dwt_coefficients"
            )
            or
            not self.dwt_coefficients
        ):

            self.load_dwt_coefficients()

        if not hasattr(
            self,
            "ll_coefficients"
        ):

            self.load_ll_coefficients()

        full_ll = (
            self.ll_coefficients.copy()
        )

        original_dwt = {
            "LH":
                self.dwt_coefficients[
                    "LH"
                ].copy(),

            "HL":
                self.dwt_coefficients[
                    "HL"
                ].copy(),

            "HH":
                self.dwt_coefficients[
                    "HH"
                ].copy()
        }

        modified_dwt = {

            "LH":
                np.asarray(
                    center_workspace[
                        "modified"
                    ]["LH"],
                    dtype=np.float64
                ).copy(),

            "HL":
                np.asarray(
                    center_workspace[
                        "modified"
                    ]["HL"],
                    dtype=np.float64
                ).copy(),

            "HH":
                np.asarray(
                    center_workspace[
                        "modified"
                    ]["HH"],
                    dtype=np.float64
                ).copy()
        }

        pre_dwt = {
            key:
                value.copy()
            for key, value
            in original_dwt.items()
        }

        pre_workspace = {

            "original": {
                "LH":
                    original_dwt["LH"].copy(),
                "HL":
                    original_dwt["HL"].copy(),
                "HH":
                    original_dwt["HH"].copy()
            },

            "modified": {
                "LH":
                    original_dwt["LH"].copy(),
                "HL":
                    original_dwt["HL"].copy(),
                "HH":
                    original_dwt["HH"].copy()
            },

            "embedding_mask": {
                "LH":
                    np.zeros(
                        original_dwt["LH"].shape,
                        dtype=bool
                    ),
                "HL":
                    np.zeros(
                        original_dwt["HL"].shape,
                        dtype=bool
                    ),
                "HH":
                    np.zeros(
                        original_dwt["HH"].shape,
                        dtype=bool
                    )
            }
        }

        post_workspace = {

            "original": {
                "LH":
                    original_dwt["LH"].copy(),
                "HL":
                    original_dwt["HL"].copy(),
                "HH":
                    original_dwt["HH"].copy()
            },

            "modified": {
                "LH":
                    modified_dwt["LH"].copy(),
                "HL":
                    modified_dwt["HL"].copy(),
                "HH":
                    modified_dwt["HH"].copy()
            },

            "embedding_mask": {
                "LH":
                    np.zeros(
                        original_dwt["LH"].shape,
                        dtype=bool
                    ),
                "HL":
                    np.zeros(
                        original_dwt["HL"].shape,
                        dtype=bool
                    ),
                "HH":
                    np.zeros(
                        original_dwt["HH"].shape,
                        dtype=bool
                    )
            }
        }

        pre_image = np.asarray(
            self.inverse_dwt_reconstruction(
                pre_workspace
            ),
            dtype=np.float64
        )

        post_image = np.asarray(
            self.inverse_dwt_reconstruction(
                post_workspace
            ),
            dtype=np.float64
        )

        post_ll, (
            post_lh,
            post_hl,
            post_hh
        ) = pywt.dwt2(
            post_image,
            "db2"
        )

        post_dwt = {

            "LH":
                np.asarray(
                    post_lh,
                    dtype=np.float64
                ),

            "HL":
                np.asarray(
                    post_hl,
                    dtype=np.float64
                ),

            "HH":
                np.asarray(
                    post_hh,
                    dtype=np.float64
                )
        }

        min_r = min(
            pre_dwt["LH"].shape[0],
            post_dwt["LH"].shape[0]
        )

        min_c = min(
            pre_dwt["LH"].shape[1],
            post_dwt["LH"].shape[1]
        )

        pre_dwt = {
            key:
                value[
                    :min_r,
                    :min_c
                ]
            for key, value
            in pre_dwt.items()
        }

        post_dwt = {
            key:
                value[
                    :min_r,
                    :min_c
                ]
            for key, value
            in post_dwt.items()
        }

        # print(
        #     "DWT ACTUAL CHANGE CHECK"
        # )

        # for subband in [
        #     "LH",
        #     "HL",
        #     "HH"
        # ]:

        #     difference = (
        #         post_dwt[subband]
        #         -
        #         pre_dwt[subband]
        #     )

        #     changed = int(
        #         np.count_nonzero(
        #             np.abs(
        #                 difference
        #             )
        #             >
        #             1e-12
        #         )
        #     )

        #     print(
        #         f"{subband} | "
        #         f"MSE="
        #         f"{np.mean(difference ** 2):.12e} | "
        #         f"Max="
        #         f"{np.max(np.abs(difference)):.12e} | "
        #         f"Changed={changed}"
        #     )

        baseline_dwt = (
            self.calculate_baseline_roundtrip_dwt()
        )

        baseline_dwt = {
            key:
                value[
                    :min_r,
                    :min_c
                ]
            for key, value
            in baseline_dwt.items()
        }

        neighbor_impact = (
            self.calculate_actual_neighbor_impact(
                post_dwt=post_dwt,
                region_id=region_id
            )
        )

        spatial_neighbor_impact = (
            self.calculate_spatial_neighbor_impact(
                pre_image=pre_image,
                post_image=post_image,
                region_id=region_id
            )
        )

        center_post_features = (
            self.calculate_post_dwt_region_features(
                post_dwt=post_dwt,
                region_id=region_id
            )
        )

        neighbor_post_features = (
            self.calculate_post_neighbor_features(
                post_dwt=post_dwt,
                region_id=region_id
            )
        )

        neighbor_pre_features = (
            self.calculate_post_neighbor_features(
                post_dwt=pre_dwt,
                region_id=region_id
            )
        )

        neighbor_feature_changes = (
            self.calculate_neighbor_feature_changes(
                neighbor_pre_features,
                neighbor_post_features
            )
        )

        return {

            "center_post_features":
                center_post_features,

            "neighbor_post_features":
                neighbor_post_features,

            "neighbor_feature_changes":
                neighbor_feature_changes,

            "actual_neighbor_impact":
                neighbor_impact,

            "spatial_neighbor_impact":
                spatial_neighbor_impact,

            "pre_image":
                pre_image,

            "reconstructed_image":
                post_image,

            "pre_dwt":
                baseline_dwt,

            "post_dwt":
                post_dwt

        }

    
    
    def build_actual_post_embedding_feature_row(
        self,
        scenario,
        actual_analysis,
        pre_features=None
    ):

        region_id = scenario[
            "region_id"
        ]

        row = {

            "region_id":
                region_id,

            "scenario_id":
                scenario[
                    "scenario_id"
                ],

            "payload_size":
                int(
                    scenario[
                        "payload_size"
                    ]
                ),

            "capacity":
                int(
                    scenario[
                        "capacity"
                    ]
                ),

            "capacity_ratio":
                float(
                    scenario[
                        "capacity_ratio"
                    ]
                )
        }

        if pre_features:

            for key, value in (
                pre_features.items()
            ):

                try:

                    value = float(
                        value
                    )

                    if np.isfinite(value):

                        row[
                            f"pre_{key}"
                        ] = value

                except (
                    ValueError,
                    TypeError
                ):

                    continue

        center_features = (
            actual_analysis[
                "center_post_features"
            ]
        )

        for key, value in (
            center_features.items()
        ):

            try:

                value = float(
                    value
                )

                if np.isfinite(value):

                    row[
                        f"post_center_{key}"
                    ] = value

            except (
                ValueError,
                TypeError
            ):

                continue

        neighbor_features = (
            actual_analysis[
                "neighbor_post_features"
            ]
        )

        for direction, features in (
            neighbor_features.items()
        ):

            if features is None:
                continue

            for key, value in (
                features.items()
            ):

                try:

                    value = float(
                        value
                    )

                    if np.isfinite(value):

                        row[
                            f"post_neighbor_"
                            f"{direction}_{key}"
                        ] = value

                except (
                    ValueError,
                    TypeError
                ):

                    continue

        actual_impact = (
            actual_analysis[
                "actual_neighbor_impact"
            ]
        )

        neighbor_change_values = []

        for direction, impact in (
            actual_impact.items()
        ):

            if not impact.get(
                "available",
                False
            ):

                continue

            mse = float(
                impact.get(
                    "mse",
                    0.0
                )
            )

            mae = float(
                impact.get(
                    "mae",
                    0.0
                )
            )

            psnr = float(
                impact.get(
                    "psnr",
                    float("inf")
                )
            )

            variance_change = float(
                impact.get(
                    "variance_change",
                    0.0
                )
            )

            maximum_pixel_change = float(
                impact.get(
                    "maximum_pixel_change",
                    0.0
                )
            )

            changed_pixel_density = float(
                impact.get(
                    "changed_pixel_density",
                    0.0
                )
            )

            correlation = float(
                impact.get(
                    "correlation",
                    1.0
                )
            )

            impact_score = float(
                impact.get(
                    "impact_score",
                    0.0
                )
            )

            row[
                f"neighbor_{direction}_mse"
            ] = mse

            row[
                f"neighbor_{direction}_mae"
            ] = mae

            if np.isfinite(psnr):

                row[
                    f"neighbor_{direction}_psnr"
                ] = psnr

            row[
                f"neighbor_{direction}_variance_change"
            ] = variance_change

            row[
                f"neighbor_{direction}_maximum_pixel_change"
            ] = maximum_pixel_change

            row[
                f"neighbor_{direction}_changed_pixel_density"
            ] = changed_pixel_density

            row[
                f"neighbor_{direction}_correlation"
            ] = correlation

            row[
                f"neighbor_{direction}_impact_score"
            ] = impact_score

            neighbor_change_values.append(
                impact_score
            )
        row[
            "actual_neighbor_mean_impact"
        ] = (
            float(
                np.mean(
                    neighbor_change_values
                )
            )
            if neighbor_change_values
            else 0.0
        )

        row[
            "actual_neighbor_maximum_impact"
        ] = (
            float(
                np.max(
                    neighbor_change_values
                )
            )
            if neighbor_change_values
            else 0.0
        )

        return row


    def calculate_actual_post_embedding_quality(
        self,
        pre_image,
        post_image
    ):

        pre = np.asarray(
            pre_image,
            dtype=np.float64
        )

        post = np.asarray(
            post_image,
            dtype=np.float64
        )

        if pre.shape != post.shape:

            raise ValueError(
                "Pre and post images must have "
                "the same shape."
            )

        difference = (
            post - pre
        )

        mse = float(
            np.mean(
                difference ** 2
            )
        )

        if mse <= 1e-12:

            psnr = float("inf")

        else:

            psnr = float(
                10.0
                *
                np.log10(
                    (255.0 ** 2)
                    /
                    mse
                )
            )

        pre_mean = float(
            np.mean(pre)
        )

        post_mean = float(
            np.mean(post)
        )

        pre_variance = float(
            np.var(pre)
        )

        post_variance = float(
            np.var(post)
        )

        mean_change = float(
            abs(
                post_mean
                -
                pre_mean
            )
        )

        variance_change = float(
            abs(
                post_variance
                -
                pre_variance
            )
        )

        return {

            "image_mse":
                mse,

            "image_psnr":
                psnr,

            "mean_change":
                mean_change,

            "variance_change":
                variance_change,

            "maximum_pixel_change":
                float(
                    np.max(
                        np.abs(
                            difference
                        )
                    )
                ),

            "changed_pixels":
                int(
                    np.count_nonzero(
                        np.abs(
                            difference
                        ) > 1e-12
                    )
                ),

            "total_pixels":
                int(
                    pre.size
                ),

            "pixel_change_density":
                float(
                    np.count_nonzero(
                        np.abs(
                            difference
                        ) > 1e-12
                    )
                    /
                    pre.size
                )
                if pre.size
                else 0.0
        }


    def calculate_actual_center_quality(
        self,
        pre_image,
        post_image,
        region_id,
        workspace
    ):

        bounds = (
            self.get_dwt_region_bounds(
                region_id
            )
        )

        coefficient_metrics = {}

        for subband in [
            "LH",
            "HL",
            "HH"
        ]:

            pre_region = (
                workspace[
                    "original"
                ][subband][
                    bounds["row_start"]:
                    bounds["row_end"],
                    bounds["column_start"]:
                    bounds["column_end"]
                ]
            )

            post_region = (
                workspace[
                    "modified"
                ][subband][
                    bounds["row_start"]:
                    bounds["row_end"],
                    bounds["column_start"]:
                    bounds["column_end"]
                ]
            )

            metrics = (
                self.calculate_region_quality_metrics(
                    pre_region,
                    post_region
                )
            )

            for key, value in metrics.items():

                coefficient_metrics[
                    f"{subband}_{key}"
                ] = value

        image_quality = (
            self.calculate_actual_post_embedding_quality(
                pre_image,
                post_image
            )
        )

        return {

            "center_coefficient_metrics":
                coefficient_metrics,

            "center_image_quality":
                image_quality
        }


    def merge_actual_post_quality_features(
        self,
        row,
        quality
    ):

        if not quality:
            return row

        for key, value in (
            quality.items()
        ):

            if isinstance(
                value,
                dict
            ):

                for subkey, subvalue in (
                    value.items()
                ):

                    try:

                        subvalue = float(
                            subvalue
                        )

                        if np.isfinite(
                            subvalue
                        ):

                            row[
                                f"{key}_{subkey}"
                            ] = subvalue

                    except (
                        ValueError,
                        TypeError
                    ):

                        continue

            else:

                try:

                    value = float(
                        value
                    )

                    if np.isfinite(value):

                        row[
                            key
                        ] = value

                except (
                    ValueError,
                    TypeError
                ):

                    continue

        return row

    def calculate_actual_embedding_consequence(
        self,
        row
    ):

        image_mse = float(
            row.get(
                "image_mse",
                0.0
            )
        )

        image_psnr = float(
            row.get(
                "image_psnr",
                100.0
            )
        )

        pixel_density = float(
            row.get(
                "pixel_change_density",
                0.0
            )
        )

        maximum_pixel_change = float(
            row.get(
                "maximum_pixel_change",
                0.0
            )
        )

        neighbor_mean = float(
            row.get(
                "actual_neighbor_mean_impact",
                0.0
            )
        )

        neighbor_maximum = float(
            row.get(
                "actual_neighbor_maximum_impact",
                0.0
            )
        )

        coefficient_mse_values = []
        coefficient_variance_values = []
        coefficient_density_values = []
        coefficient_maximum_values = []

        for subband in [
            "LH",
            "HL",
            "HH"
        ]:

            mse = row.get(
                f"center_coefficient_metrics_{subband}_mse"
            )

            variance = row.get(
                f"center_coefficient_metrics_{subband}_variance_change"
            )

            density = row.get(
                f"center_coefficient_metrics_{subband}_changed_density"
            )

            maximum = row.get(
                f"center_coefficient_metrics_{subband}_maximum_change"
            )

            if mse is not None:
                coefficient_mse_values.append(
                    float(mse)
                )

            if variance is not None:
                coefficient_variance_values.append(
                    float(variance)
                )

            if density is not None:
                coefficient_density_values.append(
                    float(density)
                )

            if maximum is not None:
                coefficient_maximum_values.append(
                    float(maximum)
                )

        coefficient_mse = (
            float(
                np.mean(
                    coefficient_mse_values
                )
            )
            if coefficient_mse_values
            else 0.0
        )

        # print(
        #     "DEBUG MSE KEYS:",
        #     {
        #         k: row[k]
        #         for k in row
        #         if "mse" in str(k).lower()
        #         and any(
        #             band in str(k)
        #             for band in ["LH", "HL", "HH"]
        #         )
        #     }
        # )

        # print(
        #     "DEBUG coefficient_mse_values:",
        #     coefficient_mse_values
        # )

        coefficient_variance = (
            float(
                np.mean(
                    coefficient_variance_values
                )
            )
            if coefficient_variance_values
            else 0.0
        )

        coefficient_density = (
            float(
                np.mean(
                    coefficient_density_values
                )
            )
            if coefficient_density_values
            else 0.0
        )

        coefficient_maximum = (
            float(
                np.mean(
                    coefficient_maximum_values
                )
            )
            if coefficient_maximum_values
            else 0.0
        )

        image_mse_score = float(
            np.clip(
                image_mse,
                0.0,
                1.0
            )
        )

        image_quality_loss = float(
            np.clip(
                1.0
                -
                (
                    np.clip(
                        image_psnr,
                        0.0,
                        100.0
                    )
                    /
                    100.0
                ),
                0.0,
                1.0
            )
        )

        coefficient_mse_score = float(
            np.clip(
                coefficient_mse,
                0.0,
                1.0
            )
        )

        coefficient_variance_score = float(
            np.clip(
                coefficient_variance,
                0.0,
                1.0
            )
        )

        coefficient_density_score = float(
            np.clip(
                coefficient_density,
                0.0,
            1.0
            )
        )

        coefficient_maximum_score = float(
            np.clip(
                coefficient_maximum
                /
                255.0,
                0.0,
                1.0
            )
        )

        pixel_density_score = float(
            np.clip(
                pixel_density,
                0.0,
                1.0
            )
        )

        maximum_pixel_score = float(
            np.clip(
                maximum_pixel_change
                /
                255.0,
                0.0,
                1.0
            )
        )

        neighbor_score = float(
            np.clip(
                (
                    0.65
                    *
                    neighbor_mean
                )
                +
                (
                    0.35
                    *
                    neighbor_maximum
                ),
                0.0,
                1.0
            )
        )

        consequence = (

            0.20
            *
            coefficient_mse_score

            +

            0.10
            *
            coefficient_variance_score

            +

            0.10
            *
            coefficient_density_score

            +

            0.05
            *
            coefficient_maximum_score

            +

            0.20
            *
            image_mse_score

            +

            0.15
            *
            image_quality_loss

            +

            0.05
            *
            pixel_density_score

            +

            0.05
            *
            maximum_pixel_score

            +

            0.10
            *
            neighbor_score
        )

        consequence = float(
            np.clip(
                consequence,
                0.0,
                1.0
            )
        )

        safety = float(
            1.0
            -
            consequence
        )

        return {

            "actual_consequence_score":
                consequence,

            "actual_safety_score":
                safety,

            "actual_center_distortion":
                coefficient_mse_score,

            "actual_neighbor_impact":
                neighbor_score,

            "actual_quality_loss":
                image_quality_loss,

            "actual_pixel_change":
                pixel_density_score
        }


    def calculate_actual_embedding_probabilities(
        self,
        consequence_score
    ):
        consequence_score = float(
            np.clip(
                consequence_score,
                0.0,
                1.0
            )
        )

        # Balanced values to allow proper tier distribution
        sharpness = 8.0
        threshold = 0.40

        safe_probability = float(
            1.0
            /
            (
                1.0
                +
                np.exp(
                    sharpness
                    *
                    (
                        consequence_score
                        -
                        threshold
                    )
                )
            )
        )

        risk_probability = float(
            1.0
            -
            safe_probability
        )

        return {
            "actual_safe_probability": safe_probability,
            "actual_risk_probability": risk_probability
        }


    def classify_actual_embedding_result(
        self,
        safety_score,
        safe_probability
    ):

        if (
            safety_score >= 0.85
            and
            safe_probability >= 0.85
        ):

            return "PRIMARY"

        elif (
            safety_score >= 0.55
            and
            safe_probability >= 0.55
        ):

            return "SECONDARY"

        elif (
            safety_score >= 0.25
            and
            safe_probability >= 0.25
        ):

            return "TERTIARY"

        else:

            return "REJECT"


    def finalize_actual_post_embedding_row(
        self,
        row
    ):

        consequence = (
            self.calculate_actual_embedding_consequence(
                row
            )
        )

        probabilities = (
            self.calculate_actual_embedding_probabilities(
                consequence[
                    "actual_consequence_score"
                ]
            )
        )

        classification = (
            self.classify_actual_embedding_result(
                consequence[
                    "actual_safety_score"
                ],
                probabilities[
                    "actual_safe_probability"
                ]
            )
        )

        row.update(
            consequence
        )

        row.update(
            probabilities
        )

        row[
            "actual_embedding_class"
        ] = classification

        row[
            "actual_embedding_safe"
        ] = int(
            classification != "REJECT"
        )

        return row


    def append_actual_post_embedding_row(
        self,
        row
    ):

        row = (
            self.finalize_actual_post_embedding_row(
                row
            )
        )

        if not hasattr(
            self,
            "actual_post_dataset"
        ):

            self.actual_post_dataset = (
                pd.DataFrame()
            )

        self.actual_post_dataset = pd.concat(
            [
                self.actual_post_dataset,
                pd.DataFrame(
                    [row]
                )
            ],
            ignore_index=True
        )

        return row


    def save_actual_post_embedding_dataset(
        self
    ):

        if (
            not hasattr(
                self,
                "actual_post_dataset"
            )
            or
            self.actual_post_dataset.empty
        ):

            raise ValueError(
                "Actual post-embedding dataset is empty."
            )

        output_path = os.path.join(
            self.output_dir,
            "actual_post_embedding_dataset.csv"
        )

        self.actual_post_dataset.to_csv(
            output_path,
            index=False
        )

        print(
            f"Actual Post Dataset Rows : "
            f"{len(self.actual_post_dataset)}"
        )

        print(
            f"Actual Post Dataset Features : "
            f"{len(self.actual_post_dataset.columns)}"
        )

        print(
            f"Actual Post Dataset Saved : "
            f"{output_path}"
        )

        return output_path

    def calculate_actual_post_embedding_gradients(
        self
    ):

        if (
            not hasattr(
                self,
                "actual_post_dataset"
            )
            or
            self.actual_post_dataset.empty
        ):

            return pd.DataFrame()

        dataframe = (
            self.actual_post_dataset.copy()
        )

        if "region_id" not in dataframe.columns:

            return dataframe

        dataframe.sort_values(
            by=[
                "region_id",
                "payload_size"
            ],
            inplace=True
        )

        gradient_columns = [

            "actual_consequence_score",

            "actual_safety_score",

            "actual_safe_probability",

            "actual_risk_probability",

            "actual_center_distortion",

            "actual_neighbor_impact",

            "actual_quality_loss"
        ]

        for column in gradient_columns:

            gradient_name = (
                f"{column}_gradient"
            )

            dataframe[
                gradient_name
            ] = np.nan

            for region_id, indices in (
                dataframe.groupby(
                    "region_id"
                ).groups.items()
            ):

                region_indices = list(
                    indices
                )

                previous_index = None

                for current_index in (
                    region_indices
                ):

                    if previous_index is None:

                        previous_index = (
                            current_index
                        )

                        continue

                    current_payload = float(
                        dataframe.loc[
                            current_index,
                            "payload_size"
                        ]
                    )

                    previous_payload = float(
                        dataframe.loc[
                            previous_index,
                            "payload_size"
                        ]
                    )

                    payload_delta = (
                        current_payload
                        -
                        previous_payload
                    )

                    if abs(
                        payload_delta
                    ) < 1e-12:

                        previous_index = (
                            current_index
                        )

                        continue

                    current_value = pd.to_numeric(
                        dataframe.loc[
                            current_index,
                            column
                        ],
                        errors="coerce"
                    )

                    previous_value = pd.to_numeric(
                        dataframe.loc[
                            previous_index,
                            column
                        ],
                        errors="coerce"
                    )

                    if (
                        pd.isna(
                            current_value
                        )
                        or
                        pd.isna(
                            previous_value
                        )
                    ):

                        previous_index = (
                            current_index
                        )

                        continue

                    dataframe.loc[
                        current_index,
                        gradient_name
                    ] = (
                        float(
                            current_value
                            -
                            previous_value
                        )
                        /
                        payload_delta
                    )

                    previous_index = (
                        current_index
                    )

        return dataframe


    def calculate_actual_safe_capacity(
        self,
        consequence_threshold=0.30,
        safety_threshold=0.25,
        probability_threshold=0.25
    ):

        if (
            not hasattr(
                self,
                "actual_post_dataset"
            )
            or
            self.actual_post_dataset.empty
        ):

            return pd.DataFrame()

        dataframe = (
            self.actual_post_dataset.copy()
        )

        safe_mask = (

            (
                dataframe[
                    "actual_consequence_score"
                ]
                <= consequence_threshold
            )

            &

            (
                dataframe[
                    "actual_safety_score"
                ]
                >= safety_threshold
            )

            &

            (
                dataframe[
                    "actual_safe_probability"
                ]
                >= probability_threshold
            )
        )

        dataframe[
            "actual_safe"
        ] = safe_mask.astype(
            int
        )

        capacity_rows = []

        for region_id, group in (
            dataframe.groupby(
                "region_id"
            )
        ):

            safe_group = group[
                group[
                    "actual_safe"
                ] == 1
            ]

            if safe_group.empty:

                capacity_rows.append({

                    "region_id":
                        region_id,

                    "actual_safe_capacity":
                        0,

                    "actual_max_safe_probability":
                        float(
                            group[
                                "actual_safe_probability"
                            ].max()
                        ),

                    "actual_min_consequence":
                        float(
                            group[
                                "actual_consequence_score"
                            ].min()
                        ),

                    "actual_max_safety":
                        float(
                            group[
                                "actual_safety_score"
                            ].max()
                        ),

                    "actual_status":
                        "REJECT"
                })

                continue

            best = safe_group.loc[
                safe_group[
                    "payload_size"
                ].idxmax()
            ]

            capacity_rows.append({

                "region_id":
                    region_id,

                "actual_safe_capacity":
                    int(
                        best[
                            "payload_size"
                        ]
                    ),

                "actual_max_safe_probability":
                    float(
                        best[
                            "actual_safe_probability"
                        ]
                    ),

                "actual_min_consequence":
                    float(
                        safe_group[
                            "actual_consequence_score"
                        ].min()
                    ),

                "actual_max_safety":
                    float(
                        safe_group[
                            "actual_safety_score"
                        ].max()
                    ),

                "actual_status":
                    "SAFE"
            })

        capacity_dataframe = pd.DataFrame(
            capacity_rows
        )

        output_path = os.path.join(
            self.output_dir,
            "actual_safe_capacity.csv"
        )

        capacity_dataframe.to_csv(
            output_path,
            index=False
        )

        return capacity_dataframe


    def calculate_actual_post_region_ranking(
        self
    ):

        if (
            not hasattr(
                self,
                "actual_post_dataset"
            )
            or
            self.actual_post_dataset.empty
        ):

            return pd.DataFrame()

        dataframe = (
            self.actual_post_dataset.copy()
        )

        ranking = (
            dataframe
            .sort_values(
                by=[
                    "actual_safe_probability",
                    "actual_safety_score",
                    "actual_consequence_score"
                ],
                ascending=[
                    False,
                    False,
                    True
                ]
            )
            .drop_duplicates(
                subset=[
                    "region_id"
                ],
                keep="first"
            )
            .copy()
        )

        ranking[
            "actual_region_score"
        ] = (

            0.35
            *
            ranking[
                "actual_safe_probability"
            ]

            +

            0.25
            *
            ranking[
                "actual_safety_score"
            ]

            +

            0.20
            *
            (
                1.0
                -
                ranking[
                    "actual_consequence_score"
                ]
            )

            +

            0.10
            *
            (
                1.0
                -
                ranking[
                    "actual_neighbor_impact"
                ]
            )

            +

            0.10
            *
            ranking[
                "capacity_ratio"
            ]
        )

        ranking[
            "actual_region_score"
        ] = np.clip(
            ranking[
                "actual_region_score"
            ],
            0.0,
            1.0
        )

        ranking.sort_values(
            by=[
                "actual_region_score"
            ],
            ascending=False,
            inplace=True
        )

        ranking.reset_index(
            drop=True,
            inplace=True
        )

        ranking[
            "actual_rank"
        ] = (
            np.arange(
                len(ranking)
            )
            + 1
        )

        output_path = os.path.join(
            self.output_dir,
            "actual_post_region_ranking.csv"
        )

        ranking.to_csv(
            output_path,
            index=False
        )

        return ranking


    def finalize_actual_post_embedding_dataset(
        self
    ):

        if (
            not hasattr(
                self,
                "actual_post_dataset"
            )
            or
            self.actual_post_dataset.empty
        ):

            raise ValueError(
                "Actual post-embedding dataset is empty."
            )

        self.actual_post_dataset = (
            self.calculate_actual_post_embedding_gradients()
        )

        capacity = (
            self.calculate_actual_safe_capacity()
        )

        ranking = (
            self.calculate_actual_post_region_ranking()
        )

        dataset_path = (
            self.save_actual_post_embedding_dataset()
        )

        capacity_path = os.path.join(
            self.output_dir,
            "actual_safe_capacity.csv"
        )

        ranking_path = os.path.join(
            self.output_dir,
            "actual_post_region_ranking.csv"
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "ACTUAL POST-EMBEDDING DATASET COMPLETED"
        )

        print(
            "=" * 80
        )

        print(
            f"Rows       : "
            f"{len(self.actual_post_dataset)}"
        )

        print(
            f"Features   : "
            f"{len(self.actual_post_dataset.columns)}"
        )

        print(
            f"Regions    : "
            f"{self.actual_post_dataset['region_id'].nunique()}"
        )

        print(
            f"Dataset    : "
            f"{dataset_path}"
        )

        print(
            f"Capacity   : "
            f"{capacity_path}"
        )

        print(
            f"Ranking    : "
            f"{ranking_path}"
        )

        print(
            "=" * 80
        )

        return {

            "dataset":
                self.actual_post_dataset,

            "safe_capacity":
                capacity,

            "ranking":
                ranking
        }

    def build_actual_post_ml_targets(
        self
    ):

        if (
            not hasattr(
                self,
                "actual_post_dataset"
            )
            or
            self.actual_post_dataset.empty
        ):

            raise ValueError(
                "Actual post-embedding dataset is empty."
            )

        dataframe = (
            self.actual_post_dataset.copy()
        )

        dataframe[
            "target_safe_probability"
        ] = dataframe[
            "actual_safe_probability"
        ]

        dataframe[
            "target_risk_probability"
        ] = dataframe[
            "actual_risk_probability"
        ]

        dataframe[
            "target_consequence"
        ] = dataframe[
            "actual_consequence_score"
        ]

        dataframe[
            "target_safety"
        ] = dataframe[
            "actual_safety_score"
        ]

        dataframe[
            "target_distortion"
        ] = dataframe[
            "actual_center_distortion"
        ]

        dataframe[
            "target_neighbor_impact"
        ] = dataframe[
            "actual_neighbor_impact"
        ]

        primary_threshold = 0.85

        secondary_threshold = 0.55

        tertiary_threshold = 0.25

        print(
            "Probability Thresholds"
        )

        print(
            f"Primary   : "
            f"{primary_threshold:.6f}"
        )

        print(
            f"Secondary : "
            f"{secondary_threshold:.6f}"
        )

        print(
            f"Tertiary  : "
            f"{tertiary_threshold:.6f}"
        )

        dataframe[
            "target_class"
        ] = "REJECT"

        dataframe.loc[
            (
                dataframe[
                    "actual_safety_score"
                ] >= primary_threshold
            )
            &
            (
                dataframe[
                    "actual_safe_probability"
                ] >= 0.85
            ),
            "target_class"
        ] = "PRIMARY"

        dataframe.loc[
            (
                dataframe[
                    "actual_safety_score"
                ] < primary_threshold
            )
            &
            (
                dataframe[
                    "actual_safety_score"
                ] >= secondary_threshold
            )
            &
            (
                dataframe[
                    "actual_safe_probability"
                ] >= 0.55
            ),
            "target_class"
        ] = "SECONDARY"

        dataframe.loc[
            (
                dataframe[
                    "actual_safety_score"
                ] < secondary_threshold
            )
            &
            (
                dataframe[
                    "actual_safety_score"
                ] >= tertiary_threshold
            )
            &
            (
                dataframe[
                    "actual_safe_probability"
                ] >= 0.25
            ),
            "target_class"
        ] = "TERTIARY"

        dataframe[
            "target_safe"
        ] = dataframe[
            "actual_safety_score"
        ].astype(
            float
        )

        class_mapping = {
            "PRIMARY": 3,
            "SECONDARY": 2,
            "TERTIARY": 1,
            "REJECT": 0
        }

        dataframe[
            "target_class_id"
        ] = (
            dataframe[
                "target_class"
            ]
            .map(
                class_mapping
            )
            .fillna(0)
            .astype(
                int
            )
        )

        target_path = os.path.join(
            self.output_dir,
            "actual_post_embedding_ml_targets.csv"
        )

        dataframe.to_csv(
            target_path,
            index=False
        )

        self.actual_ml_targets = (
            dataframe
        )

        print(
            f"Actual ML Targets Saved : "
            f"{target_path}"
        )

        return dataframe

    def build_actual_post_ml_feature_dataset(
        self
    ):

        if (
            not hasattr(
                self,
                "actual_post_dataset"
            )
            or
            self.actual_post_dataset.empty
        ):

            raise ValueError(
                "Actual post-embedding dataset is empty."
            )

        dataframe = (
            self.actual_post_dataset.copy()
        )

        target_columns = [

            "actual_safe_probability",

            "actual_risk_probability",

            "actual_consequence_score",

            "actual_safety_score",

            "actual_center_distortion",

            "actual_neighbor_impact",

            "actual_quality_loss",

            "actual_pixel_change",

            "actual_embedding_class",

            "actual_embedding_safe"
        ]

        identifier_columns = [

            "region_id",

            "scenario_id"
        ]

        excluded = (
            set(
                target_columns
                +
                identifier_columns
            )
        )

        feature_columns = [

            column

            for column in dataframe.columns

            if column not in excluded
        ]

        feature_data = {}

        for column in feature_columns:

            numeric = pd.to_numeric(
                dataframe[
                    column
                ],
                errors="coerce"
            )

            if numeric.notna().all():

                feature_data[
                    column
                ] = numeric.astype(
                    np.float64
                ).values

        feature_data[
            "region_id"
        ] = dataframe[
            "region_id"
        ].values

        feature_data[
            "scenario_id"
        ] = dataframe[
            "scenario_id"
        ].values

        for column in target_columns:

            if column in dataframe.columns:

                feature_data[
                    column
                ] = dataframe[
                    column
                ].values

        feature_dataframe = pd.DataFrame(
            feature_data,
            index=dataframe.index
        )

        output_path = os.path.join(
            self.output_dir,
            "actual_post_embedding_ml_dataset.csv"
        )

        feature_dataframe.to_csv(
            output_path,
            index=False
        )

        self.actual_ml_dataset = (
            feature_dataframe
        )

        print(
            f"Actual ML Dataset Rows     : "
            f"{len(feature_dataframe)}"
        )

        print(
            f"Actual ML Dataset Features : "
            f"{len(feature_dataframe.columns)}"
        )

        print(
            f"Actual ML Dataset Saved     : "
            f"{output_path}"
        )

        return feature_dataframe


    def build_actual_post_prediction_targets(
        self
    ):

        if not hasattr(
            self,
            "actual_ml_dataset"
        ):

            self.build_actual_post_ml_feature_dataset()

        dataframe = (
            self.actual_ml_dataset.copy()
        )

        targets = [

            "actual_safe_probability",

            "actual_risk_probability",

            "actual_consequence_score",

            "actual_safety_score",

            "actual_center_distortion",

            "actual_neighbor_impact",

            "actual_quality_loss",

            "actual_pixel_change",

            "actual_embedding_safe"
        ]

        available_targets = [

            column

            for column in targets

            if column in dataframe.columns
        ]

        prediction_targets = dataframe[
            available_targets
        ].copy()

        prediction_path = os.path.join(
            self.output_dir,
            "actual_post_prediction_targets.csv"
        )

        prediction_targets.to_csv(
            prediction_path,
            index=False
        )

        print(
            f"Prediction Targets Saved : "
            f"{prediction_path}"
        )

        return prediction_targets


    def calculate_actual_region_prediction_summary(
        self
    ):

        if (
            not hasattr(
                self,
                "actual_post_dataset"
            )
            or
            self.actual_post_dataset.empty
        ):

            return pd.DataFrame()

        dataframe = (
            self.actual_post_dataset.copy()
        )

        aggregation = {

            "payload_size":
                "max",

            "actual_safe_probability":
                "max",

            "actual_risk_probability":
                "min",

            "actual_consequence_score":
                "min",

            "actual_safety_score":
                "max",

            "actual_center_distortion":
                "min",

            "actual_neighbor_impact":
                "min",

            "actual_quality_loss":
                "min",

            "actual_safe":
                "max"
        }

        aggregation = {

            key: value

            for key, value
            in aggregation.items()

            if key in dataframe.columns
        }

        summary = (
            dataframe
            .groupby(
                "region_id",
                as_index=False
            )
            .agg(
                aggregation
            )
        )

        summary.rename(
            columns={
                "payload_size":
                    "maximum_tested_payload",

                "actual_safe_probability":
                    "maximum_safe_probability",

                "actual_risk_probability":
                    "minimum_risk_probability",

                "actual_consequence_score":
                    "minimum_consequence",

                "actual_safety_score":
                    "maximum_safety",

                "actual_center_distortion":
                    "minimum_center_distortion",

                "actual_neighbor_impact":
                    "minimum_neighbor_impact",

                "actual_quality_loss":
                    "minimum_quality_loss"
            },
            inplace=True
        )

        summary[
            "recommended"
        ] = (
            (
                summary[
                    "maximum_safe_probability"
                ] >= 0.70
            )
            &
            (
                summary[
                    "maximum_safety"
                ] >= 0.70
            )
            &
            (
                summary[
                    "minimum_consequence"
                ] <= 0.30
            )
        )

        summary.sort_values(
            by=[
                "recommended",
                "maximum_safe_probability",
                "maximum_safety"
            ],
            ascending=[
                False,
                False,
                False
            ],
            inplace=True
        )

        summary.reset_index(
            drop=True,
            inplace=True
        )

        summary[
            "prediction_rank"
        ] = (
            np.arange(
                len(summary)
            )
            + 1
        )

        output_path = os.path.join(
            self.output_dir,
            "actual_region_prediction_summary.csv"
        )

        summary.to_csv(
            output_path,
            index=False
        )

        return summary

    def run_actual_post_embedding_dataset_pipeline(
        self,
        embedding_function
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            "ACTUAL POST-EMBEDDING DATASET PIPELINE"
        )

        print(
            "=" * 80
        )

        self.initialize_post_embedding_pipeline()

        seen_regions = set()

        # experiments = [
        #     experiment
        #     for experiment in self.experiment_plan
        #     if 700 <= int(
        #         experiment["region_id"].split("_")[-1]
        #     ) <= 709
        # ]
        experiments = self.experiment_plan
        results = (
            self.execute_post_embedding_experiments(
                embedding_function=
                    embedding_function,
                experiments=
                    experiments
            )
        )

        if not results:

            raise RuntimeError(
                "No post-embedding experiments completed."
            )

        actual_rows = []

        for result in results:

            scenario = result.get(
                "scenario"
            )

            if not scenario:
                continue

            embedding_result = result.get(
                "embedding_result"
            )

            if not embedding_result:
                continue

            region_id = scenario[
                "region_id"
            ]

            workspace = (
                embedding_result.get(
                    "workspace"
                )
            )

            if workspace is None:

                workspace = (
                    embedding_result.get(
                        "center_workspace"
                    )
                )

            if workspace is None:
                continue

            try:
                actual_analysis = embedding_result.get("actual_post_analysis")
                if actual_analysis is None:
                    actual_analysis = self.calculate_actual_post_embedding_analysis(
                        workspace,
                        region_id
                    )

                pre_features = self.load_region_pre_features(region_id)

                row = self.build_actual_post_embedding_feature_row(
                    scenario,
                    actual_analysis,
                    pre_features
                )

                pre_workspace = copy.deepcopy(workspace)
                pre_workspace["modified"] = {
                    "LH": workspace["original"]["LH"].copy(),
                    "HL": workspace["original"]["HL"].copy(),
                    "HH": workspace["original"]["HH"].copy()
                }

                pre_image = self.inverse_dwt_reconstruction(pre_workspace)
                post_image = actual_analysis["reconstructed_image"]

                center_quality = self.calculate_actual_center_quality(
                    pre_image,
                    post_image,
                    region_id,
                    workspace
                )

                row = self.merge_actual_post_quality_features(
                    row,
                    center_quality
                )

                image_quality = self.calculate_actual_post_embedding_quality(
                    pre_image,
                    post_image
                )

                row = self.merge_actual_post_quality_features(
                    row,
                    image_quality
                )
                row = self.append_actual_post_embedding_row(row)
                actual_rows.append(row)

            except Exception as error:

                print(
                    f"Actual analysis failed | "
                    f"Region {region_id} | "
                    f"Scenario "
                    f"{scenario.get('scenario_id')} | "
                    f"{error}"
                )

        if not actual_rows:

            raise RuntimeError(
                "No actual post-embedding rows generated."
            )

        decision_scores = np.asarray(
            [
                float(
                    row.get(
                        "actual_safety_score",
                        row.get(
                            "embedding_safety_score",
                            0.0
                        )
                    )
                )
                for row in actual_rows
            ],
            dtype=np.float64
        )

        primary_threshold = float(
            np.percentile(
                decision_scores,
                85
            )
        )

        secondary_threshold = float(
            np.percentile(
                decision_scores,
                55
            )
        )

        tertiary_threshold = float(
            np.percentile(
                decision_scores,
                25
            )
        )

        print(
            "Safety Score Thresholds"
        )

        print(
            f"Primary   : "
            f"{primary_threshold:.6f}"
        )

        print(
            f"Secondary : "
            f"{secondary_threshold:.6f}"
        )

        print(
            f"Tertiary  : "
            f"{tertiary_threshold:.6f}"
        )

        for row in actual_rows:

            safety_score = float(
                row.get(
                    "actual_safety_score",
                    row.get(
                        "total_safety",
                        0.0
                    )
                )
            )

            if safety_score >= primary_threshold:

                classification = "PRIMARY"

            elif safety_score >= secondary_threshold:

                classification = "SECONDARY"

            elif safety_score >= tertiary_threshold:

                classification = "TERTIARY"

            else:

                classification = "REJECT"

            row[
                "actual_embedding_class"
            ] = classification

            row[
                "actual_embedding_safe"
            ] = int(
                classification != "REJECT"
            )

        final_dataset = (
            self.finalize_actual_post_embedding_dataset()
        )

        self.build_actual_post_ml_targets()

        self.build_actual_post_ml_feature_dataset()

        self.build_actual_post_prediction_targets()

        prediction_summary = (
            self.calculate_actual_region_prediction_summary()
        )

        final_path = os.path.join(
            self.output_dir,
            "actual_post_embedding_complete_dataset.csv"
        )

        self.actual_post_dataset.to_csv(
            final_path,
            index=False
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "ACTUAL POST-EMBEDDING PIPELINE COMPLETED"
        )

        print(
            "=" * 80
        )

        print(
            f"Rows       : "
            f"{len(self.actual_post_dataset)}"
        )

        print(
            f"Features   : "
            f"{len(self.actual_post_dataset.columns)}"
        )

        print(
            f"Regions    : "
            f"{self.actual_post_dataset['region_id'].nunique()}"
        )

        print(
            f"Scenarios  : "
            f"{self.actual_post_dataset['scenario_id'].nunique()}"
        )

        print(
            f"Dataset    : "
            f"{final_path}"
        )

        print(
            f"ML Dataset : "
            f"{os.path.join(self.output_dir, 'actual_post_embedding_ml_dataset.csv')}"
        )

        print(
            "=" * 80
        )

        return {

            "actual_dataset":
                self.actual_post_dataset,

            "ml_dataset":
                self.actual_ml_dataset,

            "targets":
                self.actual_ml_targets,

            "prediction_summary":
                prediction_summary,

            "final_dataset":
                final_path,

            "final_output":
                final_dataset
        }


    def save_complete_post_embedding_manifest(
        self
    ):

        manifest = {

            "dataset":
                "actual_post_embedding_complete_dataset.csv",

            "ml_dataset":
                "actual_post_embedding_ml_dataset.csv",

            "targets":
                "actual_post_embedding_ml_targets.csv",

            "prediction_targets":
                "actual_post_prediction_targets.csv",

            "safe_capacity":
                "actual_safe_capacity.csv",

            "region_ranking":
                "actual_post_region_ranking.csv",

            "region_summary":
                "actual_region_prediction_summary.csv",

            "embedding_subbands":
                [
                    "LH",
                    "HL",
                    "HH"
                ],

            "ignored_subbands":
                [
                    "LL"
                ],

            "post_embedding_analysis":
                True,

            "center_analysis":
                True,

            "eight_neighbor_analysis":
                True,

            "pre_post_comparison":
                True,

            "distortion_analysis":
                True,

            "quality_analysis":
                True,

            "gradient_analysis":
                True,

            "safe_capacity_analysis":
                True,

            "probability_analysis":
                True,

            "ml_target_generation":
                True
        }

        manifest_path = os.path.join(
            self.output_dir,
            "post_embedding_dataset_manifest.json"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                manifest,
                file,
                indent=4
            )

        print(
            f"Post Dataset Manifest Saved : "
            f"{manifest_path}"
        )

        return manifest_path

    def run_complete_post_embedding_dataset_generation(
        self,
        embedding_function
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            "POST-EMBEDDING DATASET GENERATION"
        )

        print(
            "=" * 80
        )

        result = (
            self.run_actual_post_embedding_dataset_pipeline(
                embedding_function
            )
        )
        self.combine_all_post_embedding_datasets()

        manifest_path = (
            self.save_complete_post_embedding_manifest()
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
            f"Complete Dataset : "
            f"{result['final_dataset']}"
        )

        print(
            f"ML Dataset       : "
            f"{os.path.join(self.output_dir, 'actual_post_embedding_ml_dataset.csv')}"
        )

        print(
            f"Targets          : "
            f"{os.path.join(self.output_dir, 'actual_post_embedding_ml_targets.csv')}"
        )

        print(
            f"Safe Capacity    : "
            f"{os.path.join(self.output_dir, 'actual_safe_capacity.csv')}"
        )

        print(
            f"Region Ranking   : "
            f"{os.path.join(self.output_dir, 'actual_post_region_ranking.csv')}"
        )

        print(
            f"Manifest         : "
            f"{manifest_path}"
        )

        print(
            "=" * 80
        )

        return result


    def get_dataset_for_feature_engineering(
        self
    ):

        path = os.path.join(
            self.output_dir,
            "actual_post_embedding_ml_dataset.csv"
        )

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                f"ML dataset not found: {path}"
            )

        dataframe = pd.read_csv(
            path
        )

        if dataframe.empty:

            raise ValueError(
                "ML dataset is empty."
            )

        numeric_columns = (
            dataframe
            .select_dtypes(
                include=[np.number]
            )
            .columns
            .tolist()
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "FEATURE ENGINEERING INPUT READY"
        )

        print(
            "=" * 80
        )

        print(
            f"Rows             : "
            f"{len(dataframe)}"
        )

        print(
            f"Columns          : "
            f"{len(dataframe.columns)}"
        )

        print(
            f"Numeric Features : "
            f"{len(numeric_columns)}"
        )

        print(
            f"Regions          : "
            f"{dataframe['region_id'].nunique()}"
            if "region_id" in dataframe.columns
            else "Regions          : N/A"
        )

        print(
            f"Scenarios        : "
            f"{dataframe['scenario_id'].nunique()}"
            if "scenario_id" in dataframe.columns
            else "Scenarios        : N/A"
        )

        print(
            "=" * 80
        )

        return dataframe

    def validate_feature_engineering_input(
        self
    ):

        dataframe = (
            self.get_dataset_for_feature_engineering()
        )

        required_identifiers = [
            "region_id",
            "scenario_id"
        ]

        for column in required_identifiers:

            if column not in dataframe.columns:

                raise ValueError(
                    f"Required identifier missing: "
                    f"{column}"
                )

        target_columns = [

            "actual_safe_probability",

            "actual_risk_probability",

            "actual_consequence_score",

            "actual_safety_score",

            "actual_center_distortion",

            "actual_neighbor_impact",

            "actual_quality_loss",

            "actual_pixel_change",

            "actual_embedding_safe"
        ]

        missing_targets = [

            column

            for column in target_columns

            if column not in dataframe.columns
        ]

        if missing_targets:

            raise ValueError(
                "Required prediction targets missing: "
                +
                ", ".join(
                    missing_targets
                )
            )

        numeric_columns = (
            dataframe
            .select_dtypes(
                include=[np.number]
            )
            .columns
        )

        invalid_columns = []

        for column in numeric_columns:

            values = pd.to_numeric(
                dataframe[
                    column
                ],
                errors="coerce"
            )

            if values.isna().any():

                invalid_columns.append(
                    column
                )

                continue

            if not np.all(
                np.isfinite(
                    values.to_numpy()
                )
            ):

                invalid_columns.append(
                    column
                )

        if invalid_columns:

            raise ValueError(
                "Invalid numeric columns detected: "
                +
                ", ".join(
                    invalid_columns
                )
            )

        return True


    def save_feature_engineering_input_manifest(
        self
    ):

        dataframe = (
            self.get_dataset_for_feature_engineering()
        )

        numeric_features = (
            dataframe
            .select_dtypes(
                include=[np.number]
            )
            .columns
            .tolist()
        )

        identifier_features = [

            column

            for column in [
                "region_id",
                "scenario_id"
            ]

            if column in dataframe.columns
        ]

        target_features = [

            column

            for column in [

                "actual_safe_probability",

                "actual_risk_probability",

                "actual_consequence_score",

                "actual_safety_score",

                "actual_center_distortion",

                "actual_neighbor_impact",

                "actual_quality_loss",

                "actual_pixel_change",

                "actual_embedding_safe"

            ]

            if column in dataframe.columns
        ]

        manifest = {

            "dataset":
                "actual_post_embedding_ml_dataset.csv",

            "rows":
                int(
                    len(dataframe)
                ),

            "columns":
                int(
                    len(dataframe.columns)
                ),

            "numeric_features":
                int(
                    len(numeric_features)
                ),

            "regions":
                int(
                    dataframe[
                        "region_id"
                    ].nunique()
                ),

            "scenarios":
                int(
                    dataframe[
                        "scenario_id"
                    ].nunique()
                ),

            "identifier_features":
                identifier_features,

            "target_features":
                target_features,

            "feature_engineering_status":
                "READY",

            "next_stage":
                "FEATURE_ENGINEERING"
        }

        path = os.path.join(
            self.output_dir,
            "feature_engineering_input_manifest.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                manifest,
                file,
                indent=4
            )

        print(
            f"Feature Engineering Manifest Saved : "
            f"{path}"
        )

        return path


    def prepare_final_ml_handoff(
        self
    ):

        self.validate_feature_engineering_input()

        dataframe = (
            self.get_dataset_for_feature_engineering()
        )

        manifest_path = (
            self.save_feature_engineering_input_manifest()
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "ML FEATURE ENGINEERING HANDOFF READY"
        )

        print(
            "=" * 80
        )

        print(
            f"Dataset Rows       : "
            f"{len(dataframe)}"
        )

        print(
            f"Dataset Columns    : "
            f"{len(dataframe.columns)}"
        )

        print(
            f"Regions            : "
            f"{dataframe['region_id'].nunique()}"
        )

        print(
            f"Scenarios          : "
            f"{dataframe['scenario_id'].nunique()}"
        )

        print(
            "LL                 : IGNORED"
        )

        print(
            "DWT Embedding      : LH + HL + HH"
        )

        print(
            "Post Embedding     : INCLUDED"
        )

        print(
            "8-Neighbor Impact  : INCLUDED"
        )

        print(
            "Pre/Post Features  : INCLUDED"
        )

        print(
            "Gradient Features  : INCLUDED"
        )

        print(
            "Distortion Targets : INCLUDED"
        )

        print(
            "Probability Targets: INCLUDED"
        )

        print(
            "Safety Targets     : INCLUDED"
        )

        print(
            f"Manifest            : "
            f"{manifest_path}"
        )

        print(
            "=" * 80
        )

        return dataframe

    def prepare_feature_engineering_dataframe(
        self
    ):

        dataframe = (
            self.get_dataset_for_feature_engineering()
            .copy()
        )

        identifier_columns = [
            "region_id",
            "scenario_id"
        ]

        target_columns = [

            "actual_safe_probability",

            "actual_risk_probability",

            "actual_consequence_score",

            "actual_safety_score",

            "actual_center_distortion",

            "actual_neighbor_impact",

            "actual_quality_loss",

            "actual_pixel_change",

            "actual_embedding_safe"
        ]

        excluded_columns = set(
            identifier_columns
            +
            target_columns
        )

        feature_columns = [

            column

            for column in dataframe.columns

            if column not in excluded_columns
        ]

        feature_dataframe = pd.DataFrame(
            index=dataframe.index
        )

        for column in feature_columns:

            numeric = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

            if numeric.notna().all():

                feature_dataframe[
                    column
                ] = numeric.astype(
                    np.float64
                )

        feature_dataframe.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan,
            inplace=True
        )

        feature_dataframe.fillna(
            feature_dataframe.median(
                numeric_only=True
            ),
            inplace=True
        )

        self.feature_engineering_features = (
            feature_dataframe
        )

        self.feature_engineering_targets = (
            dataframe[
                [
                    column
                    for column in target_columns
                    if column in dataframe.columns
                ]
            ].copy()
        )

        self.feature_engineering_identifiers = (
            dataframe[
                [
                    column
                    for column in identifier_columns
                    if column in dataframe.columns
                ]
            ].copy()
        )

        print(
            f"Feature Engineering Input Rows     : "
            f"{len(feature_dataframe)}"
        )

        print(
            f"Feature Engineering Input Features : "
            f"{len(feature_dataframe.columns)}"
        )

        return (
            feature_dataframe,
            self.feature_engineering_targets,
            self.feature_engineering_identifiers
        )


    def remove_constant_features(
        self,
        dataframe
    ):

        variance = dataframe.var(
            numeric_only=True
        )

        constant_columns = (
            variance[
                variance <= 1e-12
            ]
            .index
            .tolist()
        )

        reduced = dataframe.drop(
            columns=constant_columns,
            errors="ignore"
        )

        print(
            f"Constant Features Removed : "
            f"{len(constant_columns)}"
        )

        return reduced, constant_columns


    def remove_high_correlation_features(
        self,
        dataframe,
        threshold=0.95
    ):

        if dataframe.empty:

            return dataframe, []

        correlation = (
            dataframe
            .corr()
            .abs()
        )

        upper = correlation.where(
            np.triu(
                np.ones(
                    correlation.shape
                ),
                k=1
            ).astype(
                bool
            )
        )

        correlated_columns = [

            column

            for column in upper.columns

            if any(
                upper[
                    column
                ]
                > threshold
            )
        ]

        reduced = dataframe.drop(
            columns=correlated_columns,
            errors="ignore"
        )

        print(
            f"Highly Correlated Features Removed : "
            f"{len(correlated_columns)}"
        )

        return reduced, correlated_columns


    def scale_feature_dataframe(
        self,
        dataframe
    ):

        from sklearn.preprocessing import StandardScaler

        if dataframe.empty:

            return dataframe, None

        scaler = StandardScaler()

        scaled_values = scaler.fit_transform(
            dataframe
        )

        scaled_dataframe = pd.DataFrame(
            scaled_values,
            columns=dataframe.columns,
            index=dataframe.index
        )

        self.feature_scaler = scaler

        return (
            scaled_dataframe,
            scaler
        )


    def run_basic_feature_engineering(
        self
    ):

        (
            features,
            targets,
            identifiers
        ) = (
            self.prepare_feature_engineering_dataframe()
        )

        original_count = (
            len(
                features.columns
            )
        )

        features, constant_columns = (
            self.remove_constant_features(
                features
            )
        )

        features, correlated_columns = (
            self.remove_high_correlation_features(
                features
            )
        )

        scaled_features, scaler = (
            self.scale_feature_dataframe(
                features
            )
        )

        self.engineered_features = (
            features
        )

        self.scaled_features = (
            scaled_features
        )

        self.removed_constant_features = (
            constant_columns
        )

        self.removed_correlated_features = (
            correlated_columns
        )

        output_path = os.path.join(
            self.output_dir,
            "engineered_features.csv"
        )

        features.to_csv(
            output_path,
            index=False
        )

        scaled_path = os.path.join(
            self.output_dir,
            "scaled_engineered_features.csv"
        )

        scaled_features.to_csv(
            scaled_path,
            index=False
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "BASIC FEATURE ENGINEERING COMPLETED"
        )

        print(
            "=" * 80
        )

        print(
            f"Original Features  : "
            f"{original_count}"
        )

        print(
            f"Final Features     : "
            f"{len(features.columns)}"
        )

        print(
            f"Constant Removed   : "
            f"{len(constant_columns)}"
        )

        print(
            f"Correlation Removed: "
            f"{len(correlated_columns)}"
        )

        print(
            f"Engineered Dataset : "
            f"{output_path}"
        )

        print(
            f"Scaled Dataset     : "
            f"{scaled_path}"
        )

        print(
            "=" * 80
        )

        return {
            "features":
                features,

            "scaled_features":
                scaled_features,

            "targets":
                targets,

            "identifiers":
                identifiers
        }

    def calculate_feature_importance_lr(
        self,
        features,
        targets
    ):

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score

        results = {}

        target_columns = [
            column
            for column in [
                "actual_safe_probability",
                "actual_risk_probability",
                "actual_consequence_score",
                "actual_safety_score",
                "actual_center_distortion",
                "actual_neighbor_impact",
                "actual_quality_loss",
                "actual_pixel_change"
            ]
            if column in targets.columns
        ]

        for target in target_columns:

            y = pd.to_numeric(
                targets[target],
                errors="coerce"
            )

            valid = y.notna()

            X = features.loc[
                valid
            ]

            y = y.loc[
                valid
            ]

            if len(y) < 3:
                continue

            model = LinearRegression()

            model.fit(
                X,
                y
            )

            prediction = model.predict(
                X
            )

            coefficients = pd.Series(
                model.coef_,
                index=X.columns
            )

            results[target] = {

                "r2":
                    float(
                        r2_score(
                            y,
                            prediction
                        )
                    ),

                "mse":
                    float(
                        mean_squared_error(
                            y,
                            prediction
                        )
                    ),

                "intercept":
                    float(
                        model.intercept_
                    ),

                "coefficients":
                    coefficients.to_dict(),

                "absolute_coefficients":
                    coefficients.abs().to_dict()
            }

        return results


    def calculate_feature_importance_rf(
        self,
        features,
        targets
    ):

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        results = {}

        target_columns = [
            column
            for column in [
                "actual_safe_probability",
                "actual_risk_probability",
                "actual_consequence_score",
                "actual_safety_score",
                "actual_center_distortion",
                "actual_neighbor_impact",
                "actual_quality_loss",
                "actual_pixel_change"
            ]
            if column in targets.columns
        ]

        for target in target_columns:

            y = pd.to_numeric(
                targets[target],
                errors="coerce"
            )

            valid = y.notna()

            X = features.loc[
                valid
            ]

            y = y.loc[
                valid
            ]

            if len(y) < 3:
                continue

            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X,
                y
            )

            prediction = model.predict(
                X
            )

            importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            )

            results[target] = {

                "r2":
                    float(
                        r2_score(
                            y,
                            prediction
                        )
                    ),

                "mse":
                    float(
                        mean_squared_error(
                            y,
                            prediction
                        )
                    ),

                "feature_importance":
                    importance.to_dict()
            }

        return results


    def calculate_feature_importance_gb(
        self,
        features,
        targets
    ):

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        results = {}

        target_columns = [
            column
            for column in [
                "actual_safe_probability",
                "actual_risk_probability",
                "actual_consequence_score",
                "actual_safety_score",
                "actual_center_distortion",
                "actual_neighbor_impact",
                "actual_quality_loss",
                "actual_pixel_change"
            ]
            if column in targets.columns
        ]

        for target in target_columns:

            y = pd.to_numeric(
                targets[target],
                errors="coerce"
            )

            valid = y.notna()

            X = features.loc[
                valid
            ]

            y = y.loc[
                valid
            ]

            if len(y) < 3:
                continue

            model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )

            model.fit(
                X,
                y
            )

            prediction = model.predict(
                X
            )

            importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            )

            results[target] = {

                "r2":
                    float(
                        r2_score(
                            y,
                            prediction
                        )
                    ),

                "mse":
                    float(
                        mean_squared_error(
                            y,
                            prediction
                        )
                    ),

                "feature_importance":
                    importance.to_dict()
            }

        return results


    def combine_model_feature_importance(
        self,
        lr_results,
        rf_results,
        gb_results
    ):

        combined = {}

        targets = set(
            lr_results
        ) | set(
            rf_results
        ) | set(
            gb_results
        )

        for target in targets:

            feature_scores = {}

            lr_features = (
                lr_results
                .get(
                    target,
                    {}
                )
                .get(
                    "absolute_coefficients",
                    {}
                )
            )

            rf_features = (
                rf_results
                .get(
                    target,
                    {}
                )
                .get(
                    "feature_importance",
                    {}
                )
            )

            gb_features = (
                gb_results
                .get(
                    target,
                    {}
                )
                .get(
                    "feature_importance",
                    {}
                )
            )

            all_features = (
                set(
                    lr_features
                )
                |
                set(
                    rf_features
                )
                |
                set(
                    gb_features
                )
            )

            for feature in all_features:

                lr_score = float(
                    lr_features.get(
                        feature,
                        0.0
                    )
                )

                rf_score = float(
                    rf_features.get(
                        feature,
                        0.0
                    )
                )

                gb_score = float(
                    gb_features.get(
                        feature,
                        0.0
                    )
                )

                feature_scores[
                    feature
                ] = {

                    "lr":
                        lr_score,

                    "rf":
                        rf_score,

                    "gb":
                        gb_score,

                    "combined":
                        (
                            0.33
                            * lr_score
                            +
                            0.33
                            * rf_score
                            +
                            0.34
                            * gb_score
                        )
                }

            combined[
                target
            ] = feature_scores

        return combined


    def run_model_feature_importance_analysis(
        self
    ):

        if not hasattr(
            self,
            "engineered_features"
        ):

            raise ValueError(
                "Run basic feature engineering first."
            )

        features = (
            self.engineered_features
        )

        targets = (
            self.feature_engineering_targets
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "MODEL FEATURE IMPORTANCE ANALYSIS"
        )

        print(
            "=" * 80
        )

        print(
            "Running Linear Regression..."
        )

        lr_results = (
            self.calculate_feature_importance_lr(
                features,
                targets
            )
        )

        print(
            "Linear Regression Completed"
        )

        print(
            "Running Random Forest..."
        )

        rf_results = (
            self.calculate_feature_importance_rf(
                features,
                targets
            )
        )

        print(
            "Random Forest Completed"
        )

        print(
            "Running Gradient Boosting..."
        )

        gb_results = (
            self.calculate_feature_importance_gb(
                features,
                targets
            )
        )

        print(
            "Gradient Boosting Completed"
        )

        combined = (
            self.combine_model_feature_importance(
                lr_results,
                rf_results,
                gb_results
            )
        )

        results = {

            "linear_regression":
                lr_results,

            "random_forest":
                rf_results,

            "gradient_boosting":
                gb_results,

            "combined":
                combined
        }

        output_path = os.path.join(
            self.output_dir,
            "model_feature_importance.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4
            )

        print(
            f"Feature Importance Saved : "
            f"{output_path}"
        )

        self.model_feature_importance = (
            results
        )

        return results

    def select_top_features_from_combined_importance(
        self,
        top_k=100
    ):

        if not hasattr(
            self,
            "model_feature_importance"
        ):

            raise ValueError(
                "Run model feature importance analysis first."
            )

        combined = (
            self.model_feature_importance[
                "combined"
            ]
        )

        feature_scores = {}

        for target, features in (
            combined.items()
        ):

            for feature, values in (
                features.items()
            ):

                score = float(
                    values.get(
                        "combined",
                        0.0
                    )
                )

                feature_scores.setdefault(
                    feature,
                    []
                ).append(
                    score
                )

        ranking = []

        for feature, scores in (
            feature_scores.items()
        ):

            scores = np.asarray(
                scores,
                dtype=np.float64
            )

            ranking.append({

                "feature":
                    feature,

                "mean_importance":
                    float(
                        np.mean(
                            scores
                        )
                    ),

                "maximum_importance":
                    float(
                        np.max(
                            scores
                        )
                    ),

                "target_count":
                    int(
                        len(scores)
                    ),

                "importance_sum":
                    float(
                        np.sum(
                            scores
                        )
                    )
            })

        ranking_dataframe = (
            pd.DataFrame(
                ranking
            )
        )

        if ranking_dataframe.empty:

            raise ValueError(
                "No feature importance values available."
            )

        ranking_dataframe.sort_values(
            by=[
                "mean_importance",
                "maximum_importance",
                "importance_sum"
            ],
            ascending=[
                False,
                False,
                False
            ],
            inplace=True
        )

        ranking_dataframe.reset_index(
            drop=True,
            inplace=True
        )

        ranking_dataframe[
            "rank"
        ] = (
            np.arange(
                len(
                    ranking_dataframe
                )
            )
            + 1
        )

        selected = (
            ranking_dataframe
            .head(
                int(top_k)
            )
            .copy()
        )

        selected_features = (
            selected[
                "feature"
            ]
            .tolist()
        )

        self.feature_importance_ranking = (
            ranking_dataframe
        )

        self.selected_features = (
            selected_features
        )

        ranking_path = os.path.join(
            self.output_dir,
            "feature_importance_ranking.csv"
        )

        selected_path = os.path.join(
            self.output_dir,
            "selected_features.csv"
        )

        ranking_dataframe.to_csv(
            ranking_path,
            index=False
        )

        selected.to_csv(
            selected_path,
            index=False
        )

        print(
            f"Features Ranked : "
            f"{len(ranking_dataframe)}"
        )

        print(
            f"Features Selected : "
            f"{len(selected_features)}"
        )

        print(
            f"Ranking Saved : "
            f"{ranking_path}"
        )

        print(
            f"Selected Features Saved : "
            f"{selected_path}"
        )

        return selected_features


    def build_selected_feature_dataset(
        self
    ):

        if not hasattr(
            self,
            "selected_features"
        ):

            raise ValueError(
                "Select features first."
            )

        features = (
            self.engineered_features[
                self.selected_features
            ].copy()
        )

        identifiers = (
            self.feature_engineering_identifiers
            .copy()
        )

        targets = (
            self.feature_engineering_targets
            .copy()
        )

        selected_dataset = pd.concat(
            [
                identifiers.reset_index(
                    drop=True
                ),
                features.reset_index(
                    drop=True
                ),
                targets.reset_index(
                    drop=True
                )
            ],
            axis=1
        )

        output_path = os.path.join(
            self.output_dir,
            "selected_feature_dataset.csv"
        )

        selected_dataset.to_csv(
            output_path,
            index=False
        )

        self.selected_feature_dataset = (
            selected_dataset
        )

        print(
            f"Selected Dataset Rows     : "
            f"{len(selected_dataset)}"
        )

        print(
            f"Selected Dataset Columns : "
            f"{len(selected_dataset.columns)}"
        )

        print(
            f"Selected Dataset Saved   : "
            f"{output_path}"
        )

        return selected_dataset

    def build_complete_ml_input_dataset(
        self
    ):

        if not hasattr(
            self,
            "selected_feature_dataset"
        ):

            raise ValueError(
                "Selected feature dataset has not been built."
            )

        dataset = (
            self.selected_feature_dataset
            .copy()
        )

        if dataset.empty:

            raise ValueError(
                "Selected feature dataset is empty."
            )

        dataset.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan,
            inplace=True
        )

        numeric_columns = (
            dataset
            .select_dtypes(
                include=[np.number]
            )
            .columns
        )

        for column in numeric_columns:

            median = dataset[
                column
            ].median()

            if pd.isna(
                median
            ):

                median = 0.0

            dataset[
                column
            ] = dataset[
                column
            ].fillna(
                median
            )

        dataset.reset_index(
            drop=True,
            inplace=True
        )

        self.complete_ml_input_dataset = (
            dataset
        )

        output_path = os.path.join(
            self.output_dir,
            "complete_ml_input_dataset.csv"
        )

        dataset.to_csv(
            output_path,
            index=False
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "COMPLETE ML INPUT DATASET GENERATED"
        )

        print(
            "=" * 80
        )

        print(
            f"Rows       : "
            f"{len(dataset)}"
        )

        print(
            f"Columns    : "
            f"{len(dataset.columns)}"
        )

        print(
            f"Regions    : "
            f"{dataset['region_id'].nunique()}"
            if "region_id" in dataset.columns
            else "Regions    : N/A"
        )

        print(
            f"Scenarios  : "
            f"{dataset['scenario_id'].nunique()}"
            if "scenario_id" in dataset.columns
            else "Scenarios  : N/A"
        )

        print(
            f"Saved      : "
            f"{output_path}"
        )

        print(
            "=" * 80
        )

        return dataset


    def validate_complete_ml_input_dataset(
        self
    ):

        if not hasattr(
            self,
            "complete_ml_input_dataset"
        ):

            raise ValueError(
                "Complete ML input dataset has not been generated."
            )

        dataset = (
            self.complete_ml_input_dataset
        )

        if dataset.empty:

            raise ValueError(
                "Complete ML input dataset is empty."
            )

        required_identifiers = [

            "region_id",

            "scenario_id"
        ]

        for column in required_identifiers:

            if column not in dataset.columns:

                raise ValueError(
                    f"Missing required identifier: "
                    f"{column}"
                )

        required_targets = [

            "actual_safe_probability",

            "actual_risk_probability",

            "actual_consequence_score",

            "actual_safety_score",

            "actual_center_distortion",

            "actual_neighbor_impact",

            "actual_quality_loss",

            "actual_pixel_change",

            "actual_embedding_safe"
        ]

        missing_targets = [

            column

            for column in required_targets

            if column not in dataset.columns
        ]

        if missing_targets:

            raise ValueError(
                "Missing ML targets: "
                +
                ", ".join(
                    missing_targets
                )
            )

        numeric_columns = (
            dataset
            .select_dtypes(
                include=[np.number]
            )
            .columns
        )

        for column in numeric_columns:

            values = pd.to_numeric(
                dataset[
                    column
                ],
                errors="coerce"
            )

            if values.isna().any():

                raise ValueError(
                    f"Invalid numeric values in "
                    f"column: {column}"
                )

            if not np.all(
                np.isfinite(
                    values.to_numpy()
                )
            ):

                raise ValueError(
                    f"Non-finite values in "
                    f"column: {column}"
                )

        return True


    def save_complete_ml_input_manifest(
        self
    ):

        dataset = (
            self.complete_ml_input_dataset
        )

        feature_columns = [

            column

            for column in dataset.columns

            if column not in [

                "region_id",

                "scenario_id",

                "actual_safe_probability",

                "actual_risk_probability",

                "actual_consequence_score",

                "actual_safety_score",

                "actual_center_distortion",

                "actual_neighbor_impact",

                "actual_quality_loss",

                "actual_pixel_change",

                "actual_embedding_safe"
            ]
        ]

        target_columns = [

            column

            for column in [

                "actual_safe_probability",

                "actual_risk_probability",

                "actual_consequence_score",

                "actual_safety_score",

                "actual_center_distortion",

                "actual_neighbor_impact",

                "actual_quality_loss",

                "actual_pixel_change",

                "actual_embedding_safe"

            ]

            if column in dataset.columns
        ]

        manifest = {

            "dataset_name":
                "complete_ml_input_dataset.csv",

            "rows":
                int(
                    len(dataset)
                ),

            "columns":
                int(
                    len(dataset.columns)
                ),

            "regions":
                int(
                    dataset[
                        "region_id"
                    ].nunique()
                ),

            "scenarios":
                int(
                    dataset[
                        "scenario_id"
                    ].nunique()
                ),

            "feature_count":
                int(
                    len(feature_columns)
                ),

            "target_count":
                int(
                    len(target_columns)
                ),

            "feature_columns":
                feature_columns,

            "target_columns":
                target_columns,

            "identifiers":
                [
                    "region_id",
                    "scenario_id"
                ],

            "ll_subband_used":
                False,

            "embedding_subbands":
                [
                    "LH",
                    "HL",
                    "HH"
                ],

            "center_features":
                True,

            "eight_neighbor_features":
                True,

            "pre_embedding_features":
                True,

            "post_embedding_features":
                True,

            "pre_post_delta_features":
                True,

            "distortion_features":
                True,

            "quality_features":
                True,

            "gradient_features":
                True,

            "safe_capacity_features":
                True,

            "probability_targets":
                True,

            "ml_handoff":
                "READY"
        }

        manifest_path = os.path.join(
            self.output_dir,
            "complete_ml_input_manifest.json"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                manifest,
                file,
                indent=4
            )

        print(
            f"ML Input Manifest Saved : "
            f"{manifest_path}"
        )

        return manifest_path


    def finalize_dataset_builder(
        self
    ):

        dataset = (
            self.build_complete_ml_input_dataset()
        )

        self.validate_complete_ml_input_dataset()

        manifest = (
            self.save_complete_ml_input_manifest()
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "DATASET BUILDER FINALIZED"
        )

        print(
            "=" * 80
        )

        print(
            f"Dataset Rows       : "
            f"{len(dataset)}"
        )

        print(
            f"Dataset Columns    : "
            f"{len(dataset.columns)}"
        )

        print(
            "Train/Test Split   : NOT DONE"
        )

        print(
            "ML Training        : NOT DONE"
        )

        print(
            "Feature Selection  : DEFERRED TO ML MODULE"
        )

        print(
            "Dataset Status     : READY FOR ML"
        )

        print(
            f"Manifest           : "
            f"{manifest}"
        )

        print(
            "=" * 80
        )

        return dataset

    def run_dataset_builder_final(
        self
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            "ML DATASET BUILDER - FINAL RUN"
        )

        print(
            "=" * 80
        )

        try:

            print(
                "\n[1] Loading previous module outputs..."
            )

            self.load_previous_outputs()

            print(
                "[OK] Previous outputs loaded."
            )

            print(
                "\n[2] Building master dataset..."
            )

            self.build_master_dataset()

            print(
                "[OK] Master dataset built."
            )

            print(
                "\n[3] Building post-embedding dataset..."
            )

            if not hasattr(
                self,
                "actual_post_dataset"
            ):

                print(
                    "[INFO] Actual post-embedding dataset "
                    "will be generated by the post-embedding stage."
                )

            print(
                "\n[4] Building complete ML input dataset..."
            )

            if not hasattr(
                self,
                "selected_feature_dataset"
            ):

                self.build_selected_feature_dataset()

            dataset = (
                self.finalize_dataset_builder()
            )

            print(
                "\n"
                + "=" * 80
            )

            print(
                "ML DATASET BUILDER COMPLETED"
            )

            print(
                "=" * 80
            )

            print(
                f"Final Dataset Shape : "
                f"{dataset.shape}"
            )

            print(
                "Ready for Feature Engineering / ML Module."
            )

            print(
                "=" * 80
            )

            return dataset

        except Exception as error:

            print(
                "\n"
                + "=" * 80
            )

            print(
                "ML DATASET BUILDER FAILED"
            )

            print(
                "=" * 80
            )

            print(
                str(error)
            )

            print(
                "=" * 80
            )

            raise


    def get_final_ml_dataset_path(
        self
    ):

        path = os.path.join(
            self.output_dir,
            "complete_ml_input_dataset.csv"
        )

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                f"Final ML dataset not found: {path}"
            )

        return path


    def get_final_ml_dataset(
        self
    ):

        path = (
            self.get_final_ml_dataset_path()
        )

        dataframe = pd.read_csv(
            path
        )

        if dataframe.empty:

            raise ValueError(
                "Final ML dataset is empty."
            )

        return dataframe

    def verify_final_ml_dataset_integrity(
        self
    ):

        dataset = (
            self.get_final_ml_dataset()
        )

        required_columns = [

            "region_id",

            "scenario_id",

            "actual_safe_probability",

            "actual_risk_probability",

            "actual_consequence_score",

            "actual_safety_score",

            "actual_center_distortion",

            "actual_neighbor_impact",

            "actual_quality_loss",

            "actual_pixel_change",

            "actual_embedding_safe"
        ]

        missing = [

            column

            for column in required_columns

            if column not in dataset.columns
        ]

        if missing:

            raise ValueError(
                "Final dataset missing required columns: "
                +
                ", ".join(
                    missing
                )
            )

        numeric_columns = (
            dataset
            .select_dtypes(
                include=[np.number]
            )
            .columns
        )

        invalid_numeric = []

        for column in numeric_columns:

            values = pd.to_numeric(
                dataset[
                    column
                ],
                errors="coerce"
            )

            if values.isna().any():

                invalid_numeric.append(
                    column
                )

                continue

            if not np.all(
                np.isfinite(
                    values.to_numpy()
                )
            ):

                invalid_numeric.append(
                    column
                )

        if invalid_numeric:

            raise ValueError(
                "Invalid numeric columns: "
                +
                ", ".join(
                    invalid_numeric
                )
            )

        duplicate_rows = int(
            dataset.duplicated().sum()
        )

        duplicate_scenarios = 0

        if {
            "region_id",
            "scenario_id"
        }.issubset(
            dataset.columns
        ):

            duplicate_scenarios = int(
                dataset.duplicated(
                    subset=[
                        "region_id",
                        "scenario_id"
                    ]
                ).sum()
            )

        integrity = {

            "status":
                "VALID",

            "rows":
                int(
                    len(dataset)
                ),

            "columns":
                int(
                    len(dataset.columns)
                ),

            "regions":
                int(
                    dataset[
                        "region_id"
                    ].nunique()
                ),

            "scenarios":
                int(
                    dataset[
                        "scenario_id"
                    ].nunique()
                ),

            "duplicate_complete_rows":
                duplicate_rows,

            "duplicate_region_scenarios":
                duplicate_scenarios,

            "missing_required_columns":
                missing,

            "invalid_numeric_columns":
                invalid_numeric,

            "ready_for_feature_engineering":
                True,

            "train_test_split_performed":
                False,

            "ml_training_performed":
                False
        }

        path = os.path.join(
            self.output_dir,
            "final_ml_dataset_integrity.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                integrity,
                file,
                indent=4
            )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "FINAL ML DATASET INTEGRITY CHECK"
        )

        print(
            "=" * 80
        )

        print(
            f"Status             : "
            f"{integrity['status']}"
        )

        print(
            f"Rows               : "
            f"{integrity['rows']}"
        )

        print(
            f"Columns            : "
            f"{integrity['columns']}"
        )

        print(
            f"Regions            : "
            f"{integrity['regions']}"
        )

        print(
            f"Scenarios          : "
            f"{integrity['scenarios']}"
        )

        print(
            f"Duplicate Rows     : "
            f"{integrity['duplicate_complete_rows']}"
        )

        print(
            f"Duplicate Scenarios: "
            f"{integrity['duplicate_region_scenarios']}"
        )

        print(
            "Ready for Feature Engineering : YES"
        )

        print(
            "Train/Test Split               : NO"
        )

        print(
            f"Integrity Report               : "
            f"{path}"
        )

        print(
            "=" * 80
        )

        return integrity
    
    def finalize_ml_dataset_handoff(
        self
    ):

        dataset = (
            self.get_final_ml_dataset()
        )

        integrity = (
            self.verify_final_ml_dataset_integrity()
        )

        handoff = {

            "status":
                "READY_FOR_ML",

            "dataset_path":
                self.get_final_ml_dataset_path(),

            "rows":
                int(
                    len(dataset)
                ),

            "columns":
                int(
                    len(dataset.columns)
                ),

            "region_count":
                int(
                    dataset[
                        "region_id"
                    ].nunique()
                ),

            "scenario_count":
                int(
                    dataset[
                        "scenario_id"
                    ].nunique()
                ),

            "feature_engineering":
                "NEXT_STAGE",

            "train_test_split":
                "NOT_PERFORMED",

            "model_training":
                "NOT_PERFORMED",

            "feature_selection":
                "NOT_PERFORMED",

            "targets_included": [

                "actual_safe_probability",

                "actual_risk_probability",

                "actual_consequence_score",

                "actual_safety_score",

                "actual_center_distortion",

                "actual_neighbor_impact",

                "actual_quality_loss",

                "actual_pixel_change",

                "actual_embedding_safe"
            ],

            "dwt_embedding_bands": [

                "LH",

                "HL",

                "HH"
            ],

            "ll_ignored":
                True,

            "integrity":
                integrity
        }

        path = os.path.join(
            self.output_dir,
            "ml_dataset_handoff.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                handoff,
                file,
                indent=4
            )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "ML DATASET HANDOFF READY"
        )

        print(
            "=" * 80
        )

        print(
            f"Dataset : "
            f"{handoff['dataset_path']}"
        )

        print(
            f"Rows    : "
            f"{handoff['rows']}"
        )

        print(
            f"Columns : "
            f"{handoff['columns']}"
        )

        print(
            f"Regions : "
            f"{handoff['region_count']}"
        )

        print(
            f"Scenarios : "
            f"{handoff['scenario_count']}"
        )

        print(
            "Feature Engineering : NEXT"
        )

        print(
            "Train/Test Split     : NOT HERE"
        )

        print(
            "Model Training       : NOT HERE"
        )

        print(
            f"Handoff Manifest     : "
            f"{path}"
        )

        print(
            "=" * 80
        )

        return handoff

    def combine_all_post_embedding_datasets(self):

        files = [
            "actual_post_embedding_complete_dataset.csv",
            "actual_post_embedding_ml_dataset.csv",
            "actual_post_embedding_ml_targets.csv",
            "actual_post_prediction_targets.csv",
            "actual_safe_capacity.csv",
            "actual_post_region_ranking.csv"
        ]

        frames = []

        for filename in files:

            path = os.path.join(
                self.output_dir,
                filename
            )

            if not os.path.exists(path):
                continue

            dataframe = pd.read_csv(path)

            if not dataframe.empty:
                frames.append(dataframe)

        if not frames:
            raise RuntimeError(
                "No post-embedding datasets found."
            )

        combined = frames[0].copy()

        for dataframe in frames[1:]:

            if (
                "region_id" in combined.columns
                and
                "scenario_id" in combined.columns
                and
                "region_id" in dataframe.columns
                and
                "scenario_id" in dataframe.columns
            ):

                keys = [
                    "region_id",
                    "scenario_id"
                ]

            elif (
                "region_id" in combined.columns
                and
                "region_id" in dataframe.columns
            ):

                keys = [
                    "region_id"
                ]

            else:
                continue

            dataframe = dataframe.copy()

            duplicate_columns = [
                column
                for column in dataframe.columns
                if column in combined.columns
                and column not in keys
            ]

            dataframe = dataframe.drop(
                columns=duplicate_columns
            )

            combined = combined.merge(
                dataframe,
                on=keys,
                how="left"
            )

        final_path = os.path.join(
            self.output_dir,
            "actual_post_embedding_final_dataset.csv"
        )

        combined.to_csv(
            final_path,
            index=False
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "FINAL COMBINED POST-EMBEDDING DATASET"
        )

        print(
            "=" * 80
        )

        print(
            f"Rows       : {len(combined)}"
        )

        print(
            f"Features   : {len(combined.columns)}"
        )

        print(
            f"Dataset    : {final_path}"
        )

        print(
            "=" * 80
        )

        return combined


    def run_final_dataset_stage(
        self
    ):

        dataset = (
            self.finalize_dataset_builder()
        )

        self.verify_final_ml_dataset_integrity()

        handoff = (
            self.finalize_ml_dataset_handoff()
        )

        return {

            "dataset":
                dataset,

            "handoff":
                handoff
        }

    def run(
        self
    ):

        print(
            "\n"
            + "=" * 100
        )

        print(
            "ML DATASET BUILDER"
        )

        print(
            "=" * 100
        )

        try:

            print(
                "\nBuilding final ML dataset..."
            )

            result = (
                self.run_final_dataset_stage()
            )

            print(
                "\n"
                + "=" * 100
            )

            print(
                "ML DATASET BUILDER COMPLETED"
            )

            print(
                "=" * 100
            )

            dataset = result[
                "dataset"
            ]

            print(
                f"Final Dataset Shape : "
                f"{dataset.shape}"
            )

            print(
                f"Final Dataset       : "
                f"{self.get_final_ml_dataset_path()}"
            )

            print(
                "Status              : READY FOR ML"
            )

            print(
                "=" * 100
            )

            return result

        except Exception as error:

            print(
                "\n"
                + "=" * 100
            )

            print(
                "ML DATASET BUILDER FAILED"
            )

            print(
                "=" * 100
            )

            print(
                f"ERROR : {error}"
            )

            print(
                "=" * 100
            )

            raise

    def actual_embedding_function(
        self,
        workspace,
        embedding_request
    ):

        self.validate_dwt_workspace(
            workspace
        )

        payload_bits = embedding_request.get(
            "payload_bits"
        )

        embedding_positions = embedding_request.get(
            "embedding_positions"
        )

        if payload_bits is None:
            raise ValueError(
                "Embedding request does not contain payload_bits."
            )

        if embedding_positions is None:
            raise ValueError(
                "Embedding request does not contain embedding_positions."
            )

        if len(payload_bits) != len(
            embedding_positions
        ):
            raise ValueError(
                "Payload bits and embedding positions "
                "must have the same length."
            )

        modified = {
            "LH":
                workspace["original"]["LH"].copy(),

            "HL":
                workspace["original"]["HL"].copy(),

            "HH":
                workspace["original"]["HH"].copy()
        }

        for bit, position in zip(
            payload_bits,
            embedding_positions
        ):

            subband = position[
                "subband"
            ]

            row = int(
                position["row"]
            )

            column = int(
                position["column"]
            )

            coefficient = float(
                modified[
                    subband
                ][
                    row,
                    column
                ]
            )

            coefficient_integer = int(
                round(
                    coefficient
                )
            )

            coefficient_integer = (
                coefficient_integer
                & ~1
            ) | int(bit)

            modified[
                subband
            ][
                row,
                column
            ] = float(
                coefficient_integer
            )

        return {
            "modified": modified
        }

if __name__ == "__main__":

    engine = PostEmbeddingAnalysis()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "POST-EMBEDDING ANALYSIS STARTING"
    )

    print(
        "=" * 80
    )

    engine.initialize_post_embedding_pipeline()

    print(
        "\nPre-Embedding Dataset Ready."
    )

    print(
        f"Rows        : {len(engine.dataset)}"
    )

    print(
        f"Columns     : {len(engine.dataset.columns)}"
    )

    print(
        f"Experiments : {len(engine.experiment_plan)}"
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
        engine.run_complete_post_embedding_dataset_generation(
            engine.actual_embedding_function
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
    