import os


class MLConfiguration:

    def __init__(self):

        self.output_directory = os.path.join(
            "ml_module",
            "output"
        )

        self.dataset_directory = os.path.join(
            self.output_directory,
            "dataset_builder"
        )


        self.pointer_file = os.path.join(
            self.output_directory,
            "ml_pointer.json"
        )

        self.create_directories()

    def create_directories(self):

        directories = [

            self.output_directory,

            self.dataset_directory,


        ]

        for directory in directories:

            os.makedirs(
                directory,
                exist_ok=True
            )