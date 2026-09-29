import os
import json
import hashlib
import numpy as np


class MLRandomManager:

    def __init__(self):

        self.seed_file = (

            "qrng/output/history_check/final_verified_seeds/"
            "linear_regression_seed_verified.txt"

        )

        self.pointer_file = (

            "ml_module/output/ml_pointer.json"

        )

        self.pointer_key = "linear_regression_pointer"

        self.seed_bits = ""

        self.pointer = 0

        self.load_seed()

        self.load_pointer()


    def load_seed(self):

        if not os.path.exists(

            self.seed_file

        ):

            raise FileNotFoundError(

                self.seed_file

            )

        with open(

            self.seed_file,

            "r",

            encoding="utf-8"

        ) as file:

            data = file.read()

        self.seed_bits = "".join(

            character

            for character in data

            if character in "01"

        )

        if len(

            self.seed_bits

        ) == 0:

            raise ValueError(

                "Seed file is empty."

            )


    def load_pointer(self):

        directory = os.path.dirname(

            self.pointer_file

        )

        os.makedirs(

            directory,

            exist_ok=True

        )

        if not os.path.exists(

            self.pointer_file

        ):

            self.pointer = 0

            self.save_pointer()

            return

        with open(

            self.pointer_file,

            "r",

            encoding="utf-8"

        ) as file:

            pointer_data = json.load(file)

        self.pointer = pointer_data.get(

            self.pointer_key,

            0

        )


    def save_pointer(self):

        data = {}

        if os.path.exists(

            self.pointer_file

        ):

            with open(

                self.pointer_file,

                "r",

                encoding="utf-8"

            ) as file:

                try:

                    data = json.load(file)

                except:

                    data = {}

        data[

            self.pointer_key

        ] = self.pointer

        with open(

            self.pointer_file,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                data,

                file,

                indent=4

            )


    def read_bits(

        self,

        number_of_bits

    ):

        if (

            self.pointer +

            number_of_bits

        ) > len(

            self.seed_bits

        ):

            self.pointer = 0

        output = self.seed_bits[

            self.pointer:

            self.pointer +

            number_of_bits

        ]

        self.pointer += number_of_bits

        self.save_pointer()

        return output


    def get_integer(

        self,

        minimum,

        maximum

    ):

        binary = self.read_bits(

            32

        )

        integer = int(

            binary,

            2

        )

        value = minimum + (

            integer %

            (

                maximum -

                minimum +

                1

            )

        )

        return value


    def get_float(self):

        binary = self.read_bits(

            32

        )

        integer = int(

            binary,

            2

        )

        return integer / (

            2 ** 32

        )


    def get_boolean(self):

        return bool(

            self.get_integer(

                0,

                1

            )

        )


    def random_choice(

        self,

        values

    ):

        if len(

            values

        ) == 0:

            raise ValueError(

                "Empty sequence."

            )

        index = self.get_integer(

            0,

            len(values) - 1

        )

        return values[index]

    def random_sample(

        self,

        values,

        sample_size

    ):

        values = list(values)

        output = []

        sample_size = min(

            sample_size,

            len(values)

        )

        while len(

            output

        ) < sample_size:

            index = self.get_integer(

                0,

                len(values) - 1

            )

            output.append(

                values.pop(index)

            )

        return output


    def random_permutation(

        self,

        values

    ):

        values = list(values)

        output = []

        while len(values) > 0:

            index = self.get_integer(

                0,

                len(values) - 1

            )

            output.append(

                values.pop(index)

            )

        return output


    def shuffle_numpy(

        self,

        array

    ):

        indices = list(

            range(

                len(array)

            )

        )

        indices = self.random_permutation(

            indices

        )

        return array[

            indices

        ]


    def split_indices(

        self,

        total_size,

        training_ratio=0.70,

        validation_ratio=0.15,

        testing_ratio=0.15

    ):

        indices = list(

            range(total_size)

        )

        indices = self.random_permutation(

            indices

        )

        train_end = round(
            total_size *
            training_ratio
        )

        validation_end = train_end + round(
            total_size *
            validation_ratio
        )

        training = indices[

            :train_end

        ]

        validation = indices[

            train_end:

            validation_end

        ]

        testing = indices[

            validation_end:

        ]

        return (

            training,

            validation,

            testing

        )


    def generate_random_state(self):

        return self.get_integer(

            1,

            2147483647

        )


    def generate_hash(self):

        binary = self.read_bits(

            256

        )

        return hashlib.sha256(

            binary.encode()

        ).hexdigest()


    def generate_probability_vector(

        self,

        length

    ):

        probabilities = []

        for _ in range(length):

            probabilities.append(

                self.get_float()

            )

        probabilities = np.array(

            probabilities,

            dtype=np.float64

        )

        total = np.sum(

            probabilities

        )

        if total == 0:

            probabilities += 1.0

            total = np.sum(

                probabilities

            )

        probabilities /= total

        return probabilities


    def reset_pointer(self):

        self.pointer = 0

        self.save_pointer()


    def get_pointer(self):

        return self.pointer


    def get_remaining_bits(self):

        return len(

            self.seed_bits

        ) - self.pointer


    def get_seed_length(self):

        return len(

            self.seed_bits

        )


if __name__ == "__main__":

    manager = MLRandomManager()

    print(

        "\nQRNG Integer :",

        manager.get_integer(

            1,

            100

        )

    )

    print(

        "QRNG Float :",

        manager.get_float()

    )

    print(

        "QRNG Random State :",

        manager.generate_random_state()

    )

    print(

        "Remaining Bits :",

        manager.get_remaining_bits()

    )