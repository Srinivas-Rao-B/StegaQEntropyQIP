import os


class BackendManager:

    def __init__(self, service):

        self.service = service

        self.backends = None

        self.backend = None

        self.output_folder = "qrng/output"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )


    def discover_backends(self):

        print("\n" + "=" * 70)
        print("BACKEND DISCOVERY")
        print("=" * 70)

        self.backends = self.service.backends(

            simulator=False,

            operational=True

        )

        print(f"\nAvailable Quantum Backends : {len(self.backends)}\n")

        for backend in self.backends:

            status = backend.status()

            print(f"Backend Name      : {backend.name}")

            print(f"Version           : {backend.backend_version}")

            print(f"Pending Jobs      : {status.pending_jobs}")

            print(f"Operational       : {status.operational}")

            print("-" * 60)

        return self.backends


    def select_backend(self):

        print("\nSelecting Least Busy Backend...\n")

        self.backend = min(

            self.backends,

            key=lambda backend: backend.status().pending_jobs

        )

        print("Selected Backend Successfully.\n")

        return self.backend


    def backend_information(self):

        print("\n" + "=" * 70)
        print("BACKEND INFORMATION")
        print("=" * 70)

        status = self.backend.status()

        target = self.backend.target

        print(f"\nBackend Name          : {self.backend.name}")

        print(f"Backend Version       : {self.backend.backend_version}")

        print(f"Operational           : {status.operational}")

        print(f"Pending Jobs          : {status.pending_jobs}")

        print(f"Physical Qubits       : {self.backend.num_qubits}")

        print(f"Instruction Count     : {len(target.instructions)}")

        print()

        print("Native Instructions")

        print("-" * 60)

        instruction_set = set()

        for instruction in target.instructions:

            gate = instruction[0]

            try:

                gate_name = str(gate.name)

            except Exception:

                gate_name = str(gate)

            instruction_set.add(gate_name)

        for gate in sorted(list(instruction_set)):

            print(gate)

        print()

        with open(

            os.path.join(

                self.output_folder,

                "backend_information.txt"

            ),

            "w"

        ) as file:

            file.write(

                f"Backend Name : {self.backend.name}\n"

            )

            file.write(

                f"Version : {self.backend.backend_version}\n"

            )

            file.write(

                f"Operational : {status.operational}\n"

            )

            file.write(

                f"Pending Jobs : {status.pending_jobs}\n"

            )

            file.write(

                f"Physical Qubits : {self.backend.num_qubits}\n\n"

            )

            file.write(

                "Native Instructions\n"

            )

            file.write(

                "-" * 60 + "\n"

            )

            for gate in sorted(instruction_set):

                file.write(

                    gate + "\n"

                )

        print("Backend Information Saved Successfully.")

        return self.backend
    
    def run(self):

        self.discover_backends()

        self.select_backend()

        self.backend_information()

        return self.backend