from qiskit_ibm_runtime import QiskitRuntimeService


class IBMConnection:

    def __init__(self):

        self.service = None

    def connect(self):

        print("\n" + "=" * 70)
        print("IBM QUANTUM CONNECTION")
        print("=" * 70)

        print("\nConnecting to IBM Quantum Platform...")

        self.service = QiskitRuntimeService(
            channel="ibm_quantum_platform"
        )

        print("Connection Established Successfully.")

        return self.service
    
    def run(self):

        return self.connect()