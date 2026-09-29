from .ibm_connection import IBMConnection
from .backend_manager import BackendManager
from .quantum_circuit import QuantumCircuitManager
from .state_analysis import StateAnalysis
from .transpiler import QuantumTranspiler
from .quantum_execution import QuantumExecution
from .randomness_tests import RandomnessTests
from .qrng_pool import QRNGPool
from .seed_manager import SeedManager
from .qrng_summary import QRNGSummary


class QRNGRandomnessEngine:

    def __init__(self):

        self.connection = None

        self.backend = None

        self.circuit_manager = None

        self.state_analysis = None

        self.transpiler = None

        self.execution = None

        self.randomness = None

        self.circuit_manager=None

        self.pool = None

        self.seed_manager = None

        self.summary = None


    def banner(self):

        print("\n" + "=" * 70)

        print("STEGAQENTROPY")

        print("IBM QUANTUM RANDOMNESS ENGINE")

        print("=" * 70)


    def run(self):

        self.banner()

        print("\nSTEP 1 : IBM CONNECTION")

        self.connection = IBMConnection()

        service = self.connection.run()

        input("\nPress ENTER to continue...")



        print("\nSTEP 2 : BACKEND")

        backend_manager = BackendManager(service)

        self.backend = backend_manager.run()

        input("\nPress ENTER to continue...")



        print("\nSTEP 3 : QUANTUM CIRCUIT")
        

        self.circuit_manager = QuantumCircuitManager(

            self.backend

        )
        analysis_circuit, measurement_circuit = self.circuit_manager.run()
    

        input("\nPress ENTER to continue...")



        print("\nSTEP 4 : STATE ANALYSIS")

        analysis_circuit = self.circuit_manager.analysis_circuit

        self.state_analysis = StateAnalysis(
            analysis_circuit
        )

        self.state_analysis.run()



        print("\nSTEP 5 : TRANSPILATION")

        self.transpiler = QuantumTranspiler(

            self.backend,

            measurement_circuit

        )

        transpiled_circuit = self.transpiler.run()

        input("\nPress ENTER to continue...")



        print("\nSTEP 6 : IBM EXECUTION")

        self.execution = QuantumExecution(

            self.backend,

            transpiled_circuit

        )

        self.execution.run()

        input("\nPress ENTER to continue...")



        print("\nSTEP 7 : RANDOMNESS TESTS")

        self.randomness = RandomnessTests(

            "qrng/output/qrng_pool.txt"

        )

        self.randomness.run()

        input("\nPress ENTER to continue...")



        print("\nSTEP 8 : QRNG POOL")

        self.pool = QRNGPool(

            self.execution

        )

        self.pool.run()

        input("\nPress ENTER to continue...")



        print("\nSTEP 9 : SEED MANAGER")

        self.seed_manager = SeedManager(

            "qrng/output/qrng_pool.txt"

        )

        self.seed_manager.run()

        input("\nPress ENTER to continue...")



        print("\nSTEP 10 : FINAL SUMMARY")

        self.summary = QRNGSummary(

            
            self.backend,

            self.execution,

            self.circuit_manager,

            self.randomness,

            self.pool,

            self.seed_manager

        )

        self.summary.run()

        print("\n" + "=" * 70)

        print("QRNG SUBSYSTEM EXECUTED SUCCESSFULLY")

        print("=" * 70)

        return {

            "backend": self.backend,

            "execution": self.execution,

             "circuit_manager": self.circuit_manager,

             "randomness": self.randomness,

            "pool": self.pool,

            "seed_manager": self.seed_manager,

        }


if __name__ == "__main__":

    engine = QRNGRandomnessEngine()

    engine.run()