from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    SamplerV2
)

print("=" * 70)
print("IBM QUANTUM CONNECTION TEST")
print("=" * 70)

print("\nConnecting to IBM Quantum...")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform"
)

print("Connected Successfully.")

print("\nSearching Available Backends...")

backends = service.backends(
    simulator=False,
    operational=True
)

print(f"Available Backends : {len(backends)}")

for backend in backends:
    print(f"- {backend.name}")

print("\nSelecting Least Busy Backend...")

backend = min(
    backends,
    key=lambda b: b.status().pending_jobs
)

print("\nSelected Backend")
print("-" * 40)
print(f"Name              : {backend.name}")
print(f"Version           : {backend.backend_version}")
print(f"Pending Jobs      : {backend.status().pending_jobs}")
print(f"Operational       : {backend.status().operational}")

print("\nCreating Quantum Circuit...")

qc = QuantumCircuit(1)

qc.h(0)

qc.measure_all()

print(qc)

print("\nTranspiling Circuit...")

transpiled = transpile(
    qc,
    backend=backend,
    optimization_level=3
)

print("Transpilation Completed.\n")

print(transpiled)

print("\nCircuit Information")
print("-" * 40)

print(f"Depth             : {transpiled.depth()}")
print(f"Width             : {transpiled.width()}")
print(f"Size              : {transpiled.size()}")

print("\nGate Counts")

print(transpiled.count_ops())

print("\nCreating Sampler...")

sampler = SamplerV2(
    mode=backend
)

print("Sampler Created.")

print("\nSubmitting Job...")

job = sampler.run(
    [transpiled],
    shots=1024
)

print("Job Submitted Successfully.")

print(f"Job ID : {job.job_id()}")

print("\nWaiting for IBM Quantum...")

result = job.result()

print("\nJob Finished Successfully.")

print("\n================ RESULT ================\n")

print(result)

print("\nIBM Quantum Test Completed Successfully.")

# ---------------------------------------------------------------------------------------------

service = QiskitRuntimeService(
    channel="ibm_quantum_platform"
)

job = service.job(
    "d9c9ash6dkoc73fhclng"
)
print(job.status())

# ---------------------------------------------------------------------------------------------


from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform")

print(service.active_account())

# ---------------------------------------------------------------------------------------------

from qiskit_ibm_runtime import QiskitRuntimeService

print(QiskitRuntimeService.saved_accounts())