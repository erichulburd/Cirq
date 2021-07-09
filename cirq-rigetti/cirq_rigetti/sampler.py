##############################################################################
# Copyright 2021 The Cirq Developers
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
from typing import List

from pyquil.api import QuantumComputer
import cirq
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti import circuit_sweep_executors as executors


_default_executor = executors.with_quilc_compilation_and_cirq_parameter_resolution


class RigettiQCSSampler(cirq.Sampler):
    """Construct a sampler for running on Rigetti QCS QuantumComputer. This class supports
    running circuits on QCS quantum hardware as well as pyQuil's quantum virtual machine (QVM).

    Args:
        quantum_computer: A `pyquil.api.QuantumComputer` against which to run the `cirq.Circuit` s.
        executor: A callable that first uses the below `transformer` on `cirq.Circuit` s and
            then executes the transformed circuit on the `quantum_computer`. You may pass your
            own callable or any static method on `CircuitSweepExecutors`.
        transformer: A callable that transforms the `cirq.Circuit` into a `pyquil.Program`.
            You may pass your own callable or any static method on `CircuitTransformers`.
    """

    def __init__(
        self,
        quantum_computer: QuantumComputer,
        executor: executors.CircuitSweepExecutor = _default_executor,
        transformer: transformers.CircuitTransformer = transformers.default,
    ):
        self._quantum_computer = quantum_computer
        self.executor = executor
        self.transformer = transformer

    def run_sweep(
        self,
        program: cirq.Circuit,
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> List[cirq.Result]:
        """This will evaluate results on the circuit for every set of parameters in `params`.

        Args:
            program: Circuit to evaluate for each set of parameters in `params`.
            params: `cirq.Sweepable` of parameters which this function passes to
                `cirq.protocols.resolve_parameters` for evaluating the circuit.
            repetitions: Number of times to run each iteration through the `params`. For a given
                set of parameters, the `cirq.Result` will include a measurement for each repetition.

        Returns:
            A list of `cirq.Result` s.
        """

        resolvers = [r for r in cirq.to_resolvers(params)]
        return self.executor(
            quantum_computer=self._quantum_computer,
            circuit=program,
            resolvers=resolvers,
            repetitions=repetitions,
            transformer=self.transformer,
        )
