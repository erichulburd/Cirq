from typing import cast, Tuple
import cirq
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSService, RigettiQCSSampler, circuit_sweep_executors


def test_parametric_circuit_through_service(
    parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable]
) -> None:
    """
    test that RigettiQCSService can run a basic parametric circuit on the QVM and return an accurate
    ``cirq.study.Result``.
    """
    circuit, sweepable = parametric_circuit_with_params

    qc = get_qc('9q-square', as_qvm=True)
    service = RigettiQCSService(
        quantum_computer=qc,
    )

    # set the seed so we get a deterministic set of results.
    qvm = cast(QVM, qc.qam)
    qvm.random_seed = 0

    repetitions = 10
    param_resolvers = [r for r in cirq.study.to_resolvers(sweepable)]
    result = service.run(
        circuit=circuit,
        repetitions=repetitions,
        param_resolver=param_resolvers[1],
    )
    assert isinstance(result, cirq.study.Result)
    assert sweepable[1] == result.params

    assert 'm' in result.measurements
    assert (repetitions, 1) == result.measurements['m'].shape

    counter = result.histogram(key='m')
    assert 4 == counter[0]
    assert 6 == counter[1]


def test_parametric_circuit_through_sampler(
    parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable]
) -> None:
    """
    test that RigettiQCSSampler can run a basic parametric circuit on the QVM and return an accurate
    list of ``cirq.study.Result``.
    """
    circuit, sweepable = parametric_circuit_with_params

    qc = get_qc('9q-square', as_qvm=True)
    sampler = RigettiQCSSampler(quantum_computer=qc)

    # set the seed so we get a deterministic set of results.
    qvm = cast(QVM, qc.qam)
    qvm.random_seed = 0

    repetitions = 10
    results = sampler.run_sweep(
        program=circuit,
        params=sweepable,
        repetitions=repetitions
    )
    assert len(sweepable) == len(results)

    expected_results = [
        (10, 0),
        (4, 6),
        (0, 10),
        (4, 6),
        (10, 0),
    ]
    for i, result in enumerate(results):
        assert isinstance(result, cirq.study.Result)
        assert sweepable[i] == result.params

        assert 'm' in result.measurements
        assert (repetitions, 1) == result.measurements['m'].shape

        counter = result.histogram(key='m')
        assert expected_results[i][0] == counter[0]
        assert expected_results[i][1] == counter[1]


def test_parametric_circuit_through_sampler_with_parametric_compilation(
    parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable]
) -> None:
    """
    test that RigettiQCSSampler can run a basic parametric circuit on the QVM using parametric
    compilation and return an accurate list of ``cirq.study.Result``.
    """
    circuit, sweepable = parametric_circuit_with_params

    qc = get_qc('9q-square', as_qvm=True)
    sampler = RigettiQCSSampler(
        quantum_computer=qc,
        executor=circuit_sweep_executors.with_quilc_parametric_compilation,
    )

    # set the seed so we get a deterministic set of results.
    qvm = cast(QVM, qc.qam)
    qvm.random_seed = 10

    repetitions = 10
    results = sampler.run_sweep(
        program=circuit,
        params=sweepable,
        repetitions=repetitions
    )
    assert len(sweepable) == len(results)

    expected_results = [
        (10, 0),
        (8, 2),
        (0, 10),
        (8, 2),
        (10, 0),
    ]
    for i, result in enumerate(results):
        assert isinstance(result, cirq.study.Result)
        assert sweepable[i] == result.params

        assert 'm' in result.measurements
        assert (repetitions, 1) == result.measurements['m'].shape

        counter = result.histogram(key='m')
        assert expected_results[i][0] == counter[0]
        assert expected_results[i][1] == counter[1]