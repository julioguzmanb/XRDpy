"""Non-modal simulation tasks with immutable inputs and main-thread plotting."""
from copy import deepcopy
from dataclasses import dataclass
import matplotlib.pyplot as plt

from ....analysis.gui.widgets.task_output_dialog import run_task_with_output_dialog
from ...deferred_plotting import capture_plots
from ...parallel import execution_options
from ...parallel import cancellation, check_cancelled, SimulationCancelled
import multiprocessing
from ...results import SimulationResult, targeting_result


@dataclass
class SimulationRequest:
    function: object
    kwargs: dict


def simulation_request(function, **kwargs):
    return SimulationRequest(function, kwargs)


def scan_result(request, output, metadata):
    name = metadata['function']
    values = request.kwargs
    if name == 'simulate_reciprocal':
        return SimulationResult('reciprocal', output, metadata)
    if name == 'target_hkl_near_pixel_fixed_energy':
        return targeting_result(output, metadata)
    if name.startswith('scan_two_'):
        if output.get('format') == 'mapping-slices-v1':
            metadata['points_are_complete'] = True
            return SimulationResult('parameter_mapping', output, metadata)
        from ...mapping_filter import mapping_context
        metadata['require_detector_hit'] = bool(values.get('require_detector_hit', False))
        metadata['points_are_complete'] = not metadata['require_detector_hit']
        metadata['mapping_context'] = mapping_context(values)
        prefix = 'motor' if 'motor1_name' in values else 'param'
        metadata['axes'] = [values[prefix + str(i) + '_name'] for i in (1, 2)]
        metadata['grid_steps'] = [abs(values[prefix + str(i) + '_range'][2]) for i in (1, 2)]
        return SimulationResult('parameter_mapping', {'points': output}, metadata)
    if name.startswith('detector_rotations_collecting'):
        points = output.success_detector_rotations or {}
        if 'scan_ranges' in values:
            names = list(values['scan_ranges'])
            metadata['motor_names'] = names
            metadata['motor_steps'] = {key: abs(value[2]) for key, value in values['scan_ranges'].items()}
            points = {key: {name: [row[name] for row in rows] for name in names}
                      for key, rows in points.items()}
        else:
            metadata['grid_steps'] = [abs(values['angle_range'][2])] * 2
        return SimulationResult('detector_mapping', {'points': points}, metadata)
    return None


def start_simulation(tab, request, button, *, metadata, on_result=None, parallel=False, workers=None):
    if getattr(tab, '_simulation_busy', False):
        return False
    request = SimulationRequest(request.function, deepcopy(request.kwargs))
    metadata = deepcopy(metadata)
    name = metadata['function']

    cancel_event = multiprocessing.get_context('spawn').Event()
    def compute():
        try:
            with cancellation(cancel_event), execution_options(parallel, workers), capture_plots() as plots:
                output = request.function(**request.kwargs)
                check_cancelled()
        except SimulationCancelled:
            print('Simulation stopped safely; incomplete results were discarded.')
            return {'cancelled': True}
        result = scan_result(request, output, metadata)
        return result, plots

    def failed(error):
        tab.last_error_message = error.strip().splitlines()[-1]
        tab.last_error_traceback = error
        tab.run_completed.emit(False, name)

    def completed(payload):
        if isinstance(payload, dict) and payload.get('cancelled'):
            tab.run_completed.emit(False, name)
            return
        result, plots = payload
        with plt.ion():
            if on_result is not None and result is not None:
                on_result(result)
            # Mapping plots are recreated with the selected display mode.
            if result is None or result.kind in ('targeting', 'reciprocal'):
                for function, args, kwargs in plots:
                    function(*args, **kwargs)
        tab.run_completed.emit(True, name)

    tab._simulation_busy = True
    button.setEnabled(False)
    dialog = run_task_with_output_dialog(tab, 'Simulation: ' + name, compute,
                                         on_success=completed, on_error=failed,
                                         cancel_callback=cancel_event.set,
                                         auto_close_on_success=True)
    tab._simulation_dialog = dialog

    def finished():
        tab._simulation_busy = False
        button.setEnabled(True)
        dialog._worker.func = None

    dialog._thread.finished.connect(finished)
    return True
