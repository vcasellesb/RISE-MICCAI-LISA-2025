import torch
import multiprocessing
import queue
from torch.multiprocessing import Event, Queue, Manager
from time import sleep

from src.preprocessing.preprocessing_inference import preprocess_case


def preprocess_fromfiles_save_to_queue(
    list_of_files: list[str],
    output_filenames: list[str] | None,
    brain_segs_list: list[str],
    preprocessing_kwargs: dict,
    synthsr_kwargs: dict,
    target_queue: Queue,
    done_event: Event,
    abort_event: Event,
    tmpdir: str
):

    try:
        for i, f in enumerate(list_of_files):
            data, data_properties = preprocess_case(lowfield_scan=f,
                                                    brain_seg_path=brain_segs_list[i],
                                                    preprocessing_kwargs=preprocessing_kwargs,
                                                    synthsr_kwargs=synthsr_kwargs,
                                                    tmpdir=tmpdir)

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {
                'data': data,
                'data_properties': data_properties,
                'ofile': output_filenames[i] if output_filenames is not None else None
            }
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        # print(Exception, e)
        abort_event.set()
        raise e

def preprocessing_iterator_from_list(
    list_of_files: list[str],
    output_filenames: list[str] | None,
    brain_seg_paths: list[str],
    preprocessing_kwargs: dict,
    synthsr_kwargs: dict,
    num_processes: int,
    tmpdir,
    pin_memory: bool = False
):

    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_files), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)

        args=(
            list_of_files[i::num_processes],
            output_filenames[i::num_processes] if output_filenames is not None else None,
            brain_seg_paths[i::num_processes],
            preprocessing_kwargs,
            synthsr_kwargs,
            queue,
            event,
            abort_event,
            tmpdir
        )

        pr = context.Process(target=preprocess_fromfiles_save_to_queue, args=args, daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # import IPython;IPython.embed()
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = (
                all([i.is_alive() or j.is_set() for i, j in zip(processes, done_events)])
                and not abort_event.is_set()
            )
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]
