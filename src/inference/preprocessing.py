import torch
import multiprocessing
import queue
from torch.multiprocessing import Event, Queue, Manager
from time import sleep

from src.preprocessing.preprocessing import preprocess_case


def preprocess_fromfiles_save_to_queue(
    dict_of_case_dicts: dict[str, dict[str, tuple[str | None] | str | None]],
    output_filenames_truncated: list[str] | None,
    preprocessing_kwargs: dict,
    target_queue: Queue,
    done_event: Event,
    abort_event: Event
):

    try:
        for i, k in enumerate(dict_of_case_dicts):
            data, seg, data_properties = preprocess_case(image_paths=dict_of_case_dicts[k]['images'],
                                                         seg_path = dict_of_case_dicts[k]['seg'],
                                                         preprocessing_kwargs=preprocessing_kwargs)

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {
                'data': data, 
                'data_properties': data_properties,
                'ofile': output_filenames_truncated[i] if output_filenames_truncated is not None else None,
                'identifier': k
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


def preprocessing_iterator_fromfiles(
        dict_of_case_dicts: dict[str, dict[str, tuple[str | None] | str | None]],
        output_filenames_truncated: list[str] | None,
        preprocessing_kwargs: dict,
        num_processes: int,
        pin_memory: bool = False
    ):

    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(dict_of_case_dicts), num_processes)
    identifiers = list(dict_of_case_dicts.keys())
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)

        these_keys = identifiers[i::num_processes]

        args=(
            {k: v for k, v in dict_of_case_dicts.items() if k in these_keys},
            output_filenames_truncated[i::num_processes] if output_filenames_truncated is not None else None,
            preprocessing_kwargs,
            queue,
            event,
            abort_event
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
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
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
