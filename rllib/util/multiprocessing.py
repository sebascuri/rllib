"""Multi-Processing Utilities."""
import torch.multiprocessing as mp


def run_parallel_returns(
    function, args_list, num_cpu=None, max_process_time=300, max_timeouts=4
):
    """Run a function in parallel and gather the return value of the function.

    Parameters
    ----------
    function: callable.
        Function ot execute in parallel.
    args_list: List[Tuple]
        List of tuples of arguments to pass to the function.
    num_cpu: int, optional.
        Number of cpus to run in parallel the code.
    max_process_time: int, optional.
        Maximum number of seconds to run each process.
    max_timeouts: int, optional.
        Maximum number of timeouts tolerated before it raises an Error.

    Returns
    -------
    results: List[Any]
        It returns a list of all the return values of the function in parallel.
        You are in charge of the gathering.

    """
    # Base case
    if max_timeouts == 0:
        return None
    num_cpu = mp.cpu_count() if num_cpu is None else num_cpu

    if len(args_list) == 1:
        results = [function(*args_list[0])]
    elif num_cpu <= 1:
        results = []
        for args in args_list:
            results.append(function(*args))
    else:
        pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
        parallel_runs = [pool.apply_async(function, args=args) for args in args_list]
        try:
            results = [p.get(timeout=max_process_time) for p in parallel_runs]
        except Exception as e:
            print(str(e))
            print("Timeout Error raised... Trying again")
            pool.close()
            pool.terminate()
            pool.join()
            return run_parallel_returns(
                function, args_list, num_cpu, max_process_time, max_timeouts - 1
            )

        pool.close()
        pool.terminate()
        pool.join()

    return results


def modify_parallel(function, args_list, num_cpu=None):
    """Run a function that mutates a variable inside the args_list parallel.

    Parameters
    ----------
    function: callable.
        Function ot execute in parallel.
    args_list: List[Tuple]
        List of tuples of arguments to pass to the function.
    num_cpu: int, optional.
        Number of cpus to run in parallel the code.

    Returns
    -------
    None.

    Notes
    -----
    Remember to call tensor.share_memory_() or module.share_memory() before calling this
    function.
    """
    num_cpu = mp.cpu_count() if num_cpu is None else num_cpu
    num_calls = len(args_list)
    if num_calls == 1:
        function(*args_list[0])
    else:
        processes = []
        for rank in range(num_calls):
            p = mp.Process(target=function, args=(*args_list[rank],))
            processes.append(p)

        for p in processes:
            p.join()
