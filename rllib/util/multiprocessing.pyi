"""Multi-Processing Utilities."""
from typing import Any, Callable, List, Optional, Tuple

import torch.multiprocessing as mp

def run_parallel_returns(
    function: Callable[..., Any],
    args_list: List[Tuple],
    num_cpu: Optional[int] = ...,
    max_process_time: int = ...,
    max_timeouts: int = ...,
) -> List: ...
def modify_parallel(
    function: Callable[..., None], args_list: List[Tuple], num_cpu: Optional[int] = ...
) -> None: ...
