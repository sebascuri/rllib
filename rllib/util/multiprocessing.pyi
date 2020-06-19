"""Multi-Processing Utilities."""
import torch.multiprocessing as mp
from typing import List, Tuple, Callable, Any, Optional

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
