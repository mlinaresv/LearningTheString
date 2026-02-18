from joblib import Parallel
import contextlib
import joblib
from joblib.externals.loky import get_reusable_executor

class ParallelWithLogger(Parallel):

    def __init__(self, n_jobs=None, backend=None, return_as="list", verbose=0, timeout=None, pre_dispatch='2 * n_jobs',
                 batch_size='auto',
                 temp_folder=None, max_nbytes='1M', mmap_mode='r', prefer=None, require=None, logger=None):
        super().__init__(n_jobs=n_jobs, backend=backend, return_as=return_as, verbose=verbose, timeout=timeout,
                         pre_dispatch=pre_dispatch, batch_size=batch_size, temp_folder=temp_folder,
                         max_nbytes=max_nbytes, mmap_mode=mmap_mode, prefer=prefer, require=require)
        if logger and verbose:
            logger.info('Instantiating Parallel with logger.')
        self.logger = logger

    def _print(self, msg):
        if not self.verbose:
            return
        if self.logger:
            self.logger.info(msg)
        else:
            super()._print(msg)


@contextlib.contextmanager
def joblib_tqdm(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def force_worker_pool_shutdown():
    get_reusable_executor().shutdown(wait=True)
