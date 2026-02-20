from __future__ import print_function
"""
A gpu management system derived from
the simple discretionary locking system for /dev/nvidia devices.
by Iain Murray, November 2009, January 2010, January 2011.

Axel Roebel, January 2018.

This script implements the main functionalities of the manage_gpus system

The system distinguishes three priorities

hard locks: non interruptible gpu processes
soft locks: interruptile gpu processes for all users alled to create hard locks
no locks: gpu processes without locks will be removed whenever a soft or hard lock is requested

"""


from typing import List, Optional
import sys
import os, os.path
import subprocess
import logging
import pwd
import warnings
import time
import atexit
import psutil
import signal

def create_logger(filename, log_level=logging.INFO, file_mode="a",
                  log_format='%(asctime)s :'+str(os.getpid())+': %(name)s::%(levelname)s::%(message)s',
                  log_date_format='%m/%d/%Y %H:%M:%S') :
    """
    returns a logging object with level configured according to env varibale PY_LOGGING_LEVEL
        with default log_level as specified via argument log_level.
        the name of the logger is specified as argument name.
    """

    logger = logging.getLogger("manage_gpus")
    logger.propagate = False # do not propagate to parent logger (which here is the root logger)

    fileh = logging.FileHandler(filename, mode=file_mode)
    fileh.setFormatter(logging.Formatter(log_format,
                                             datefmt=log_date_format))
    fileh.setLevel(log_level)
    logger.addHandler(fileh)

    logger.setLevel(log_level)
    return logger

# ensure consistence order between cuda and pynvml device numbering
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
try:
    pwd.getpwnam('gpu_manager')
    has_gpu_manager = True
    os.environ["CUDA_VISIBLE_DEVICES"]=""
except KeyError:
    # no gpu_manager so don't touch CUDA_VISIBILE devices
    has_gpu_manager = False

if False and os.path.exists('/dev/shm'):
    # /dev/shm on linux machines is a RAM disk, so is definitely cleared

    # ATTENTION: files and dirs under /dev/shm can be cleaned also when the system is running
    # by system services! don't use this anymore
    _lock_file_dir = '/dev/shm/'

elif os.path.exists('/data/anasynth_nonbp/gpu_locks'):
    _lock_file_dir = '/data/anasynth_nonbp/gpu_locks'
else:
    _lock_file_dir = '/tmp'

_lock_file_dir_lock = os.path.join(_lock_file_dir, "locked")
_gpu_lock_pattern = os.path.join(_lock_file_dir, "gpu_lock_{0:d}")
_guard_info_pattern = os.path.join(_lock_file_dir, "guard_info_{0:d}")
gpu_lock_file_start_string="GPULOCK"

def process_id_from_lock_file_name(lock_file_name) -> Optional[int]:
    """retrieve process id of the process owning the lock from lock file name
     wil return None in case the lock file name does not match the expected pattern (for exmplae for locks taken with versions up to 1.10.2)
    :param  lock_file_name: str containing the lock file name
    :return Optional[int]: the process id of the process owning the lock or None if information is not available.
    """
    basename = os.path.basename(lock_file_name)
    start_part = basename.split("_")[0]
    try :
        if start_part.startswith(gpu_lock_file_start_string):
            procid_str = start_part.split(":")[-1]            
            return int(procid_str)
    except Exception:
        pass
    return None

try:
    uid=int(os.environ['SUDO_UID'])
    sudotag="sudo"
except KeyError:
    uid=os.getuid()
    sudotag=""

username=pwd.getpwuid(uid).pw_name
if sudotag:
    username=username+"_"+sudotag


_dev_prefix = '/dev/nvidia'

_python_prefix = sys.exec_prefix
_manager_bin_dir = os.path.expanduser("~gpu_manager/bin")
_bin_dir = os.path.join(_python_prefix, "bin")
manager_cmd_name = "manage_gpus_cmds.py"
manage_with_sudo_name = "manage_gpu_use_sudo"
my_bin_dir = None
management_script = None

class NoGpuManager(Exception):
    def __init__(self):

        # Call the base class constructor with the parameters it needs
        super(NoGpuManager, self).__init__("this system does not support locking gpus, use all available gpus on your own risk")

class NoGpuAvailable(Exception):
    def __init__(self, msg):

        # Call the base class constructor with the parameters it needs
        super(NoGpuAvailable, self).__init__(msg)


if os.path.exists(os.path.join("/usr/local/bin", manager_cmd_name)):
    my_bin_dir = "/usr/local/bin"
elif os.path.exists(os.path.join(_manager_bin_dir, manager_cmd_name)):
    my_bin_dir=_manager_bin_dir
elif os.path.exists(os.path.join(_bin_dir, manager_cmd_name )):
    my_bin_dir = _bin_dir
else :
    import site
    for dd in site.getsitepackages():
        dd=os.path.dirname(dd)
        found = os.path.exists(os.path.join(dd, 'bin', manager_cmd_name))
        while dd != '/' and (not found):
            dd = os.path.dirname(dd)
            found = os.path.exists(os.path.join(dd, 'bin', manager_cmd_name))
        if found:
            my_bin_dir=os.path.join(dd, 'bin', manager_cmd_name)

_logger = None

def get_logger():
    return (_logger)


if my_bin_dir is not None:
    _logger=create_logger(os.path.join(_lock_file_dir, ".manager_" + username + ".log"))

    management_script=os.path.join(my_bin_dir, manager_cmd_name)
    try:
        sudo_bin_dir = os.path.dirname(os.readlink(management_script))
    except Exception:
        sudo_bin_dir = my_bin_dir

    use_sudo = (os.path.islink(os.path.join(sudo_bin_dir, manage_with_sudo_name))
                or os.path.isfile(os.path.join(sudo_bin_dir, manage_with_sudo_name)))
    if use_sudo:
        sudo=["sudo", "-u", "gpu_manager"]
    else:
        sudo = []

    _logger.info("manage_gpus script {} sudo {}".format(management_script, use_sudo))

def get_user_table(filename="gpu_user_config.txt"):
    import pwd

    user_data = {}
    with open(os.path.join(sys.prefix,"share", "manage_gpus", filename), "r") as fi:
        for ll in fi :
            if ll[0] != "#" and ('=' in ll):
                name, num = ll.split('=')
                user_data[pwd.getpwnam(name).pw_uid] = int(num)

    return user_data


def get_user_table_with_names(filename="gpu_user_config.txt"):
    user_data = {}
    with open(os.path.join(sys.prefix,"share", "manage_gpus", filename), "r") as fi:
        for ll in fi :
            if ll[0] != "#" and ('=' in ll):
                name, num = ll.split('=')
                user_data[name] = int(num)

    return user_data


# Get ID's of NVIDIA boards or None if there are none or if the managementscript is not available.
def board_ids():
    """Returns integer board ids that are free to use on this machine.
    returns None in case no GPUs exist
    raises NoGpuManager if GPUs are avaiable but no management facilities exists 
    """
    if os.path.exists(_dev_prefix+"0") and management_script is not None:
        if not has_gpu_manager:
            raise NoGpuManager()
        ret = subprocess.call(sudo + [management_script, "--board_ids"],
                              stdout=open('/dev/null', 'w'), stderr= subprocess.STDOUT, shell=False)

        # decode available board_ids, that are returned as bin codes
        gpu_ids_hard=[]
        for gpu_id in range(5):
            if 2 ** (2 * gpu_id) & ret:
                gpu_ids_hard.append(gpu_id)
        if gpu_ids_hard:
            return gpu_ids_hard

        gpu_ids_soft=[]
        for gpu_id in range(5):
            if 2 ** (2 * gpu_id + 1)& ret:
                gpu_ids_soft.append(gpu_id)

        return gpu_ids_soft
    else:
        return None


def _obtain_lock(id, hard=True, verbose=False, keep_guard_log=False):
    """Attempts to lock id, returning success as True/False."""
    soft_arg = ["--kind"]
    if hard:
        soft_arg = []

    keep_arg = []
    if keep_guard_log:
        keep_arg = ["--keep_guard_log"]

    verbose_arg =[]
    if verbose:
        verbose_arg = ["--verbose"]

    cmd = sudo + [management_script, "--lock_id", "{0:d}".format(id)] + soft_arg + keep_arg + verbose_arg
    _logger.info("_obtain_lock cmd:" + " ".join(cmd))
    try:
        # On POSIX systems symlink creation is atomic, so this should be a
        # robust locking operation:
        if subprocess.call(cmd, shell=False):
            _logger.info("cmd failed")
            return False

        if not os.environ["CUDA_VISIBLE_DEVICES"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]+",{0:d}".format(id)
        _logger.info("locked {} -> visible devices set to <{}>".format(id, os.environ["CUDA_VISIBLE_DEVICES"]))
        return True
    except Exception as ex:
        import traceback
        tb=traceback.format_exc()
        print("exception in _obtain_lock:", str(ex), tb, file=sys.stderr)
        _logger.info("exception in _obtain_lock:"+ str(ex) + tb)
        return False


def obtain_lock_id(id = -1, hard=True, verbose=False, keep_guard_log=True):
    """
    Finds a free id, locks it and returns integer id, or -1 if none free.

    A process is spawned that will free the lock automatically when the
    process with the pid of the current python process terminates.
    """

    if not has_gpu_manager:
        raise NoGpuManager()

    if id  < 0:
        id = _obtain_lock_id_to_hog(hard=hard, keep_guard_log=keep_guard_log, verbose=verbose)
        if id < 0:
            return -1
    else:
        if not _obtain_lock(id, hard=hard, keep_guard_log=keep_guard_log, verbose=verbose):
            return -1

    # register cleanup handler
    atexit.register(_free_lock, id)
    return id


def _obtain_lock_id_to_hog(hard=True, verbose=False, keep_guard_log=True):
    """
    Finds a free id, locks it and returns integer id, or -1 if none free.

    * Lock must be freed manually *
    """
    soft_arg = ["--kind"]
    if hard:
        soft_arg = []
    keep_arg = []
    if keep_guard_log:
        keep_arg = ["--keep_guard_log"]
    verbose_arg =[]
    if verbose:
        verbose_arg = ["--verbose"]
    
    cmd = sudo + [management_script, "--lock_id"] + soft_arg + keep_arg + verbose_arg
    _logger.info("_obtain_lock_id_to_hog cmd:" + " ".join(cmd))

    try:
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=False, universal_newlines=True)
        out, err = pr.communicate()
        out = out.strip()
        if verbose:
            print("_obtain_lock_id_to_hog: got <{0}>".format( out), file=sys.stderr)
        if pr.returncode or len(out) < 1:
            _logger.info("no log available - manager returned: {0}".format(err))
            print("no lock available - manager returned: {0}".format(err), file=sys.stderr)
            return -1
        id = int(out.split(":")[-1])

        if not os.environ.get("CUDA_VISIBLE_DEVICES", None):
            os.environ["CUDA_VISIBLE_DEVICES"] = "{0:d}".format(id)
        else:
            gpu_ids = set([int(ii) for ii in os.environ["CUDA_VISIBLE_DEVICES"].split(",")])
            gpu_ids.add(id)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in sorted(gpu_ids)])
 
        _logger.info("locked {} -> visible devices set to <{}>".format(id, os.environ["CUDA_VISIBLE_DEVICES"]))
        return id
    except Exception as ex:
        import traceback
        tb=traceback.format_exc()
        print("exception in _obtain_lock_id_to_hog:", str(ex), tb, file=sys.stderr)
        return -1

    return -1


def get_gpu_lock(gpu_device_id=-1, soft=False):
    """
    lock a gpu and return its gpu id

    :param gpu_device_id: the gpu id to lock, if -1 is specified any free gpu will be locked.
    :param soft: request a soft lock even if a hard lock would be possible
    :return: the id of the locked gpu, raises NoGpuAvailable if the specified GPU cannot be obtained
    """

    gpu_id_locked=obtain_lock_id(id=gpu_device_id, hard=not soft)
    if gpu_id_locked < 0:
        # lock removal has time delay of 2 so be sure to have the lock of the last run removed we wait
        # for 3 s here
        time.sleep(3)
        gpu_id_locked=obtain_lock_id(id=gpu_device_id, hard=not soft)
    if gpu_id_locked < 0:
        if gpu_device_id < 0:
            raise NoGpuAvailable("No GPUs available for locking")
        else:
            raise NoGpuAvailable("cannot obtain the selected GPU {0}".format(str(gpu_device_id)))

    return gpu_id_locked


def free_lock(id) :
    warnings.warn("manage_gpus::free_lock:: as tensorflow will never free a GPU, the free_lock call is no longer working, unlocking of GPUs will be performed automatically once your process exits.")
    return False


def _free_lock(id):
    """
    free a GPU lock
    :param id: lock id to free
    :return: True in case of success
    """
    ret = -1
    cmd = sudo + [management_script, "--remove_id", str(id)]

    _logger.info("finish all subprocesses ...")

    proc = None
    try:
        proc = psutil.Process()
    except psutil.NoSuchProcess:
        # process has died in the meantime
        _logger.error("strange we are here and our process does no longer exist ?", file=sys.stderr)
        
    proc_list = kill_all_childs(proc, signal.SIGTERM, True)
    _, alive=psutil.wait_procs(proc_list, timeout=3)
    if alive:
        for proc in alive:

            kill_all_childs(proc, signal.SIGKILL, True, recursive=False)
        _, alive=psutil.wait_procs(proc_list, timeout=3)
        if alive:
            _logger.info("the following processes resisted:")
            for proc in alive:
                _logger.info("  "+str(proc))
        

    if alive:
        # wait for all subprocesses to finish
        done = False
        _logger.info("wait for processes that cannot be killed...")
        while not done:
            try: 
                res = os.wait()
                _logger.info(f"wait results: <{res}>")
                time.sleep(2)
            except Exception as ex:
                _logger.info(f"wait exception: <{ex}>")
                done = True
            
        _logger.info("done.")
    _logger.info("run free_lock cmd:" + " ".join(cmd))

    try:
        ret = subprocess.call(cmd, shell=False)
        if ret == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].replace("{0},".format(str(id)), "")\
                .replace("{0}".format(str(id)), "")
            _logger.info("done - visible devices set to <{}>".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        else:
            _logger.error("failed - visible devices are <{}>".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    except Exception as ex:
        import traceback
        tb=traceback.format_exc()
        print("exception in free_lock:", str(ex), tb, file=sys.stderr)
        _logger.error("exception <{}> <{}>".format(str(ex), tb))
        return
    return ret == 0

def get_sig_name(sigused):
    return str(sigused).split(".")[-1]


def kill_all_childs(process:psutil.Process, sigused, verbose=False, recursive=True, kill_proc = False):

    ret = []
    if "python" in process.name() and any([ True for aa in process.cmdline() if "multiprocessing.resource_tracker" in aa.lower()]):
        # this is a multiprocessing resource tracker process
        # it is expected to cleanup once we quit we should not bother with it
        _logger.info(f"skip killing of multiprocessing resource tracker process pid:{process.pid} name:{process.name()}")
        return ret

    if recursive:
        procs = process.children() 
        for pp in procs:
            ret += kill_all_childs(pp, sigused, verbose=verbose, kill_proc=True)

    if kill_proc:        
        signam = get_sig_name(sigused)
        ret.append(process)
        try:
            if verbose:
                print("send {2} to process: pid {0:d} name: {1} ".format(process.pid, process.name(), signam), file=sys.stderr)
                sys.stderr.flush()
            # simulate CTRL C
            process.send_signal(sigused)
        except psutil.NoSuchProcess:
            # process has died in the meantime
            pass
        except Exception as ex:
            print("cannot send signal to process {0:d} name {1}".format(process.pid, process.name()))

    return ret


def retrieve_my_gpu_locks() -> List[int]:
    """ Search the processes running on the GPUs and recover the CUDA_VISIBLE_DEVICES environment variable
    for all gpus that are used by one of your own parent processes running on the GPU
    
    The function will add the GPU indices to the CUDA_VISIBLE_DEVICES environment variable
    
    :return: the list of all GPU indices that are accessible by the process including the onces that where already owned by the process
    """
    if not has_gpu_manager:
        raise NoGpuManager()

    max_num_gpus_to_try = 10

    my_process = psutil.Process()
    my_pid = my_process.pid
    my_parents = [my_pid]
    a_parent_process = my_process.parent()

    while a_parent_process is not None:
        if isinstance(a_parent_process.cmdline, list):
            parent_cmdline = a_parent_process.cmdline
        else:
            parent_cmdline = a_parent_process.cmdline()
        if parent_cmdline:   # sometimes, zombie or unknown (e.g. [kworker/8:2H])
            if "python" in os.path.basename(parent_cmdline[0]):              
                my_parents.append(a_parent_process.pid)
        a_parent_process = a_parent_process.parent()

    # search whether any of the processes running on the gpus are in the list of my parents
    # if they are gather the GPU id 
    my_gpus = set()       
    for gpu_id in range(max_num_gpus_to_try):
        gpu_lock_link = _gpu_lock_pattern.format(gpu_id)
        if os.path.islink(gpu_lock_link):
            pp = os.path.realpath(gpu_lock_link)
            ss = os.path.basename(pp).split("_")
            if len(ss) > 2 and ss[0].startswith(gpu_lock_file_start_string) and "_".join(ss[1:-1]) == username :
                try:
                    owner_process_pid_str = ss[0].split(":")[-1]                    
                    owner_process_pid = int(owner_process_pid_str)
                    if owner_process_pid in my_parents:
                        my_gpus.add(gpu_id)
                except Exception as ex:
                    pass

    if os.environ.get("CUDA_VISIBLE_DEVICES", None):
        my_gpus.update([int(gpu_id) for gpu_id in os.environ["CUDA_VISIBLE_DEVICES"].split(",")])
    my_gpus = sorted(my_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in my_gpus])
 
    return my_gpus
