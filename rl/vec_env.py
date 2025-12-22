import multiprocessing as mp
import numpy as np
import gymnasium as gym
import traceback

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    ob, _ = env.reset()
                remote.send((ob, reward, terminated or truncated, info))
            elif cmd == 'reset':
                ob, _ = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                attr_name, value = data
                if isinstance(value, CloudpickleWrapper):
                    value = value.x
                setattr(env, attr_name, value)
                remote.send(True) # Ack
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    except Exception:
        print('SubprocVecEnv worker: got Exception')
        traceback.print_exc()
    finally:
        env.close()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv:
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses.
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.num_envs = nenvs

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def env_method(self, method_name, *args, **kwargs):
        for remote in self.remotes:
            remote.send(('env_method', (method_name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]

    def get_attr(self, attr_name):
        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]

    def set_attr(self, attr_name, value):
        wrapper = CloudpickleWrapper(value)
        for remote in self.remotes:
            remote.send(('set_attr', (attr_name, wrapper)))
        return [remote.recv() for remote in self.remotes]
