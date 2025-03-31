import numpy as np
from gym.envs.classic_control.acrobot import AcrobotEnv

class CustomAcrobotEnv(AcrobotEnv):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
    def step(self, action):
        ns = self._rk4(self._dsdt, self.state, [0, self.dt])
        self.state = ns[-1]
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        return self._get_ob(), reward, terminal, {}

    def _rk4(self, derivs, y0, t):
        Ny = len(y0)
        yout = np.zeros((len(t), Ny), np.float64)  # Changed np.float_ to np.float64
        yout[0] = y0
        for i in range(len(t) - 1):
            thistime = t[i]
            dt = t[i + 1] - thistime
            dt2 = dt / 2.0
            y0 = yout[i]
            k1 = np.asarray(derivs(y0, thistime))
            k2 = np.asarray(derivs(y0 + dt2 * k1, thistime + dt2))
            k3 = np.asarray(derivs(y0 + dt2 * k2, thistime + dt2))
            k4 = np.asarray(derivs(y0 + dt * k3, thistime + dt))
            yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout
    def render(self, mode='human'):
        if self.render_mode == 'rgb_array':
            return self.render(mode='rgb_array')
        else:
            return super().render(mode)