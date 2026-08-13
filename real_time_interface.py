import threading
import time
import tkinter as tk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from simulator import create_simulator
from OTSystem.plant import Qube


def create_real_time_simulator():
    return create_simulator(
        "LQR",
        "KalmanEstimator",
        "full_auto",
        "InjectAndListen",
        "NoisePacketGenerator",
        "RandomStep",
        3,
        1,
        {"bias": 0.0, "std": 0.1},
        {"probability": 0.5},
    )


class VisualApp:
    def __init__(self, master, timestep_ms=None):
        self.master = master
        self.fig = Figure(figsize=(8, 9))
        gs = self.fig.add_gridspec(5, 1, height_ratios=[2, 1, 1, 1, 0.3])

        self.ax: Any = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax.grid(False)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        self.theta_ax = self.fig.add_subplot(gs[1, 0])
        self.alpha_ax = self.fig.add_subplot(gs[2, 0], sharex=self.theta_ax)
        self.control_ax = self.fig.add_subplot(gs[3, 0], sharex=self.theta_ax)
        self.detection_ax = self.fig.add_subplot(gs[4, 0], sharex=self.theta_ax)
        self.theta_ax.set_ylabel("theta (deg)")
        self.alpha_ax.set_ylabel("alpha (deg)")
        self.control_ax.set_ylabel("u (V)")
        self.detection_ax.set_xlabel("time (s)")
        self.theta_ax.tick_params(labelbottom=False)
        self.alpha_ax.tick_params(labelbottom=False)
        self.control_ax.tick_params(labelbottom=False)
        self.theta_ax.grid(False)
        self.alpha_ax.grid(False)
        self.control_ax.grid(False)
        self.detection_ax.grid(False)
        self.detection_ax.set_ylabel("detect")
        self.detection_ax.set_yticks([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.simulation = create_real_time_simulator()
        self.plant: Any = self.simulation.ot_system.plant
        self.timestep_ms = int(self.plant.ts * 1000) if timestep_ms is None else timestep_ms

        bottom_frame = tk.Frame(self.master, bg="white")
        bottom_frame.pack(side="bottom", fill="x")
        font_spec = "TkDefaultFont 18"

        self.runstop_canvas = tk.Canvas(bottom_frame, width=40, height=40, highlightthickness=0, bd=0, relief="flat", bg="white")
        self.runstop_canvas.pack(side="left", padx=6, pady=6)
        self._play_id = self.runstop_canvas.create_text(20, 20, text="▶", fill="black", font=font_spec)
        self._stop_id = self.runstop_canvas.create_text(20, 20, text="●", fill="black", font=font_spec)
        self.running = True
        self.runstop_canvas.itemconfigure(self._play_id, state="hidden")
        self.runstop_canvas.itemconfigure(self._stop_id, state="normal")
        self.runstop_canvas.bind("<Button-1>", self._toggle_running)

        self.restart_canvas = tk.Canvas(bottom_frame, width=40, height=40, highlightthickness=0, bd=0, relief="flat", bg="white")
        self.restart_canvas.pack(side="left", padx=6, pady=6)
        self._restart_id = self.restart_canvas.create_text(20, 20, text="↻", fill="black", font=font_spec)
        self.restart_canvas.bind("<Button-1>", self._restart_simulation)

        origin = np.array([0.0, 0.0, 0.0])
        endpoints = self.plant.endpoints
        p0 = endpoints[0]
        p1 = endpoints[1]

        self.line0: Any = self.ax.plot([origin[0], p0[0]], [origin[1], p0[1]], [origin[2], p0[2]], lw=3)[0]
        self.line1: Any = self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], lw=3)[0]

        max_range = self.plant.L0 + self.plant.L1 + 0.1
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        self.ax.set_zlim(-max_range, max_range)

        self.time_hist = [0.0]
        self.theta_hist = [np.degrees(self.plant.theta)]
        self.alpha_hist = [np.degrees(self.plant.alpha)]
        self.theta_est_hist = [None]
        self.alpha_est_hist = [None]
        self.control_hist = [self.plant.voltage]
        self.detection_times = []
        self.detection_vals = []
        self.prev_detected = int(getattr(self.simulation.ot_system, "number_of_detections", 0))

        self.theta_line, = self.theta_ax.plot(self.time_hist, self.theta_hist, lw=1, color="tab:blue", label="theta (true)")
        self.theta_est_line, = self.theta_ax.plot([], [], lw=1, color="tab:orange", ls="--", label="theta (est)")
        self.alpha_line, = self.alpha_ax.plot(self.time_hist, self.alpha_hist, lw=1, color="tab:blue", label="alpha (true)")
        self.alpha_est_line, = self.alpha_ax.plot([], [], lw=1, color="tab:orange", ls="--", label="alpha (est)")
        self.control_line, = self.control_ax.plot(self.time_hist, self.control_hist, lw=1, color="tab:green", label="control (V)")
        self.theta_ax.legend(loc="upper right")
        self.alpha_ax.legend(loc="upper right")
        self.control_ax.legend(loc="upper right")

        self.detection_base_line: Any = self.detection_ax.plot(self.time_hist, [0.0 for _ in self.time_hist], lw=1, color="lightgray")[0]
        self.detection_scatter: Any = self.detection_ax.scatter([], [], color="red", s=30)

        self._sim_thread = None
        self._stop_event = threading.Event()
        self._start_simulation_thread()

    def update(self):
        self.canvas.draw_idle()

    def _start_simulation_thread(self):
        sim_thread = getattr(self, "_sim_thread", None)
        if sim_thread is not None and sim_thread.is_alive():
            return
        self._stop_event.clear()
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

    def _simulation_loop(self):
        while not self._stop_event.is_set() and getattr(self, "running", False):
            t0 = time.perf_counter()
            self.simulation.step()

            plant: Qube = self.simulation.ot_system.plant # type: ignore
            state_est = self.simulation.state_estimate
            theta_est = None if state_est is None else np.degrees(state_est[2])
            alpha_est = None if state_est is None else np.degrees(state_est[3])
            endpoints = plant.endpoints.copy()

            self.master.after(
                0,
                lambda endpoints=endpoints, plant=plant, theta=np.degrees(plant.theta), alpha=np.degrees(plant.alpha), control=plant.voltage, theta_est=theta_est, alpha_est=alpha_est: self._apply_snapshot(endpoints, theta, alpha, control, theta_est, alpha_est),
            )

            elapsed = time.perf_counter() - t0
            to_sleep = plant.ts - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def _apply_snapshot(self, endpoints, theta, alpha, control, theta_est, alpha_est):
        if not getattr(self, "running", False):
            return

        origin = np.array([0.0, 0.0, 0.0])
        p0 = endpoints[0]
        p1 = endpoints[1]
        plant = self.simulation.ot_system.plant

        t = self.time_hist[-1] + plant.ts
        self.time_hist.append(t)
        self.theta_hist.append(theta)
        self.alpha_hist.append(alpha)
        self.theta_est_hist.append(theta_est)
        self.alpha_est_hist.append(alpha_est)
        self.control_hist.append(control)

        time_window_size = 1.0
        max_len = int(time_window_size / plant.ts) + 5
        if len(self.time_hist) > max_len:
            self.time_hist = self.time_hist[-max_len:]
            self.theta_hist = self.theta_hist[-max_len:]
            self.alpha_hist = self.alpha_hist[-max_len:]
            self.theta_est_hist = self.theta_est_hist[-max_len:]
            self.alpha_est_hist = self.alpha_est_hist[-max_len:]
            self.control_hist = self.control_hist[-max_len:]
            t_cutoff = max(0.0, self.time_hist[-1] - time_window_size)
            keep_idx = [i for i, tt in enumerate(self.detection_times) if tt >= t_cutoff]
            self.detection_times = [self.detection_times[i] for i in keep_idx]
            self.detection_vals = [self.detection_vals[i] for i in keep_idx]

        self.theta_line.set_data(self.time_hist, self.theta_hist)
        self.alpha_line.set_data(self.time_hist, self.alpha_hist)
        self.control_line.set_data(self.time_hist, self.control_hist)

        theta_est_times = [tt for tt, value in zip(self.time_hist, self.theta_est_hist) if value is not None]
        theta_est_vals = [value for value in self.theta_est_hist if value is not None]
        alpha_est_times = [tt for tt, value in zip(self.time_hist, self.alpha_est_hist) if value is not None]
        alpha_est_vals = [value for value in self.alpha_est_hist if value is not None]
        self.theta_est_line.set_data(theta_est_times, theta_est_vals)
        self.alpha_est_line.set_data(alpha_est_times, alpha_est_vals)

        current_detected = int(getattr(self.simulation.ot_system, "number_of_detections", self.prev_detected))
        if current_detected > self.prev_detected:
            for _ in range(current_detected - self.prev_detected):
                self.detection_times.append(t)
                self.detection_vals.append(0.0)
            self.prev_detected = current_detected

        self.detection_base_line.set_data(self.time_hist, [0.0 for _ in self.time_hist])
        if self.detection_times:
            self.detection_scatter.set_offsets(np.column_stack((self.detection_times, self.detection_vals)))
        else:
            self.detection_scatter.set_offsets(np.empty((0, 2)))

        self.line0.set_data([origin[0], p0[0]], [origin[1], p0[1]])
        self.line0.set_3d_properties([origin[2], p0[2]])
        self.line1.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        self.line1.set_3d_properties([p0[2], p1[2]])

        t_now = self.time_hist[-1]
        xmin = max(0.0, t_now - time_window_size)
        self.theta_ax.set_xlim(xmin, t_now)
        self.theta_ax.relim()
        self.theta_ax.autoscale_view(scalex=False)
        self.alpha_ax.relim()
        self.alpha_ax.autoscale_view(scalex=False)
        self.control_ax.relim()
        self.control_ax.autoscale_view(scalex=False)
        self.detection_ax.relim()
        self.detection_ax.autoscale_view(scalex=False)

        self.canvas.draw_idle()

    def _toggle_running(self, event=None):
        if getattr(self, "running", False):
            self.running = False
            self._stop_event.set()
            self.runstop_canvas.itemconfigure(self._play_id, state="normal")
            self.runstop_canvas.itemconfigure(self._stop_id, state="hidden")
        else:
            self.running = True
            self.runstop_canvas.itemconfigure(self._play_id, state="hidden")
            self.runstop_canvas.itemconfigure(self._stop_id, state="normal")
            self._start_simulation_thread()

    def _restart_simulation(self, event=None):
        self.running = False
        self._stop_event.set()
        sim_thread = getattr(self, "_sim_thread", None)
        if sim_thread is not None and sim_thread.is_alive():
            sim_thread.join(timeout=0.2)

        self.simulation = create_real_time_simulator()
        self.plant = self.simulation.ot_system.plant
        self.timestep_ms = int(self.plant.ts * 1000)

        self.time_hist = [0.0]
        self.theta_hist = [np.degrees(self.plant.theta)]
        self.alpha_hist = [np.degrees(self.plant.alpha)]
        self.theta_est_hist = [None]
        self.alpha_est_hist = [None]
        self.control_hist = [self.plant.voltage]
        self.detection_times = []
        self.detection_vals = []
        self.prev_detected = int(getattr(self.simulation.ot_system, "number_of_detections", 0))

        endpoints = self.plant.endpoints
        p0 = endpoints[0]
        p1 = endpoints[1]
        origin = np.array([0.0, 0.0, 0.0])
        self.line0.set_data([origin[0], p0[0]], [origin[1], p0[1]])
        self.line0.set_3d_properties([origin[2], p0[2]])  # type: ignore[attr-defined]
        self.line1.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        self.line1.set_3d_properties([p0[2], p1[2]])  # type: ignore[attr-defined]

        self.theta_line.set_data(self.time_hist, self.theta_hist)
        self.theta_est_line.set_data([], [])
        self.alpha_line.set_data(self.time_hist, self.alpha_hist)
        self.alpha_est_line.set_data([], [])
        self.control_line.set_data(self.time_hist, self.control_hist)
        self.detection_base_line.set_data(self.time_hist, [0.0 for _ in self.time_hist])
        self.detection_scatter.set_offsets(np.empty((0, 2)))

        max_range = self.plant.L0 + self.plant.L1 + 0.1
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        self.ax.set_zlim(-max_range, max_range)

        self.running = True
        self._stop_event.clear()
        self.runstop_canvas.itemconfigure(self._play_id, state="hidden")
        self.runstop_canvas.itemconfigure(self._stop_id, state="normal")
        self._start_simulation_thread()
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    root.wm_title("Qube 3D Visual")
    app = VisualApp(root)
    root.mainloop()