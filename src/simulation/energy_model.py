"""
energy_model.py
---------------
Complete energy model for CF-SAF (Equations 8-11 in the paper).

Covers three cost components:
  1. CPU execution energy (dynamic + idle)
  2. Communication energy (gateway Tx/Rx)
  3. Task migration energy (fog-to-fog transfer)
"""


class EnergyModel:
    """
    Parameters (from config.yaml / paper Section IV-D):
      p_tx_w      : Gateway transmit power (0.5 W, LoRaWAN class)
      p_rx_w      : Gateway receive power  (0.3 W)
      p_tx_wan_w  : WAN transmit power     (2.1 W)
      ff_bw_mbps  : Fog-to-fog bandwidth   (100 Mbps)
      gf_bw_mbps  : Gateway-to-fog bandwidth
    """

    def __init__(self,
                 p_tx_w:     float = 0.50,
                 p_rx_w:     float = 0.30,
                 p_tx_wan_w: float = 2.10,
                 ff_bw_mbps: float = 100.0,
                 gf_bw_mbps: float = 1000.0):
        self.p_tx      = p_tx_w
        self.p_rx      = p_rx_w
        self.p_tx_wan  = p_tx_wan_w
        self.ff_bw     = ff_bw_mbps * 1e6 / 8  # bits/s -> bytes/s
        self.gf_bw     = gf_bw_mbps * 1e6 / 8

    # ------------------------------------------------------------------
    # Equation 8 — CPU execution energy
    # ------------------------------------------------------------------
    def cpu_energy(self, exec_time_s: float,
                   p_dyn_w: float, p_idle_w: float,
                   epoch_s: float, n_tasks: int) -> float:
        """
        E_cpu = P_dyn * E_j + P_idle * (T_epoch - E_j) / n_tasks
        """
        idle_share = max(0.0, epoch_s - exec_time_s) / max(n_tasks, 1)
        return p_dyn_w * exec_time_s + p_idle_w * idle_share

    # ------------------------------------------------------------------
    # Equation 9 — Communication energy
    # ------------------------------------------------------------------
    def comm_energy(self, rho_kb: float,
                    download_latency_s: float,
                    is_cloud: bool = False) -> float:
        """
        E_comm = P_tx * T_upload + P_rx * T_download
        T_upload = rho / B_gf  (or B_wan for cloud)
        """
        rho_bytes = rho_kb * 1024.0
        bw = self.gf_bw if not is_cloud else (self.p_tx_wan * 1e6 / 8)
        t_upload = rho_bytes / max(bw, 1.0)
        p_tx = self.p_tx if not is_cloud else self.p_tx_wan
        return p_tx * t_upload + self.p_rx * download_latency_s

    # ------------------------------------------------------------------
    # Equation 10 — Task migration energy
    # ------------------------------------------------------------------
    def migration_energy(self, rho_kb: float) -> float:
        """
        E_mig = P_tx * rho / B_ff
        Only queued (not executing) tasks migrate.
        """
        rho_bytes = rho_kb * 1024.0
        t_transfer = rho_bytes / max(self.ff_bw, 1.0)
        return self.p_tx * t_transfer

    # ------------------------------------------------------------------
    # Equation 11 — Total per-task energy
    # ------------------------------------------------------------------
    def total_energy(self, exec_time_s: float,
                     rho_kb: float,
                     download_latency_s: float,
                     p_dyn_w: float,
                     p_idle_w: float,
                     epoch_s: float,
                     n_tasks: int,
                     migrated: bool = False,
                     is_cloud: bool = False) -> float:
        """
        E_j = E_cpu + E_comm + E_mig
        """
        e_cpu  = self.cpu_energy(exec_time_s, p_dyn_w, p_idle_w,
                                 epoch_s, n_tasks)
        e_comm = self.comm_energy(rho_kb, download_latency_s, is_cloud)
        e_mig  = self.migration_energy(rho_kb) if migrated else 0.0
        return e_cpu + e_comm + e_mig
