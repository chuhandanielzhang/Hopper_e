"""
Domain Randomization 模块
参考 Hopper_rl_t-master 实现，用于 Sim-to-Real Zero-Shot

使用方法:
    from utils.domain_randomization import DomainRandomizer
    
    randomizer = DomainRandomizer(model, data)
    randomizer.randomize_all()  # 每次 reset 时调用
"""

import numpy as np
import mujoco


class DomainRandomConfig:
    """Domain Randomization 配置（与 hopper_rl_t-master 完全一致）"""
    
    # ========== 质量/惯量随机化 (与 RL 版本一致) ==========
    randomize_base_mass = True
    added_mass_range = [-0.5, 0.5]  # kg (与 RL 一致)
    
    randomize_base_com = True
    com_displacement_range = [-0.12, 0.12]  # m (与 RL 一致: ±12cm)
    
    randomize_base_inertia = False  # 与 RL 一致: 默认关闭
    base_inertia_ratio_range = [0.8, 1.2]
    
    randomize_all_link_mass = True
    link_mass_ratio = [0.8, 1.2]
    
    randomize_all_link_inertia = True
    link_inertia_ratio_range = [0.8, 1.2]
    
    # ========== 摩擦随机化 (与 RL 版本一致) ==========
    randomize_friction = False  # 与 RL 一致: 默认关闭
    friction_range = [0.5, 1.25]
    
    # ========== 电机随机化 (与 RL 版本一致) ==========
    randomize_motor_strength = True
    motor_strength_ratio_range = [0.8, 1.2]
    
    randomize_joint_damping = False  # RL 版本用 pd_ratio 而不是 damping
    damping_ratio_range = [0.8, 1.2]
    
    # 新增：电机零点偏移 (与 RL 一致)
    add_motor_offset = True
    motor_offset_range = [-0.05, 0.05]  # rad (与 RL 一致)
    
    # ========== 传感器偏差 (与 RL 版本一致) ==========
    randomize_imu_offset = True
    imu_rpy_offset_range = [-0.05, 0.05]  # rad (与 RL 一致: ±0.05rad)
    
    # ========== 观测噪声 ==========
    add_observation_noise = True
    noise_scales = {
        'joint_pos': 0.015,    # rad
        'joint_vel': 1.5,      # rad/s
        'lin_vel': 0.1,        # m/s
        'ang_vel': 0.5,        # rad/s
        'quat': 0.01,          # 
    }
    
    # ========== 通信延迟 ==========
    add_delay = True
    delay_prob = 0.5
    delay_steps_range = [0, 2]  # 0~2 steps @ 1kHz = 0~2ms


class DomainRandomizer:
    """
    MuJoCo Domain Randomization
    
    在每次 episode reset 时调用 randomize_all() 来随机化物理参数
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 config: DomainRandomConfig = None):
        self.model = model
        self.data = data
        self.config = config or DomainRandomConfig()
        
        # 保存原始参数
        self._save_original_params()
        
        # 当前随机化的参数
        self.motor_strength = np.ones(3)
        self.motor_offset = np.zeros(3)  # 电机零点偏移 (与 RL 一致)
        self.imu_offset = np.zeros(3)  # rpy offset
        self.observation_delay_buffer = []
        
    def _save_original_params(self):
        """保存原始物理参数，用于随机化时作为基准"""
        self.original_masses = self.model.body_mass.copy()
        self.original_inertias = self.model.body_inertia.copy()
        self.original_friction = self.model.geom_friction.copy()
        self.original_damping = self.model.dof_damping.copy()
        
    def randomize_all(self):
        """随机化所有参数（在 episode reset 时调用）"""
        cfg = self.config
        
        if cfg.randomize_base_mass:
            self._randomize_base_mass()
            
        if cfg.randomize_base_com:
            self._randomize_base_com()
            
        if cfg.randomize_base_inertia:
            self._randomize_base_inertia()
            
        if cfg.randomize_all_link_mass:
            self._randomize_link_masses()
            
        if cfg.randomize_all_link_inertia:
            self._randomize_link_inertias()
            
        if cfg.randomize_friction:
            self._randomize_friction()
            
        if cfg.randomize_motor_strength:
            self._randomize_motor_strength()
            
        if cfg.randomize_joint_damping:
            self._randomize_joint_damping()
            
        if cfg.add_motor_offset:
            self._randomize_motor_offset()
            
        if cfg.randomize_imu_offset:
            self._randomize_imu_offset()
            
    def _randomize_base_mass(self):
        """随机化机体质量"""
        rng = self.config.added_mass_range
        delta_mass = np.random.uniform(rng[0], rng[1])
        
        # 找到 base_link body
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_id >= 0:
            self.model.body_mass[base_id] = self.original_masses[base_id] + delta_mass
            
    def _randomize_base_com(self):
        """随机化机体质心"""
        rng = self.config.com_displacement_range
        delta_com = np.random.uniform(rng[0], rng[1], 3)
        
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_id >= 0:
            self.model.body_ipos[base_id] += delta_com * np.array([0.5, 0.3, 0.3])
            
    def _randomize_base_inertia(self):
        """随机化机体惯量"""
        rng = self.config.base_inertia_ratio_range
        ratio = np.random.uniform(rng[0], rng[1], 3)
        
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_id >= 0:
            self.model.body_inertia[base_id] = self.original_inertias[base_id] * ratio
            
    def _randomize_link_masses(self):
        """随机化所有连杆质量"""
        rng = self.config.link_mass_ratio
        for i in range(self.model.nbody):
            if i > 0:  # 跳过 world body
                ratio = np.random.uniform(rng[0], rng[1])
                self.model.body_mass[i] = self.original_masses[i] * ratio
                
    def _randomize_link_inertias(self):
        """随机化所有连杆惯量"""
        rng = self.config.link_inertia_ratio_range
        for i in range(self.model.nbody):
            if i > 0:
                ratio = np.random.uniform(rng[0], rng[1], 3)
                self.model.body_inertia[i] = self.original_inertias[i] * ratio
                
    def _randomize_friction(self):
        """随机化地面摩擦"""
        rng = self.config.friction_range
        
        # 找到 ground geom
        ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        if ground_id >= 0:
            ratio = np.random.uniform(rng[0], rng[1])
            self.model.geom_friction[ground_id, 0] = self.original_friction[ground_id, 0] * ratio
            
    def _randomize_motor_strength(self):
        """随机化电机出力"""
        rng = self.config.motor_strength_ratio_range
        self.motor_strength = np.random.uniform(rng[0], rng[1], 3)
        
    def _randomize_joint_damping(self):
        """随机化关节阻尼"""
        rng = self.config.damping_ratio_range
        for i in range(self.model.nv):
            ratio = np.random.uniform(rng[0], rng[1])
            self.model.dof_damping[i] = self.original_damping[i] * ratio
            
    def _randomize_motor_offset(self):
        """随机化电机零点偏移 (与 RL 版本一致)"""
        rng = self.config.motor_offset_range
        self.motor_offset = np.random.uniform(rng[0], rng[1], 3)
        
    def _randomize_imu_offset(self):
        """随机化 IMU 零点偏移"""
        rng = self.config.imu_rpy_offset_range
        self.imu_offset = np.random.uniform(rng[0], rng[1], 3)
        
    def apply_motor_strength(self, torques: np.ndarray) -> np.ndarray:
        """应用电机强度随机化到扭矩"""
        return torques * self.motor_strength
    
    def apply_motor_offset(self, joint_pos: np.ndarray) -> np.ndarray:
        """应用电机零点偏移到关节位置 (与 RL 版本一致)"""
        return joint_pos - self.motor_offset
    
    def apply_imu_offset(self, rpy: np.ndarray) -> np.ndarray:
        """应用 IMU 偏移到姿态"""
        return rpy + self.imu_offset
    
    def add_observation_noise(self, obs: dict) -> dict:
        """给观测添加噪声"""
        if not self.config.add_observation_noise:
            return obs
            
        noisy_obs = obs.copy()
        scales = self.config.noise_scales
        
        if 'joint_pos' in obs:
            noise = np.random.randn(3) * scales['joint_pos']
            noisy_obs['joint_pos'] = obs['joint_pos'] + noise
            
        if 'joint_vel' in obs:
            noise = np.random.randn(3) * scales['joint_vel']
            noisy_obs['joint_vel'] = obs['joint_vel'] + noise
            
        if 'lin_vel' in obs:
            noise = np.random.randn(3) * scales['lin_vel']
            noisy_obs['lin_vel'] = obs['lin_vel'] + noise
            
        if 'ang_vel' in obs:
            noise = np.random.randn(3) * scales['ang_vel']
            noisy_obs['ang_vel'] = obs['ang_vel'] + noise
            
        if 'quat' in obs:
            noise = np.random.randn(4) * scales['quat']
            noisy_obs['quat'] = obs['quat'] + noise
            noisy_obs['quat'] /= np.linalg.norm(noisy_obs['quat'])
            
        return noisy_obs
    
    def apply_delay(self, value: np.ndarray) -> np.ndarray:
        """应用通信延迟"""
        if not self.config.add_delay:
            return value
            
        # 随机决定是否延迟
        if np.random.rand() > self.config.delay_prob:
            return value
            
        # 添加到延迟缓冲
        self.observation_delay_buffer.append(value.copy())
        
        # 随机延迟步数
        delay_steps = np.random.randint(
            self.config.delay_steps_range[0],
            self.config.delay_steps_range[1] + 1
        )
        
        # 返回延迟的值
        if len(self.observation_delay_buffer) > delay_steps:
            return self.observation_delay_buffer[-delay_steps-1]
        else:
            return self.observation_delay_buffer[0]
            
    def reset(self):
        """重置（新 episode 开始时调用）"""
        self.observation_delay_buffer = []
        self.randomize_all()


# ========== 使用示例 ==========
if __name__ == "__main__":
    import mujoco
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path("mjcf/hopper_serial.xml")
    data = mujoco.MjData(model)
    
    # 创建随机化器
    randomizer = DomainRandomizer(model, data)
    
    # 每次 reset 时调用
    randomizer.reset()
    
    print("Motor strength:", randomizer.motor_strength)
    print("IMU offset:", randomizer.imu_offset)
    
    # 在控制循环中应用
    torque = np.array([1.0, 2.0, 3.0])
    torque_randomized = randomizer.apply_motor_strength(torque)
    print("Original torque:", torque)
    print("Randomized torque:", torque_randomized)

