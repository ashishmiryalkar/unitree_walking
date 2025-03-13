# import argparse

# from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Training for wheeled quadruped robot.")
# parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.classic.unitree_g1_walking.mdp as mdp

from .terrains import PYRAMID_STAIRS

# from .terrains import PYRAMID_STAIRS

def robot_configuration():
    """Configuration for the Unitree G1 Humanoid robot."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                ".*_elbow_pitch_joint": 0.87,
                "left_shoulder_roll_joint": 0.16,
                "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.16,
                "right_shoulder_pitch_joint": 0.35,
                "left_one_joint": 1.0,
                "right_one_joint": -1.0,
                "left_two_joint": 0.52,
                "right_two_joint": -0.52,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    "torso_joint",
                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                    "torso_joint": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0,
                    ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0,
                    ".*_knee_joint": 5.0,
                    "torso_joint": 5.0,
                },
                armature={
                    ".*_hip_.*": 0.01,
                    ".*_knee_joint": 0.01,
                    "torso_joint": 0.01,
                },
            ),
            "feet": ImplicitActuatorCfg(
                effort_limit=20,
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness=20.0,
                damping=2.0,
                armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
                armature={
                    ".*_shoulder_.*": 0.01,
                    ".*_elbow_.*": 0.01,
                    ".*_five_joint": 0.001,
                    ".*_three_joint": 0.001,
                    ".*_six_joint": 0.001,
                    ".*_four_joint": 0.001,
                    ".*_zero_joint": 0.001,
                    ".*_one_joint": 0.001,
                    ".*_two_joint": 0.001,
                },
            ),
        },
    )

@configclass
class MySceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        # terrain_type="plane",
        terrain_generator=PYRAMID_STAIRS,
        # collision_group=-1,
        # physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        # debug_vis=False,
        visual_material=None,
        debug_vis=False,
    )

    robot = robot_configuration()

    lights = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            "torso_joint",
        ],
        # scale={".*": 360},
        scale={
            ".*_ankle_pitch_joint": 45.0, #
            ".*_ankle_roll_joint": 15.0, #
            ".*_hip_yaw_joint": 150.0, #
            ".*_hip_roll_joint": 90.0, #
            ".*_hip_pitch_joint": 180.0,  #
            ".*_knee_joint": 90.0, # 
            "torso_joint": 160.0, # 
        }
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_pitch_link", "left_ankle_roll_link", "right_ankle_pitch_link", "right_ankle_roll_link"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    # (3) Reward for non-upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.5})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.7, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )

    velocity_penalty = RewTerm(
        func=mdp.root_vel, weight=0.7, params={"target_lin_vel": 0.5, "target_ang_vel": None}
    )

    # joint limits
    
    # feet distance
    far_feet_distance = RewTerm(
        func=mdp.feet_distance, weight=0.1, params={"threshold": 0.4}
    )

    # power_consumption = RewTerm(
    #     func=mdp.power_consumption, weight=0.1
    # )

    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})

     # (3) Terminate if feet distance is greater than 0.6 m
    feet_distance = DoneTerm(func=mdp.terminate_feet_distance, params={"threshold":0.6})

    min_feet_y_distance = DoneTerm(func=mdp.terminate_feet_min_distance, params={"threshold":0.05})


@configclass
class unitreeGroundPlaneEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
