import os

import numpy as np
import pybullet as p
from gymnasium.spaces import Box, Dict

from .env import AssistiveEnv


class FeedingEnv(AssistiveEnv):
    def __init__(self, robot_type="pr2", human_control=False):
        super().__init__(
            robot_type=robot_type,
            task="feeding",
            human_control=human_control,
            frame_skip=10,
            time_step=0.01,
            action_robot_len=7,
            action_human_len=(4 if human_control else 0),
            obs_robot_len=25,
            obs_human_len=(23 if human_control else 0),
        )

    def step(self, action):
        self.take_step(
            action,
            robot_arm="right",
            gains=self.config("robot_gains"),
            forces=self.config("robot_forces"),
            human_gains=0.0005,
        )

        robot_force_on_human, spoon_force_on_human = self.get_total_force()
        total_force_on_human = robot_force_on_human + spoon_force_on_human
        # reward_food, food_mouth_velocities, food_hit_human_reward = self.get_food_rewards()
        # end_effector_velocity = np.linalg.norm(p.getBaseVelocity(self.spoon, physicsClientId=self.id)[0])  # noqa
        obs = self._get_obs([spoon_force_on_human], [robot_force_on_human, spoon_force_on_human])

        # # Get human preferences
        # preferences_score, pref_info = self.human_preferences(
        #     end_effector_velocity=end_effector_velocity,
        #     total_force_on_human=robot_force_on_human,
        #     tool_force_at_target=spoon_force_on_human,
        #     food_hit_human_reward=food_hit_human_reward,
        #     food_mouth_velocities=food_mouth_velocities,
        #     verbose=True,
        # )

        # spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
        # spoon_pos = np.array(spoon_pos)

        # reward_distance_mouth_target = -np.linalg.norm(
        #     self.target_pos - spoon_pos
        # )  # Penalize robot for distance between the spoon and human mouth.
        # reward_action = -np.sum(np.square(action))  # Penalize actions

        # reward = (
        #     self.config("distance_weight") * reward_distance_mouth_target
        #     + self.config("action_weight") * reward_action
        #     + self.config("food_reward_weight") * reward_food
        #     + preferences_score
        # )

        # if self.gui and reward_food != 0:
        #     print("Task success:", self.task_success, "Food reward:", reward_food)

        reward, reward_info = self.compute_reward(action)

        info = {
            "total_force_on_human": total_force_on_human,
            "task_success": int(self.task_success >= self.total_food_count * self.config("task_success_threshold")),
            "action_robot_len": self.action_robot_len,
            "action_human_len": self.action_human_len,
            "obs_robot_len": self.obs_robot_len,
            "obs_human_len": self.obs_human_len,
        }

        info.update(reward_info)

        # info.update(
        #     {
        #         "r_food": reward_food,
        #         "r_distance_mouth_target": reward_distance_mouth_target,
        #         "r_action": reward_action,
        #         # Human preferences
        #         "r_high_target_forces": pref_info["Reward/high_target_forces"],
        #         "r_velocity": pref_info["Reward/velocity"],
        #         "r_force_nontarget": pref_info["Reward/force_nontarget"],
        #         "r_food_velocities": pref_info["Reward/food_velocities"],
        #         "r_food_hit_human": food_hit_human_reward,
        #     }
        # )
        done = False

        if self.record_video:
            self.record_video_frame()

        return obs, reward, done, info

    def get_total_force(self):
        robot_force_on_human = 0
        spoon_force_on_human = 0
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
            robot_force_on_human += c[9]
        for c in p.getContactPoints(bodyA=self.spoon, bodyB=self.human, physicsClientId=self.id):
            spoon_force_on_human += c[9]
        return robot_force_on_human, spoon_force_on_human

    def get_food_rewards(self):
        # Check all food particles to see if they have left the spoon or entered the person's mouth
        # Give the robot a reward or penalty depending on food particle status
        food_reward = 0
        food_hit_human_reward = 0
        food_mouth_velocities = []
        foods_to_remove = []
        for f in self.foods:
            food_pos, food_orient = p.getBasePositionAndOrientation(f, physicsClientId=self.id)
            distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
            if distance_to_mouth < 0.02:
                # Delete particle and give robot a reward
                food_reward += 20
                self.task_success += 1
                food_velocity = np.linalg.norm(p.getBaseVelocity(f, physicsClientId=self.id)[0])
                food_mouth_velocities.append(food_velocity)
                foods_to_remove.append(f)
                p.resetBasePositionAndOrientation(
                    f, self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1], physicsClientId=self.id
                )
                continue
            elif (
                food_pos[-1] < 0.5
                or len(p.getContactPoints(bodyA=f, bodyB=self.table, physicsClientId=self.id)) > 0
                or len(p.getContactPoints(bodyA=f, bodyB=self.bowl, physicsClientId=self.id)) > 0
            ):
                # Delete particle and give robot a penalty for spilling food
                food_reward -= 5
                foods_to_remove.append(f)
                continue
            if (
                len(p.getContactPoints(bodyA=f, bodyB=self.human, physicsClientId=self.id)) > 0
                and f not in self.foods_hit_person
            ):
                # Record that this food particle just hit the person, so that we can penalize the robot
                self.foods_hit_person.append(f)
                food_hit_human_reward -= 1
        self.foods = [f for f in self.foods if f not in foods_to_remove]
        return food_reward, food_mouth_velocities, food_hit_human_reward

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(
            p.getLinkState(
                self.robot,
                15 if self.robot_type == "pr2" else 0,
                computeForwardKinematics=True,
                physicsClientId=self.id,
            )[0]
        )
        spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
        robot_right_joint_states = p.getJointStates(
            self.robot, jointIndices=self.robot_right_arm_joint_indices, physicsClientId=self.id
        )
        robot_right_joint_positions = np.array([x[0] for x in robot_right_joint_states])
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
        if self.human_control:
            human_pos = np.array(p.getBasePositionAndOrientation(self.human, physicsClientId=self.id)[0])
            human_joint_states = p.getJointStates(
                self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id
            )
            human_joint_positions = np.array([x[0] for x in human_joint_states])

        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[
            :2
        ]

        robot_obs = np.concatenate([
            spoon_pos - torso_pos,
            spoon_orient,
            spoon_pos - self.target_pos,
            robot_right_joint_positions,
            head_pos - torso_pos,
            head_orient,
            forces,
        ]).ravel()
        if self.human_control:
            human_obs = np.concatenate([
                spoon_pos - human_pos,
                spoon_orient,
                spoon_pos - self.target_pos,
                human_joint_positions,
                head_pos - human_pos,
                head_orient,
                forces_human,
            ]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    @property
    def randomness_values(self):
        return self._randomness_values

    @randomness_values.setter
    def randomness_values(self, value):
        self._randomness_values = value

    def reset(self, randomness_values=None):
        self._randomness_values = randomness_values
        if self._randomness_values is None:
            self._randomness_values = [
                self.np_random.choice(["male", "female"]),
                self.np_random.choice(["none", "limits", "weakness", "tremor"]),
                self.np_random.choice(["none", "limits", "weakness"]),
                self.np_random.uniform(0.5, 1.0),
                self.np_random.uniform(0.25, 1.0),
                self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)),
                self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)),
                self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)),
                self.np_random.uniform(-0.05, 0.05),
                self.np_random.uniform(-0.05, 0.05),
                self.np_random.uniform(-0.05, 0.05, size=3),
                self.np_random.uniform(-0.5, 0),
                self.np_random.uniform(-0.5, 0.5),
                np.deg2rad(self.np_random.uniform(-30, 30)),
            ]

        super().reset(randomness_values=self._randomness_values)
        self.setup_timing()
        self.task_success = 0
        (
            self.human,
            self.wheelchair,
            self.robot,
            self.robot_lower_limits,
            self.robot_upper_limits,
            self.human_lower_limits,
            self.human_upper_limits,
            self.robot_right_arm_joint_indices,
            self.robot_left_arm_joint_indices,
            self.gender,
        ) = self.world_creation.create_new_world(
            furniture_type="wheelchair",
            static_human_base=True,
            human_impairment="no_tremor",
            print_joints=False,
            gender="random",
        )
        self.robot_lower_limits = self.robot_lower_limits[self.robot_right_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_right_arm_joint_indices]
        # define robot arm init positions
        self.robot_left_arm_init_joint_positions = [0, 0, 0, 0, 0, 0, 0]
        self.robot_right_arm_init_joint_positions = [0, 0, 0, 0, 0, 0, 0]

        if self.robot_type == "pr2":
            self.robot_left_arm_init_joint_positions = [1.75, 1.25, 1.5, -0.5, 1, 0, 1]
            self.robot_right_arm_init_joint_positions = [-1.75, 1.25, -1.5, -0.5, -1, 0, -1]

        if self.robot_type == "baxter":
            self.robot_left_arm_init_joint_positions = [0.75, 1, 0.5, 0.5, 1, -0.5, 0]
            self.robot_right_arm_init_joint_positions = [-0.75, 1, -0.5, 0.5, -1, -0.5, 0]

        self.reset_robot_joints()
        if self.robot_type == "jaco":
            wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(
                self.wheelchair, physicsClientId=self.id
            )
            p.resetBasePositionAndOrientation(
                self.robot,
                np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]),
                p.getQuaternionFromEuler([0, 0, -np.pi / 2.0], physicsClientId=self.id),
                physicsClientId=self.id,
            )
            base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

        joints_positions = [
            (6, np.deg2rad(-90)),
            (16, np.deg2rad(-90)),
            (28, np.deg2rad(-90)),
            (31, np.deg2rad(80)),
            (35, np.deg2rad(-90)),
            (38, np.deg2rad(80)),
        ]
        joints_positions += [
            (21, self._randomness_values[5]),
            (22, self._randomness_values[6]),
            (23, self._randomness_values[7]),
        ]
        self.human_controllable_joint_indices = [20, 21, 22, 23]
        self.world_creation.setup_human_joints(
            self.human,
            joints_positions,
            (
                self.human_controllable_joint_indices
                if (self.human_control or self.world_creation.human_impairment == "tremor")
                else []
            ),
            use_static_joints=True,
            human_reactive_force=None,
        )
        p.resetBasePositionAndOrientation(
            self.human, [0, 0.03, 0.89 if self.gender == "male" else 0.86], [0, 0, 0, 1], physicsClientId=self.id
        )
        human_joint_states = p.getJointStates(
            self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id
        )
        self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]

        # Place a bowl of food on a table
        self.table = p.loadURDF(
            os.path.join(self.world_creation.directory, "table", "table_tall.urdf"),
            basePosition=[0.35, -0.9, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id),
            physicsClientId=self.id,
        )
        self.bowl_scale = 0.75
        visual_filename = os.path.join(self.world_creation.directory, "dinnerware", "bowl_reduced_compressed.obj")
        collision_filename = os.path.join(self.world_creation.directory, "dinnerware", "bowl_vhacd.obj")
        bowl_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=[self.bowl_scale] * 3, physicsClientId=self.id
        )
        bowl_collision = p.createCollisionShape(
            shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=[self.bowl_scale] * 3, physicsClientId=self.id
        )
        bowl_pos = np.array([-0.15, -0.55, 0.75]) + np.array(
            [self._randomness_values[8], self._randomness_values[9], 0]
        )
        self.bowl = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=bowl_collision,
            baseVisualShapeIndex=bowl_visual,
            basePosition=bowl_pos,
            baseOrientation=p.getQuaternionFromEuler([np.pi / 2.0, 0, 0], physicsClientId=self.id),
            baseInertialFramePosition=[0, 0.04 * self.bowl_scale, 0],
            useMaximalCoordinates=False,
            physicsClientId=self.id,
        )

        shoulder_pos, shoulder_orient = p.getLinkState(
            self.human, 5, computeForwardKinematics=True, physicsClientId=self.id
        )[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 7, computeForwardKinematics=True, physicsClientId=self.id)[
            :2
        ]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[
            :2
        ]
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[
            :2
        ]

        # Set target on mouth
        self.mouth_pos = [0, -0.11, 0.03] if self.gender == "male" else [0, -0.1, 0.03]
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[
            :2
        ]
        target_pos, target_orient = p.multiplyTransforms(
            head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id
        )
        self.target_pos = np.array(target_pos)
        sphere_collision = -1
        sphere_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1], physicsClientId=self.id
        )
        self.target = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.target_pos,
            useMaximalCoordinates=False,
            physicsClientId=self.id,
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=1.10,
            cameraYaw=40,
            cameraPitch=-45,
            cameraTargetPosition=[-0.2, 0, 0.75],
            physicsClientId=self.id,
        )

        target_pos = np.array(bowl_pos) + np.array([0, -0.1, 0.4]) + self.randomness_values[10]
        if self.robot_type == "pr2":
            target_orient = p.getQuaternionFromEuler([np.pi / 2.0, 0, 0], physicsClientId=self.id)
            self.position_robot_toc(
                self.robot,
                54,
                [(target_pos, target_orient), (self.target_pos, None)],
                [(self.target_pos, target_orient)],
                self.robot_right_arm_joint_indices,
                self.robot_lower_limits,
                self.robot_upper_limits,
                ik_indices=range(15, 15 + 7),
                pos_offset=np.array([0.1, 0.2, 0]),
                max_ik_iterations=200,
                step_sim=True,
                check_env_collisions=False,
                human_joint_indices=self.human_controllable_joint_indices,
                human_joint_positions=self.target_human_joint_positions,
                fixed_random_x_position=self.randomness_values[11],
                fixed_random_y_position=self.randomness_values[12],
                fixed_random_rotation=self.randomness_values[13],
            )
            self.world_creation.set_gripper_open_position(self.robot, position=0.03, left=False, set_instantly=True)
            self.spoon = self.world_creation.init_tool(
                self.robot,
                mesh_scale=[0.08] * 3,
                pos_offset=[0, -0.03, -0.11],
                orient_offset=p.getQuaternionFromEuler([-0.2, 0, 0], physicsClientId=self.id),
                left=False,
                maximal=False,
            )
        elif self.robot_type == "jaco":
            target_orient = p.getQuaternionFromEuler(np.array([np.pi / 2.0, 0, np.pi / 2.0]), physicsClientId=self.id)
            self.util.ik_random_restarts(
                self.robot,
                8,
                target_pos,
                target_orient,
                self.world_creation,
                self.robot_right_arm_joint_indices,
                self.robot_lower_limits,
                self.robot_upper_limits,
                ik_indices=[0, 1, 2, 3, 4, 5, 6],
                max_iterations=1000,
                max_ik_random_restarts=40,
                random_restart_threshold=0.01,
                step_sim=True,
                check_env_collisions=True,
            )
            self.world_creation.set_gripper_open_position(self.robot, position=1.33, left=False, set_instantly=True)
            self.spoon = self.world_creation.init_tool(
                self.robot,
                mesh_scale=[0.08] * 3,
                pos_offset=[0.1, -0.0225, 0.03],
                orient_offset=p.getQuaternionFromEuler([-0.1, -np.pi / 2.0, 0], physicsClientId=self.id),
                left=False,
                maximal=False,
            )
        else:
            target_orient = p.getQuaternionFromEuler(np.array([np.pi / 2.0, 0, np.pi / 2.0]), physicsClientId=self.id)
            if self.robot_type == "baxter":
                self.position_robot_toc(
                    self.robot,
                    26,
                    [(target_pos, target_orient)],
                    [(self.target_pos, target_orient)],
                    self.robot_right_arm_joint_indices,
                    self.robot_lower_limits,
                    self.robot_upper_limits,
                    ik_indices=range(1, 8),
                    pos_offset=np.array([0, 0.2, 0.975]),
                    max_ik_iterations=200,
                    step_sim=True,
                    check_env_collisions=False,
                    human_joint_indices=self.human_controllable_joint_indices,
                    human_joint_positions=self.target_human_joint_positions,
                    fixed_random_x_position=self.randomness_values[11],
                    fixed_random_y_position=self.randomness_values[12],
                    fixed_random_rotation=self.randomness_values[13],
                )
            else:
                self.position_robot_toc(
                    self.robot,
                    19,
                    [(target_pos, target_orient), (self.target_pos, None)],
                    [(self.target_pos, target_orient)],
                    self.robot_right_arm_joint_indices,
                    self.robot_lower_limits,
                    self.robot_upper_limits,
                    ik_indices=[0, 2, 3, 4, 5, 6, 7],
                    pos_offset=np.array([-0.1, 0.2, 0.975]),
                    max_ik_iterations=200,
                    step_sim=True,
                    check_env_collisions=False,
                    human_joint_indices=self.human_controllable_joint_indices,
                    human_joint_positions=self.target_human_joint_positions,
                    fixed_random_x_position=self.randomness_values[11],
                    fixed_random_y_position=self.randomness_values[12],
                    fixed_random_rotation=self.randomness_values[13],
                )
            self.world_creation.set_gripper_open_position(self.robot, position=0.0, left=False, set_instantly=True)
            self.spoon = self.world_creation.init_tool(
                self.robot,
                mesh_scale=[0.08] * 3,
                pos_offset=[-0.1, 0.12, -0.02],
                orient_offset=p.getQuaternionFromEuler([np.pi / 2.0 - 0.1, 0, np.pi / 2.0], physicsClientId=self.id),
                left=False,
                maximal=False,
            )

        p.resetBasePositionAndOrientation(
            self.bowl,
            bowl_pos,
            p.getQuaternionFromEuler([np.pi / 2.0, 0, 0], physicsClientId=self.id),
            physicsClientId=self.id,
        )

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.human, physicsClientId=self.id)

        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)

        # Generate food
        spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
        spoon_pos = np.array(spoon_pos)
        food_radius = 0.005
        food_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=food_radius, physicsClientId=self.id)
        # food_visual = -1
        food_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=food_radius, rgbaColor=[1, 0, 1, 1], physicsClientId=self.id
        )
        food_mass = 0.001
        food_count = 2 * 2 * 2
        batch_positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    batch_positions.append(
                        np.array([i * 2 * food_radius - 0.005, j * 2 * food_radius, k * 2 * food_radius + 0.02])
                        + spoon_pos
                    )
        last_food_id = p.createMultiBody(
            baseMass=food_mass,
            baseCollisionShapeIndex=food_collision,
            baseVisualShapeIndex=food_visual,
            basePosition=[0, 0, 0],
            useMaximalCoordinates=False,
            batchPositions=batch_positions,
            physicsClientId=self.id,
        )
        self.foods = list(range(last_food_id - food_count + 1, last_food_id + 1))
        self.foods_hit_person = []
        self.total_food_count = len(self.foods)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop food in the spoon
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        return self._get_obs([0], [0, 0])

    def update_targets(self):
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[
            :2
        ]
        target_pos, target_orient = p.multiplyTransforms(
            head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id
        )
        self.target_pos = np.array(target_pos)
        p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)

    @property
    def reward_space(self):
        """
        Returns the range of permissible weights for each reward term.
        The primary term is fixed to [1.0, 1.0].
        The others range from [0.0, X], where X < 1.0.
        """
        return Dict(
            {
                "r_food": Box(low=0.5, high=1.0, shape=(), dtype=float),
                "r_distance_mouth_target": Box(low=0.5, high=1.0, shape=(), dtype=float),
                "r_food_velocities": Box(low=0.0, high=1.0, shape=(), dtype=float),
                "r_velocity": Box(low=0.0, high=1.0, shape=(), dtype=float),
                "r_food_hit_human": Box(low=0.0, high=1.0, shape=(), dtype=float),
                "r_force_nontarget": Box(low=0.0, high=0.1, shape=(), dtype=float),
                "r_action": Box(low=0.0, high=0.1, shape=(), dtype=float),
                "r_return_home": Box(low=0.0, high=1.0, shape=(), dtype=float),
            }
        )

    @property
    def default_reward_weights(self):
        """
        Default weight configuration for each reward term.
        """
        return {
            "r_food": 1.0,  # Primary (fixed)
            "r_distance_mouth_target": 1.0,  # Secondary
            "r_food_velocities": 1.0,  # Secondary
            "r_velocity": 0.25,  # Secondary
            "r_food_hit_human": 1.0,  # Secondary
            "r_force_nontarget": 0.01,  # Secondary
            "r_action": 0.01,  # Secondary
            "r_return_home": 0.5,  # Secondary
        }

    def compute_reward(self, action):
        """
        Computes the total reward as a weighted sum of different criteria:
        1) food_in_mouth: Reward or penalty from get_food_rewards()
        2) distance_to_mouth: Negative distance between spoon and mouth
        3) tilt_alignment: Negative deviation of spoon orientation from a desired tilt
        4) spoon_force_on_human: Negative force on human
        5) end_effector_velocity: Negative spoon velocity for smooth movement
        6) food_hit_person: Negative if any food particle hits person incorrectly

        Returns:
            total_reward (float): The scalar reward
            rewards_dict (dict): A dictionary of each reward term
        """
        # 1) Food-based rewards and penalties
        food_reward, food_mouth_velocities, food_hit_human_reward = self.get_food_rewards()

        # 2) Contact force with the human
        robot_force_on_human, spoon_force_on_human = self.get_total_force()
        total_force_on_human = robot_force_on_human + spoon_force_on_human

        # 3) Food velocities
        food_velocities = np.sum(food_mouth_velocities) if len(food_mouth_velocities) > 0 else 0

        # 4) Smoothness of motion (end-effector velocity)
        end_effector_velocity = np.linalg.norm(p.getBaseVelocity(self.spoon, physicsClientId=self.id)[0])

        # 5) Distance between spoon and mouth
        spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
        spoon_pos = np.array(spoon_pos)
        distance_spoon_to_mouth = np.linalg.norm(spoon_pos - self.target_pos)

        # 6) Action smoothness
        action_smoothness = -np.sum(np.square(action))

        # 7) return to home configuration
        # Current right-arm joint positions (used for measuring distance to "home" pose).
        robot_right_joint_states = p.getJointStates(
            self.robot, self.robot_right_arm_joint_indices, physicsClientId=self.id
        )
        robot_right_joint_positions = np.array([x[0] for x in robot_right_joint_states])

        # Reward for returning to the home configuration after all food is fed.
        # In this simple example, we measure the norm of the right-arm joint positions
        # from 0, and give a shaped reward if the distance is small.
        if self.task_success == self.total_food_count:
            dist_to_home = np.linalg.norm(robot_right_joint_positions - self.robot_right_arm_init_joint_positions)
            # Give a small shaped reward: clamp below 0 to ensure positivity only if close
        else:
            dist_to_home = 0.0

        # Prepare each raw reward/penalty term
        r_food_in_mouth = food_reward  # May be positive (food success) or negative (spillage)
        r_distance_to_mouth = -distance_spoon_to_mouth  # Negative distance => smaller distance => higher reward
        r_spoon_force_on_human = -total_force_on_human  # Negative for contact force
        r_end_effector_velocity = -end_effector_velocity  # Negative for fast movement
        r_food_hit_person = food_hit_human_reward  # Already negative if hits the person
        r_food_velocities = -food_velocities  # Negative for fast velocities
        r_action_smoothness = action_smoothness  # Already negative for noisy actions
        r_return_home = -dist_to_home  # Reward for returning to home

        # Calculate final weighted sum
        total_reward = 0.0
        total_reward += self.default_reward_weights["r_food"] * r_food_in_mouth
        total_reward += self.default_reward_weights["r_food_velocities"] * r_food_velocities
        total_reward += self.default_reward_weights["r_distance_mouth_target"] * r_distance_to_mouth
        total_reward += self.default_reward_weights["r_force_nontarget"] * r_spoon_force_on_human
        total_reward += self.default_reward_weights["r_velocity"] * r_end_effector_velocity
        total_reward += self.default_reward_weights["r_food_hit_human"] * r_food_hit_person
        total_reward += self.default_reward_weights["r_action"] * r_action_smoothness
        total_reward += self.default_reward_weights["r_return_home"] * r_return_home

        # Return the total reward and a dictionary of each term
        rewards_dict = {
            "r_food": r_food_in_mouth,
            "r_distance_mouth_target": r_distance_to_mouth,
            "r_action": r_action_smoothness,
            "r_velocity": r_end_effector_velocity,
            "r_force_nontarget": r_spoon_force_on_human,
            "r_food_velocities": r_food_velocities,
            "r_food_hit_human": r_food_hit_person,
            "r_return_home": r_return_home,
        }

        return total_reward, rewards_dict
