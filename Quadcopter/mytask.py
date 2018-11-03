import numpy as np
from physics_sim import PhysicsSim
import itertools

class MyTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    # --------------------------------------------------------------------------
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.init_pose = init_pose

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 100
        self.action_high = 900
        self.action_size = 4

        self.count = 1
        self.episode_reward = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # --------------------------------------------------------------------------
    def get_reward(self, done, rotor_speeds):
        """Uses current pose of sim to return reward."""

        """differences = 0
        checks = list(range(0, 4))
        combos = itertools.combinations(checks, 2)
        avg = 0
        for combo in combos:
            differences += abs(rotor_speeds[combo[0]] - rotor_speeds[combo[1]])
            avg += 1
        differences += 30
        differences /= avg

        vertical_diff = self.sigmoid(np.power(self.target_pos[2] - self.sim.pose[2], 2).sum())
        x_y_diff = self.sigmoid(np.power(self.target_pos[:2] - self.sim.pose[:2], 2).sum())

        reward = (.4*(1/differences)) + (.04*(1/vertical_diff)) + (.04*(1/x_y_diff))"""
        
        reward = 0
        
        if self.sim.v[0] != 0: reward -= 10
        if self.sim.v[1] != 0: reward -= 10
        if self.sim.v[2] > 0: reward += 50
        reward -= self.sim.pose[3:6].sum()
        
        if done and self.sim.pose[2] < 1: reward = -1
        
        lateral_distance = ((self.sim.pose[0] - self.target_pos[0]) + (self.sim.pose[1] - self.target_pos[1])) ** 2
        reward -= lateral_distance
        vertical_distance = max((self.sim.pose[2] - self.target_pos[2]) ** 2, .00001)
        if vertical_distance != 0: reward += 100 / vertical_distance
        
        return np.clip(reward, -1, 1)

    # --------------------------------------------------------------------------
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        self.count += 1
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done, rotor_speeds)
            pose_all.append(self.sim.pose)
        self.episode_reward += reward
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    # --------------------------------------------------------------------------
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.count = 1
        self.episode_reward = 0
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state