import os
import sys
from pathlib import Path

import habitat_sim

repo = '/home/adminer/agentRAG/HOV-SG'
sys.path.append(str(Path(repo) / 'hovsg/data/hm3dsem'))
from habitat_utils import make_cfg, load_poses_from_file, save_obs  # noqa: E402


def main() -> None:
    root_dataset_dir = '/home/adminer/agentRAG/experiment_data/hm3d'
    scene_dir = '00824-Dd4bFSTQ8gi'
    scene_name = scene_dir.split('-')[-1]
    scene_data_dir = f'{root_dataset_dir}/val/{scene_dir}'
    save_dir = f'/home/adminer/agentRAG/experiment_data/hm3dsem_walks/val/{scene_dir}'
    pose_file = f'{repo}/hovsg/data/hm3dsem/metadata/poses/{scene_dir}.txt'
    os.makedirs(save_dir, exist_ok=True)

    sim_settings = {
        'scene': os.path.join(scene_data_dir, scene_name + '.glb'),
        'default_agent': 0,
        'sensor_height': 1.5,
        'color_sensor': True,
        'depth_sensor': True,
        'semantic_sensor': True,
        'lidar_sensor': False,
        'move_forward': 0.2,
        'move_backward': 0.2,
        'turn_left': 5,
        'turn_right': 5,
        'look_up': 5,
        'look_down': 5,
        'look_left': 5,
        'look_right': 5,
        'width': 640,
        'height': 480,
        'enable_physics': False,
        'seed': 42,
        'lidar_fov': 360,
        'depth_img_for_lidar_n': 20,
        'img_save_dir': save_dir,
    }

    os.environ['MAGNUM_LOG'] = 'quiet'
    os.environ['HABITAT_SIM_LOG'] = 'quiet'

    sim_cfg = make_cfg(sim_settings, root_dataset_dir, scene_data_dir, scene_name)
    sim = habitat_sim.Simulator(sim_cfg)
    agent = sim.initialize_agent(sim_settings['default_agent'])
    agent_state = habitat_sim.AgentState()
    agent_state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(agent_state)

    poses_list = load_poses_from_file(pose_file)
    max_frames = min(60, len(poses_list))
    print(f'poses={len(poses_list)}, use={max_frames}')
    for i, pose in enumerate(poses_list[:max_frames]):
        st = agent.get_state()
        st.sensor_states['color_sensor'].position = pose[:3]
        st.sensor_states['color_sensor'].rotation = pose[3:]
        st.sensor_states['depth_sensor'].position = pose[:3]
        st.sensor_states['depth_sensor'].rotation = pose[3:]
        st.sensor_states['semantic'].position = pose[:3]
        st.sensor_states['semantic'].rotation = pose[3:]
        agent.set_state(st, reset_sensors=True, infer_sensor_states=False)
        obs = sim.get_sensor_observations(0)
        save_obs(save_dir, sim_settings, obs, pose, i)

    for sub in ['rgb', 'depth', 'semantic', 'pose']:
        p = Path(save_dir) / sub
        cnt = len(list(p.glob('*'))) if p.exists() else 0
        print(sub, cnt)


if __name__ == '__main__':
    main()
