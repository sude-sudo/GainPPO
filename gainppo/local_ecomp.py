"""Audited LiftTest factory. Run in an environment with EComp + openpi_client installed."""
from .ecomp_adapter import ECompBackend


def make_backend(config):
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    if config['ecomp_env'] != 'LiftTest':
        raise ValueError('Only audited LiftTest factory provided; implement explicit adapter for other tasks')
    controller=load_composite_controller_config(config['controller_config'])
    def make():
        return suite.make('LiftTest', robots='UR5e',controller_configs=controller,
            has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,
            render_gpu_device_id=config['render_gpu_device_id'],
            camera_names=('agentview','robot0_eye_in_hand','frontview'),
            camera_widths=256,camera_heights=256,control_freq=10,
            horizon=config['max_episode_steps']+config['wait_steps'],ignore_done=False)
    policy=WebsocketClientPolicy(host=config['vla_host'],port=config['vla_port'],timeout=config['vla_timeout'],max_attempts=1)
    return ECompBackend(make,policy,horizon=config['max_episode_steps'],wait_steps=config['wait_steps'],
                        prompt=config['prompt'],remote_frozen_confirmed=config['remote_vla_frozen_confirmed'])
