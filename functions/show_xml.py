import time
from pathlib import Path

import mujoco
import mujoco.viewer


def _name_to_id(model, obj_type, name, label):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"{label} '{name}' was not found in the XML model.")
    return obj_id


def _default_follow_body(model):
    if model.nbody <= 1:
        raise ValueError("The model has no body to follow.")
    return 1


def _configure_fixed_camera(viewer, model, camera_name, distance, azimuth, elevation, lookat):
    if camera_name is not None:
        camera_id = _name_to_id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
            "Camera",
        )
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        return

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    viewer.cam.distance = float(distance)
    viewer.cam.azimuth = float(azimuth)
    viewer.cam.elevation = float(elevation)

    if lookat is not None:
        viewer.cam.lookat[:] = lookat


def _configure_follow_camera(viewer, model, follow_body, distance, azimuth, elevation, lookat):
    if follow_body is None:
        body_id = _default_follow_body(model)
    elif isinstance(follow_body, int):
        body_id = follow_body
    else:
        body_id = _name_to_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            follow_body,
            "Body",
        )

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body_id
    viewer.cam.fixedcamid = -1
    viewer.cam.distance = float(distance)
    viewer.cam.azimuth = float(azimuth)
    viewer.cam.elevation = float(elevation)

    if lookat is not None:
        viewer.cam.lookat[:] = lookat


def show_xml(
    xml_path,
    camera_mode="fixed",
    camera_name=None,
    follow_body=None,
    distance=3.0,
    azimuth=90.0,
    elevation=-20.0,
    lookat=None,
    run_seconds=None,
    show_left_ui=True,
    show_right_ui=True,
):
    """
    Open a MuJoCo viewer for an XML model with a custom camera setup.

    Parameters
    ----------
    xml_path : str or Path
        Path to the MuJoCo XML file.
    camera_mode : str
        'fixed' for a static/free camera or 'follow' for a tracking camera.
    camera_name : str | None
        Name of a camera already defined in the XML. Only used in fixed mode.
    follow_body : str | int | None
        Body name or body id to follow in follow mode.
    distance, azimuth, elevation : float
        Camera orbit parameters for free/tracking cameras.
    lookat : sequence[float] | None
        Optional 3D point used by the camera when applicable.
    run_seconds : float | None
        Auto-close after this many wall-seconds. If None, keep running until
        the viewer window is closed.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if lookat is not None and len(lookat) != 3:
        raise ValueError("lookat must be a 3-element sequence like [x, y, z].")

    run_until = None if run_seconds is None else time.time() + float(run_seconds)

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=show_left_ui,
        show_right_ui=show_right_ui,
    ) as viewer:
        with viewer.lock():
            if camera_mode == "fixed":
                _configure_fixed_camera(
                    viewer,
                    model,
                    camera_name=camera_name,
                    distance=distance,
                    azimuth=azimuth,
                    elevation=elevation,
                    lookat=lookat,
                )
            elif camera_mode == "follow":
                _configure_follow_camera(
                    viewer,
                    model,
                    follow_body=follow_body,
                    distance=distance,
                    azimuth=azimuth,
                    elevation=elevation,
                    lookat=lookat,
                )
            else:
                raise ValueError("camera_mode must be either 'fixed' or 'follow'.")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            if run_until is not None and time.time() >= run_until:
                break

            time.sleep(model.opt.timestep)

    return model, data
