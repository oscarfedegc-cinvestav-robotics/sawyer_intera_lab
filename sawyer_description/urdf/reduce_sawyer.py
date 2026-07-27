import os

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

URDF_PATH = "/home/nexus/Documents/GitHub/sawyer_intera_lab/sawyer_description/urdf/sawyer.urdf"


def print_joint_summary(model: pin.Model, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"nq = {model.nq}")
    print(f"nv = {model.nv}")
    print("\nJoints:")
    for i, name in enumerate(model.names):
        print(f"{i:2d}: {name}")


def print_frame_summary(model: pin.Model, max_frames: int | None = None) -> None:
    print("\nFrames:")
    frames = model.frames if max_frames is None else model.frames[:max_frames]
    for i, frame in enumerate(frames):
        print(f"{i:3d}: {frame.name} | parent_joint={frame.parentJoint} | type={int(frame.type)}")


def find_frames_by_name(model: pin.Model, frame_name: str) -> list[int]:
    ids = []
    for i, frame in enumerate(model.frames):
        if frame.name == frame_name:
            ids.append(i)
    return ids


def build_reduced_sawyer_model(model: pin.Model, joints_to_lock_names: list[str]) -> pin.Model:
    q0 = pin.neutral(model)

    joints_to_lock_ids = []
    for joint_name in joints_to_lock_names:
        if joint_name not in model.names:
            raise ValueError(f"Joint '{joint_name}' not found in model.names")
        joints_to_lock_ids.append(model.getJointId(joint_name))

    reduced_model = pin.buildReducedModel(model, joints_to_lock_ids, q0)
    return reduced_model


def build_symbolic_functions(model: pin.Model, ee_frame_id: int) -> dict:
    if ee_frame_id < 0 or ee_frame_id >= len(model.frames):
        raise ValueError(f"Invalid ee_frame_id: {ee_frame_id}")

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    q = ca.SX.sym("q", cmodel.nq, 1)
    v = ca.SX.sym("v", cmodel.nv, 1)
    a = ca.SX.sym("a", cmodel.nv, 1)
    tau = ca.SX.sym("tau", cmodel.nv, 1)
    dq = ca.SX.sym("dq", cmodel.nv, 1)
    q_ref = ca.SX.sym("q_ref", cmodel.nq, 1)

    cpin.forwardKinematics(cmodel, cdata, q, v, a)
    cpin.updateFramePlacements(cmodel, cdata)

    oMf = cdata.oMf[ee_frame_id]
    p_ee = oMf.translation
    R_ee = oMf.rotation

    J_ee = cpin.computeFrameJacobian(
        cmodel,
        cdata,
        q,
        ee_frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )

    M = cpin.crba(cmodel, cdata, q)
    h = cpin.nonLinearEffects(cmodel, cdata, q, v)
    g = cpin.computeGeneralizedGravity(cmodel, cdata, q)
    tau_id = cpin.rnea(cmodel, cdata, q, v, a)
    ddq = cpin.aba(cmodel, cdata, q, v, tau)
    q_next = cpin.integrate(cmodel, q, dq)
    q_err = cpin.difference(cmodel, q_ref, q)

    fk_pos_fun = ca.Function("fk_position", [q], [p_ee], ["q"], ["p_ee"])
    fk_rot_fun = ca.Function("fk_rotation", [q], [R_ee], ["q"], ["R_ee"])
    jac_fun = ca.Function("jacobian_ee", [q], [J_ee], ["q"], ["J_ee"])
    M_fun = ca.Function("mass_matrix", [q], [M], ["q"], ["M"])
    h_fun = ca.Function("nonlinear_effects", [q, v], [h], ["q", "v"], ["h"])
    g_fun = ca.Function("gravity", [q], [g], ["q"], ["g"])
    rnea_fun = ca.Function("inverse_dynamics", [q, v, a], [tau_id], ["q", "v", "a"], ["tau"])
    aba_fun = ca.Function("forward_dynamics", [q, v, tau], [ddq], ["q", "v", "tau"], ["ddq"])
    integrate_fun = ca.Function("integrate_q", [q, dq], [q_next], ["q", "dq"], ["q_next"])
    difference_fun = ca.Function("difference_q", [q_ref, q], [q_err], ["q_ref", "q"], ["q_err"])

    return {
        "model": model,
        "cmodel": cmodel,
        "ee_frame_id": ee_frame_id,
        "ee_frame_name": model.frames[ee_frame_id].name,
        "fk_position": fk_pos_fun,
        "fk_rotation": fk_rot_fun,
        "jacobian_ee": jac_fun,
        "mass_matrix": M_fun,
        "nonlinear_effects": h_fun,
        "gravity": g_fun,
        "inverse_dynamics": rnea_fun,
        "forward_dynamics": aba_fun,
        "integrate_q": integrate_fun,
        "difference_q": difference_fun,
    }


if __name__ == "__main__":
    if not os.path.isfile(URDF_PATH):
        raise FileNotFoundError(URDF_PATH)

    full_model = pin.buildModelFromUrdf(URDF_PATH)
    print_joint_summary(full_model, "FULL MODEL")

    reduced_model = build_reduced_sawyer_model(full_model, ["head_pan"])
    print_joint_summary(reduced_model, "REDUCED MODEL WITHOUT head_pan")
    print_frame_summary(reduced_model)

    candidates = find_frames_by_name(reduced_model, "right_hand")
    print("\nright_hand candidates:", candidates)

    # Use one of the printed candidate ids.
    # From your output, start with 40.
    ee_frame_id = 40

    sym = build_symbolic_functions(reduced_model, ee_frame_id)

    nq = sym["cmodel"].nq
    nv = sym["cmodel"].nv

    q0 = ca.DM.zeros(nq, 1)
    v0 = ca.DM.zeros(nv, 1)
    a0 = ca.DM.zeros(nv, 1)
    tau0 = ca.DM.zeros(nv, 1)

    print("\n=== TEST SYMBOLIC FUNCTIONS ===")
    print("Reduced nq:", nq)
    print("Reduced nv:", nv)
    print("Using ee_frame_id:", ee_frame_id)
    print("Using ee_frame_name:", sym["ee_frame_name"])

    p_ee = sym["fk_position"](q0)
    J_ee = sym["jacobian_ee"](q0)
    M = sym["mass_matrix"](q0)
    h = sym["nonlinear_effects"](q0, v0)
    g = sym["gravity"](q0)
    tau_id = sym["inverse_dynamics"](q0, v0, a0)
    ddq = sym["forward_dynamics"](q0, v0, tau0)

    print("\nEnd-effector position:")
    print(p_ee)

    print("\nJacobian shape:")
    print(J_ee.shape)

    print("\nMass matrix shape:")
    print(M.shape)

    print("\nNonlinear effects:")
    print(h)

    print("\nGravity:")
    print(g)

    print("\nInverse dynamics:")
    print(tau_id)

    print("\nForward dynamics:")
    print(ddq)
