import cadquery as cq


def create_pvc_pipe(diameter, length):
    """Creates a PVC pipe model."""
    outer_radius = diameter / 2
    inner_radius = outer_radius - 0.1  # Assuming 0.1" wall thickness

    pipe = cq.Workplane("XY").circle(outer_radius).circle(inner_radius).extrude(length)
    return pipe


def create_coupler(diameter, length=2):
    """Creates a PVC coupler fitting."""
    return create_pvc_pipe(diameter + 0.2, length)  # Slightly wider for fitting


def create_elbow(diameter, angle=90):
    """Creates a PVC elbow fitting."""
    outer_radius = diameter / 2
    inner_radius = outer_radius - 0.1

    elbow = (cq.Workplane("XY")
             .workplane(offset=outer_radius)
             .circle(outer_radius).circle(inner_radius)
             .sweep(cq.Workplane("XZ").radiusArc((outer_radius, outer_radius), angle)))

    return elbow


def create_tee(diameter):
    """Creates a PVC tee fitting."""
    main_pipe = create_pvc_pipe(diameter, 4)
    branch = create_pvc_pipe(diameter, 2).rotate((0, 0, 0), (0, 1, 0), 90)
    return main_pipe.union(branch)


def create_cross(diameter):
    """Creates a PVC cross fitting."""
    pipe1 = create_pvc_pipe(diameter, 4)
    pipe2 = create_pvc_pipe(diameter, 4).rotate((0, 0, 0), (0, 1, 0), 90)
    return pipe1.union(pipe2)


def create_sideout(diameter):
    """Creates a PVC side-out fitting."""
    main_pipe = create_pvc_pipe(diameter, 4)
    side_branch = create_pvc_pipe(diameter, 2).rotate((0, 0, 0), (1, 0, 0), 45)
    return main_pipe.union(side_branch)


def export_step(model, filename):
    """Exports a model as a STEP file."""
    cq.exporters.export(model, filename)


def generate_step_files():
    sizes = {0.5: 21.34, 1: 33.40, 2: 60.33}  # Nominal size to actual OD mapping in mm  # Pipe diameters in inches
    fittings = {
        "pipe": create_pvc_pipe,
        "coupler": create_coupler,
        "elbow": create_elbow,
        "tee": create_tee,
        "cross": create_cross,
        "sideout": create_sideout
    }

    for nominal_size, actual_od in sizes.items():
        for name, func in fittings.items():
            model = func(actual_od)  # Convert inches to mm
            filename = f"pvc_{name}_{size}in.step"
            export_step(model, filename)
            print(f"Exported: {filename}")


if __name__ == "__main__":
    generate_step_files()