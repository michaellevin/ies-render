from module import IES_Thumbnail_Generator

if __name__ == "__main__":
    ies_paths = [
        "examples/vertical_angles.ies",
        "examples/horiz_angles.ies",
        "examples/ies-lights-pack/area-light.ies",
    ]
    tb = IES_Thumbnail_Generator(ies_paths[2])
    # tb.generate(size=1024, horizontal_angle=0, distance=0.3, blur_radius=0.5, save=True)
    tb.render(size=1024, horizontal_angle=0, distance=0.0, blur_radius=0, save=True)
