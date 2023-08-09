"""Game control related."""

SIMPLE_ACTIONS = [["right"], ["right", "A"]]

def get_env_name(world, stage, quality=0):
    """Returns the name of the environment, quality=0 means best by default."""
    assert 1 <= world <= 8, "world must be in [1, 8]"
    assert 1 <= stage <= 4, "stage must be in [1, 4]"
    return f"SuperMarioBros-{world}-{stage}-v{quality}"
