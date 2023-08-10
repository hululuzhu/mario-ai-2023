"""Everything related to visualization in colab notebook."""
from PIL import Image, ImageOps
import game_control
import game_env
import os
from ipywidgets import widgets, HBox
from tqdm.notebook import tqdm


_GIF_PATH = "/tmp/mario_ai_gifs"

def visualize_multi_stages(start_end_worlds, start_end_stages, is_human_view=True):
    if not os.path.exists(_GIF_PATH):
        os.mkdir(_GIF_PATH)
    """Runs multiple stages and returns horizontal visual boxes for each world."""
    assert len(start_end_worlds) == len(start_end_stages) == 2, (
        "must be pairs for worlds and stages")
    hboxs = []
    for w in tqdm(range(*start_end_worlds)):
        stage_names = [
            game_control.get_env_name(w, s) for s in range(*start_end_stages)]
        stage_names = [k[:-1] + '1' for k in stage_names]
        multi_env = game_env.ShuffleEnv(
           stage_names, game_control.SIMPLE_ACTIONS, is_human_view=is_human_view)
        for i in range(*start_end_stages):
            glance_env(multi_env, f"{_GIF_PATH}/glance{w}{i}.gif",
                       is_human_view=is_human_view)
        hboxs.append(wrapper_gifs_horizontal([
           f"{_GIF_PATH}/glance{w}{i}.gif" for i in range(*start_end_stages)]))
    return hboxs


def glance_env(env, gif_path, is_human_view=True, sample_steps=40, action=0):
    """Take a glance of the world and save as gif."""
    images = []
    obs = env.reset()
    done = False
    for i in range(sample_steps):
        screen = env.render(mode='rgb_array')
        if is_human_view:
          images.append(Image.fromarray(screen))
        else:
          images.append(_cast_to_ai_view(screen))
        for _ in range(4): # skip frames for efficiency
            if done:
                try:
                  # try multi-env within-session reset first
                  env.env.reset()
                except:
                  env.reset()
                break
            obs, reward, done, info = env.step(action)
    if len(images) > 1:
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], loop=0, duration=1)
    else:
        raise("Bad environment, cannot move")


def _cast_to_ai_view(screen_frame):
    im = Image.fromarray(screen_frame)
    im = im.resize((84, 84,))
    return ImageOps.grayscale(im)


def wrapper_gifs_horizontal(gift_list):
    """Wraps a list of gifs into a horizontal box to prepare for visualization."""
    hbox = HBox([widgets.Image(value=open(g, 'rb').read()) for g in gift_list])
    return hbox
