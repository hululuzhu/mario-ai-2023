"""Everything related to visualization in colab notebook."""
from PIL import Image
import game_env
from ipywidgets import widgets, HBox


def glance_env(env, gif_path, sample_steps=40, action=0):
    """Take a glance of the world and save as gif."""
    images = []
    obs = env.reset()
    done = False
    for i in range(sample_steps):
        screen = env.render(mode='rgb_array')
        images.append(Image.fromarray(screen))
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


def wrapper_gifs_horizontal(gift_list):
    """Wraps a list of gifs into a horizontal box to prepare for visualization."""
    hbox = HBox([widgets.Image(value=open(g, 'rb').read()) for g in gift_list])
    return hbox
