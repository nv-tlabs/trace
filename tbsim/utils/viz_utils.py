import numpy as np

import os

from scipy.signal import savgol_filter
import subprocess
import matplotlib as mpl
mpl.use('Agg') # NOTE: very important to avoid matplotlib memory leak when looping and plotting
import matplotlib.pyplot as plt

import matplotlib.collections as mcoll
import matplotlib.patches as patches

from trajdata.simulation.sim_df_cache import SimulationDataFrameCache
from trajdata import UnifiedDataset

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.geometry_utils import transform_points

COLORS = {
    "agent_contour": "#247BA0",
    "agent_fill": "#56B1D8",
    "ego_contour": "#911A12",
    "ego_fill": "#FE5F55",
}


def get_agt_color(agt_idx):
    # agt_colors = ["grey", "purple", "blue", "green", "orange", "red"]
    agt_colors = ["grey", "orchid", "royalblue", "limegreen", "gold", "salmon"]
    return agt_colors[agt_idx % len(agt_colors)]

def get_agt_cmap(agt_idx):
    agt_colors = ["Greys", "Purples", "Blues", "Greens", "Oranges", "Reds"]
    return agt_colors[agt_idx % len(agt_colors)]

def get_group_color(agt_idx):
    # agt_colors = ["grey", "purple", "blue", "green", "orange", "red"]
    group_colors = ["red", "green", "blue", "orange"]
    return group_colors[agt_idx % len(group_colors)]

def colorline(
        ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def get_trajdata_renderer(desired_data, data_dirs,
                            raster_size=500, px_per_m=5, 
                            rebuild_maps=False,
                            cache_location='./unified_data_cache'):
    kwargs = dict(
        cache_location=cache_location,
        desired_data=desired_data,
        data_dirs=data_dirs,
        future_sec=(1.5, 1.5),
        history_sec=(1.0, 1.0),
        incl_map=True, # so that maps will be cached with correct resolution
        map_params={
                "px_per_m": px_per_m,
                "map_size_px": raster_size,
                "return_rgb": True,
        },
        num_workers=os.cpu_count(),
        desired_dt=0.1,
        rebuild_cache=rebuild_maps,
        rebuild_maps=rebuild_maps
    )

    dataset = UnifiedDataset(**kwargs)
    renderer = UnifiedRenderer(dataset, raster_size=raster_size, resolution=px_per_m)

    return renderer

class UnifiedRenderer(object):
    def __init__(self, dataset, raster_size=500, resolution=2):
        self.dataset = dataset
        self.raster_size = raster_size
        self.resolution = resolution
        num_total_scenes = dataset.num_scenes()
        scene_info = dict()
        for i in range(num_total_scenes):
            si = dataset.get_scene(i)
            scene_info[si.name] = si
        self.scene_info = scene_info

    def do_render_map(self, scene_name):
        """ whether the map needs to be rendered """
        return SimulationDataFrameCache.are_maps_cached(self.dataset.cache_path,
                                                              self.scene_info[scene_name].env_name)

    def render(self, ras_pos, ras_yaw, scene_name):
        scene_info = self.scene_info[scene_name]
        cache = SimulationDataFrameCache(
            self.dataset.cache_path,
            scene_info,
            0,
            self.dataset.augmentations,
        )
        render_map = self.do_render_map(scene_name)

        state_im = np.ones((self.raster_size, self.raster_size, 3))
        if render_map:
            patch_data, _, _ = cache.load_map_patch(
                ras_pos[0],
                ras_pos[1],
                self.raster_size,
                self.resolution,
                (0, 0),
                ras_yaw,
                return_rgb=True
            )
            # drivable area
            state_im[patch_data[0] > 0] = np.array([200, 211, 213]) / 255.
            # road/lane dividers
            state_im[patch_data[1] > 0] = np.array([164, 184, 196]) / 255.
            # crosswalks and sidewalks
            state_im[patch_data[2] > 0] = np.array([96, 117, 138]) / 255.

        raster_from_agent = np.array([
            [self.resolution, 0, 0.5 * self.raster_size],
            [0, self.resolution, 0.5 * self.raster_size],
            [0, 0, 1]
        ])

        world_from_agent: np.ndarray = np.array(
            [
                [np.cos(ras_yaw), np.sin(ras_yaw), ras_pos[0]],
                [-np.sin(ras_yaw), np.cos(ras_yaw), ras_pos[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        agent_from_world = np.linalg.inv(world_from_agent)

        raster_from_world = raster_from_agent @ agent_from_world

        del cache

        return state_im, raster_from_world

def draw_trajectories(ax, trajectories, raster_from_world, linewidth):
    raster_trajs = transform_points(trajectories, raster_from_world)
    for traj in raster_trajs:
        colorline(
            ax,
            traj[..., 0],
            traj[..., 1],
            cmap="viridis",
            linewidth=linewidth
        )

def draw_action_traj(ax, action_traj, raster_from_world, world_from_agent, linewidth,alpha=0.9):
    world_trajs = GeoUtils.batch_nd_transform_points_np(action_traj[np.newaxis],world_from_agent[np.newaxis])
    raster_trajs = GeoUtils.batch_nd_transform_points_np(world_trajs, raster_from_world[np.newaxis])
    raster_trajs = TensorUtils.join_dimensions(raster_trajs,0,2)
    colorline(
        ax,
        raster_trajs[..., 0],
        raster_trajs[..., 1],
        cmap="viridis",
        linewidth=linewidth,
        alpha=alpha,
    )

def draw_action_samples(ax, action_samples, raster_from_world, world_from_agent, linewidth,alpha=0.5, cmap="RdPu"):
    world_trajs = GeoUtils.batch_nd_transform_points_np(action_samples, world_from_agent[np.newaxis,np.newaxis])
    raster_trajs = GeoUtils.batch_nd_transform_points_np(world_trajs, raster_from_world[np.newaxis,np.newaxis])
    for traj_i in range(raster_trajs.shape[1]):
        colorline(
            ax,
            raster_trajs[:, traj_i, 0],
            raster_trajs[:, traj_i, 1],
            cmap=cmap,
            linewidth=linewidth,
            alpha=alpha,
        )

def draw_agent_boxes_plt(ax, pos, yaw, extent, raster_from_agent,
                         outline_colors=None,
                         outline_widths=None,
                         fill_colors=None,
                         mark_agents=None):
    if fill_colors is not None:
        assert len(fill_colors) == pos.shape[0]
    if outline_colors is not None:
        assert len(outline_colors) == pos.shape[0]
    if outline_widths is not None:
        assert len(outline_widths) == pos.shape[0]
    boxes = GeoUtils.get_box_world_coords_np(pos, yaw, extent)
    boxes_raster = transform_points(boxes, raster_from_agent)
    boxes_raster = boxes_raster.reshape((-1, 4, 2))
    if mark_agents is not None:
        # tuple of agt_idx, marker style, color
        mark_agents = {
            v[0] : (v[1], v[2]) for v in mark_agents
        }
    else:
        mark_agents = dict()
    for bi, b in enumerate(boxes_raster):
        cur_fill_color = get_agt_color(bi) if fill_colors is None else fill_colors[bi]
        cur_outline_color = "grey" if outline_colors is None else outline_colors[bi]
        cur_outline_width = 0.5 if outline_widths is None else outline_widths[bi]
        rect = patches.Polygon(b, fill=True, color=cur_fill_color, zorder=1)
        rect_border = patches.Polygon(b, fill=False, color=cur_outline_color, zorder=1, linewidth=cur_outline_width)
        ax.add_patch(rect)
        ax.add_patch(rect_border)

        if bi in mark_agents:
            mark_pos = np.mean(b, axis=0)
            ax.scatter(mark_pos[0], mark_pos[1], marker=mark_agents[bi][0], color=mark_agents[bi][1], s=10.0, zorder=1)

        # identify agt idx
        # mark_pos = np.mean(b, axis=0)
        # ax.text(mark_pos[0], mark_pos[1], str(bi), c='r')

def draw_constraint(ax, loc, rel_time, max_time, raster_from_world, world_from_agent, marker_color='r'):
    if world_from_agent is not None:
        world_loc = GeoUtils.batch_nd_transform_points_np(loc[np.newaxis], world_from_agent)
    else:
        world_loc = loc[np.newaxis]
    raster_loc = GeoUtils.batch_nd_transform_points_np(world_loc, raster_from_world)[0]
    if rel_time is not None:
        cmap = mpl.cm.get_cmap('viridis')
        ax.plot(raster_loc[0:1], raster_loc[1:2], 'x', color=cmap(float(rel_time)/max_time))
    # ax.scatter(raster_loc[0:1], raster_loc[1:2], color=marker_color, s=16.0, edgecolors='red', zorder=3)
    ax.scatter(raster_loc[0:1], raster_loc[1:2], color=marker_color, s=16.0, zorder=3)


def draw_scene_data(ax, scene_name, scene_data, starting_frame, rasterizer, 
                    guidance_config=None,
                    draw_trajectory=True,
                    draw_action=True,
                    draw_diffusion_step=None,
                    n_step_action=5,
                    draw_action_sample=False,
                    traj_len=200,
                    ras_pos=None,
                    linewidth=3.0):
    t = starting_frame
    if ras_pos is None:
        ras_pos = np.mean(scene_data["centroid"][:,0], axis=0)

    state_im, raster_from_world = rasterizer.render(
        ras_pos=ras_pos,
        ras_yaw=0,
        scene_name=scene_name
    )
    extent_scale = 1.0

    ax.imshow(state_im)

    if draw_action_sample == True and "action_sample_positions" in scene_data:
        NA = scene_data["action_sample_positions"].shape[0]
        for aidx in range(NA):
            draw_action_samples(
                ax,
                action_samples=scene_data["action_sample_positions"][aidx, t],
                raster_from_world=raster_from_world,
                # actions are always wrt to the frame that they were planned in
                world_from_agent = scene_data["world_from_agent"][aidx,::n_step_action][int(t/n_step_action)],
                linewidth=linewidth*0.5,
                alpha=0.3,
                cmap=get_agt_cmap(aidx),
            )

    if draw_action and "action_traj_positions" in scene_data:
        NA = scene_data["action_traj_positions"].shape[0]
        for aidx in range(NA):
            draw_action_traj(
                ax,
                action_traj=scene_data["action_traj_positions"][aidx, t],
                raster_from_world=raster_from_world,
                # actions are always wrt to the frame that they were planned in
                world_from_agent = scene_data["world_from_agent"][aidx,::n_step_action][int(t/n_step_action)],
                linewidth=linewidth*0.75
            )

    if draw_trajectory and "centroid" in scene_data:
        draw_trajectories(
            ax,
            trajectories=scene_data["centroid"][:, t:t+traj_len],
            raster_from_world=raster_from_world,
            linewidth=linewidth
        )

    if draw_diffusion_step is not None and "diffusion_steps_traj" in scene_data:
        NA = scene_data["diffusion_steps_traj"].shape[0]
        for aidx in range(NA):
            draw_action_traj(
                ax,
                action_traj=scene_data["diffusion_steps_traj"][aidx, t, :, draw_diffusion_step, :2], # positions
                raster_from_world=raster_from_world,
                # actions are always wrt to the frame that they were planned in
                world_from_agent = scene_data["world_from_agent"][aidx,::n_step_action][int(t/n_step_action)],
                linewidth=linewidth*0.5
            )

    # agent drawing colors (may be modified by guidance config)
    fill_colors = np.array([get_agt_color(aidx) for aidx in range(scene_data["centroid"].shape[0])])
    outline_colors = np.array([COLORS["agent_contour"] for _ in range(scene_data["centroid"].shape[0])])
    outline_widths = np.array([0.5 for _ in range(scene_data["centroid"].shape[0])])
    mark_agents = []

    if guidance_config is not None:
        social_group_cnt = 0
        for cur_guide in guidance_config:
            if cur_guide['name'] == 'target_pos_at_time':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    # only plot if waypoint is coming in future
                    rel_time = cur_guide['params']['target_time'][aidx] - t
                    if rel_time >= 0:
                        draw_constraint(ax, 
                                        np.array(cur_guide['params']['target_pos'][aidx]),
                                        # conver to "global" timestamp
                                        rel_time,
                                        n_step_action,
                                        raster_from_world,
                                        # constraints are always wrt to the local planning frame of agent
                                        scene_data["world_from_agent"][saidx,::n_step_action][int(t/n_step_action)],
                                        marker_color=get_agt_color(saidx))
            elif cur_guide['name'] == 'target_pos':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    # only plot if waypoint is coming in future
                    draw_constraint(ax, 
                                    np.array(cur_guide['params']['target_pos'][aidx]),
                                    None,
                                    n_step_action,
                                    raster_from_world,
                                    # constraints are always wrt to the local planning frame of agent
                                    scene_data["world_from_agent"][saidx,::n_step_action][int(t/n_step_action)],
                                    marker_color=get_agt_color(saidx))
            elif cur_guide['name'] == 'global_target_pos_at_time':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    # only plot if waypoint is coming in future
                    rel_time = cur_guide['params']['target_time'][aidx] - t
                    if rel_time >= 0:
                        draw_constraint(ax, 
                                        np.array(cur_guide['params']['target_pos'][aidx]),
                                        # conver to "global" timestamp
                                        rel_time,
                                        n_step_action,
                                        raster_from_world,
                                        # global constraints are already in world frame
                                        None,
                                        marker_color=get_agt_color(saidx))
            elif cur_guide['name'] == 'global_target_pos':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    draw_constraint(ax, 
                                    np.array(cur_guide['params']['target_pos'][aidx]),
                                    None,
                                    n_step_action,
                                    raster_from_world,
                                    None,
                                    marker_color=get_agt_color(saidx))
            elif cur_guide['name'] == 'social_group':
                cur_group_color = get_group_color(social_group_cnt)
                outline_colors[cur_guide['agents']] = cur_group_color
                outline_widths[cur_guide['agents']] = 1.0
                # mark to denote leader
                mark_agents.append((cur_guide['params']['leader_idx'], "*", cur_group_color))

                social_group_cnt += 1

    draw_agent_boxes_plt(
        ax,
        pos=scene_data["centroid"][:, t],
        yaw=scene_data["yaw"][:, [t]],
        extent=scene_data["extent"][:, t, :2] * extent_scale,
        raster_from_agent=raster_from_world,
        outline_colors=outline_colors.tolist(),
        outline_widths=outline_widths.tolist(),
        fill_colors=fill_colors.tolist(),
        mark_agents=mark_agents,
    )

    ax.set_xlim([0, state_im.shape[1]])
    ax.set_ylim([0, state_im.shape[0]])
    if not rasterizer.do_render_map(scene_name):
        ax.grid(True)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    else:
        ax.axis("off")
        # ax.grid(False)
        ax.grid(True)

    del state_im

def preprocess(scene_data, filter_yaw=False):
    data = dict()
    for k in scene_data.keys():
        data[k] = scene_data[k][:].copy()

    if filter_yaw:
        data["yaw"] = savgol_filter(data["yaw"], 11, 3)
    return data

def create_video(img_path_form, out_path, fps):
    '''
    Creates a video from a format for frame e.g. 'data_out/frame%04d.png'.
    Saves in out_path.
    '''
    subprocess.run(['ffmpeg', '-y', '-r', str(fps), '-i', img_path_form, '-vf' , "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    '-vcodec', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out_path])


def scene_to_video(rasterizer, scene_data, scene_name, output_dir,
                    guidance_config=None,
                    filter_yaw=False,
                    fps=10,
                    n_step_action=5,
                    first_frame_only=False,
                    viz_traj=True,
                    sim_num=0):
    scene_data = preprocess(scene_data, filter_yaw)
    frames = [0] if first_frame_only else range(scene_data["centroid"].shape[1])
    for frame_i in frames:
        fig, ax = plt.subplots()
        draw_scene_data(
            ax,
            scene_name,
            scene_data,
            frame_i,
            rasterizer,
            guidance_config=guidance_config,
            draw_trajectory=False,
            draw_action=viz_traj,
            draw_action_sample=viz_traj,
            n_step_action=n_step_action,
            traj_len=20,
            linewidth=2.0,
            ras_pos=np.mean(scene_data["centroid"][:, 0], axis=0)
        )

        if first_frame_only:
            ffn = os.path.join(output_dir, "{sname}_{simnum:04d}_{framei:03d}.png").format(sname=scene_name, simnum=sim_num, framei=frame_i)
        else:
            video_dir = os.path.join(output_dir, scene_name + '_%04d' % (sim_num))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            ffn = os.path.join(video_dir, "{:03d}.png").format(frame_i)
        plt.savefig(ffn, dpi=200, bbox_inches="tight", pad_inches=0)
        print("Figure written to {}".format(ffn))
        fig.clf()
        plt.close(fig)

    if not first_frame_only:
        create_video(os.path.join(video_dir, "%03d.png"), video_dir + ".mp4", fps=fps)

def draw_diffusion_prof(ax, scene_data, frame_i, diffusion_step,
                        val_inds=[4,5],
                        val_names=['acc', 'yawvel']):
    assert len(val_inds) == len(val_names)
    t = frame_i
    NT = scene_data["diffusion_steps_traj"].shape[2]
    NA = scene_data["diffusion_steps_traj"].shape[0]
    for aidx in range(NA):
        for cidx, vinfo in enumerate(zip(val_inds, val_names)):
            vidx, vname = vinfo
            ax[aidx,cidx].plot(np.arange(NT), scene_data["diffusion_steps_traj"][aidx, t, :, diffusion_step, vidx],
                                c=plt.rcParams['axes.prop_cycle'].by_key()['color'][aidx % 9])
            ax[aidx,cidx].set_ylabel(vname + " agent %d" % (aidx))

def scene_diffusion_video(rasterizer, scene_data, scene_name, output_dir,
                             n_step_action=5,
                             viz_traj=True,
                             viz_prof=False):
    scene_data = preprocess(scene_data, False)
    assert "diffusion_steps_traj" in scene_data
    print(scene_data["diffusion_steps_traj"].shape)
    num_diff_steps = scene_data["diffusion_steps_traj"].shape[-2]
    NA = scene_data["diffusion_steps_traj"].shape[0]
    # first acceleration profiles
    if viz_prof:
        for frame_i in range(0, 1): #range(0, scene_data["diffusion_steps_traj"].shape[1], n_step_action):
            video_dir = os.path.join(output_dir, scene_name + "_diffusion_ctrl_%03d" % (frame_i))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            for diff_step in range(num_diff_steps):
                num_col = 3
                fig, ax = plt.subplots(NA, num_col)
                draw_diffusion_prof(
                    ax,
                    scene_data,
                    frame_i,
                    diff_step,
                    val_inds=[2, 4, 5], # which indices of the state to plot
                    val_names=['vel', 'acc', 'yawvel']
                )
                plt.subplots_adjust(right=1.3)
                ffn = os.path.join(video_dir, "{:05d}.png").format(diff_step)
                plt.savefig(ffn, dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                print("Figure written to {}".format(ffn))

            per_frame_len_sec = 4.0
            create_video(os.path.join(video_dir, "%05d.png"), video_dir + ".mp4", fps=(num_diff_steps/per_frame_len_sec))

    if viz_traj:
        # then state trajectories (resulting from accel)
        for frame_i in range(0, 1): #range(0, scene_data["diffusion_steps_traj"].shape[1], n_step_action):
            video_dir = os.path.join(output_dir, scene_name + "_diffusion_%03d" % (frame_i))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            for diff_step in range(num_diff_steps):
                fig, ax = plt.subplots()
                draw_scene_data(
                    ax,
                    scene_name,
                    scene_data,
                    frame_i,
                    rasterizer,
                    draw_trajectory=False,
                    draw_action=False,
                    draw_action_sample=False,
                    draw_diffusion_step=diff_step,
                    n_step_action=n_step_action,
                    traj_len=20,
                    linewidth=2.0,
                    ras_pos=scene_data["centroid"][0, 0]
                )
                ffn = os.path.join(video_dir, "{:05d}.png").format(diff_step)
                plt.savefig(ffn, dpi=212, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)
                print("Figure written to {}".format(ffn))

            per_frame_len_sec = 4.0
            create_video(os.path.join(video_dir, "%05d.png"), video_dir + ".mp4", fps=(num_diff_steps/per_frame_len_sec))

def visualize_guided_rollout(output_dir, rasterizer, si, scene_data,
                            guidance_config=None,
                            filter_yaw=False,
                            fps=10,
                            n_step_action=5,
                            viz_diffusion_steps=False,
                            first_frame_only=False,
                            viz_traj=True,
                            sim_num=0):
    '''
    guidance configs are for the given scene ONLY.
    '''
    if viz_diffusion_steps:
        print('Visualizing diffusion for %s...' % (si))
        scene_diffusion_video(rasterizer, scene_data, si, output_dir,
                                n_step_action=n_step_action,
                                viz_prof=False,
                                viz_traj=True)
    print('Visualizing rollout for %s...' % (si))
    scene_to_video(rasterizer, scene_data, si, output_dir,
                    guidance_config=guidance_config,
                    filter_yaw=filter_yaw,
                    fps=fps,
                    n_step_action=n_step_action,
                    first_frame_only=first_frame_only,
                    viz_traj=viz_traj,
                    sim_num=sim_num)
