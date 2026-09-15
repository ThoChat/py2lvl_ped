import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from shapely import wkt
import sys

GS_SCALING_FACTOR = 0.26 / (2 * 0.3 * 1.65)


def create_visualisator(simulation_file_name, saving_file_name=""):
    conn = sqlite3.connect(simulation_file_name)
    cursor = conn.cursor()

    # get all stored geometries
    cursor.execute("SELECT wkt FROM geometry")
    rows = cursor.fetchall()

    geometries = [wkt.loads(r[0]) for r in rows]

    # Get the frame data
    cursor.execute("SELECT * FROM trajectory_data ORDER BY frame")
    frame_data = cursor.fetchall()

    # Get bounding box
    def get_meta(key):
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        return float(cursor.fetchone()[0])

    xmin = get_meta("xmin")
    xmax = get_meta("xmax")
    ymin = get_meta("ymin")
    ymax = get_meta("ymax")

    max_frame = max(frame_data, key=lambda x: x[0])[0]

    # Figure + axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=60, roll=0)

    # Play/Pause button
    ax_play = plt.axes([0.8, 0.025, 0.1, 0.04])
    button_play = Button(ax_play, "Play")
    is_playing = True

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.55, 0.03])
    slider = Slider(ax_slider, "Frame", 1, max_frame, valinit=1, valfmt="%0.0f")

    # Set view params
    def set_plot_view_param():
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(0, 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_zticklabels([])

    # Plot function
    def plot_frame(frame):
        ax.cla()
        set_plot_view_param()

        # All agents in this frame
        agent_data = [row for row in frame_data if row[0] == frame]

        for agent in agent_data:

            pos_x, pos_y = agent[2], agent[3]
            pos_gs_x, pos_gs_y = agent[6], agent[7]
            height = agent[10]
            radius = agent[11]

            # Create the upper body circle
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_pos_x = pos_x + radius * np.cos(theta)
            circle_pos_y = pos_y + radius * np.sin(theta)
            circle_pos_z = np.full_like(circle_pos_x, 1.5)  # Keep circle at 1.5m

            ax.plot(
                circle_pos_x,
                circle_pos_y,
                circle_pos_z,
                "k--",
                linewidth=1,
            )

            # Create the lower circle ~ BoS
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_gs_x = pos_gs_x + radius * GS_SCALING_FACTOR * height * np.cos(theta)
            circle_gs_y = pos_gs_y + radius * GS_SCALING_FACTOR * height * np.sin(theta)
            circle_gs_z = np.full_like(circle_gs_x, 0)  # Keep circle on the ground

            ax.plot(
                circle_gs_x,
                circle_gs_y,
                circle_gs_z,
                "k--",
                linewidth=1,
            )

            # Draw a link between the circles
            ax.plot(
                [pos_x, pos_gs_x],
                [pos_y, pos_gs_y],
                [1.5, 0],
                "b-",
            )

            # Plot all the geometries on the ground
            for geom in geometries:
                if geom.geom_type == "Polygon":
                    x, y = geom.exterior.xy
                    z = np.zeros_like(x)
                    ax.plot(x, y, z, color="blue", alpha=0.6, linewidth=1, zorder=1)
                    for interior in geom.interiors:
                        x, y = interior.xy
                        ax.plot(x, y, color="black", alpha=0.6, linewidth=1, zorder=1)

        # ax.set_title(f"Frame {frame}")

    # Slider update callback
    def slider_update(val):
        frame = int(slider.val)
        plot_frame(frame)
        fig.canvas.draw_idle()

    slider.on_changed(slider_update)

    # Animate callback
    def animate(i):
        nonlocal is_playing
        if is_playing:
            current_frame = int(slider.val)
            next_frame = (current_frame + 1) if current_frame < max_frame else 1
            slider.set_val(next_frame)

    # Button callback
    def toggle_play(event):
        nonlocal is_playing
        is_playing = not is_playing
        button_play.label.set_text("Pause" if is_playing else "Play")

    button_play.on_clicked(toggle_play)

    # Initialize plot
    plot_frame(1)

    # Animation loop
    ani = animation.FuncAnimation(
        fig, animate, frames=max_frame, interval=50, cache_frame_data=False
    )

    if saving_file_name != "":
        writer = animation.FFMpegWriter(fps=30, metadata=dict(artist="Me"))
        ani.save(saving_file_name, writer=writer)
    else:
        plt.show()


if __name__ == "__main__":
    file_name = sys.argv[1]
    saving_file_name = sys.argv[2] if len(sys.argv) > 2 else ""
    create_visualisator(file_name, saving_file_name)